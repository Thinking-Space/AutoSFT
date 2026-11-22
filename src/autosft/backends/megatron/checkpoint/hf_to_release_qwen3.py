#!/usr/bin/env python3
"""Import local Qwen3 HuggingFace safetensors as a Megatron release checkpoint.

The training image can load a release checkpoint and fine-tune it with a
different data/tensor/context parallel layout. This importer writes one
legacy ``mp_rank_XX`` file per requested tensor-parallel rank; it does not
write optimizer state because the caller uses ``--finetune``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from autosft.backends.megatron.convert import validate_hf_template  # noqa: E402
from autosft.backends.megatron.model_profiles import get_model_profile  # noqa: E402


def _split(tensor, parts: int, dim: int, *, name: str):
    if tensor.shape[dim] % parts:
        raise ValueError(
            f"{name} dimension {tensor.shape[dim]} is not divisible by TP={parts}"
        )
    # clone(): a dim-0 chunk is already contiguous, so contiguous() alone
    # would return a view and torch.save would write the whole unsplit
    # storage into every rank file.
    return tuple(piece.contiguous().clone() for piece in tensor.chunk(parts, dim=dim))


def _full_model(get_tensor, config):
    """Map HF Qwen3 names to unsharded Megatron Qwen3 release names."""
    import torch

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    heads = config["num_attention_heads"]
    groups = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    if heads % groups:
        raise ValueError("attention heads must be divisible by KV groups")
    model = {}

    def read(name, shape):
        tensor = get_tensor(name)
        if tuple(tensor.shape) != tuple(shape) or not tensor.is_floating_point():
            raise ValueError(
                f"invalid HF tensor {name}: expected floating-point {shape}"
            )
        return tensor.contiguous()

    model["embedding.word_embeddings.weight"] = read(
        "model.embed_tokens.weight", (config["vocab_size"], hidden)
    )
    model["decoder.final_layernorm.weight"] = read("model.norm.weight", (hidden,))
    model["decoder.final_layernorm._extra_state"] = torch.empty(0, dtype=torch.uint8)
    for layer in range(config["num_hidden_layers"]):
        source = f"model.layers.{layer}"
        target = f"decoder.layers.{layer}"
        query = read(f"{source}.self_attn.q_proj.weight", (heads * head_dim, hidden))
        key = read(f"{source}.self_attn.k_proj.weight", (groups * head_dim, hidden))
        value = read(f"{source}.self_attn.v_proj.weight", (groups * head_dim, hidden))
        model[f"{target}.self_attention.linear_qkv.weight"] = torch.cat(
            [
                query.reshape(groups, -1, hidden),
                key.reshape(groups, head_dim, hidden),
                value.reshape(groups, head_dim, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)
        mappings = (
            (
                "self_attention.linear_qkv.layer_norm_weight",
                "input_layernorm.weight",
                (hidden,),
            ),
            (
                "self_attention.linear_proj.weight",
                "self_attn.o_proj.weight",
                (hidden, heads * head_dim),
            ),
            (
                "self_attention.q_layernorm.weight",
                "self_attn.q_norm.weight",
                (head_dim,),
            ),
            (
                "self_attention.k_layernorm.weight",
                "self_attn.k_norm.weight",
                (head_dim,),
            ),
            (
                "mlp.linear_fc1.layer_norm_weight",
                "post_attention_layernorm.weight",
                (hidden,),
            ),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight", (hidden, intermediate)),
        )
        for destination, origin, shape in mappings:
            model[f"{target}.{destination}"] = read(f"{source}.{origin}", shape)
        model[f"{target}.mlp.linear_fc1.weight"] = torch.cat(
            [
                read(f"{source}.mlp.gate_proj.weight", (intermediate, hidden)),
                read(f"{source}.mlp.up_proj.weight", (intermediate, hidden)),
            ]
        )
        for module in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "self_attention.q_layernorm",
            "self_attention.k_layernorm",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ):
            model[f"{target}.{module}._extra_state"] = torch.empty(0, dtype=torch.uint8)
    return model


def _rank_model(full: dict, config: dict, rank: int, tp: int) -> dict:
    """Shard column/row parallel weights for one tensor-parallel rank."""
    import torch

    intermediate = config["intermediate_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    groups = config["num_key_value_heads"]
    if heads % tp or groups % tp:
        raise ValueError(
            f"heads ({heads}) and KV groups ({groups}) must divide TP={tp}"
        )

    out = {
        "embedding.word_embeddings.weight": _split(
            full["embedding.word_embeddings.weight"],
            tp,
            0,
            name="embedding.word_embeddings.weight",
        )[rank],
        "decoder.final_layernorm.weight": full["decoder.final_layernorm.weight"],
        "decoder.final_layernorm._extra_state": full[
            "decoder.final_layernorm._extra_state"
        ],
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"decoder.layers.{layer}"
        out[f"{prefix}.self_attention.linear_qkv.weight"] = _split(
            full[f"{prefix}.self_attention.linear_qkv.weight"],
            tp,
            0,
            name=f"{prefix}.linear_qkv",
        )[rank]
        out[f"{prefix}.self_attention.linear_proj.weight"] = _split(
            full[f"{prefix}.self_attention.linear_proj.weight"],
            tp,
            1,
            name=f"{prefix}.linear_proj",
        )[rank]
        # SwiGLU stores [gate; up]. Each TP rank owns matching slices of
        # both projections, rather than one contiguous slice of their concat.
        gate, up = full[f"{prefix}.mlp.linear_fc1.weight"].split(intermediate, dim=0)
        out[f"{prefix}.mlp.linear_fc1.weight"] = torch.cat(
            (
                _split(gate, tp, 0, name=f"{prefix}.gate_proj")[rank],
                _split(up, tp, 0, name=f"{prefix}.up_proj")[rank],
            ),
            dim=0,
        )
        out[f"{prefix}.mlp.linear_fc2.weight"] = _split(
            full[f"{prefix}.mlp.linear_fc2.weight"],
            tp,
            1,
            name=f"{prefix}.linear_fc2",
        )[rank]
        for name in (
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.q_layernorm.weight",
            "self_attention.k_layernorm.weight",
            "mlp.linear_fc1.layer_norm_weight",
        ):
            out[f"{prefix}.{name}"] = full[f"{prefix}.{name}"]
        for module in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "self_attention.q_layernorm",
            "self_attention.k_layernorm",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ):
            out[f"{prefix}.{module}._extra_state"] = full[
                f"{prefix}.{module}._extra_state"
            ]

    if not config["tie_word_embeddings"]:
        out["output_layer.weight"] = _split(
            full["output_layer.weight"], tp, 0, name="output_layer.weight"
        )[rank]
    return out


def import_checkpoint(
    source: Path, output: Path, tensor_parallel_size: int
) -> list[Path]:
    import torch
    from safetensors import safe_open

    profile = get_model_profile()
    config = validate_hf_template(source, profile)
    required = {
        "head_dim": 128,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"unsupported Qwen3 config: {name} must equal {expected}")
    if config.get("rope_scaling") is not None:
        raise ValueError(
            "scaled RoPE is not supported by this fixed training configuration"
        )
    if tensor_parallel_size < 1:
        raise ValueError("tensor parallel size must be positive")
    if output.exists():
        raise FileExistsError(f"checkpoint destination already exists: {output}")
    shard_paths = sorted(source.glob("*.safetensors"))
    if not shard_paths:
        raise ValueError("the HF model must include local safetensors weights")

    with ExitStack() as stack:
        readers = {}
        for shard in shard_paths:
            reader = stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
            for key in reader.keys():
                if key in readers:
                    raise ValueError(f"duplicate HF tensor: {key}")
                readers[key] = reader
        consumed: set[str] = set()

        def get_tensor(name):
            if name not in readers:
                raise ValueError(f"missing HF tensor: {name}")
            consumed.add(name)
            return readers[name].get_tensor(name)

        full = _full_model(get_tensor, config)
        if config["tie_word_embeddings"]:
            if "lm_head.weight" in readers:
                if not torch.equal(
                    get_tensor("lm_head.weight"),
                    full["embedding.word_embeddings.weight"],
                ):
                    raise ValueError("tied HF output and embedding weights differ")
        else:
            if "lm_head.weight" not in readers:
                raise ValueError("untied Qwen3 config requires lm_head.weight")
            full["output_layer.weight"] = get_tensor("lm_head.weight").contiguous()
        if set(readers) != consumed:
            raise ValueError(f"unmapped HF tensors: {sorted(set(readers) - consumed)}")

        destinations: list[Path] = []
        for rank in range(tensor_parallel_size):
            model = _rank_model(full, config, rank, tensor_parallel_size)
            destination = (
                output / "release" / f"mp_rank_{rank:02d}" / "model_optim_rng.pt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model, "checkpoint_version": 3.0, "iteration": 0}, destination
            )
            destinations.append(destination)
    (output / "import_manifest.json").write_text(
        json.dumps(
            {
                "format": "megatron-release",
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": 1,
                "model_id": profile.model_id,
                "config": config,
                "source_shards": [path.name for path in shard_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "latest_checkpointed_iteration.txt").write_text(
        "release\n", encoding="utf-8"
    )
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    destinations = import_checkpoint(
        args.model.resolve(),
        args.output.resolve(),
        args.tensor_parallel_size,
    )
    print(
        json.dumps(
            {"status": "converted", "files": [str(path) for path in destinations]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
