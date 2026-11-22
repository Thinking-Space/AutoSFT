#!/usr/bin/env python3
"""Initialize a TP=1, PP=1 Megatron Qwen3 checkpoint from local public HF weights."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from autosft.core.common import REPO_ROOT as ROOT

from autosft.backends.megatron.convert import validate_hf_template
from autosft.backends.megatron.model_profiles import get_model_profile


def convert_weights(get_tensor, config):
    import torch

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    heads = config["num_attention_heads"]
    groups = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    if heads % groups:
        raise ValueError("Attention heads must be divisible by KV groups")
    model = {}

    def read(name, shape):
        tensor = get_tensor(name)
        if tuple(tensor.shape) != tuple(shape) or not tensor.is_floating_point():
            raise ValueError(
                f"Invalid HF tensor {name}: expected floating-point {shape}"
            )
        return tensor.contiguous()

    model["embedding.word_embeddings.weight"] = read(
        "model.embed_tokens.weight",
        (config["vocab_size"], hidden),
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


def import_checkpoint(source: Path, output: Path):
    import torch
    from safetensors import safe_open

    profile = get_model_profile()
    config = validate_hf_template(source, profile)
    required = {
        "head_dim": 128,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"Unsupported Qwen3 config: {name} must equal {expected}")
    if config.get("rope_scaling") is not None:
        raise ValueError(
            "Scaled RoPE is not supported by this fixed training configuration"
        )
    shard_paths = sorted(source.glob("*.safetensors"))
    if not shard_paths:
        raise ValueError("The HF model must include local safetensors weights")
    if output.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {output}")
    with ExitStack() as stack:
        tensors = {}
        for shard in shard_paths:
            reader = stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
            for key in reader.keys():
                if key in tensors:
                    raise ValueError(f"Duplicate HF tensor: {key}")
                tensors[key] = reader
        consumed = set()

        def get_tensor(name):
            consumed.add(name)
            return tensors[name].get_tensor(name)

        model = convert_weights(get_tensor, config)
        if "lm_head.weight" in tensors:
            if not torch.equal(
                get_tensor("lm_head.weight"), model["embedding.word_embeddings.weight"]
            ):
                raise ValueError("Tied HF output and embedding weights differ")
        if set(tensors) != consumed:
            raise ValueError(f"Unmapped HF tensors: {sorted(set(tensors) - consumed)}")
        destination = output / "release/mp_rank_00/model_optim_rng.pt"
        destination.parent.mkdir(parents=True)
        torch.save(
            {"model": model, "checkpoint_version": 3.0, "iteration": 0}, destination
        )
    (output / "import_manifest.json").write_text(
        json.dumps(
            {
                "format": "megatron-release",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "model_type": "qwen3",
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
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(import_checkpoint(args.model.resolve(), args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
