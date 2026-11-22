"""Qwen3-1.7B-Base architecture and portable local initialization assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex


@dataclass(frozen=True)
class ModelProfile:
    size: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    tie_word_embeddings: bool
    tokenizer_model: str
    load_checkpoint: str

    @property
    def model_id(self) -> str:
        return f"Qwen/Qwen3-{self.size}-Base"

    @property
    def context_length(self) -> int:
        # The local Qwen3 Base assets currently used by this run all declare
        # 32K positions. The CLI resolver below prefers the exact value in the
        # selected template when a config is available.
        return 32768

    @property
    def tensor_parallel_size(self) -> int:
        return 1

    @property
    def context_parallel_size(self) -> int:
        return 4

    @property
    def sft_memory_gib(self) -> int:
        return 128

    @property
    def hf_config(self) -> dict:
        return {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    @property
    def conversion_args(self) -> tuple[str, ...]:
        args = (
            "--num-layers",
            str(self.num_hidden_layers),
            "--hidden-size",
            str(self.hidden_size),
            "--ffn-hidden-size",
            str(self.intermediate_size),
            "--num-attention-heads",
            str(self.num_attention_heads),
            "--num-query-groups",
            "8",
            "--kv-channels",
            "128",
        )
        if not self.tie_word_embeddings:
            args += ("--untie-embeddings-and-output-weights",)
        return args

    @property
    def training_args(self) -> tuple[str, ...]:
        return (
            *self.conversion_args,
            "--vocab-size",
            "151936",
            "--make-vocab-size-divisible-by",
            "1",
        )

    @property
    def training_script(self) -> str:
        return "examples/qwen3/train_qwen3_1.7b.128k.sh"


MODEL_PROFILES = {
    "1.7b": ModelProfile(
        "1.7B",
        2048,
        6144,
        28,
        16,
        True,
        "models/Qwen3-1.7B-Base",
        "models/qwen3_1.7b_base.tp1",
    ),
}


def get_model_profile(model: str | None = None) -> ModelProfile:
    value = model if model is not None else os.environ.get("AUTO_SFT_SFT_MODEL", "1.7b")
    key = value.strip().lower()
    for alias, profile in MODEL_PROFILES.items():
        if key in (
            alias,
            profile.model_id.lower(),
            profile.model_id.split("/")[1].lower(),
        ):
            return profile
    raise ValueError(
        f"Unsupported AUTO_SFT_SFT_MODEL={value!r}; this local release supports 1.7b "
        "(or Qwen/Qwen3-1.7B-Base)."
    )


def resolve_model_assets(profile: ModelProfile | None = None) -> tuple[str, str]:
    profile = profile or get_model_profile()
    root = Path(__file__).resolve().parents[4]
    template = (
        os.environ.get("AUTO_SFT_SFT_TOKENIZER_MODEL")
        or os.environ.get("AUTO_SFT_BASE_MODEL_PATH")
        or root / profile.tokenizer_model
    )
    checkpoint = (
        os.environ.get("AUTO_SFT_SFT_BASE_CKPT")
        or os.environ.get("AUTO_SFT_BASE_CHECKPOINT")
        or root / profile.load_checkpoint
    )
    return (
        str(Path(template).expanduser().resolve()),
        str(Path(checkpoint).expanduser().resolve()),
    )


def resolve_context_length(profile: ModelProfile | None = None) -> int:
    """Read the selected template's context limit, with a profile fallback."""
    profile = profile or get_model_profile()
    template, _ = resolve_model_assets(profile)
    try:
        config = json.loads(
            (Path(template) / "config.json").read_text(encoding="utf-8")
        )
        value = (
            config.get("max_position_embeddings") if isinstance(config, dict) else None
        )
        if type(value) is int and value > 0:
            return value
    except (OSError, ValueError):
        pass
    return profile.context_length


def validate_model_assets(
    profile: ModelProfile | None = None,
    *,
    tokenizer_model: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Fail before starting a run if its local template/checkpoint is absent."""
    profile = profile or get_model_profile()
    resolved_template, resolved_checkpoint = resolve_model_assets(profile)
    template = Path(tokenizer_model or resolved_template).expanduser().resolve()
    checkpoint = Path(checkpoint_path or resolved_checkpoint).expanduser().resolve()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (template / name).is_file():
            raise ValueError(
                f"{profile.model_id}: missing {template / name}; set "
                "AUTO_SFT_SFT_TOKENIZER_MODEL to the matching local HuggingFace model directory."
            )
    try:
        config = json.loads((template / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid model config: {template / 'config.json'}") from exc
    for field, expected in profile.hf_config.items():
        if not isinstance(config, dict) or config.get(field) != expected:
            raise ValueError(
                f"{profile.model_id}: {template / 'config.json'} must have {field}={expected!r}."
            )
    if not checkpoint.is_dir():
        raise ValueError(
            f"{profile.model_id}: base checkpoint directory not found: {checkpoint}; "
            "set AUTO_SFT_SFT_BASE_CKPT to the matching Megatron base checkpoint directory."
        )
    marker = checkpoint / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise ValueError(
            f"{profile.model_id}: checkpoint is missing latest_checkpointed_iteration.txt: {checkpoint}"
        )
    step = marker.read_text(encoding="utf-8").strip()
    if step == "release":
        if not (checkpoint / "release/mp_rank_00/model_optim_rng.pt").is_file():
            raise ValueError(
                f"{profile.model_id}: release checkpoint has no rank 0 weights: {checkpoint}"
            )
    elif step.isdecimal() and int(step) > 0:
        iteration = checkpoint / f"iter_{int(step):07d}"
        if not iteration.is_dir() or not any(iteration.iterdir()):
            raise ValueError(
                f"{profile.model_id}: checkpoint iteration is missing or empty: {iteration}"
            )
    else:
        raise ValueError(f"{profile.model_id}: invalid checkpoint marker: {marker}")
    manifest_path = checkpoint / "import_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid Megatron import manifest: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid Megatron import manifest: {manifest_path}")
        # The original public importer recorded the exact config before it
        # included model_id. Accept those complete 1.7B imports on recovery.
        imported_config = manifest.get("config", {})
        model_id = manifest.get("model_id")
        if (
            model_id is None
            and isinstance(imported_config, dict)
            and all(
                imported_config.get(key) == value
                for key, value in profile.hf_config.items()
            )
        ):
            model_id = profile.model_id
        if model_id != profile.model_id:
            raise ValueError(
                f"{profile.model_id}: checkpoint manifest model_id does not match: {manifest_path}"
            )
        configured_tp = int(
            os.environ.get("AUTO_SFT_SFT_TP", str(profile.tensor_parallel_size))
        )
        if manifest.get("tensor_parallel_size") != configured_tp:
            raise ValueError(
                f"{profile.model_id}: checkpoint TP does not match training TP={configured_tp}: {manifest_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--field",
        choices=(
            "model_id",
            "tokenizer_model",
            "load_checkpoint",
            "training_script",
            "training_args",
            "conversion_args",
            "context_length",
            "tensor_parallel_size",
            "context_parallel_size",
            "sft_memory_gib",
        ),
    )
    output.add_argument(
        "--shell",
        action="store_true",
        help="Print shell-quoted run environment exports.",
    )
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args()
    try:
        profile = get_model_profile()
        if args.check_assets:
            validate_model_assets(profile)
        template, checkpoint = resolve_model_assets(profile)
    except ValueError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    values = {
        "model_id": profile.model_id,
        "tokenizer_model": template,
        "load_checkpoint": checkpoint,
        "training_script": profile.training_script,
        "training_args": profile.training_args,
        "conversion_args": profile.conversion_args,
        "context_length": resolve_context_length(profile),
        "tensor_parallel_size": profile.tensor_parallel_size,
        "context_parallel_size": profile.context_parallel_size,
        "sft_memory_gib": profile.sft_memory_gib,
    }
    if args.shell:
        for key, value in (
            ("AUTO_SFT_SFT_MODEL", profile.model_id),
            ("AUTO_SFT_SFT_TOKENIZER_MODEL", template),
            ("AUTO_SFT_SFT_BASE_CKPT", checkpoint),
        ):
            print(f"export {key}={shlex.quote(value)}")
    elif args.field:
        value = values[args.field]
        print(shlex.join(value) if isinstance(value, tuple) else value)
    else:
        print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
