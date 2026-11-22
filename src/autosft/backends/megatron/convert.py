#!/usr/bin/env python3
"""Convert one complete Megatron Qwen3 Base round checkpoint to HF format.

The converter deliberately has a small, fail-closed interface.  A round is
identified by the checkpoint's final marker, and conversion destinations are
owned by that round and step.  This prevents an old experiment's model or
template from being silently reused by the current run.
"""

from __future__ import annotations
from autosft.config.components import executor_enabled

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from autosft.core.common import REPO_ROOT, utc_now, write_json
import autosft.backends.megatron.conversion_environment as _conversion_environment
from autosft.agents.executor_handoff import refresh_executor_handoff
from autosft.backends.megatron.model_profiles import (
    ModelProfile,
    get_model_profile,
    resolve_model_assets,
)


# Keep standalone conversion on the same public checkout as training.  The
# previous stub default only worked for import-only fixtures and made a fresh
# user who followed `fetch_megatron.py` fail at export unless they happened to
# set AUTO_SFT_MEGATRON_DIR manually.
DEFAULT_MEGATRON_DIR = REPO_ROOT / "third_party/megatron"
DEFAULT_HF_TEMPLATE = Path(resolve_model_assets(get_model_profile("1.7b"))[0])
ConversionEnvironmentError = _conversion_environment.ConversionEnvironmentError
SHARED_CONVERT_PYTHON = _conversion_environment.SHARED_CONVERT_PYTHON

REQUIRED_TEMPLATE_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json")
OPTIONAL_TEMPLATE_FILES = (
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
)
# Default Qwen3 checkpoint conversion settings.
QWEN3_1P7B_CONFIG: dict[str, Any] = dict(get_model_profile("1.7b").hf_config)
# SFT answers end with <|im_end|>. Include it even when the base model's
# generation settings only declare <|endoftext|> as EOS.
QWEN3_IM_END_ID = 151645
_POSITIVE_DECIMAL = re.compile(r"^[1-9][0-9]*$")
_STORAGE_ROUND = re.compile(r"^round_([1-9][0-9]*)$")


class ConversionContractError(ValueError):
    """An input does not satisfy the current conversion contract."""


def checkpoint_step(value: str) -> str:
    """Argparse type for a canonical positive decimal checkpoint step."""

    if not isinstance(value, str) or _POSITIVE_DECIMAL.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "checkpoint step must be a canonical positive integer"
        )
    return value


def check_convert_python(python_bin: str | Path, **kwargs: object) -> str:
    """Delegate to the shared checker while keeping a patchable entry point."""

    return _conversion_environment.check_convert_python(python_bin, **kwargs)


def resolve_convert_python(
    *,
    check_compatibility: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str:
    """Choose the configured, current, or fallback conversion environment."""

    return _conversion_environment.resolve_convert_python(
        check_compatibility=check_compatibility,
        shared_python=SHARED_CONVERT_PYTHON,
        env=env,
        cwd=cwd,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one complete Megatron Qwen3 Base checkpoint to a "
            "HuggingFace model directory."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Megatron checkpoint root containing the final marker.",
    )
    parser.add_argument(
        "--ckpt-step",
        required=True,
        type=checkpoint_step,
        help="Canonical final checkpoint step, e.g. 2.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="HF model output directory."
    )
    parser.add_argument(
        "--hf-template",
        required=True,
        type=Path,
        help="Complete local HuggingFace template matching AUTO_SFT_SFT_MODEL.",
    )
    parser.add_argument(
        "--megatron-dir",
        type=Path,
        default=Path(
            os.environ.get("AUTO_SFT_MEGATRON_DIR", str(DEFAULT_MEGATRON_DIR))
        ),
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Conversion interpreter (default: first compatible "
            "configured/shared/current interpreter)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair-invalid-manifest",
        action="store_true",
        help="Quarantine an invalid conversion directory and rebuild from the verified checkpoint/template.",
    )
    return parser.parse_args()


def _canonical(path: Path | str) -> str:
    if not isinstance(path, (str, Path)) or not str(path):
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ConversionContractError(f"{label} must be a JSON object: {path}")
    return payload


def validate_hf_template(
    template: Path | str,
    profile: ModelProfile | None = None,
) -> dict[str, Any]:
    """Validate the selected Qwen3 Base HF template.

    The model identity is taken from the configuration rather than from a
    user-controlled directory name.  This permits a run-local copy while
    still rejecting any architecture other than the run's selected model.
    """

    profile = profile or get_model_profile()
    resolved = Path(template).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise ConversionContractError(f"HF template directory not found: {resolved}")
    missing = [
        name for name in REQUIRED_TEMPLATE_FILES if not (resolved / name).is_file()
    ]
    if missing:
        raise ConversionContractError(
            f"{profile.model_id} HF template is missing required files: "
            + ", ".join(missing)
        )
    config = _json_object(resolved / "config.json", label="HF template config.json")
    for field, expected in profile.hf_config.items():
        actual = config.get(field)
        if actual != expected:
            raise ConversionContractError(
                f"HF template config field {field!r} must be {expected!r}; "
                f"got {actual!r}"
            )
    # tokenizer_config is consumed by downstream evaluation.  Check that it is
    # readable JSON, while leaving tokenizer.json's implementation-specific
    # schema to the tokenizer library.
    _json_object(
        resolved / "tokenizer_config.json", label="HF template tokenizer_config.json"
    )
    return config


def _iteration_dir(checkpoint: Path, step: str) -> Path:
    return checkpoint / f"iter_{int(step):07d}"


def validate_checkpoint_contract(
    checkpoint: Path | str,
    requested_step: str,
    *,
    require_iteration_dir: bool = True,
) -> tuple[str, Path]:
    """Validate marker, final-step equality, and final iteration directory."""

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=False)
    try:
        requested = checkpoint_step(str(requested_step))
    except argparse.ArgumentTypeError as exc:
        raise ConversionContractError(str(exc)) from exc
    if not checkpoint_path.is_dir():
        raise ConversionContractError(
            f"checkpoint directory not found: {checkpoint_path}"
        )
    marker_path = checkpoint_path / "latest_checkpointed_iteration.txt"
    if not marker_path.is_file():
        raise ConversionContractError(f"checkpoint marker not found: {marker_path}")
    try:
        marker = marker_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConversionContractError(
            f"cannot read checkpoint marker: {marker_path}"
        ) from exc
    try:
        final = checkpoint_step(marker)
    except argparse.ArgumentTypeError as exc:
        raise ConversionContractError(
            f"checkpoint marker must be a canonical positive integer: {marker!r}"
        ) from exc
    if requested != final:
        raise ConversionContractError(
            f"requested checkpoint step {requested} does not equal final marker {final}"
        )
    iteration_dir = _iteration_dir(checkpoint_path, final)
    if require_iteration_dir and not iteration_dir.is_dir():
        raise ConversionContractError(
            f"final checkpoint directory is missing: {iteration_dir}"
        )
    return final, iteration_dir


def validate_run_path_contract(
    run_dir: Path | str,
    *,
    checkpoint: Path | str,
    output: Path | str,
    step: str,
) -> str:
    """Require the canonical checkpoint and conversion paths for one run."""

    try:
        canonical_step = checkpoint_step(str(step))
    except argparse.ArgumentTypeError as exc:
        raise ConversionContractError(str(exc)) from exc
    run = Path(run_dir).expanduser().resolve(strict=False)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=False)
    output_path = Path(output).expanduser().resolve(strict=False)
    try:
        relative = checkpoint_path.relative_to(run / "sft")
    except ValueError as exc:
        raise ConversionContractError(
            "checkpoint must be AUTO_SFT_RUN_DIR/sft/round_<n>/checkpoints"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[1] != "checkpoints":
        raise ConversionContractError(
            "checkpoint must be AUTO_SFT_RUN_DIR/sft/round_<n>/checkpoints"
        )
    round_match = _STORAGE_ROUND.fullmatch(relative.parts[0])
    if round_match is None:
        raise ConversionContractError(
            "checkpoint must use canonical round_<positive integer> storage"
        )
    round_id = round_match.group(1)
    expected_checkpoint = run / "sft" / f"round_{round_id}" / "checkpoints"
    if checkpoint_path != expected_checkpoint:
        raise ConversionContractError(
            f"checkpoint path is not canonical for AUTO_SFT_RUN_DIR: {checkpoint_path}"
        )
    expected_output = (
        run / "ckpt_eval" / f"round_{round_id}" / f"step_{canonical_step}" / "hf_model"
    )
    if output_path != expected_output:
        raise ConversionContractError(
            "output must be AUTO_SFT_RUN_DIR/ckpt_eval/round_<n>/"
            f"step_{canonical_step}/hf_model; got {output_path}"
        )
    return round_id


def conversion_matches(
    existing: dict,
    *,
    checkpoint: Path,
    ckpt_step: str,
    hf_template: Path,
    profile: ModelProfile | None = None,
) -> bool:
    """Return whether a manifest is an exact current-input match."""

    profile = profile or get_model_profile()
    # Existing manifests without model identity predate multi-model support.
    manifest_model = existing.get("model_id", get_model_profile("1.7b").model_id)
    return (
        _canonical(existing.get("checkpoint", "")) == _canonical(checkpoint)
        and existing.get("ckpt_step") == ckpt_step
        and _canonical(existing.get("hf_template", "")) == _canonical(hf_template)
        and manifest_model == profile.model_id
    )


def conversion_complete(output: Path, profile: ModelProfile | None = None) -> bool:
    """Check that a converted directory has the selected model/config contract."""

    from autosft.orchestration.completion_evidence import model_assets_complete

    return model_assets_complete(output, profile)


def ensure_chat_stop_token(output: Path) -> None:
    """Make the converted model stop at <|im_end|> as well as its EOS."""

    path = output / "generation_config.json"
    config = (
        _json_object(path, label="generation_config.json") if path.is_file() else {}
    )
    eos = config.get("eos_token_id")
    if eos is None:
        eos = _json_object(output / "config.json", label="config.json").get(
            "eos_token_id"
        )
    ids = [eos] if isinstance(eos, int) else list(eos or [])
    if QWEN3_IM_END_ID in ids:
        return
    config["eos_token_id"] = [QWEN3_IM_END_ID, *ids]
    write_json(path, config)


def _read_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionContractError(
            f"invalid existing conversion manifest: {path}"
        ) from exc
    if not isinstance(candidate, dict):
        raise ConversionContractError(
            f"existing conversion manifest must be an object: {path}"
        )
    return candidate


def _manifest_for_args(
    *,
    args: argparse.Namespace,
    python_bin: str,
    command: list[str],
    final_step: str,
    iteration_dir: Path,
    profile: ModelProfile,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "checkpoint": _canonical(args.checkpoint),
        "checkpoint_final_step": final_step,
        "checkpoint_iteration": _canonical(iteration_dir),
        "ckpt_step": final_step,
        "output": _canonical(args.output),
        "hf_template": _canonical(args.hf_template),
        "model_id": profile.model_id,
        "megatron_dir": _canonical(args.megatron_dir),
        "python": python_bin,
        "command": command,
        "dry_run": bool(args.dry_run),
    }


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if args.dry_run:
        _convert(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The lock lives outside the output so quarantining an interrupted
    # conversion cannot let a second process start writing the same model.
    with args.output.with_name(f".{args.output.name}.conversion.lock").open(
        "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _convert(args, lock_fd=lock.fileno())


def _convert(args: argparse.Namespace, *, lock_fd: int | None = None) -> None:
    try:
        profile = get_model_profile()
        final_step, iteration_dir = validate_checkpoint_contract(
            args.checkpoint,
            args.ckpt_step,
            require_iteration_dir=not args.dry_run,
        )
        validate_hf_template(args.hf_template, profile)
        run_dir_value = os.environ.get("AUTO_SFT_RUN_DIR", "").strip()
        if run_dir_value:
            validate_run_path_contract(
                run_dir_value,
                checkpoint=args.checkpoint,
                output=args.output,
                step=final_step,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    converter = (
        REPO_ROOT
        / "src/autosft/backends/megatron/checkpoint/dist_ckpt_to_hf_qwen3_cpu.py"
    )
    python_bin = args.python or resolve_convert_python()
    command = [
        python_bin,
        str(converter),
        "--load",
        _canonical(args.checkpoint),
        "--iteration",
        final_step,
        "--save",
        _canonical(args.output),
        *profile.conversion_args,
    ]

    existing_manifest_path = args.output / "conversion_manifest.json"
    repair = None
    try:
        existing_manifest = _read_existing_manifest(existing_manifest_path)
    except ConversionContractError as exc:
        if not getattr(args, "repair_invalid_manifest", False) or args.dry_run:
            raise SystemExit(
                str(exc) + "; use --repair-invalid-manifest to preserve it and rebuild"
            ) from exc
        # Invalid JSON cannot authenticate its old source. Never adopt its
        # weights: preserve the entire directory, then rebuild from the inputs
        # validated above. Refuse overlapping source/output trees.
        for source in (args.checkpoint, args.hf_template):
            source = Path(source).resolve()
            if source.is_relative_to(args.output) or args.output.is_relative_to(source):
                raise SystemExit(
                    "repair output must not overlap checkpoint or template"
                )
        quarantine = args.output.with_name(
            f"{args.output.name}.invalid-{uuid.uuid4().hex}"
        )
        repair = {
            "quarantined_output": str(quarantine),
            "invalid_manifest_sha256": hashlib.sha256(
                existing_manifest_path.read_bytes()
            ).hexdigest(),
            "reason": "explicit rebuild after invalid conversion manifest",
            "checkpoint": _canonical(args.checkpoint),
            "ckpt_step": final_step,
            "hf_template": _canonical(args.hf_template),
            "requested_at": utc_now(),
        }
        existing_manifest = None
    if existing_manifest is not None:
        if conversion_matches(
            existing_manifest,
            checkpoint=args.checkpoint,
            ckpt_step=final_step,
            hf_template=args.hf_template,
            profile=profile,
        ) and _canonical(existing_manifest.get("output", "")) == _canonical(
            args.output
        ):
            if conversion_complete(args.output, profile) and not args.dry_run:
                ensure_chat_stop_token(args.output)
                print(
                    json.dumps(
                        {
                            "status": "existing_conversion_found",
                            "model_dir": str(args.output),
                            "manifest": str(existing_manifest_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                _refresh_handoff_if_needed()
                return
            if not args.dry_run:
                repair = {
                    "quarantined_output": str(
                        args.output.with_name(
                            f"{args.output.name}.incomplete-{uuid.uuid4().hex}"
                        )
                    ),
                    "reason": "rebuild incomplete conversion with matching inputs",
                    "checkpoint": _canonical(args.checkpoint),
                    "ckpt_step": final_step,
                    "hf_template": _canonical(args.hf_template),
                    "requested_at": utc_now(),
                }
        else:
            raise SystemExit(
                f"conversion output belongs to different inputs: {existing_manifest_path}; "
                "use the canonical round/step output"
            )

    if args.dry_run:
        print("+ " + " ".join(command))
        return

    if not converter.is_file():
        raise SystemExit(f"conversion script not found: {converter}")
    env = _conversion_environment.conversion_subprocess_env(
        megatron_dir=args.megatron_dir
    )
    try:
        if args.python:
            python_bin = check_convert_python(
                args.python, env=env, cwd=args.megatron_dir
            )
        else:
            python_bin = resolve_convert_python(
                check_compatibility=True,
                env=env,
                cwd=args.megatron_dir,
            )
    except ConversionEnvironmentError as exc:
        raise SystemExit(str(exc)) from exc
    command[0] = python_bin
    if repair is not None:
        for source in (args.checkpoint, args.hf_template):
            source = Path(source).resolve()
            if source.is_relative_to(args.output) or args.output.is_relative_to(source):
                raise SystemExit(
                    "repair output must not overlap checkpoint or template"
                )
        quarantine = Path(repair["quarantined_output"])
        args.output.rename(quarantine)
        write_json(quarantine / "conversion_repair.json", repair)
    subprocess.run(
        command,
        cwd=args.megatron_dir,
        env=env,
        check=True,
        pass_fds=() if lock_fd is None else (lock_fd,),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_TEMPLATE_FILES, *OPTIONAL_TEMPLATE_FILES):
        src = args.hf_template / name
        if src.is_file():
            shutil.copy2(src, args.output / name)
    ensure_chat_stop_token(args.output)
    if not conversion_complete(args.output, profile):
        missing = [
            name
            for name in REQUIRED_TEMPLATE_FILES
            if not (args.output / name).is_file()
        ]
        suffix = f": {', '.join(missing)}" if missing else ""
        raise SystemExit(
            f"converted model does not satisfy {profile.model_id} contract{suffix}"
        )

    manifest = _manifest_for_args(
        args=args,
        python_bin=python_bin,
        command=command,
        final_step=final_step,
        iteration_dir=iteration_dir,
        profile=profile,
    )
    if repair is not None:
        manifest["repair"] = repair
    write_json(args.output / "conversion_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    _refresh_handoff_if_needed()


def _refresh_handoff_if_needed() -> None:
    run_dir = os.environ.get("AUTO_SFT_RUN_DIR")
    if run_dir and executor_enabled():
        try:
            refresh_executor_handoff(Path(run_dir))
        except Exception as exc:
            print(f"WARN: failed to refresh executor handoff: {exc}")


if __name__ == "__main__":
    main()
