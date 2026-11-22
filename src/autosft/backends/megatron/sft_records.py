"""Frozen submission inputs and retryable accounting for local SFT rounds."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from autosft.core.common import write_json
from autosft.core.round_ids import logical_id_from_storage, storage_round_id
from autosft.orchestration.submission_registry import (
    find_sft_submission,
    sft_submission_conflicts,
    submission_task,
)


def intent_path(run: Path, run_name: str) -> Path:
    return run / "sft_records" / storage_round_id(run_name) / "submission.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_submission(run: Path, manifest: dict, recipe: Path) -> dict:
    """Persist inputs before any task can start; caller holds submission_lock."""
    path = intent_path(run, manifest["run_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = dict(manifest)
    recipe_copy = path.parent / "data-recipe.yaml"
    # The recipe is JSON-compatible even when its source spelling is YAML.
    from autosft.core.common import load_structured

    write_json(recipe_copy, load_structured(recipe))
    frozen["recipe_snapshot"] = str(recipe_copy)
    frozen["recipe_sha256"] = _digest(recipe_copy)
    data = frozen.get("data_manifest") or {}
    usage = data.get("syn_usage") or {}
    if usage.get("attributed_rows", 0):
        source = Path(data["datasets"][0]["path"])
        destination = path.parent / "syn_training_data.jsonl"
        shutil.copyfile(source, destination)
        if _digest(destination) != data["datasets"][0]["sha256"]:
            raise ValueError("training data changed while freezing Synthesis usage")
        frozen["syn_data_snapshot"] = str(destination)
    recipe_for_recovery(run, frozen)
    write_json(path, frozen)
    return frozen


def load_submission(run: Path, run_name: str, submission: dict) -> dict:
    """Read durable inputs without preparing data again or changing a live task."""
    path = intent_path(run, run_name)
    if not path.exists():
        path = run / "run_manifest.json"  # Compatible with pre-intent runs.
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"submitted SFT is missing its recovery record: {path}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("run_name") != run_name
        or manifest.get("dry_run") is True
    ):
        raise ValueError(f"SFT recovery record does not identify {run_name}: {path}")
    required = (
        "dataset_path",
        "train_iters",
        "seq_length",
        "sft_output_dir",
        "data_manifest",
    )
    if any(key not in manifest for key in required):
        raise ValueError(
            f"incomplete SFT recovery record: {path}; preserve the task and restore its original inputs"
        )
    conflicts = sft_submission_conflicts(
        submission,
        output_dir=Path(manifest["sft_output_dir"]),
        dataset_path=Path(manifest["dataset_path"]),
        train_iters=manifest["train_iters"],
        seq_length=manifest["seq_length"],
    )
    if (
        manifest.get("sft_task") not in {None, submission_task(submission)}
        or Path(manifest["sft_output_dir"]).resolve()
        != (run / "sft" / run_name).resolve()
        or conflicts
    ):
        raise ValueError("SFT recovery identity mismatch: " + "; ".join(conflicts))
    return manifest


def recipe_for_recovery(run: Path, manifest: dict) -> Path:
    path = Path(manifest.get("recipe_snapshot") or manifest["recipe"])
    if not path.resolve().is_relative_to(run.resolve()) or not path.is_file():
        raise ValueError("SFT recipe snapshot is missing or outside this run")
    expected = manifest.get("recipe_sha256")
    if expected and _digest(path) != expected:
        raise ValueError("SFT recipe snapshot changed after submission")
    from autosft.core.common import load_structured

    recipe = load_structured(path)
    data = manifest["data_manifest"]
    if str(recipe.get("round_id")) != logical_id_from_storage(
        manifest["run_name"]
    ) or recipe.get("recipe_id") != data.get("recipe_id"):
        raise ValueError("SFT recipe does not match the submitted data manifest")
    return path


def finish_sft_accounting(run: Path, receipt: dict) -> None:
    """Confirm Synthesis exposure only after successful training; safe to repeat on status."""
    if receipt.get("status") != "Succeeded" or "benchmark" in receipt:
        return
    run_name = receipt.get("run_name")
    if not run_name or "dataset_path" not in receipt:
        return
    path = intent_path(run, run_name)
    if not path.is_file():
        return  # Legacy preparations already appended their usage events.
    with path.with_name(".accounting.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submission = find_sft_submission(
            run / "sft_submissions.jsonl", run_name=run_name
        )
        if submission is None or submission_task(submission) != receipt.get("task"):
            raise ValueError(
                "successful SFT receipt does not match its durable submission"
            )
        manifest = load_submission(run, run_name, submission)
        data = manifest["data_manifest"]
        preview = data.get("syn_usage") or {}
        if not preview.get("attributed_rows", 0):
            return
        completed = path.with_name("syn_usage.json")
        if completed.exists():
            previous = json.loads(completed.read_text())
            if previous.get("task") == receipt["task"] and previous.get(
                "used_data_sha256"
            ) == preview.get("used_data_sha256"):
                return
            raise ValueError(
                "Synthesis usage completion record conflicts with the submitted SFT"
            )
        source = Path(preview["data_source_dir"])
        snapshot = Path(manifest["syn_data_snapshot"])
        if (
            not source.resolve().is_relative_to(run.resolve())
            or not snapshot.resolve().is_relative_to(path.parent.resolve())
            or _digest(snapshot) != data["datasets"][0]["sha256"]
        ):
            raise ValueError("frozen Synthesis usage inputs changed after submission")
        from autosft.synthesis import record_usage

        usage = record_usage(
            source,
            snapshot,
            training_run_id=run_name,
            selection_id=preview["selection_id"],
            snapshot_id=preview["snapshot_id"],
            recipe_ref=str(recipe_for_recovery(run, manifest)),
        )
        if usage.get("append_errors"):
            raise ValueError(
                f"Synthesis usage accounting failed: {usage['append_errors']}"
            )
        write_json(completed, {**usage, "task": receipt["task"], "status": "Succeeded"})
