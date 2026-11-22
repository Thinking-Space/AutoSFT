"""Validate a prepared round and preview its synthesis attribution."""

from __future__ import annotations
from autosft.config.components import synthesis_enabled

import json
import os
from pathlib import Path


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing or invalid Synthesis record: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Synthesis record must be an object: {path}")
    return value


def verify_syn_round_ready(run_dir: Path, round_id: int) -> dict | None:
    if not synthesis_enabled() or round_id == 1:
        return None
    decision = _object(run_dir / "syn_decision" / f"{round_id}.json")
    if (
        isinstance(decision.get("round_id"), bool)
        or str(decision.get("round_id")) != str(round_id)
        or not isinstance(decision.get("launch"), bool)
    ):
        raise ValueError(
            "Synthesis decision requires the upcoming training round_id and boolean launch"
        )
    round_path = run_dir / "synthesis" / str(round_id) / "round.json"
    if not decision["launch"]:
        if round_path.exists():
            raise ValueError(
                "Synthesis launch=false conflicts with an existing synthesis round"
            )
        return decision
    history = decision.get("launch_receipts")
    receipt = (
        history[-1]
        if isinstance(history, list) and history
        else decision.get("launch_receipt")
    )
    if (
        not isinstance(receipt, dict)
        or not receipt.get("executed_at")
        or not isinstance(receipt.get("launched_targets"), list)
        or not isinstance(receipt.get("skipped_active_targets"), list)
    ):
        raise ValueError(f"Synthesis round {round_id} has no complete launch_receipt")
    synthesis = _object(round_path)
    if (
        synthesis.get("disposition") is not None
        or decision.get("abandon_receipt") is not None
    ):
        from autosft.synthesis.job import verify_abandonment
        from autosft.synthesis.errors import SynError

        try:
            verify_abandonment(run_dir, str(round_id), decision)
        except (SynError, RuntimeError, OSError) as exc:
            raise ValueError(
                f"Synthesis round {round_id} has invalid abandonment evidence: {exc}"
            ) from exc
        return decision
    targets = synthesis.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or any(
            not isinstance(t, dict)
            or t.get("state") not in {"succeeded", "failed", "stopped"}
            for t in targets
        )
    ):
        raise ValueError(
            f"Synthesis round {round_id} still has non-terminal targets; collect status first"
        )
    if synthesis.get("state") not in {"succeeded", "partial"} or not any(
        t["state"] == "succeeded" for t in targets
    ):
        raise ValueError(f"Synthesis round {round_id} has no successful terminal batch")
    return decision


def maybe_record_syn_usage(dataset_path: Path, manifest: dict) -> dict | None:
    if not synthesis_enabled():
        return None
    round_id, run_dir = int(manifest["round_id"]), Path(manifest["run_dir"])
    decision = verify_syn_round_ready(run_dir, round_id)
    source = Path(
        os.environ.get("AUTO_SFT_SYN_DATA_SOURCE_DIR") or run_dir / "syn_data_source"
    )
    if not source.is_absolute():
        source = run_dir / source
    source = source.resolve()
    if source == run_dir.resolve() or not source.is_relative_to(run_dir.resolve()):
        raise ValueError("Synthesis Data Source must remain under this run")
    # Bootstrap precedes synthesis, and a no-launch decision can leave the
    # source uninitialized. Neither case invents a pool or a usage event.
    if not (source / "manifest.json").exists() and (
        not decision or not decision["launch"]
    ):
        return {"attributed_rows": 0, "per_batch": [], "appended": []}
    from autosft.synthesis import record_usage

    kwargs = dict(
        training_run_id=f"round_{round_id}",
        selection_id=f"final-{manifest['datasets'][0]['sha256'][:16]}",
        recipe_ref=manifest.get("recipe_path"),
    )
    usage = record_usage(source, dataset_path, append=False, **kwargs)
    if (
        decision
        and decision["launch"]
        and not decision.get("abandon_receipt")
        and usage.get("attributed_rows", 0) <= 0
    ):
        raise ValueError(
            f"Synthesis round {round_id} launched synthesis but uses no published Synthesis rows"
        )
    # Preparation can be revised or never submitted. Only successful SFT
    # confirms exposure, using this snapshot and the frozen submission data.
    usage["data_source_dir"] = str(source)
    usage["accounting_status"] = "prepared"
    return usage
