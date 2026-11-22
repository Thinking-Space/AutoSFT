#!/usr/bin/env python3
"""Build one compact, durable state snapshot for AutoSFT executors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autosft.core.common import utc_now
from autosft.backends.local.status import TERMINAL_STATUSES, latest_task_statuses
from autosft.orchestration.ledger import recipe_history_dir, rounds_awaiting_eval
from autosft.orchestration.blocked_rounds import acknowledged_rounds
from autosft.orchestration.round_gates import evaluation_settings
from autosft.data.recipe import (
    RecipeValidationError,
    load_minimal_recipe,
    recipe_sha256,
    validate_minimal_recipe,
)
from autosft.core.round_ids import (
    RoundIdError,
    logical_id_from_storage,
    logical_round_id,
    storage_round_id,
)
from autosft.benchmarks.aggregate import repeat_dirs
from autosft.orchestration.submission_registry import (
    current_eval_submissions,
    eval_manifest_paths,
    model_identity,
    record_repeat_indices,
    record_repeats,
    submission_records,
    submission_task,
)


HANDOFF_FILENAME = "executor_handoff.json"
DATA_ROUNDS_DIRNAME = "data_rounds"
RECIPE_FILENAME = "data-recipe.yaml"
RECIPE_PIN_FILENAME = "recipe_pin.json"
FAILED_STATUSES = {"Failed", "Stopped", "Killed", "Cancelled", "Canceled"}
DEFAULT_BENCHMARKS = ("ifeval",)
# How long a task may sit Queued/Starting before the executor treats the wait
# itself as reportable. Local jobs may wait for the host GPU lock while
# another job runs, so patience uses elapsed time, not poll counts. The note is gated on the
# task still being in a queue status: a Running task's age includes training
# time and says nothing about the queue.
DEFAULT_QUEUE_PATIENCE_SECONDS = 4 * 3600
QUEUE_WAIT_STATUSES = {"Queued", "Starting", "Pending"}


def queue_patience_seconds() -> int:
    raw = os.environ.get("AUTO_SFT_QUEUE_PATIENCE_SECONDS", "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_QUEUE_PATIENCE_SECONDS
    return value if value > 0 else DEFAULT_QUEUE_PATIENCE_SECONDS


def _parse_iso_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _task_age_seconds(events_file: Path, task: str | None) -> int | None:
    """Seconds since the task was created, from its recorded poll events.

    ``create_time`` is local_task's own creation stamp, recorded on every poll; the
    newest event's copy is as good as any. Returns None when no event carries
    a parseable create_time (e.g. the task was submitted but never polled).
    """
    if not task:
        return None
    created: datetime | None = None
    for record in _jsonl_records_of(events_file):
        if record.get("task") != task:
            continue
        stamp = _parse_iso_time(record.get("create_time"))
        if stamp is not None:
            created = stamp
    if created is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - created).total_seconds()))


def _queue_wait_note(status: str | None, age: int | None) -> str:
    """Patience guidance for one live task, empty when not queue-waiting.

    Gated on the task actually sitting in a queue status: a Running task's
    age includes training time and says nothing about the queue, so it gets
    no note (a 5-hour healthy training must never read as an exceeded queue
    wait). Durations are stated in minutes so a sub-hour patience override
    still renders sensibly.
    """
    if status not in QUEUE_WAIT_STATUSES or age is None:
        return ""
    patience = queue_patience_seconds()
    if age < patience:
        return (
            f" The task has been queued for {age // 60} minutes; waits under "
            f"{patience // 60} minutes are normal local GPU queueing, not a "
            "stuck task or a failure. Keep polling while your session can "
            "wait; if the queue outlasts it, persist one blocked "
            "(recoverable) report naming the queue wait — the round parks, "
            "the harness keeps polling the task, and the stage moves on its "
            "own when the local task advances."
        )
    return (
        f" The task has been queued for {age // 60} minutes, beyond the "
        f"{patience // 60}-minute queue patience — the wait itself is now "
        "anomalous. Persist one blocked (recoverable) report naming the "
        "exceeded patience and the wait duration in failure_details so the "
        "parked round is visibly waiting on a local task; the harness keeps "
        "polling the parked task."
    )


def _jsonl_records_of(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return records
    return records


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _canonical(value: object) -> str:
    if not isinstance(value, (str, Path)) or not str(value):
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def _resolve_run_path(
    value: object,
    *,
    run_dir: Path,
) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve(strict=False)


def round_recipe_path(run_dir: Path, round_id: str) -> Path:
    """Canonical per-round minimal-recipe artifact the meta agent authors."""
    return (
        Path(run_dir)
        / DATA_ROUNDS_DIRNAME
        / f"round_{logical_round_id(round_id)}"
        / RECIPE_FILENAME
    )


def newest_recipe_round(
    run_dir: Path, *, exclude: set[str] | None = None
) -> str | None:
    """Logical id of the newest round that has a per-round recipe artifact.

    Ordered by numeric round id, never by filename or mtime: ``round_10``
    sorts after ``round_9``, and a recipe rewritten in place must not change
    which round is newest.
    """
    newest: str | None = None
    for path in (Path(run_dir) / DATA_ROUNDS_DIRNAME).glob(
        f"round_*/{RECIPE_FILENAME}"
    ):
        try:
            round_id = logical_id_from_storage(
                path.parent.name,
                label="data round directory",
            )
        except RoundIdError:
            continue
        if exclude and round_id in exclude:
            continue
        if newest is None or int(round_id) > int(newest):
            newest = round_id
    return newest


def _load_round_recipe(
    recipe_path: Path,
    *,
    run_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load one round recipe, returning ``(recipe, error)``."""
    try:
        recipe = load_minimal_recipe(recipe_path)
        validate_minimal_recipe(recipe, run_dir=run_dir)
    except RecipeValidationError as exc:
        return None, f"invalid recipe: {exc}"
    except Exception as exc:  # noqa: BLE001 - load_structured raises
        # OSError / ValueError / yaml errors depending on what is on disk;
        # a handoff refresh must degrade to a reportable state, not crash.
        return None, f"unreadable recipe: {exc}"
    return recipe, None


def _archived_round_recipes(
    run_dir: Path,
    round_id: str,
) -> list[tuple[Path, str]]:
    """``(path, canonical sha)`` per archived recipe of one round, oldest first.

    The ledger archives the guard-validated recipe at every SFT submission
    (``recipe_history/iter_<NNN>.yaml``), so these are the only trustworthy
    restore sources once the round's working copy has been edited in place.
    """
    try:
        canonical_round = logical_round_id(round_id)
    except RoundIdError:
        return []
    results: list[tuple[Path, str]] = []
    for path in sorted(recipe_history_dir(run_dir).glob("iter_*.yaml")):
        try:
            recipe = load_minimal_recipe(path)
            archived_round = logical_round_id(
                recipe.get("round_id"),
                label="archived recipe round_id",
            )
        except (RecipeValidationError, RoundIdError):
            continue
        except Exception:  # noqa: BLE001 - an unreadable archive never blocks
            continue
        if archived_round != canonical_round:
            continue
        results.append((path, recipe_sha256(recipe)))
    return results


def _attach_pinned_archive(
    state: dict[str, Any],
    run_dir: Path,
    round_id: str,
    pinned_sha: str,
) -> None:
    """Point a violation state at the archived copy matching the pin, if any."""
    for archive_path, archive_sha in _archived_round_recipes(run_dir, round_id):
        if archive_sha == pinned_sha:
            state["pinned_recipe_archive"] = str(archive_path)
            return


def _data_state(
    run_dir: Path,
    *,
    round_id: str | None,
    has_sft_record: bool,
) -> dict[str, Any]:
    """Durable data_ready state for one round's open-pool artifacts.

    The stage is complete when the round dir holds ``data-recipe.yaml`` whose
    strict minimal schema passes and whose referenced ``final_data.jsonl``
    exists. The recipe digest is pinned at data_ready (``recipe_pin.json`` in
    the round dir); once an SFT submission exists for the round, a digest
    mismatch is an immutability violation, while before submission a mismatch
    means the meta agent reworked the data and the pin follows it.
    """
    state: dict[str, Any] = {
        "round_dir": None,
        "recipe": None,
        "recipe_id": None,
        "recipe_sha256": None,
        "pinned_recipe_sha256": None,
        "pin_status": None,
        "pinned_recipe_archive": None,
        "final_data": None,
        "complete": False,
        "error": None,
    }
    if round_id is None:
        state["error"] = "missing_recipe"
        return state
    recipe_path = round_recipe_path(run_dir, round_id)
    round_dir = recipe_path.parent
    state["round_dir"] = str(round_dir)
    pin_path = round_dir / RECIPE_PIN_FILENAME
    pin_payload = _read_json(pin_path)
    pinned_sha = (
        pin_payload.get("recipe_sha256")
        if pin_payload is not None and isinstance(pin_payload.get("recipe_sha256"), str)
        else None
    )
    state["pinned_recipe_sha256"] = pinned_sha
    if not recipe_path.is_file():
        state["error"] = "missing_recipe"
        if has_sft_record and pinned_sha is not None:
            state["pin_status"] = "violation"
            state["error"] = "recipe_removed_after_pin"
            _attach_pinned_archive(state, run_dir, round_id, pinned_sha)
        return state
    state["recipe"] = str(recipe_path)
    recipe, error = _load_round_recipe(recipe_path, run_dir=run_dir)
    if recipe is None:
        state["error"] = error
        if has_sft_record and pinned_sha is not None:
            state["pin_status"] = "violation"
            _attach_pinned_archive(state, run_dir, round_id, pinned_sha)
        return state
    state["recipe_id"] = str(recipe.get("recipe_id"))
    current_sha = recipe_sha256(recipe)
    state["recipe_sha256"] = current_sha
    try:
        recipe_round = logical_round_id(
            recipe.get("round_id"),
            label="recipe round_id",
        )
    except RoundIdError as exc:
        state["error"] = str(exc)
        return state
    if recipe_round != logical_round_id(round_id):
        state["error"] = (
            f"recipe round_id {recipe_round!r} does not match "
            f"round dir {round_dir.name!r}"
        )
        return state
    datasets = recipe.get("datasets") or []
    dataset_path = _resolve_run_path(
        (datasets[0] or {}).get("path"),
        run_dir=run_dir,
    )
    state["final_data"] = str(dataset_path)
    if not dataset_path.is_file():
        state["error"] = f"missing final_data.jsonl: {dataset_path}"
        return state

    if pinned_sha is None:
        state["pin_status"] = "created"
        try:
            _atomic_write(
                pin_path,
                {"recipe_sha256": current_sha, "pinned_at": utc_now()},
            )
            state["pinned_recipe_sha256"] = current_sha
        except OSError:
            pass
    elif pinned_sha == current_sha:
        state["pin_status"] = "match"
    elif has_sft_record:
        archived_shas = {sha for _, sha in _archived_round_recipes(run_dir, round_id)}
        if current_sha in archived_shas:
            # The working copy is exactly what the ledger archived at
            # submission: the pin predates a legitimate pre-submission
            # rework whose pin was not refreshed. The submitted content is authoritative;
            # the stale pin follows it instead of orphaning a trained round.
            state["pin_status"] = "repinned_to_submitted"
            try:
                _atomic_write(
                    pin_path,
                    {"recipe_sha256": current_sha, "pinned_at": utc_now()},
                )
                state["pinned_recipe_sha256"] = current_sha
            except OSError:
                pass
        else:
            # Later stages verify the recipe is unchanged: once this round
            # has an SFT submission, its recipe is immutable. The violation
            # is recoverable, not terminal: the ledger archived the recipe
            # that was actually submitted, so restoring the round's working
            # copy to that content clears the block on the next refresh and
            # the finished training is not orphaned.
            state["pin_status"] = "violation"
            state["error"] = "recipe_changed_after_sft_submission"
            _attach_pinned_archive(state, run_dir, round_id, pinned_sha)
            return state
    else:
        # No submission yet: the meta agent reworked the data (e.g. after a budget
        # rejection); the pin follows the reworked recipe.
        state["pin_status"] = "repinned"
        try:
            _atomic_write(
                pin_path,
                {"recipe_sha256": current_sha, "pinned_at": utc_now()},
            )
            state["pinned_recipe_sha256"] = current_sha
        except OSError:
            pass
    state["complete"] = True
    return state


def parse_benchmarks(raw: str | None) -> list[str]:
    tokens = [
        token.strip().lower()
        for chunk in (raw or "").split(",")
        for token in chunk.split()
        if token.strip()
    ]
    result: list[str] = []
    for token in tokens:
        if token not in result:
            result.append(token)
    return result


def _task_status_from_final(
    final_status: dict[str, Any] | None,
    task: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if not final_status or not task:
        return None, None
    for item in final_status.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        if submission_task(item) == task:
            status = item.get("local_status")
            return (status if isinstance(status, str) else None), item
    return None, None


def _conversion_state(
    run_dir: Path,
    *,
    round_id: str | None,
    checkpoint_dir: Path | None,
    latest_step: str | None,
) -> dict[str, Any]:
    if round_id is None or latest_step is None or checkpoint_dir is None:
        return {"complete": False, "model_dir": None, "manifest": None}
    try:
        canonical_round = logical_round_id(round_id)
        canonical_step = logical_round_id(
            latest_step,
            label="latest checkpoint step",
        )
    except RoundIdError:
        return {"complete": False, "model_dir": None, "manifest": None}
    path = (
        run_dir
        / "ckpt_eval"
        / f"round_{canonical_round}"
        / f"step_{canonical_step}"
        / "hf_model"
        / "conversion_manifest.json"
    )
    manifest = _read_json(path)
    if manifest is None:
        return {"complete": False, "model_dir": None, "manifest": None}
    checkpoint_key = _canonical(checkpoint_dir)
    model_dir = path.parent
    if (
        _canonical(manifest.get("checkpoint")) != checkpoint_key
        or str(manifest.get("ckpt_step") or "") != canonical_step
        or _canonical(manifest.get("output")) != _canonical(model_dir)
    ):
        return {"complete": False, "model_dir": None, "manifest": None}
    from autosft.orchestration.completion_evidence import model_assets_complete

    return {
        "complete": model_assets_complete(model_dir),
        "model_dir": str(model_dir),
        "model_identity": model_identity(model_dir),
        "manifest": str(path),
    }


def _latest_executor_failure(
    run_dir: Path,
    *,
    round_id: str | None,
    exclude: set[str] | None = None,
) -> dict[str, Any] | None:
    # Order by numeric round, not filename: sorted() puts `round_10.json`
    # before `round_9.json`, so a lexicographic scan would report a stale
    # failure once a run passes ten rounds. Non-canonical names (staging
    # copies like `round_1.executor_report.json`) are skipped entirely —
    # only files the persist command owns may surface a failure.
    def _round_key(path: Path) -> int | None:
        try:
            return int(logical_id_from_storage(path.stem))
        except RoundIdError:
            return None

    reports = sorted(
        (
            path
            for path in (run_dir / "executor_reports").glob("round_*.json")
            if _round_key(path) is not None
        ),
        key=_round_key,
    )
    for path in reversed(reports):
        if exclude and logical_id_from_storage(path.stem) in exclude:
            continue
        report = _read_json(path)
        if (
            report
            and round_id is not None
            and str(report.get("round_id") or "") != str(round_id)
        ):
            continue
        if report and report.get("failure_details"):
            return {
                "report": str(path),
                "round_id": report.get("round_id"),
                "status": report.get("status"),
                "failure_details": report.get("failure_details"),
            }
    return None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_executor_handoff(
    run_dir: Path,
    *,
    round_id: str | None = None,
    run_name: str | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Derive the finite-state executor handoff from canonical run artifacts."""
    run_dir = Path(run_dir).resolve()
    configured_benchmarks, required_repeats = evaluation_settings()
    requested_benchmarks = list(benchmarks or configured_benchmarks)

    sft_manifest = run_dir / "sft_submissions.jsonl"
    all_sft_records = [record for record in submission_records(sft_manifest)]
    for record in all_sft_records:
        storage_round_id(record.get("run_name"), label="SFT submission run_name")
    inferred = round_id is None and run_name is None
    # Disposing of a failure closes automatic recovery, including its recipe
    # and submission fallbacks. Explicit round inspection remains available.
    disposed = acknowledged_rounds(run_dir) if inferred else set()
    active_sft_records = [
        record
        for record in all_sft_records
        if logical_id_from_storage(record["run_name"]) not in disposed
    ]
    latest_record = (
        max(
            active_sft_records,
            key=lambda item: str(item.get("submitted_at") or ""),
        )
        if active_sft_records
        else None
    )
    latest_round = (
        logical_id_from_storage(
            latest_record.get("run_name"),
            label="SFT submission run_name",
        )
        if latest_record is not None
        else None
    )
    requested_round = logical_round_id(round_id) if round_id is not None else None
    target_run_name = (
        storage_round_id(run_name, label="run_name") if run_name is not None else None
    )
    if target_run_name is not None:
        run_name_round = logical_id_from_storage(target_run_name, label="run_name")
        if requested_round is not None and run_name_round != requested_round:
            raise RoundIdError("run_name must match round_id")
        requested_round = run_name_round
    if target_run_name is None and requested_round is None:
        recipe_round = newest_recipe_round(run_dir, exclude=disposed)
        if not (
            recipe_round is not None
            and latest_round is not None
            and int(recipe_round) < int(latest_round)
        ):
            requested_round = recipe_round
        # An inferred round never abandons a trained round whose scores are
        # not in the ledger yet. Inference follows the newest recipe, so a
        # root that authors round N+1 while round N is still being evaluated
        # would otherwise move current_round forward; round N's eval records
        # are then filtered out by model identity, its collect stage vanishes
        # from the handoff, and its finished evaluation is never recorded --
        # silently, because the handoff looks healthy on the next round.
        # Only inference is held back: an explicit --round-id/--run-name still
        # reaches any round, which is what recovery and finalize need.
        unevaluated = [
            round_id
            for round_id in rounds_awaiting_eval(
                run_dir,
                benchmarks=requested_benchmarks,
                repeats=required_repeats,
            )
            if round_id not in disposed
            and (requested_round is None or int(round_id) < int(requested_round))
        ]
        if unevaluated:
            requested_round = unevaluated[0]
    if target_run_name is None and requested_round is not None:
        matching_records = [
            record
            for record in all_sft_records
            if record.get("run_name") == f"round_{requested_round}"
        ]
        if matching_records:
            matching_record = max(
                matching_records,
                key=lambda item: str(item.get("submitted_at") or ""),
            )
            target_run_name = str(matching_record.get("run_name") or "")
        else:
            target_run_name = f"round_{requested_round}"

    if target_run_name:
        sft_records = [
            record
            for record in all_sft_records
            if record.get("run_name") == target_run_name
        ]
    elif latest_record is not None:
        latest_run_name = latest_record.get("run_name")
        sft_records = [
            record
            for record in all_sft_records
            if record.get("run_name") == latest_run_name
        ]
    else:
        sft_records = []
    sft_record = (
        min(sft_records, key=lambda item: str(item.get("submitted_at") or ""))
        if sft_records
        else None
    )
    selected_run_name = (
        str(sft_record.get("run_name"))
        if sft_record and sft_record.get("run_name")
        else target_run_name
    )
    current_round = requested_round
    if current_round is None and selected_run_name is not None:
        current_round = logical_id_from_storage(
            selected_run_name,
            label="selected SFT run_name",
        )
    data_ready = _data_state(
        run_dir,
        round_id=current_round,
        has_sft_record=sft_record is not None,
    )

    sft_task = submission_task(sft_record) if sft_record else None
    sft_events = run_dir / "local_events.jsonl"
    sft_event_status = latest_task_statuses(sft_events).get(sft_task or "")
    sft_final_path = run_dir / "sft_final_status.json"
    sft_final = _read_json(sft_final_path)
    sft_final_status, sft_final_task = _task_status_from_final(sft_final, sft_task)
    sft_status = (
        sft_final_status
        or sft_event_status
        or (
            str(sft_record.get("status"))
            if sft_record and sft_record.get("status")
            else None
        )
    )

    output_dir = (
        Path(str(sft_record.get("output_dir")))
        if sft_record and sft_record.get("output_dir")
        else None
    )
    checkpoint_dir = output_dir / "checkpoints" if output_dir else None
    latest_file = (
        checkpoint_dir / "latest_checkpointed_iteration.txt" if checkpoint_dir else None
    )
    latest_step = None
    if latest_file and latest_file.is_file():
        try:
            latest_step = latest_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            latest_step = None
    # Without sft_final_status.json this falls back to the on-disk checkpoint
    # marker, which says nothing about whether the loss artifacts were parsed.
    # The stage machine therefore also gates sft_collect on sft_final_task.
    checkpoint_complete = bool(
        (sft_final_task or {}).get("checkpoint_complete")
        if sft_final_task is not None
        else latest_step is not None
    )

    conversion = _conversion_state(
        run_dir,
        round_id=current_round,
        checkpoint_dir=checkpoint_dir,
        latest_step=latest_step,
    )
    model_dir = conversion.get("model_dir")
    model_key = _canonical(model_dir)
    model_identity_key = (
        str(conversion.get("model_identity"))
        if conversion.get("model_identity")
        else ""
    )

    eval_records: list[dict[str, Any]] = []
    eval_manifests = eval_manifest_paths(run_dir)
    current_eval_tasks = {
        submission_task(record) for record, _ in current_eval_submissions(run_dir)
    }
    historical_eval_tasks: list[str] = []
    final_status_by_task: dict[str, tuple[str | None, dict[str, Any], Path]] = {}
    event_statuses: dict[str, str] = {}
    for manifest in eval_manifests:
        if not manifest.is_file():
            continue
        events_file = manifest.parent / "local_events.jsonl"
        event_statuses.update(latest_task_statuses(events_file))
        final_path = manifest.parent / "eval_final_status.json"
        final_payload = _read_json(final_path)
        if final_payload:
            for item in final_payload.get("tasks") or []:
                if not isinstance(item, dict):
                    continue
                task = submission_task(item)
                if task:
                    status = item.get("local_status")
                    final_status_by_task[task] = (
                        status if isinstance(status, str) else None,
                        item,
                        final_path,
                    )
        for record in submission_records(manifest):
            task = submission_task(record)
            if task:
                historical_eval_tasks.append(task)
            if task not in current_eval_tasks:
                continue
            record_model_identity = record.get("model_identity")
            if not isinstance(record_model_identity, str) or not record_model_identity:
                continue
            if (
                not model_key
                or not model_identity_key
                or record_model_identity != model_identity_key
            ):
                continue
            eval_records.append(
                {
                    **record,
                    "_manifest": str(manifest),
                    "_events": str(events_file),
                }
            )

    tasks: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for record in eval_records:
        task = submission_task(record)
        if task is None or task in seen_tasks:
            continue
        seen_tasks.add(task)
        final_info = final_status_by_task.get(task)
        status = (
            (final_info[0] if final_info else None)
            or event_statuses.get(task)
            or (str(record.get("status")) if record.get("status") else None)
        )
        output = Path(str(record.get("output_dir") or ""))
        # A task's artifacts are complete only when every repeat it owns has
        # its scores; a legacy per-repeat record owns just its own directory.
        covered_repeats = record_repeat_indices(record)
        owned_dirs = (
            repeat_dirs(output, record_repeats(record))
            if "repeats" in record
            else [output]
        )
        aggregate_dir = (
            output.parent
            if "repeats" not in record and record.get("repeat") and required_repeats > 1
            else output
        )
        artifacts_complete = bool(
            all(
                (rep_dir / name).is_file()
                for rep_dir in owned_dirs
                for name in (
                    "metrics.json",
                    "scores.json",
                    "failures.jsonl",
                    "failure_breakdown.json",
                )
            )
        )
        tasks.append(
            {
                "kind": "eval",
                "benchmark": record.get("benchmark"),
                "repeats": covered_repeats,
                "task": task,
                "status": status,
                "terminal": status in TERMINAL_STATUSES,
                "artifacts_complete": artifacts_complete,
                "output_dir": str(output),
                "result_artifacts": {
                    "aggregate": str(aggregate_dir / "aggregate.json"),
                    "repeat_dirs": [str(path) for path in owned_dirs],
                    "submission_manifest": record["_manifest"],
                },
                "submission_manifest": record["_manifest"],
                "events": record["_events"],
                "final_status": str(final_info[2]) if final_info else None,
            }
        )

    by_benchmark: dict[str, list[dict[str, Any]]] = {
        benchmark: [] for benchmark in requested_benchmarks
    }
    for item in tasks:
        benchmark = str(item.get("benchmark") or "").lower()
        by_benchmark.setdefault(benchmark, []).append(item)

    required_repeat_set = set(range(1, required_repeats + 1))
    missing_repeats: dict[str, list[int]] = {}
    for benchmark in requested_benchmarks:
        submitted_repeats = {
            repeat for item in by_benchmark[benchmark] for repeat in item["repeats"]
        }
        absent = sorted(required_repeat_set - submitted_repeats)
        if absent:
            missing_repeats[benchmark] = absent
    missing_benchmarks = sorted(missing_repeats)
    satisfied_benchmarks = [
        benchmark
        for benchmark in requested_benchmarks
        if benchmark not in missing_repeats
        and required_repeat_set
        <= {
            repeat
            for item in by_benchmark[benchmark]
            if item.get("status") == "Succeeded"
            and item.get("artifacts_complete") is True
            for repeat in item["repeats"]
        }
    ]
    active_tasks = [
        item for item in tasks if item.get("status") not in TERMINAL_STATUSES
    ]
    failed_tasks = [item for item in tasks if item.get("status") in FAILED_STATUSES]
    duplicate_groups = []
    for benchmark, entries in sorted(by_benchmark.items()):
        by_repeat: dict[int, set[str]] = {}
        for item in entries:
            for repeat in item["repeats"]:
                by_repeat.setdefault(repeat, set()).add(item["task"])
        for repeat, repeat_tasks in sorted(by_repeat.items()):
            if len(repeat_tasks) > 1:
                duplicate_groups.append(
                    {
                        "benchmark": benchmark,
                        "repeat": repeat,
                        "tasks": sorted(repeat_tasks),
                    }
                )
    eval_submission_manifests = sorted({item["submission_manifest"] for item in tasks})
    eval_event_files = sorted({item["events"] for item in tasks})
    eval_manifest_summary = ", ".join(eval_submission_manifests)

    if sft_record is not None and data_ready.get("pin_status") == "violation":
        stage = "blocked"
        pinned_archive = data_ready.get("pinned_recipe_archive")
        if pinned_archive:
            reconcile = (
                " This is recoverable without resubmitting: the ROOT (never "
                "the executor — its only action here is persisting the "
                "blocked report) restores "
                f"{data_ready.get('round_dir')}/{RECIPE_FILENAME} from the "
                f"archived submitted copy {pinned_archive}, refreshes this "
                "handoff (the block clears once the digest matches the pin "
                "again), and re-delegates the executor; the round then "
                "continues from its existing SFT task. Never re-pin to the "
                "edited content."
            )
        else:
            reconcile = (
                " No archived copy matches the pinned digest "
                f"({data_ready.get('pinned_recipe_sha256')}); reconciliation "
                "needs the operator."
            )
        next_action = (
            "The round recipe changed after its SFT submission "
            f"({data_ready.get('error')}); report the immutability violation, "
            "do not resubmit, and do not abandon the round." + reconcile
        )
        skills_required = []
    elif sft_record is None and not data_ready["complete"]:
        # Data assembly belongs to the meta agent; the executor's
        # stages start at sft_submit. The handoff reports data_ready with the
        # failure reason so the meta agent reworks missing or malformed artifacts.
        # Row/token preparation rejections are returned in the executor report.
        stage = "data_ready"
        next_action = (
            "The meta agent must write data_rounds/round_<id>/final_data.jsonl and "
            "a valid data-recipe.yaml for the current round"
            + (
                f" (blocking reason: {data_ready['error']})"
                if data_ready.get("error")
                else ""
            )
            + "; refresh this handoff once both exist."
        )
        skills_required = []
    elif sft_record is None:
        stage = "sft_submit"
        # Name the entrypoint by its absolute path: the executor session's
        # cwd is not guaranteed to be the checkout, and a bare `src/autosft/backends/megatron/train.py`
        # cannot be resolved reliably from a run-directory cwd.
        sft_train = (
            Path(__file__).resolve().parents[3]
            / "src/autosft/backends/megatron/train.py"
        )
        next_action = (
            f"Submit once with python {sft_train} "
            f"--run-name {selected_run_name} "
            "--run-dir "
            f"{run_dir} using recipe {data_ready['recipe']}; do not use a "
            "default or relative run directory."
        )
        skills_required = ["training-submission"]
    elif sft_status in FAILED_STATUSES:
        stage = "blocked"
        next_action = (
            f"Report the terminal SFT status {sft_status}; do not resubmit task "
            f"{sft_task}."
        )
        skills_required = ["job-lifecycle"]
    elif sft_status != "Succeeded":
        stage = "sft_poll"
        next_action = (
            f"Run autosft tools jobs sft status once for live task {sft_task} using "
            f"manifest {sft_manifest}, then refresh this handoff."
            + _queue_wait_note(sft_status, _task_age_seconds(sft_events, sft_task))
        )
        skills_required = ["job-lifecycle"]
    elif not checkpoint_complete or sft_final_task is None:
        stage = "sft_collect"
        next_action = (
            f"Run autosft tools jobs sft collect once for terminal task {sft_task} "
            f"using manifest {sft_manifest}, then refresh this handoff."
        )
        skills_required = ["job-lifecycle"]
    elif not conversion.get("complete"):
        stage = "checkpoint_conversion"
        next_action = f"Convert checkpoint {checkpoint_dir} at step {latest_step} once."
        skills_required = ["checkpoint-conversion"]
    elif missing_benchmarks:
        stage = "eval_submit"
        next_action = (
            "Submit only the missing benchmarks, one task each covering all "
            f"{required_repeats} repeat(s): "
            + "; ".join(
                f"{benchmark} (repeats {', '.join(str(k) for k in missing_repeats[benchmark])}"
                f" of {required_repeats} not yet submitted)"
                for benchmark in missing_benchmarks
            )
        )
        skills_required = ["evaluation-submission"]
    elif len(satisfied_benchmarks) == len(requested_benchmarks):
        from autosft.orchestration.completion_evidence import (
            evaluation_evidence_problems,
        )
        from autosft.orchestration.ledger import read_iterations, record_round_id

        round_records = [
            item
            for item in read_iterations(run_dir)
            if record_round_id(item) == current_round
        ]
        recorded_complete = len(
            round_records
        ) == 1 and not evaluation_evidence_problems(
            round_records[0],
            run_dir,
            benchmarks=requested_benchmarks,
            repeats=required_repeats,
            expected_model_identity=model_identity_key,
        )
        missing_final_status = any(
            not any(item.get("final_status") for item in by_benchmark[benchmark])
            for benchmark in requested_benchmarks
        )
        if not round_records:
            stage = "sft_submit"
            sft_train = (
                Path(__file__).resolve().parents[3]
                / "src/autosft/backends/megatron/train.py"
            )
            next_action = (
                "The SFT iteration ledger entry is missing. Rerun the original "
                f"python {sft_train} --run-name {selected_run_name} "
                f"--run-dir {run_dir} --recipe {data_ready['recipe']} command; "
                "the durable submission intent restores the original iteration "
                "without launching a duplicate training task. Then refresh this handoff."
            )
            skills_required = ["training-submission"]
        elif missing_final_status or not recorded_complete:
            stage = "eval_collect"
            next_action = (
                "Run autosft tools jobs eval collect once for each exact manifest: "
                f"{eval_manifest_summary}. Do not poll or parse event files."
            )
            skills_required = ["job-lifecycle"]
        else:
            stage = "complete"
            next_action = "Return the persisted execution result; submit nothing."
            skills_required = []
    elif active_tasks:
        stage = "eval_poll"
        queued_notes = [
            note
            for item in active_tasks
            if (
                note := _queue_wait_note(
                    item.get("status"),
                    _task_age_seconds(Path(item["events"]), item["task"]),
                )
            )
        ]
        next_action = (
            "Run autosft tools jobs eval status once for the exact manifest(s) "
            f"{eval_manifest_summary}; live tasks: "
            + ", ".join(item["task"] for item in active_tasks)
            + ". Then refresh this handoff; do not wrap status in a shell loop."
            + (queued_notes[0] if queued_notes else "")
        )
        skills_required = ["job-lifecycle"]
    elif failed_tasks:
        stage = "blocked"
        next_action = (
            "Report terminal eval failures; do not automatically replace tasks: "
            + ", ".join(item["task"] for item in failed_tasks)
            + ". With the executor enabled, the meta agent may explicitly delegate a retry to it "
            "in its task context; the executor then uses src/autosft/benchmarks/submit.py "
            "--retry-failed-task tasks/<id>. With the executor disabled, the meta agent may use "
            "that entrypoint directly. Preserve the original round, model, "
            "benchmark, backend, limit, and repeats."
        )
        skills_required = ["job-lifecycle", "evaluation-submission"]
    else:
        stage = "eval_collect"
        next_action = (
            "Run autosft tools jobs eval collect once for each exact manifest: "
            f"{eval_manifest_summary}. Do not poll or parse event files."
        )
        skills_required = ["job-lifecycle"]

    all_known_tasks = [
        task
        for task in [
            *(submission_task(record) for record in sft_records),
            *historical_eval_tasks,
        ]
        if task
    ]
    return {
        "schema_version": 2,
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "current_round": current_round,
        "run_name": selected_run_name,
        "stage": stage,
        "next_action": next_action,
        "skills_required": skills_required,
        "data_ready": data_ready,
        "sft": {
            "task": sft_task,
            "duplicate_tasks": [
                task
                for task in dict.fromkeys(
                    submission_task(record) for record in sft_records
                )
                if task and task != sft_task
            ],
            "status": sft_status,
            "submission_manifest": str(sft_manifest),
            "events": str(sft_events),
            "final_status": str(sft_final_path) if sft_final_path.is_file() else None,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
            "checkpoint_complete": checkpoint_complete,
            "latest_step": latest_step,
        },
        "conversion": conversion,
        "evaluation": {
            "requested_benchmarks": requested_benchmarks,
            "required_repeats": required_repeats,
            "submission_manifests": eval_submission_manifests,
            "event_files": eval_event_files,
            "tasks": tasks,
            "active_tasks": [item["task"] for item in active_tasks],
            "missing_benchmarks": missing_benchmarks,
            "missing_repeats": missing_repeats,
            "satisfied_benchmarks": satisfied_benchmarks,
            "duplicate_submission_groups": duplicate_groups,
        },
        "last_executor_failure": _latest_executor_failure(
            run_dir,
            round_id=current_round,
            exclude=disposed,
        ),
        "constraints": {
            "canonical_run_dir": str(run_dir),
            "do_not_resubmit_tasks": list(dict.fromkeys(all_known_tasks)),
            "poll_terminal_tasks": False,
            "one_status_invocation_per_poll": True,
            "manual_lifecycle_loops": False,
        },
    }


def refresh_executor_handoff(
    run_dir: Path,
    *,
    round_id: str | None = None,
    run_name: str | None = None,
    benchmarks: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = Path(run_dir).resolve() / HANDOFF_FILENAME
    payload = build_executor_handoff(
        run_dir,
        round_id=round_id,
        run_name=run_name,
        benchmarks=benchmarks,
    )
    _atomic_write(path, payload)
    return path, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--run-dir", type=Path, required=True)
    refresh.add_argument("--round-id")
    refresh.add_argument("--run-name")
    refresh.add_argument("--benchmarks")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path, payload = refresh_executor_handoff(
        args.run_dir,
        round_id=args.round_id,
        run_name=args.run_name,
        benchmarks=parse_benchmarks(args.benchmarks) if args.benchmarks else None,
    )
    print(
        json.dumps(
            {
                "handoff_path": str(path),
                "current_round": payload["current_round"],
                "stage": payload["stage"],
                "next_action": payload["next_action"],
                "skills_required": payload["skills_required"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
