#!/usr/bin/env python3
"""Persist and inspect the executor's per-round report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from autosft.core.common import utc_now
from autosft.core.round_ids import RoundIdError, logical_round_id


EXECUTOR_JOB_TYPES = {"round", "recover", "finalize"}
# Terminal outcomes of a whole delegated job. ``phase_complete`` is the one
# non-terminal status: the meta agent may scope an executor delegation to a single
# stage (e.g. ``stage: sft_submit``), and a session that finished only its
# assigned stage must not claim the round succeeded — that is what previously
# let receipt recovery fabricate success for rounds whose SFT never ran.
EXECUTOR_TERMINAL_STATUSES = {"succeeded", "failed", "blocked"}
EXECUTOR_STATUSES = EXECUTOR_TERMINAL_STATUSES | {"phase_complete"}
# Stage names a phase_complete report may claim — the handoff state machine's
# vocabulary (executor_handoff.py build_executor_handoff). The executor's
# stages start at sft_submit: data assembly is the meta agent's own work and is
# never delegated.
EXECUTOR_STAGES = {
    "sft_submit",
    "sft_poll",
    "sft_collect",
    "checkpoint_conversion",
    "eval_submit",
    "eval_poll",
    "eval_collect",
}

EXECUTOR_REPORT_DIR = "executor_reports"


def _round_report_complete(payload: dict[str, Any]) -> bool:
    """True when a ``job_type=round`` report shows the full round actually ran.

    The meta agent may phase-split a round (e.g. submission first, eval collection
    later), and a partial session persists the same
    ``executor_reports/round_<n>.json`` as the finished round would. Status
    alone therefore cannot distinguish "round done" from "one phase done": a
    succeeded round is only credible when it names at least one SFT task and
    carries terminal SFT and evaluation artifacts. Scoped to job_type=round —
    finalize/recover legitimately have empty tasks.
    """
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not any(
        isinstance(task, dict) and task.get("kind") == "sft" for task in tasks
    ):
        return False
    artifacts = payload.get("result_artifacts")
    if not isinstance(artifacts, dict):
        return False
    sft = artifacts.get("sft")
    evaluation = artifacts.get("evaluation")
    return (
        isinstance(sft, dict)
        and bool(sft.get("final_status"))
        and isinstance(evaluation, dict)
        and bool(evaluation.get("final_status"))
    )


class ExecutorReportError(RuntimeError):
    pass


def executor_report_path(run_dir: Path, round_id: str) -> Path:
    """One canonical report file per round, owned by the persist command."""
    try:
        canonical_round = logical_round_id(round_id)
    except RoundIdError as exc:
        raise ExecutorReportError(str(exc)) from exc
    return Path(run_dir) / EXECUTOR_REPORT_DIR / f"round_{canonical_round}.json"


def validate_executor_report(
    payload: dict[str, Any],
    *,
    round_id: str,
) -> dict[str, Any]:
    """Check an executor report against the shape the executor actually emits.

    An executor reports what it ran and what it produced, and is forbidden
    from interpreting metrics, so it has no ``summary`` or graded
    ``key_findings`` to give. Requiring those would force it to invent
    judgements its own contract rules out.
    """
    try:
        canonical_round = logical_round_id(round_id)
    except RoundIdError as exc:
        raise ExecutorReportError(str(exc)) from exc
    payload_round = str(payload.get("round_id") or "")
    if payload_round != canonical_round:
        raise ExecutorReportError(
            f"report round_id must be {canonical_round!r}, got {payload_round!r}"
        )
    job_type = payload.get("job_type")
    if job_type not in EXECUTOR_JOB_TYPES:
        raise ExecutorReportError(
            f"executor job_type must be one of {sorted(EXECUTOR_JOB_TYPES)}, got {job_type!r}"
        )
    status = payload.get("status")
    if status not in EXECUTOR_STATUSES:
        raise ExecutorReportError(
            f"executor status must be one of {sorted(EXECUTOR_STATUSES)}, got {status!r}"
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ExecutorReportError("executor tasks must be a list")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ExecutorReportError(f"tasks[{index}] must be an object")
        for field in ("kind", "task", "status"):
            if not isinstance(task.get(field), str):
                raise ExecutorReportError(f"tasks[{index}].{field} must be a string")
    artifacts = payload.get("result_artifacts")
    if not isinstance(artifacts, dict):
        raise ExecutorReportError("executor result_artifacts must be an object")
    if (
        job_type == "round"
        and status == "succeeded"
        and not _round_report_complete(payload)
    ):
        raise ExecutorReportError(
            "a succeeded round report must name at least one sft task and carry "
            "non-null result_artifacts.sft.final_status and "
            "result_artifacts.evaluation.final_status; a session that finished "
            "only one stage must report status phase_complete with the stage it "
            "completed, or failed/blocked with failure_details"
        )
    if status == "phase_complete":
        stage = payload.get("stage")
        if stage not in EXECUTOR_STAGES:
            raise ExecutorReportError(
                "a phase_complete report must carry a stage field naming the "
                f"completed stage, one of {sorted(EXECUTOR_STAGES)}; got {stage!r}"
            )
    return {
        **payload,
        "persisted_at": utc_now(),
    }


def _require_root_or_executor_writer() -> None:
    roles = {
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {None, "", "meta", "executor"}
    if (
        roles
        or os.environ.get("AUTO_SFT_TASK_DIR")
        or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")
    ):
        raise ExecutorReportError(
            "task roles publish their own report or wait through search_tasks; "
            "the executor record is root/harness-owned"
        )


def persist_executor_report(
    run_dir: Path,
    *,
    round_id: str,
    payload: dict[str, Any],
) -> Path:
    _require_root_or_executor_writer()
    validated = validate_executor_report(payload, round_id=round_id)
    return _atomic_write(executor_report_path(run_dir, round_id), validated)


def _atomic_write(destination: Path, payload: dict[str, Any]) -> Path:
    """Write via a temp file in the same directory, then rename into place.

    A phase-split round rewrites the same report file at every stage boundary,
    so a half-written report must never be observable to a concurrent reader.
    """
    _require_root_or_executor_writer()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temp = Path(stream.name)
    os.replace(temp, destination)
    return destination


def _load_payload(input_path: Path | None) -> dict[str, Any]:
    raw = (
        input_path.read_text(encoding="utf-8")
        if input_path is not None
        else sys.stdin.read()
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExecutorReportError(f"invalid report JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutorReportError("executor report must be a JSON object")
    return payload


def receipt_from_persisted_report(
    run_dir: Path,
    *,
    round_id: str,
) -> dict[str, Any] | None:
    """Rebuild the executor's receipt from the report it already persisted.

    The receipt is written by ``codex --output-last-message``, so a session
    killed after persisting its report but before returning leaves a complete
    report and no receipt. The launcher's completion check is the receipt, so
    without this the meta agent relaunches an executor whose work is already
    durable.

    Returns ``None`` when no valid report exists — a genuinely unfinished job
    must still be resumed.
    """
    try:
        path = executor_report_path(run_dir, round_id)
    except ExecutorReportError:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    # Only a report the persist command actually wrote counts: it stamps
    # persisted_at after validating, so a staging copy cannot be mistaken for
    # a completed delegation.
    if not payload.get("persisted_at"):
        return None
    status = payload.get("status")
    job_type = payload.get("job_type")
    if status not in EXECUTOR_STATUSES or job_type not in EXECUTOR_JOB_TYPES:
        return None
    # Never fabricate a receipt from a non-terminal report: the per-round
    # report file is shared by every phase of a phase-split round, so a
    # phase_complete report proves nothing about the delegation that died.
    if status not in EXECUTOR_TERMINAL_STATUSES:
        return None
    # A succeeded round report must actually show the round ran to
    # completion. Without this, a partial (submission-only) report that
    # claimed success would be recovered into a succeeded receipt whenever
    # a later-phase session dies receiptless — and the meta agent would silently
    # skip SFT and evaluation for the round.
    if (
        status == "succeeded"
        and job_type == "round"
        and not _round_report_complete(payload)
    ):
        return None
    return {
        "status": status,
        "round_id": str(payload.get("round_id") or round_id),
        "job_type": job_type,
        "report_path": str(path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    persist_executor = subparsers.add_parser("persist-executor")
    persist_executor.add_argument("--run-dir", type=Path, required=True)
    persist_executor.add_argument("--round-id", required=True)
    persist_executor.add_argument("--input", type=Path, default=None)
    return parser.parse_args()


def _reject_gateway_staging(input_path: Path, run_dir: Path) -> None:
    """Refuse report inputs staged inside llm_calls/.

    That tree holds recorded provider jobs and is enumerated as model-call evidence. Stage report inputs under role_io/round_<id>/ alongside the session's other files.
    """
    staged = input_path.resolve()
    gateway_root = (run_dir.resolve() / "llm_calls").resolve()
    if staged == gateway_root or gateway_root in staged.parents:
        raise ExecutorReportError(
            f"do not stage the report input under {gateway_root} — that tree "
            "holds batch gateway job records, not reports; stage it at "
            "role_io/round_<n>/<role>.report_input.json under the run dir "
            "instead"
        )


def main() -> int:
    args = _parse_args()
    try:
        # Staging files that shadow canonical names become decoy state:
        # the run observed both a run-root executor_report.json (stale
        # "blocked", later read as truth by a resumed session) and a
        # executor_reports/round_1.executor_report.json matching scanner
        # globs. The persist command owns executor_reports/; inputs must
        # be staged elsewhere under the run dir.
        if args.input is not None:
            _reject_gateway_staging(args.input, args.run_dir)
            staged = args.input.resolve()
            run_dir = args.run_dir.resolve()
            if staged.parent == run_dir / EXECUTOR_REPORT_DIR:
                raise ExecutorReportError(
                    "stage the report input outside "
                    f"{run_dir / EXECUTOR_REPORT_DIR} — that directory is "
                    "owned by the persist command"
                )
            if staged.parent == run_dir and staged.name.startswith("executor_report"):
                raise ExecutorReportError(
                    "do not stage the report input as a run-root "
                    "executor_report*.json — it becomes a stale decoy for "
                    "later sessions; use role_io/round_<n>/ instead"
                )
        path = persist_executor_report(
            args.run_dir,
            round_id=args.round_id,
            payload=_load_payload(args.input),
        )
        result = {
            "status": "succeeded",
            "report_path": str(path),
            "round_id": args.round_id,
        }
    except ExecutorReportError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
