"""Track unresolved earlier-round failures and digest-bound dispositions.

Preparation proceeds only after a failed earlier round is recovered or its
exact terminal evidence is acknowledged by the meta agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from autosft.core.common import utc_now, write_json
from autosft.orchestration.ledger import read_iterations, record_round_id
from autosft.core.round_ids import (
    RoundIdError,
    logical_id_from_storage,
    logical_round_id,
)

EXECUTOR_REPORT_DIR = "executor_reports"


def _report_paths(run_dir: Path) -> list[tuple[str, Path]]:
    directory = Path(run_dir) / EXECUTOR_REPORT_DIR
    if not directory.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(directory.glob("round_*.json")):
        try:
            round_id = logical_id_from_storage(path.stem, label="executor report round")
        except RoundIdError:
            continue
        found.append((round_id, path))
    return sorted(found, key=lambda item: int(item[0]))


def blocked_rounds(run_dir: Path) -> list[dict[str, Any]]:
    """Rounds with an unrecoverable executor report or failed local SFT task.

    ``recoverable`` is the executor's own judgement about whether the round
    can continue. A recoverable block is a wait or a repairable
    precondition — a queue park reports exactly that shape — and the harness
    already handles those. Only an unrecoverable one means the round is over
    without a result.
    """
    results: list[dict[str, Any]] = []
    for round_id, path in _report_paths(run_dir):
        try:
            content = path.read_bytes()
            payload = json.loads(content)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "blocked" or not payload.get("persisted_at"):
            continue
        details = payload.get("failure_details")
        if not isinstance(details, dict):
            continue
        # Absent means unknown, and an unknown block is not evidence that the
        # round is unrecoverable; only an explicit false is.
        if details.get("recoverable") is not False:
            continue
        results.append(
            {
                "round_id": round_id,
                "report_path": str(path),
                "report_sha256": hashlib.sha256(content).hexdigest(),
                "stage": str(details.get("stage") or ""),
                "code": str(details.get("code") or ""),
                "message": str(details.get("message") or ""),
            }
        )
    reported = {item["round_id"] for item in results}
    results.extend(
        item for item in _failed_sft_rounds(run_dir) if item["round_id"] not in reported
    )
    return sorted(results, key=lambda item: int(item["round_id"]))


def _failed_sft_rounds(run_dir: Path) -> list[dict[str, Any]]:
    """Only manifest-bound terminal failure receipts can replace an executor report."""
    from autosft.backends.local.tasks import task_path
    from autosft.orchestration.submission_registry import (
        submission_records,
        submission_task,
    )

    results = []
    for entry in submission_records(Path(run_dir) / "sft_submissions.jsonl"):
        task = submission_task(entry)
        if not task or not task.startswith("tasks/local-"):
            continue
        try:
            round_id = logical_id_from_storage(
                entry.get("run_name"), label="SFT run_name"
            )
            path = task_path(run_dir, task)
            content = path.read_bytes()
            receipt = json.loads(content)
        except (OSError, ValueError):
            continue
        if not isinstance(receipt, dict) or receipt.get("status") not in {
            "Failed",
            "Stopped",
        }:
            continue
        expected_output = (Path(run_dir) / "sft" / f"round_{round_id}").resolve()
        if (
            receipt.get("task") != task
            or receipt.get("name") != task
            or receipt.get("run_name") != entry.get("run_name")
            or receipt.get("benchmark") is not None
            or not isinstance(receipt.get("dataset_path"), str)
            or not receipt["dataset_path"]
            or any(
                receipt.get(key) != entry.get(key)
                for key in (
                    "output_dir",
                    "dataset_path",
                    "train_iters",
                    "seq_length",
                    "base_model",
                )
            )
            or not isinstance(entry.get("output_dir"), str)
            or Path(entry["output_dir"]).resolve() != expected_output
        ):
            continue
        # Bind the acknowledgement to both manifest identity and current
        # failure bytes. A changed task/failure can never inherit an old ack.
        identity = json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()
        results.append(
            {
                "round_id": round_id,
                "report_path": str(path),
                "report_sha256": hashlib.sha256(identity + b"\n" + content).hexdigest(),
                "stage": "sft",
                "code": "local_sft_failed",
                "message": str(
                    receipt.get("error") or f"local SFT ended {receipt['status']}"
                ),
                "task": task,
            }
        )
    return results


def _resolved_rounds(run_dir: Path) -> set[str]:
    """Rounds that obtained evaluation results after being blocked."""
    resolved: set[str] = set()
    from autosft.orchestration.round_gates import evaluation_problems

    try:
        iterations = read_iterations(run_dir)
    except (OSError, ValueError):
        return resolved
    for record in iterations:
        try:
            round_id = record_round_id(record)
        except (ValueError, RoundIdError):
            continue
        if round_id is None:
            continue
        if not evaluation_problems(record, run_dir):
            resolved.add(round_id)
    return resolved


def _acknowledgement_path(run_dir: Path, round_id: str) -> Path:
    return (
        Path(run_dir) / EXECUTOR_REPORT_DIR / "acknowledged" / f"round_{round_id}.json"
    )


def _acknowledged(run_dir: Path, item: dict[str, Any]) -> bool:
    path = _acknowledgement_path(run_dir, item["round_id"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("round_id") == item["round_id"]
        and payload.get("report_sha256") == item["report_sha256"]
        and bool(payload.get("acknowledged_at"))
    )


def acknowledge(run_dir: Path, *, round_id: str) -> dict[str, Any]:
    """The meta agent acknowledges a specific failed execution without recording an interpretation."""
    roles = [
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    ]
    if (
        os.environ.get("AUTO_SFT_TASK_DIR")
        or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")
        or any(role and role not in {"meta"} for role in roles)
    ):
        raise ValueError(
            "only the meta agent may acknowledge a blocked executor report or failed local SFT"
        )
    canonical = logical_round_id(round_id)
    item = next(
        (item for item in blocked_rounds(run_dir) if item["round_id"] == canonical),
        None,
    )
    if item is None:
        raise ValueError(
            f"round {canonical} has no unrecoverable blocked executor report "
            "or manifest-bound failed local SFT receipt"
        )
    path = _acknowledgement_path(run_dir, canonical)
    if _acknowledged(run_dir, item):
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "round_id": canonical,
        "report_sha256": item["report_sha256"],
        "acknowledged_at": utc_now(),
    }
    write_json(path, payload)
    return payload


def acknowledged_rounds(run_dir: Path) -> set[str]:
    """Failed executions root explicitly disposed of, bound to current reports."""
    return {
        item["round_id"]
        for item in blocked_rounds(run_dir)
        if _acknowledged(run_dir, item)
    }


def unresolved_blocked_rounds(run_dir: Path) -> list[dict[str, Any]]:
    """Blocked rounds still awaiting root acknowledgement or evaluation."""
    resolved = _resolved_rounds(run_dir)
    return [
        item
        for item in blocked_rounds(run_dir)
        if item["round_id"] not in resolved and not _acknowledged(run_dir, item)
    ]


def describe(items: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"round {item['round_id']} blocked at {item['stage'] or 'an unknown stage'}"
        f" ({item['code'] or 'no code'})"
        for item in items
    )


def _gate_message(items: list[dict[str, Any]], *, requested_round: str) -> str:
    listed = "\n".join(
        f"  - round {item['round_id']}: {item['code'] or 'blocked'} at "
        f"{item['stage'] or 'unknown stage'}"
        + (f" — {item['message']}" if item["message"] else "")
        + f"\n    report: {item['report_path']}"
        for item in items
    )
    rounds = ", ".join(item["round_id"] for item in items)
    return (
        f"round {requested_round} cannot be submitted while an earlier round "
        f"is still unresolved:\n{listed}\n"
        f"That round's execution ended without a complete result, and no "
        f"decision has accounted for it. Opening a new "
        f"round now would bury it exactly as before.\n"
        f"Resolve it first, by either:\n"
        f"  1. completing recoverable conversion/evaluation for round {rounds} "
        f"through the documented lifecycle entrypoints, or\n"
        f"  2. acknowledging each failed execution:\n"
        f"     python -m autosft.orchestration.blocked_rounds acknowledge --run-dir "
        f"<run-dir> --round-id <id>\n"
        f"Use each listed round ID. The acknowledgement records the current "
        f"failure report and permits continuing without its result."
    )


def assert_no_unresolved_blocked_round(run_dir: Path, *, requested_round: str) -> None:
    """Refuse to train a *different* round while one is unresolved.

    Work on the blocked round itself stays open: a revision, a retry, or a
    resubmission of that same round is how it gets fixed.
    """
    try:
        canonical = logical_round_id(requested_round)
    except RoundIdError:
        return
    pending = [
        item
        for item in unresolved_blocked_rounds(run_dir)
        if item["round_id"] != canonical
    ]
    if pending:
        raise SystemExit("sft: " + _gate_message(pending, requested_round=canonical))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("list", "Print unresolved blocked rounds as JSON; exit 3 if any."),
        ("check", "Exit 3 when a new round would be gated."),
        (
            "acknowledge",
            "Acknowledge the current executor or local SFT failure as root.",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run-dir", type=Path, required=True)
        if name in {"check", "acknowledge"}:
            command.add_argument("--round-id", required=True)
    args = parser.parse_args(argv)

    if args.command == "acknowledge":
        try:
            payload = acknowledge(args.run_dir, round_id=args.round_id)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    items = unresolved_blocked_rounds(args.run_dir)
    if args.command == "list":
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 3 if items else 0

    try:
        assert_no_unresolved_blocked_round(args.run_dir, requested_round=args.round_id)
    except SystemExit as exc:
        print(str(exc.code), flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
