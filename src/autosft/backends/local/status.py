#!/usr/bin/env python3
"""Compact, replayable Local status events.

The lifecycle shell entrypoints use this module to avoid putting an unchanged
full local process specification into the agent context on every poll.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autosft.orchestration.submission_registry import (
    submission_records,
    submission_task,
)


TERMINAL_STATUSES = frozenset(
    {
        "Succeeded",
        "Failed",
        "Stopped",
        "Killed",
        "Cancelled",
        "Canceled",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
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
    return records


def latest_task_statuses(events_file: Path) -> dict[str, str]:
    """Return the latest non-empty status recorded for every task."""
    statuses: dict[str, str] = {}
    for record in _jsonl_records(Path(events_file)):
        task = record.get("task")
        status = record.get("status")
        if isinstance(task, str) and task and isinstance(status, str) and status:
            statuses[task] = status
    return statuses


def manifest_tasks(manifest: Path) -> list[str]:
    """Return unique submission task ids in manifest order."""
    tasks: list[str] = []
    seen: set[str] = set()
    for record in submission_records(Path(manifest)):
        task = submission_task(record)
        if task is not None and task not in seen:
            seen.add(task)
            tasks.append(task)
    return tasks


def pending_tasks(
    manifest: Path,
    events_file: Path,
    *,
    include_terminal: bool = False,
) -> list[str]:
    """Select tasks that still need polling.

    A task with no event is included once. After a terminal event is persisted,
    default manifest-driven polls leave it alone. Explicit task arguments in the
    shell wrappers bypass this selector and can still recheck a terminal task.
    """
    tasks = manifest_tasks(manifest)
    if include_terminal:
        return tasks
    statuses = latest_task_statuses(events_file)
    return [task for task in tasks if statuses.get(task) not in TERMINAL_STATUSES]


def _bounded(value: Any, *, limit: int = 2000) -> Any:
    if value is None:
        return None
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(value)
    if len(rendered) <= limit:
        return value
    return rendered[:limit] + "...<truncated>"


def compact_status(parsed: dict[str, Any], *, task: str) -> dict[str, Any]:
    """Keep only fields that can change or drive the next lifecycle action."""
    return {
        "task": task,
        "name": parsed.get("name"),
        "status": parsed.get("status"),
        "create_time": (
            parsed.get("createTime")
            or parsed.get("creationTime")
            or parsed.get("createdAt")
        ),
        "update_time": parsed.get("updateTime") or parsed.get("updatedAt"),
        "error": _bounded(
            parsed.get("error")
            or parsed.get("lastError")
            or parsed.get("failureReason")
        ),
        "artifact_paths": _bounded(
            parsed.get("artifactPaths")
            or parsed.get("artifacts")
            or parsed.get("outputs")
        ),
    }


def append_status_event(
    events_file: Path,
    *,
    task: str,
    local_task_exit_code: int,
    raw_response: str,
    source: str,
    polled_at: str | None = None,
) -> dict[str, Any]:
    """Append one compact poll event and return its human-facing summary."""
    events_file = Path(events_file)
    previous_status = latest_task_statuses(events_file).get(task)
    summary: dict[str, Any] = {
        "task": task,
        "status": None,
        "previous_status": previous_status,
        "changed": False,
        "create_time": None,
        "update_time": None,
        "error": None,
        "artifact_paths": None,
    }

    raw = raw_response.strip()
    if local_task_exit_code == 0 and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            summary["error"] = (
                f"invalid local_task JSON: {exc}; response="
                f"{raw[:2000]}{'...<truncated>' if len(raw) > 2000 else ''}"
            )
        else:
            if isinstance(parsed, dict):
                compact = compact_status(parsed, task=task)
                summary.update(
                    {
                        key: value
                        for key, value in compact.items()
                        if key != "name" or value is not None
                    }
                )
            else:
                summary["error"] = "local_task JSON response was not an object"
    elif local_task_exit_code != 0:
        summary["error"] = f"local_task exited with code {local_task_exit_code}"
    else:
        summary["error"] = "local_task returned empty stdout"

    current_status = summary.get("status")
    summary["changed"] = bool(
        isinstance(current_status, str) and current_status != previous_status
    )
    event = {
        "event": "status_poll",
        **summary,
        "polled_at": polled_at or _utc_now(),
        "source": source,
        "local_task_exit_code": local_task_exit_code,
    }
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with events_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pending = subparsers.add_parser("pending")
    pending.add_argument("--manifest", type=Path, required=True)
    pending.add_argument("--events-file", type=Path, required=True)
    pending.add_argument("--include-terminal", action="store_true")

    record = subparsers.add_parser("record")
    record.add_argument("--events-file", type=Path, required=True)
    record.add_argument("--task", required=True)
    record.add_argument("--local_task-exit-code", type=int, required=True)
    record.add_argument("--source", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "pending":
        for task in pending_tasks(
            args.manifest,
            args.events_file,
            include_terminal=args.include_terminal,
        ):
            print(task)
        return 0

    summary = append_status_event(
        args.events_file,
        task=args.task,
        local_task_exit_code=args.local_task_exit_code,
        raw_response=sys.stdin.read(),
        source=args.source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
