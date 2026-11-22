"""Durable waits for recorded batches owned by one synthesis worker.

The worker registers receipt paths under its output directory and ends its
turn. Its runtime waits outside Codex, then resumes the same thread when a job
stops. A stopped process is a wake-up condition, never proof of batch success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

from autosft.agents.root_wait import _local_path, _lock_held
from autosft.agents.search_tasks import _gateway_process_start
from autosft.synthesis._internal.filesystem import (
    atomic_write_json,
    exclusive_file_lock,
)


_MARKER = ".synthesis_batch_wait.json"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _root(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError("output_dir must be an existing directory")
    boundary = os.environ.get("AUTO_SFT_SYN_WRITE_ROOT")
    if os.environ.get("AUTO_SFT_SYN_WORKER") == "1" and (
        not boundary or root != Path(boundary).resolve()
    ):
        raise ValueError("batch waits must belong to the worker output directory")
    return root


def _snapshot(root: Path) -> tuple[dict[str, Any] | None, str]:
    path = _local_path(root, root / _MARKER)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, ""
    generation = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("invalid batch wait version")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("batch wait must contain jobs")
        for job in jobs:
            if not isinstance(job, dict) or not _LABEL.fullmatch(job.get("label", "")):
                raise ValueError("invalid batch label")
            if "pid" in job and (
                type(job["pid"]) is not int or job["pid"] <= 0
                or not isinstance(job.get("start_time"), str)
            ):
                raise ValueError("invalid batch process identity")
    except (ValueError, TypeError, UnicodeError):
        return {"error": "Invalid batch wait marker; inspect batch logs and register again."}, generation
    return payload, generation


def wait_status(output_dir: Path) -> dict[str, Any]:
    """Read registered receipt locks and process identities without mutation."""
    root = _root(output_dir)
    payload, generation = _snapshot(root)
    if payload is None:
        return {"status": "idle", "generation": generation, "jobs": []}
    if "error" in payload:
        return {"status": "ready", "generation": generation, "jobs": [], "error": payload["error"]}
    jobs = []
    for job in payload["jobs"]:
        lock = _local_path(root, root / "llm_calls" / "round_syn" / f"{job['label']}.lock")
        running = _lock_held(lock)
        if not running and "pid" in job:
            running = _gateway_process_start(root, {"round_id": "syn"}, job["label"], job["pid"]) == job["start_time"]
        jobs.append({**job, "status": "waiting" if running else "ready"})
    return {
        "status": "ready" if any(job["status"] == "ready" for job in jobs) else "waiting",
        "generation": generation,
        "jobs": jobs,
    }


def register_wait(output_dir: Path, *, label: str, job_lock: Path, job_pid: int) -> dict[str, Any]:
    """Register the exact detached receipt; retain other outstanding batches."""
    root = _root(output_dir)
    if not _LABEL.fullmatch(label):
        raise ValueError("invalid batch label")
    expected = root / "llm_calls" / "round_syn" / f"{label}.lock"
    if _local_path(root, job_lock) != expected:
        raise ValueError("job_lock must match the receipt for round_syn and this label")
    start_time = _gateway_process_start(root, {"round_id": "syn"}, label, job_pid)
    manifest_path = _local_path(root, expected.with_suffix(".manifest.json"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("round_id") != "syn" or manifest.get("label") != label:
            raise ValueError("batch manifest does not match this receipt")
    elif start_time is None:
        raise ValueError("batch process is not running and its manifest is absent; inspect the detach log")
    job: dict[str, Any] = {"label": label}
    if start_time is not None:
        job.update(pid=job_pid, start_time=start_time)
    with exclusive_file_lock(_local_path(root, root / ".synthesis_batch_wait.lock")):
        jobs = [
            {key: value for key, value in item.items() if key != "status"}
            for item in wait_status(root)["jobs"]
            if item["label"] != label and item["status"] == "waiting"
        ]
        atomic_write_json(_local_path(root, root / _MARKER), {
            "version": 1, "registration_id": uuid.uuid4().hex, "jobs": [*jobs, job],
        }, mode=0o600)
    return wait_status(root)


def acknowledge_wait(output_dir: Path, generation: str) -> None:
    """Consume only the marker observed by the runtime before its wake-up."""
    root = _root(output_dir)
    with exclusive_file_lock(_local_path(root, root / ".synthesis_batch_wait.lock")):
        _, current = _snapshot(root)
        if generation and current == generation:
            _local_path(root, root / _MARKER).unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-label", required=True)
    parser.add_argument("--job-lock", type=Path, required=True)
    parser.add_argument("--job-pid", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = register_wait(args.output_dir, label=args.job_label, job_lock=args.job_lock, job_pid=args.job_pid)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
