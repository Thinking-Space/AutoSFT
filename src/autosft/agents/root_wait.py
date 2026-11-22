"""Durable root waits for detached gateway work, independent of executor stages.

The harness polls this module without starting a model turn. A stopped job
means the root must inspect its result; it is never a success receipt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator
import uuid

from autosft.core.common import utc_now
from autosft.core.round_ids import logical_round_id
from .search_tasks import _gateway_process_start


_MARKER = "root_wait.json"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_MAX_MARKER_BYTES = 1_048_576


def _root(run_dir: Path) -> Path:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("run_dir must be an existing directory")
    return root


def _mutation_identity() -> None:
    roles = (os.environ.get(name) for name in
             ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE"))
    # The public runner names its controller ``meta`` (the reference
    # controller used ``root``).  Both are allowed to own the run-level wait;
    # executor and analysis task identities remain task-scoped and must use
    # their own wait protocol.
    if (any(role and role not in {"root", "coordinator", "meta"} for role in roles)
            or os.environ.get("AUTO_SFT_TASK_DIR")
            or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")):
        raise ValueError("only the root/meta controller may register, acknowledge or cancel a root wait")


def _local_path(root: Path, path: Path) -> Path:
    """Reject redirects, including symlinks whose targets remain inside the run."""
    path = path if path.is_absolute() else root / path
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("gateway and wait paths must remain under run_dir")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("gateway and wait paths must not contain symlinks")
    if path.resolve() != path:
        raise ValueError("gateway and wait paths must be canonical run-local paths")
    return path


@contextmanager
def _state_lock(root: Path) -> Iterator[None]:
    path = _local_path(root, root / ".root_wait.lock")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("root wait state lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _atomic_write(root: Path, payload: dict[str, Any]) -> None:
    path = _local_path(root, root / _MARKER)
    descriptor, name = tempfile.mkstemp(prefix=".root_wait.", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(root)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sync_directory(root: Path) -> None:
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _invalid(raw: bytes, detail: str) -> tuple[None, str, list[str]]:
    return None, "invalid:" + hashlib.sha256(raw).hexdigest(), [detail]


def _snapshot(root: Path) -> tuple[dict[str, Any] | None, str, list[str]]:
    """An atomic, read-only snapshot; damaged markers wake the root for repair."""
    path = root / _MARKER
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return _invalid(str(info).encode(), "Root wait marker is not a regular file; inspect or cancel it.")
            raw = stream.read(_MAX_MARKER_BYTES + 1)
    except FileNotFoundError:
        return None, "", []
    except OSError:
        try:
            identity = str(path.lstat()).encode()
        except OSError:
            identity = str(path).encode()
        return _invalid(identity, "Root wait marker cannot be read safely; inspect or cancel it.")
    try:
        if len(raw) > _MAX_MARKER_BYTES:
            raise ValueError("oversized marker")
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int
                or payload["version"] != 1
                or not isinstance(payload.get("generation"), str)
                or not _GENERATION.fullmatch(payload["generation"])
                or not isinstance(payload.get("jobs"), list) or not payload["jobs"]):
            raise ValueError("invalid marker schema")
        identities = set()
        for job in payload["jobs"]:
            if not isinstance(job, dict):
                raise ValueError("invalid job")
            round_id = logical_round_id(job.get("round_id"))
            label = job.get("job_label")
            if not isinstance(label, str) or not _LABEL.fullmatch(label):
                raise ValueError("invalid job label")
            if not isinstance(job.get("job_lock"), str) or not job["job_lock"]:
                raise ValueError("invalid job lock")
            identity = (round_id, label)
            if identity in identities:
                raise ValueError("duplicate job")
            identities.add(identity)
            if "job_pid" in job or "job_start_time" in job:
                if (type(job.get("job_pid")) is not int or job["job_pid"] <= 0
                        or not isinstance(job.get("job_start_time"), str)
                        or not job["job_start_time"].isdigit()):
                    raise ValueError("invalid process identity")
    except (ValueError, UnicodeError):
        return _invalid(raw, "Root wait marker is empty or damaged; inspect or register the wait again.")
    return payload, payload["generation"], []


def _gateway_lock(root: Path, round_id: str, label: str, supplied: Path | None,
                  *, allow_starting: bool) -> Path:
    expected = root / "llm_calls" / f"round_{round_id}" / f"{label}.lock"
    lock = _local_path(root, Path(supplied) if supplied is not None else expected)
    if lock != expected:
        raise ValueError("job_lock must name the canonical run-local gateway lock for this round and label")
    manifest_path = _local_path(root, expected.with_suffix(".manifest.json"))
    try:
        descriptor = os.open(manifest_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("gateway manifest must be a regular file")
            manifest = json.load(stream)
    except FileNotFoundError:
        if allow_starting:
            return lock
        raise ValueError("gateway manifest is not ready; provide --job-pid from the detached receipt") from None
    if (not isinstance(manifest, dict) or manifest.get("label") != label
            or logical_round_id(manifest.get("round_id")) != round_id):
        raise ValueError("gateway manifest does not match this job's label and round")
    return lock


def _lock_held(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("gateway lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)


def _job_status(root: Path, job: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    label = job["job_label"]
    round_id = logical_round_id(job["round_id"])
    try:
        lock = _gateway_lock(root, round_id, label, Path(job["job_lock"]),
                             allow_starting="job_pid" in job)
        running = _lock_held(lock)
        if not running and "job_pid" in job:
            running = (_gateway_process_start(root, {"round_id": round_id}, label, job["job_pid"])
                       == job["job_start_time"])
    except (OSError, ValueError) as error:
        return {**job, "status": "ready"}, f"Round {round_id} job {label}: wait validation failed ({type(error).__name__}); inspect its manifest and lock."
    if running:
        return {**job, "status": "waiting"}, None
    return ({**job, "status": "ready"},
            f"Round {round_id} job {label} stopped; inspect its results before deciding success or retry.")


def _clear_marker(root: Path) -> None:
    path = root / _MARKER
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError:
        # A malformed directory must not crash every successful root turn.
        # Preserve its contents for inspection without following or deleting
        # anything it contains, and free the canonical marker for recovery.
        os.replace(path, root / f".root_wait.invalid.{uuid.uuid4().hex}")
    _sync_directory(root)


def _status(root: Path) -> dict[str, Any]:
    payload, generation, events = _snapshot(root)
    if payload is None:
        return {"status": "ready" if events else "idle", "generation": generation,
                "jobs": [], "events": events}
    jobs = []
    for job in payload["jobs"]:
        current, event = _job_status(root, job)
        jobs.append(current)
        if event:
            events.append(event)
    return {"status": "ready" if events else "waiting", "generation": generation,
            "jobs": jobs, "events": events}


def wait_status(run_dir: Path) -> dict[str, Any]:
    """Read durable state without mutation; any stopped or invalid job wakes root."""
    return _status(_root(run_dir))


def wait_for_job(run_dir: Path, round_id: str | int, *, job_label: str,
                 job_lock: Path | None = None, job_pid: int | None = None) -> dict[str, Any]:
    """Register one dependency, retaining live peers and replacing this job once.

    Every registration changes generation, including an otherwise identical
    registration. A harness acknowledging the previous turn cannot erase it.
    """
    _mutation_identity()
    root = _root(run_dir)
    round_id = logical_round_id(round_id)
    if not isinstance(job_label, str) or not _LABEL.fullmatch(job_label):
        raise ValueError("job_label must contain 1-128 letters, digits, dots, underscores or hyphens and start with a letter or digit")
    with _state_lock(root):
        process = {}
        if job_pid is not None:
            start_time = _gateway_process_start(root, {"round_id": round_id}, job_label, job_pid)
            if start_time is None:
                raise ValueError("job_pid is not a running gateway runner for this run, round and label; inspect its results")
            process = {"job_pid": job_pid, "job_start_time": start_time}
        lock = _gateway_lock(root, round_id, job_label, job_lock, allow_starting=bool(process))
        job = {"round_id": round_id, "job_label": job_label, "job_lock": str(lock), **process}
        # Invalid lock types must fail registration, not silently establish a wait.
        _lock_held(lock)
        old = _status(root)
        jobs = [{key: value for key, value in previous.items() if key != "status"}
                for previous in old["jobs"] if previous["status"] == "waiting"
                and (str(previous["round_id"]), previous["job_label"]) != (round_id, job_label)]
        jobs.append(job)
        _atomic_write(root, {"version": 1, "generation": uuid.uuid4().hex,
                             "registered_at": utc_now(), "jobs": jobs})
        return _status(root)


def acknowledge_wait(run_dir: Path, generation: str) -> bool:
    """Clear only the snapshot delivered to a completed root turn."""
    _mutation_identity()
    root = _root(run_dir)
    if not isinstance(generation, str) or not generation:
        return False
    with _state_lock(root):
        _, current, _ = _snapshot(root)
        if current != generation:
            return False
        # Unlink the marker itself; never follow a damaged marker's symlink.
        _clear_marker(root)
        return True


def cancel_wait(run_dir: Path) -> dict[str, Any]:
    """Explicitly revoke all dependencies, without stopping gateway workers."""
    _mutation_identity()
    root = _root(run_dir)
    with _state_lock(root):
        _clear_marker(root)
        return _status(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("wait", "status", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        if name == "wait":
            command.add_argument("--round-id", required=True)
            command.add_argument("--job-label", required=True)
            command.add_argument("--job-lock", type=Path)
            command.add_argument("--job-pid", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "wait":
            result = wait_for_job(args.run_dir, args.round_id, job_label=args.job_label,
                                  job_lock=args.job_lock, job_pid=args.job_pid)
        elif args.command == "cancel":
            result = cancel_wait(args.run_dir)
        else:
            result = wait_status(args.run_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
