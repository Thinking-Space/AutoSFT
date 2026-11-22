"""Durable, explicitly requested analysis proposals for SFT data research.

Tasks are keyed by round and immutable task id, never just by role. Their
reports are advisory products: failed or parked tasks do not gate training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator

from autosft.config.components import analysis_enabled
from autosft.core.common import utc_now
from autosft.agents.roles import ROLE_CATALOG, TASK_ROLES
from autosft.core.round_ids import logical_round_id


SEARCH_ROLES = TASK_ROLES
TERMINAL_STATUSES = {"completed", "failed", "parked"}
TASK_ID_RE = re.compile(r"(?!.*__)[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(?<!_)\Z")
DEFAULT_MAX_ATTEMPTS = 2
_GATEWAY_RUNNER = (
    Path(__file__).resolve().parents[3] / "src/autosft/backends/providers/batch.py"
)


class _UnsettledDependency(ValueError):
    """A valid request must wait for an upstream task before it can run."""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _stat_key(stat: os.stat_result) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_file(path: str, expected: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        if _stat_key(os.fstat(stream.fileno())) != expected:
            raise ValueError(f"artifact changed while opening: {path}")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if _stat_key(os.fstat(stream.fileno())) != expected:
            raise ValueError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()


_cached_hash = lru_cache(maxsize=512)(_hash_file)


def _sha256(path: Path, *, force: bool = False) -> str:
    resolved = path.resolve()
    key = _stat_key(resolved.stat())
    # Some filesystems coalesce timestamps for rapid same-size writes. Let
    # newly written files settle before trusting their metadata as a cache key.
    force = force or time.time_ns() - max(key[-2:]) < 1_000_000_000
    return (_hash_file if force else _cached_hash)(str(resolved), key)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _lock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def task_directory(run_dir: Path, round_id: str | int, task_id: str) -> Path:
    if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(
            "task_id must start with a letter or digit and contain 1-96 letters, "
            "digits, dots, underscores or hyphens; '__' and a trailing '_' are "
            "reserved for gateway job ownership"
        )
    root = Path(run_dir).resolve()
    path = root / "role_io" / f"round_{logical_round_id(round_id)}" / "tasks" / task_id
    if not path.resolve().is_relative_to(root):
        raise ValueError("task directory must remain under run_dir")
    return path


def _mutation_identity(
    operation: str,
    run_dir: Path,
    round_id: str | int | None = None,
    task_id: str | None = None,
) -> None:
    if operation in {"request", "run", "run-pending"} and not analysis_enabled():
        raise ValueError("Analysis tasks are disabled for this run")
    role = os.environ.get("AUTO_SFT_TASK_ROLE")
    task_root = os.environ.get("AUTO_SFT_TASK_DIR")
    if role in {None, "", "meta"} and not task_root:
        return
    if role not in SEARCH_ROLES or operation not in {"persist", "wait"}:
        raise ValueError("task sessions may only persist or wait for their own task")
    if (
        not task_root
        or not round_id
        or not task_id
        or task_id != os.environ.get("AUTO_SFT_TASK_ID")
        or str(round_id) != os.environ.get("AUTO_SFT_TASK_ROUND_ID")
        or task_directory(run_dir, round_id, task_id).resolve()
        != Path(task_root).resolve()
        or _request(Path(task_root))["role"] != role
    ):
        raise ValueError("task session cannot persist or wait for another task")


def _references(
    run_dir: Path, value: Any, *, force_hash: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("artifact references must be a list")
    root = Path(run_dir).resolve()
    result = []
    for reference in value:
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            raise ValueError("each artifact reference requires a string path")
        path = Path(reference["path"])
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"artifact must be an existing run-local file: {path}")
        sha256 = _sha256(path, force=force_hash)
        if "sha256" in reference and reference["sha256"] != sha256:
            raise ValueError(f"artifact digest changed: {path}")
        pinned = {**reference, "path": str(path), "sha256": sha256}
        # The harness refreshes memory between root turns. A contributor must
        # retain the version its request consumed, even if root omitted the
        # optional snapshot flag.
        if path == root / "experiment_memory.json":
            pinned["snapshot"] = True
        result.append(pinned)
    return result


def _snapshot_references(
    run_dir: Path, directory: Path, references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for reference in references:
        source = Path(reference["path"])
        if reference.get("snapshot") is not True:
            result.append(reference)
            continue
        destination = directory / "inputs" / f"{reference['sha256']}{source.suffix}"
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if _sha256(destination) != reference["sha256"]:
                raise ValueError(f"immutable input snapshot was changed: {destination}")
        else:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".input-", dir=destination.parent
            )
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                if _sha256(Path(temporary)) != reference["sha256"]:
                    raise ValueError(f"input changed during snapshot: {source}")
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        result.append(
            {
                **reference,
                "path": str(destination),
                "source_path": str(source),
                "snapshot": True,
            }
        )
    return result


def _request(directory: Path) -> dict[str, Any]:
    request = _read_json(directory / "request.json")
    identity = {key: request[key] for key in ("round_id", "task_id", "role", "input")}
    if request.get("input_digest") != _digest(identity):
        raise ValueError(f"immutable request was changed: {directory}")
    return request


def _state(directory: Path) -> dict[str, Any]:
    path = directory / "state.json"
    return _read_json(path) if path.exists() else {"status": "pending", "attempts": 0}


def _set_state(directory: Path, **updates: Any) -> dict[str, Any]:
    with _lock(directory / ".state.lock"):
        if updates.get("status") in TERMINAL_STATUSES:
            updates.setdefault("finished_at", utc_now())
        state = {**_state(directory), **updates, "updated_at": utc_now()}
        _atomic_write(directory / "state.json", state)
        return state


def _task_context(directory: Path, request: dict[str, Any]) -> str:
    return (
        f"# Requested task {request['task_id']}\n\n"
        f"Round: {request['round_id']}\nRole: {request['role']}\n"
        f"Input digest: {request['input_digest']}\n"
        f"Task directory: {directory}\nScratch directory: {directory / 'scratch'}\n\n"
        "The request and named input versions are immutable. Use a new task id for changed work.\n"
        "Persist a report identifying this task, round, role and input_digest.\n\n"
        "```json\n" + json.dumps(request, indent=2, sort_keys=True) + "\n```\n"
    )


def request_task(
    run_dir: Path,
    round_id: str | int,
    task_id: str,
    role: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a request; identical re-registration is safe, changes need a new id."""
    _mutation_identity("request", run_dir, round_id, task_id)
    if role not in SEARCH_ROLES:
        raise ValueError(f"unknown task role: {role}")
    if not isinstance(payload, dict):
        raise ValueError("task input must be a JSON object")
    if (
        not isinstance(payload.get("objective"), str)
        or not payload["objective"].strip()
    ):
        raise ValueError("task input requires a non-empty objective")
    round_id = logical_round_id(round_id)
    directory = task_directory(run_dir, round_id, task_id)
    normalized = json.loads(_json_bytes(payload))
    normalized["input_refs"] = _references(
        run_dir, normalized.get("input_refs", []), force_hash=True
    )
    dependencies = normalized.get("dependencies", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise ValueError("dependencies must be a list of task ids from this round")
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("dependencies must not repeat task ids")
    dependency_inputs = []
    for dependency in dependencies:
        if dependency == task_id:
            raise ValueError("a task cannot depend on itself")
        source = _request(task_directory(run_dir, round_id, dependency))
        dependency_inputs.append(
            {"task_id": dependency, "input_digest": source["input_digest"]}
        )
    normalized["dependencies"] = dependencies
    normalized["dependency_inputs"] = dependency_inputs
    attempts = normalized.setdefault("max_attempts", DEFAULT_MAX_ATTEMPTS)
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("max_attempts must be an integer between 1 and 3")
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory / ".request.lock"):
        normalized["input_refs"] = _snapshot_references(
            run_dir, directory, normalized["input_refs"]
        )
        identity = {
            "round_id": round_id,
            "task_id": task_id,
            "role": role,
            "input": normalized,
        }
        request = {
            "schema_version": "1.0",
            **identity,
            "input_digest": _digest(identity),
        }
        if (directory / "request.json").exists():
            existing = _request(directory)
            if existing["input_digest"] != request["input_digest"]:
                raise ValueError(
                    f"task id {task_id!r} already has different immutable input; use a new id"
                )
        else:
            _atomic_write(directory / "request.json", request)
        (directory / "scratch").mkdir(exist_ok=True)
        context_path = directory / "task_context.md"
        context = _task_context(directory, request)
        if (
            context_path.exists()
            and context_path.read_text(encoding="utf-8") != context
        ):
            raise ValueError(
                "immutable task context was changed; request a new task id"
            )
        if not context_path.exists():
            context_path.write_text(context, encoding="utf-8")
        if not (directory / "state.json").exists():
            _set_state(
                directory, status="pending", attempts=0, detail="requested by root"
            )
    return task_status(run_dir, round_id, task_id)


def _dependency_snapshot(
    run_dir: Path, request: dict[str, Any], *, status_cache: dict | None = None
) -> list[dict[str, Any]]:
    status_cache = {} if status_cache is None else status_cache
    outcomes = []
    for dependency in request["input"]["dependency_inputs"]:
        status = task_status(
            run_dir,
            request["round_id"],
            dependency["task_id"],
            _status_cache=status_cache,
        )
        if status["input_digest"] != dependency["input_digest"]:
            raise ValueError(f"dependency input changed: {dependency['task_id']}")
        if status["status"] not in TERMINAL_STATUSES:
            raise _UnsettledDependency(
                f"dependency is not settled: {dependency['task_id']}"
            )
        outcomes.append(
            {
                key: status[key]
                for key in (
                    "task_id",
                    "role",
                    "input_digest",
                    "status",
                    "detail",
                    "report_path",
                    "report_sha256",
                )
                if key in status
            }
        )
    return outcomes


def _execution_inputs(
    run_dir: Path,
    directory: Path,
    request: dict[str, Any],
    *,
    create: bool = False,
    force_hash: bool = False,
    status_cache: dict | None = None,
) -> dict[str, Any]:
    if (directory / "task_context.md").read_text(encoding="utf-8") != _task_context(
        directory, request
    ):
        raise ValueError("immutable task context was changed; request a new task id")
    _references(run_dir, request["input"]["input_refs"], force_hash=force_hash)
    snapshot = {
        "input_digest": request["input_digest"],
        "dependencies": _dependency_snapshot(
            run_dir, request, status_cache=status_cache
        ),
    }
    path = directory / "execution_inputs.json"
    if path.exists():
        if _read_json(path) != snapshot:
            raise ValueError(
                "dependency outcomes changed after this task started; request a new task id"
            )
    elif create:
        _atomic_write(path, snapshot)
    else:
        raise ValueError("task report is missing its pinned execution inputs")
    return snapshot


def _validated_report(
    run_dir: Path,
    directory: Path,
    report: dict[str, Any],
    *,
    create_inputs: bool = False,
    force_hash: bool = False,
    status_cache: dict | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("task report must be a JSON object")
    request = _request(directory)
    for key in ("task_id", "round_id", "role", "input_digest"):
        value = (
            logical_round_id(report.get(key)) if key == "round_id" else report.get(key)
        )
        if value != request[key]:
            raise ValueError(f"report {key} does not match immutable task input")
    if not isinstance(report.get("summary"), str) or not report["summary"].strip():
        raise ValueError("report requires a non-empty summary")
    _execution_inputs(
        run_dir,
        directory,
        request,
        create=create_inputs,
        force_hash=force_hash,
        status_cache=status_cache,
    )
    normalized = {**report, "round_id": request["round_id"]}
    normalized["result_artifacts"] = _references(
        run_dir, report.get("result_artifacts"), force_hash=force_hash
    )
    return normalized


def _receipt(
    run_dir: Path, directory: Path, *, status_cache: dict | None = None
) -> dict[str, Any] | None:
    path = directory / "report.json"
    if not path.exists():
        return None
    report = _validated_report(
        run_dir, directory, _read_json(path), status_cache=status_cache
    )
    if not report.get("persisted_at"):
        raise ValueError("report is missing its persistence stamp")
    return {
        **{key: report[key] for key in ("task_id", "round_id", "role", "input_digest")},
        "status": "completed",
        "report_path": str(path),
        "report_sha256": _sha256(path),
    }


def persist_report(
    run_dir: Path, round_id: str | int, task_id: str, report: dict[str, Any]
) -> dict[str, Any]:
    _mutation_identity("persist", run_dir, round_id, task_id)
    directory = task_directory(run_dir, round_id, task_id)
    with _lock(directory / ".state.lock"):
        state = _state(directory)
        if state["status"] in {"failed", "parked"}:
            raise ValueError(
                "terminal task cannot accept a new report; request a new task id"
            )
        normalized = _validated_report(
            run_dir, directory, report, create_inputs=True, force_hash=True
        )
        normalized.pop("persisted_at", None)
        path = directory / "report.json"
        if path.exists():
            existing = _read_json(path)
            existing.pop("persisted_at", None)
            if existing != normalized:
                raise ValueError("task report is immutable; request a new task id")
        else:
            _atomic_write(path, {**normalized, "persisted_at": utc_now()})
        receipt = _receipt(run_dir, directory)
        assert receipt is not None
        _atomic_write(directory / "receipt.json", receipt)
        _atomic_write(
            directory / "state.json",
            {
                **state,
                "status": "completed",
                "detail": "report persisted",
                "updated_at": utc_now(),
                "finished_at": state.get("finished_at", utc_now()),
            },
        )
        (directory / "waiting.json").unlink(missing_ok=True)
        return receipt


def task_status(
    run_dir: Path,
    round_id: str | int,
    task_id: str,
    *,
    _status_cache: dict | None = None,
) -> dict[str, Any]:
    _status_cache = {} if _status_cache is None else _status_cache
    key = (str(round_id), task_id)
    if key in _status_cache:
        return _status_cache[key]
    directory = task_directory(run_dir, round_id, task_id)
    request = _request(directory)
    state = _state(directory)
    result = {
        **{
            key: request[key] for key in ("task_id", "round_id", "role", "input_digest")
        },
        **state,
        "task_dir": str(directory),
        "dependencies": request["input"]["dependencies"],
    }
    try:
        receipt = _receipt(run_dir, directory, status_cache=_status_cache)
    except (ValueError, OSError, KeyError) as error:
        result.update(status="failed", detail=f"invalid persisted report: {error}")
    else:
        if receipt is not None:
            result.update(receipt, detail="report persisted")
        elif state["status"] == "completed":
            result.update(
                status="failed", detail="completed task is missing its persisted report"
            )
    if state.get("started_at"):
        result["elapsed_seconds"] = round(
            max(
                0,
                (
                    datetime.fromisoformat(state.get("finished_at") or utc_now())
                    - datetime.fromisoformat(state["started_at"])
                ).total_seconds(),
            ),
            3,
        )
    if (
        result["status"] not in TERMINAL_STATUSES
        and (directory / "waiting.json").is_file()
    ):
        result["waiting"] = _waiting_state(run_dir, directory, request)
    _status_cache[key] = result
    return result


def list_tasks(
    run_dir: Path, round_id: str | int | None = None
) -> list[dict[str, Any]]:
    root = Path(run_dir).resolve() / "role_io"
    pattern = (
        f"round_{logical_round_id(round_id)}" if round_id is not None else "round_*"
    )
    tasks = []
    status_cache: dict = {}
    for path in sorted(root.glob(f"{pattern}/tasks/*/request.json")):
        logical = path.parents[2].name.removeprefix("round_")
        try:
            tasks.append(
                task_status(
                    run_dir, logical, path.parent.name, _status_cache=status_cache
                )
            )
        except (ValueError, OSError, KeyError) as error:
            tasks.append(
                {
                    "round_id": logical,
                    "task_id": path.parent.name,
                    "status": "failed",
                    "detail": str(error),
                    "task_dir": str(path.parent),
                }
            )
    return tasks


def park_task(
    run_dir: Path, round_id: str | int, task_id: str, reason: str
) -> dict[str, Any]:
    _mutation_identity("park", run_dir, round_id, task_id)
    if not reason.strip():
        raise ValueError("parking requires a reason")
    directory = task_directory(run_dir, round_id, task_id)
    with _lock(directory / ".runner.lock", blocking=False) as acquired:
        if not acquired:
            raise ValueError("task launcher is active; park after it exits")
        status = task_status(run_dir, round_id, task_id)
        if status["status"] in TERMINAL_STATUSES:
            return status
        if _session_alive(
            _state(directory).get("pid"), directory / "last_message.json"
        ):
            raise ValueError("task session is still running; park after it exits")
        _set_state(directory, status="parked", detail=reason)
        (directory / "waiting.json").unlink(missing_ok=True)
    return task_status(run_dir, round_id, task_id)


def _gateway_lock(
    run_dir: Path,
    request: dict[str, Any],
    job_lock: Any,
    job_label: Any,
    *,
    allow_starting: bool = False,
) -> Path:
    if (
        not isinstance(job_label, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_label)
        or not job_label.startswith(request["task_id"] + "__")
        or job_label == request["task_id"] + "__"
    ):
        raise ValueError("gateway job label must be <task_id>__<purpose>")
    root = Path(run_dir).resolve()
    base = root / "llm_calls" / f"round_{request['round_id']}"
    expected = base / f"{job_label}.lock"
    if not isinstance(job_lock, str) or not job_lock:
        raise ValueError("job_lock must name the task's gateway lock file")
    lock = Path(job_lock)
    lock = (lock if lock.is_absolute() else root / lock).resolve()
    if lock != expected or not base.resolve().is_relative_to(root):
        raise ValueError("job_lock must be the task's run-local gateway lock file")
    manifest_path = base / f"{job_label}.manifest.json"
    if manifest_path.resolve() != manifest_path:
        raise ValueError("gateway manifest must not redirect outside its job directory")
    try:
        manifest = _read_json(manifest_path)
    except FileNotFoundError:
        if allow_starting:
            return lock
        raise ValueError(
            "gateway manifest is not ready; use --job-pid from the detached receipt while starting"
        )
    if (
        manifest.get("label") != job_label
        or str(manifest.get("round_id")) != request["round_id"]
    ):
        raise ValueError(
            "gateway manifest does not match this task's job label and round"
        )
    return lock


def _process_snapshot(pid: Any) -> tuple[str, list[str]] | None:
    if type(pid) is not int or pid <= 0:
        return None
    proc = Path("/proc") / str(pid)
    try:
        before = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        argv = [
            os.fsdecode(arg)
            for arg in (proc / "cmdline").read_bytes().split(b"\0")
            if arg
        ]
        after = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        # stat field 22 is process start time; the suffix begins at field 3.
        if before[0] in {"Z", "X"} or after[0] in {"Z", "X"} or before[19] != after[19]:
            return None
        return after[19], argv
    except (OSError, IndexError):
        return None


def _gateway_process_start(
    run_dir: Path, request: dict[str, Any], job_label: str, pid: Any
) -> str | None:
    snapshot = _process_snapshot(pid)
    if snapshot is None:
        return None
    start_time, argv = snapshot
    if len(argv) < 3 or argv[1] != str(_GATEWAY_RUNNER):
        return None
    options = {}
    for index, arg in enumerate(argv[3:], 3):
        key, separator, value = arg.partition("=")
        if key not in {"--run-dir", "--round-id", "--label"}:
            continue
        if key in options:
            return None
        if not separator:
            if index + 1 >= len(argv):
                return None
            value = argv[index + 1]
        options[key] = value
    if (
        options.get("--label") != job_label
        or options.get("--round-id") != request["round_id"]
    ):
        return None
    job_run = options.get("--run-dir")
    if not job_run:
        return None
    job_run = Path(job_run)
    if not job_run.is_absolute():
        job_run = Path("/proc") / str(pid) / "cwd" / job_run
    try:
        if job_run.resolve() != Path(run_dir).resolve():
            return None
    except OSError:
        return None
    return start_time


def _job_is_running(lock: Path) -> bool:
    try:
        with lock.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        return False
    return False


def _waiting_job_is_running(
    run_dir: Path, request: dict[str, Any], waiting: dict[str, Any]
) -> bool:
    if _job_is_running(Path(waiting["job_lock"])):
        return True
    return (
        waiting.get("job_pid") is not None
        and _gateway_process_start(
            run_dir, request, waiting["job_label"], waiting["job_pid"]
        )
        == waiting["job_start_time"]
    )


def _waiting_state(
    run_dir: Path, directory: Path, request: dict[str, Any]
) -> dict[str, Any] | None:
    path = directory / "waiting.json"
    if not path.exists():
        return None
    waiting = _read_json(path)
    for key in ("task_id", "round_id", "input_digest"):
        if waiting.get(key) != request[key]:
            raise ValueError(f"waiting marker {key} does not match this task")
    if type(waiting.get("attempt")) is not int or waiting["attempt"] <= 0:
        raise ValueError("waiting marker requires its task attempt")
    has_process = "job_pid" in waiting or "job_start_time" in waiting
    if has_process and (
        type(waiting.get("job_pid")) is not int
        or waiting["job_pid"] <= 0
        or not isinstance(waiting.get("job_start_time"), str)
        or not waiting["job_start_time"].isdigit()
    ):
        raise ValueError(
            "waiting marker requires the verified gateway pid and process start time"
        )
    # A verified starter may exit before producing a manifest. Preserve its
    # marker so the harness can resume the task to inspect that failed start.
    _gateway_lock(
        run_dir,
        request,
        waiting.get("job_lock"),
        waiting.get("job_label"),
        allow_starting=has_process,
    )
    return waiting


def wait_task(
    run_dir: Path,
    round_id: str | int,
    task_id: str,
    *,
    job_lock: Path,
    job_label: str,
    job_progress: str = "",
    job_pid: int | None = None,
) -> dict[str, Any]:
    """Suspend an active task on its own gateway job without completing it."""
    _mutation_identity("wait", run_dir, round_id, task_id)
    directory = task_directory(run_dir, round_id, task_id)
    request = _request(directory)
    process = {}
    if job_pid is not None:
        start_time = _gateway_process_start(run_dir, request, job_label, job_pid)
        if start_time is None:
            raise ValueError(
                "gateway pid is not a running batch runner for this run, round and label; inspect its results"
            )
        process = {"job_pid": job_pid, "job_start_time": start_time}
    lock = _gateway_lock(
        run_dir, request, str(job_lock), job_label, allow_starting=bool(process)
    )
    with _lock(directory / ".state.lock"):
        state = _state(directory)
        if state["status"] != "running" or state["attempts"] <= 0:
            raise ValueError("only an active task session can wait on a gateway job")
        waiting = {
            **{key: request[key] for key in ("task_id", "round_id", "input_digest")},
            "attempt": state["attempts"],
            "job_lock": str(lock),
            "job_label": job_label,
            "job_progress": job_progress,
            "waiting_at": utc_now(),
            **process,
        }
        if not _waiting_job_is_running(run_dir, request, waiting):
            raise ValueError("gateway job is not running; inspect its results instead")
        _atomic_write(directory / "waiting.json", waiting)
        _atomic_write(
            directory / "state.json",
            {
                **state,
                "detail": f"waiting on gateway job {job_label}",
                "updated_at": utc_now(),
            },
        )
    return task_status(run_dir, round_id, task_id)


def _session_alive(pid: Any, last_message: Path) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text().rsplit(") ", 1)[1].split()[0]
        command = (proc / "cmdline").read_bytes()
    except (OSError, IndexError):
        return False
    return state not in {"Z", "X"} and os.fsencode(last_message) in command


def _resume_id(directory: Path, prompt: str, *, after_wait: bool = False) -> str | None:
    events = directory / "events.jsonl"
    previous = directory / "prompt.md"
    if (
        (directory / "last_message.json").exists() and not after_wait
    ) or not previous.exists():
        return None
    if previous.read_text(encoding="utf-8") != prompt or not events.exists():
        return None
    with events.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "thread.started":
                session = event.get("thread_id")
                if isinstance(session, str) and re.fullmatch(
                    r"[A-Za-z0-9_-]+", session
                ):
                    if any(
                        (directory / "codex_home" / "sessions").rglob(f"*{session}*")
                    ):
                        return session
                return None
    return None


def _prompt(
    repo_root: Path, directory: Path, request: dict[str, Any], inputs: dict[str, Any]
) -> str:
    if request["role"] not in SEARCH_ROLES:
        raise ValueError(f"unknown task role: {request['role']}")
    prompt_root = repo_root / "src" / "autosft" / "agents" / "prompts"
    from autosft.agents.roles import ANALYSIS_ROLE_SKILLS

    skill = (
        repo_root
        / "skills"
        / "analysis"
        / ANALYSIS_ROLE_SKILLS[request["role"]]
        / "SKILL.md"
    )
    skill_body = skill.read_text(encoding="utf-8")
    if skill_body.startswith("---\n"):
        _, separator, skill_body = skill_body[4:].partition("\n---\n")
        if not separator:
            raise ValueError(f"analysis skill has malformed front matter: {skill}")
    context = _task_context(directory, request)
    return "\n\n".join(
        [
            (prompt_root / "analysis.md").read_text(encoding="utf-8"),
            skill_body.strip(),
            (prompt_root / "task_report_contract.md").read_text(encoding="utf-8"),
            context,
            "## Pinned dependency outcomes\n\n"
            + json.dumps(inputs, indent=2, sort_keys=True),
        ]
    )


def _start_session(
    repo_root: Path, run_dir: Path, directory: Path, prompt: str
) -> subprocess.Popen[bytes]:
    request = _request(directory)
    source_home_value = os.environ.get("CODEX_HOME")
    if not source_home_value:
        raise ValueError("CODEX_HOME must name the prepared run-local Codex home")
    source_home = Path(source_home_value)
    if not (source_home / "autosft.config.toml").is_file():
        raise ValueError("AutoSFT Codex profile was not prepared under CODEX_HOME")
    if not shutil.which("codex"):
        raise ValueError("codex CLI is not installed or is not on PATH")
    home = directory / "codex_home"
    home.mkdir(mode=0o700, exist_ok=True)
    for name in ("auth.json", "config.toml", "autosft.config.toml"):
        source = source_home / name
        if source.is_file():
            shutil.copyfile(source, home / name)
            (home / name).chmod(0o600)
    after_wait = _state(directory).get("resume_after_wait", False)
    resume_id = _resume_id(directory, prompt, after_wait=after_wait)
    last_message = directory / "last_message.json"
    if resume_id and after_wait and last_message.exists():
        last_message.rename(directory / f"last_message.json.wait_{time.time_ns()}")
    if resume_id is None:
        for name in ("events.jsonl", "last_message.json"):
            path = directory / name
            if path.exists():
                path.rename(
                    directory
                    / f"{name}.attempt_{_state(directory)['attempts']}_{time.time_ns()}"
                )
    (directory / "prompt.md").write_text(prompt, encoding="utf-8")
    hook_command = (
        f"python3 {shlex.quote(str(repo_root / 'scripts/agents/policy_hook.py'))}"
    )
    hook_config = (
        'hooks.PreToolUse=[{matcher="^(shell_command|exec_command|Bash|shell|exec|local_shell_call|'
        'apply_patch|Edit|Write|spawn_agent|Task)$",hooks=[{type="command",command='
        + json.dumps(hook_command)
        + ",timeout=30}]}]"
    )
    args = [
        "codex",
        "-C",
        str(repo_root),
        "--profile",
        "autosft",
        "--dangerously-bypass-approvals-and-sandbox",
        "--dangerously-bypass-hook-trust",
        "--model",
        os.environ.get("AUTO_SFT_RESEARCH_MODEL", "gpt-5.5"),
        "-c",
        "model_reasoning_effort="
        + json.dumps(os.environ.get("AUTO_SFT_REASONING_EFFORT", "medium")),
        "-c",
        "features.multi_agent=true",
        "-c",
        "agents.interrupt_message=true",
        "--disable",
        "unified_exec",
        "-c",
        hook_config,
        "-c",
        'web_search="disabled"',
        "-c",
        "shell_environment_policy.ignore_default_excludes=true",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--output-last-message",
        str(directory / "last_message.json"),
    ]
    if resume_id:
        args.extend(
            [
                "resume",
                resume_id,
                "Resume this immutable task. Re-read its request and durable task state, "
                "continue unfinished work, persist the task report, and return only its receipt.",
            ]
        )
    else:
        args.append(prompt)
    environment = {
        **os.environ,
        "CODEX_HOME": str(home),
        "AUTO_SFT_RUN_DIR": str(Path(run_dir).resolve()),
        "AUTO_SFT_REPO_ROOT": str(repo_root),
        "AUTO_SFT_SEARCH_TASK_DIR": str(directory),
        "AUTO_SFT_ROLE": request["role"],
        "AUTO_SFT_TASK_ROLE": request["role"],
        "AUTO_SFT_TASK_DIR": str(directory),
        "AUTO_SFT_TASK_ID": request["task_id"],
        "AUTO_SFT_TASK_ROUND_ID": request["round_id"],
        "PYTHONPATH": str(repo_root / "src")
        + (
            os.pathsep + os.environ["PYTHONPATH"]
            if os.environ.get("PYTHONPATH")
            else ""
        ),
    }
    # Direct file output and a detached process preserve the session across
    # a caller timeout; recovery trusts only this task's own pid and log.
    with (directory / "events.jsonl").open("ab" if resume_id else "wb") as events:
        return subprocess.Popen(
            args,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=events,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _usage_from_logs(directory: Path, attempts: int) -> dict[str, Any]:
    fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    totals = dict.fromkeys(fields, 0)
    turns = measured = 0
    logs = []
    paths = sorted(directory.glob("events.jsonl.attempt_*"))
    if (directory / "events.jsonl").exists():
        paths.append(directory / "events.jsonl")
    for path in paths:
        log_turns = log_measured = 0
        error = None
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (ValueError, RecursionError):
                        continue
                    if (
                        not isinstance(event, dict)
                        or event.get("type") != "turn.completed"
                    ):
                        continue
                    log_turns += 1
                    usage = event.get("usage")
                    if not isinstance(usage, dict) or any(
                        type(usage.get(field)) is not int or usage[field] < 0
                        for field in fields
                    ):
                        continue
                    log_measured += 1
                    for field in fields:
                        totals[field] += usage[field]
        except OSError as exception:
            error = str(exception)
        logs.append(
            {
                "path": str(path),
                "turn_count": log_turns,
                "turns_with_usage": log_measured,
                **({"error": error} if error else {}),
            }
        )
        turns += log_turns
        measured += log_measured
    status = (
        "unknown"
        if measured == 0
        else (
            "partial"
            if measured < turns
            or measured < attempts
            or any(log["turns_with_usage"] == 0 or log.get("error") for log in logs)
            else "observed"
        )
    )
    return {
        "status": status,
        **{field: totals[field] if measured else None for field in fields},
        "turn_count": turns if paths else None,
        "turns_with_usage": measured,
        "launched_attempts": attempts,
        "logs": logs,
        "collected_at": utc_now(),
        "scope": "Recorded turn.completed usage across all task attempt logs; interrupted turns "
        "without usage and any separately logged subagents are unmeasured.",
    }


def _collect_codex_usage(directory: Path) -> None:
    try:
        with _lock(directory / ".usage.lock", blocking=False) as acquired:
            if not acquired:
                return
            state = _state(directory)
            if (
                state.get("codex_usage") is not None
                or state["status"] not in TERMINAL_STATUSES
            ):
                return
            if _session_alive(state.get("pid"), directory / "last_message.json"):
                return
            _set_state(
                directory,
                codex_usage=_usage_from_logs(directory, state.get("attempts", 0)),
            )
    except (OSError, ValueError):
        pass  # Usage availability must not change the task's delivery result.


def _reap_session(process: subprocess.Popen[bytes], directory: Path) -> None:
    process.wait()
    try:
        _collect_codex_usage(directory)
    except OSError:
        pass  # A future harness pass recovers collection after a caller exits.


def run_task(
    run_dir: Path,
    repo_root: Path,
    round_id: str | int,
    task_id: str,
    *,
    timeout_seconds: float = 1800,
    poll_seconds: float = 2,
) -> dict[str, Any]:
    """Run/recover one requested task, with durable bounded launch attempts."""
    _mutation_identity("run", run_dir, round_id, task_id)
    directory = task_directory(run_dir, round_id, task_id)
    started = time.monotonic()
    with _lock(directory / ".runner.lock", blocking=False) as acquired:
        if not acquired:
            return task_status(run_dir, round_id, task_id)
        try:
            request = _request(directory)
            status = task_status(run_dir, round_id, task_id)
            if status["status"] in TERMINAL_STATUSES:
                _collect_codex_usage(directory)
                return task_status(run_dir, round_id, task_id)
            inputs = _execution_inputs(
                run_dir, directory, request, create=True, force_hash=True
            )
            prompt = _prompt(Path(repo_root).resolve(), directory, request, inputs)
            process = None
            while True:
                receipt = _receipt(run_dir, directory)
                if receipt is not None:
                    if process is not None:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            threading.Thread(
                                target=_reap_session,
                                args=(process, directory),
                                daemon=True,
                            ).start()
                    _atomic_write(directory / "receipt.json", receipt)
                    _set_state(directory, status="completed", detail="report persisted")
                    _collect_codex_usage(directory)
                    return task_status(run_dir, round_id, task_id)
                state = _state(directory)
                alive = (
                    process.poll() is None
                    if process is not None
                    else _session_alive(
                        state.get("pid"), directory / "last_message.json"
                    )
                )
                waiting = _waiting_state(run_dir, directory, request)
                job_running = waiting is not None and _waiting_job_is_running(
                    run_dir, request, waiting
                )
                if waiting is not None and not alive and not job_running:
                    # Credit each normal wait once, even if recovery crashes
                    # between recording the continuation and clearing its marker.
                    resumed = state.get("wait_resumes", 0)
                    if state.get("waiting_resumed_attempt") != waiting["attempt"]:
                        resumed += 1
                    _set_state(
                        directory,
                        wait_resumes=resumed,
                        waiting_resumed_attempt=waiting["attempt"],
                        resume_after_wait=True,
                        pid=None,
                        detail="gateway job finished; resuming task",
                    )
                    (directory / "waiting.json").unlink()
                    state = _state(directory)
                if alive or job_running:
                    if (
                        timeout_seconds
                        and time.monotonic() - started >= timeout_seconds
                    ):
                        if process is not None:
                            threading.Thread(
                                target=_reap_session,
                                args=(process, directory),
                                daemon=True,
                            ).start()
                        return task_status(run_dir, round_id, task_id)
                    time.sleep(max(0.01, poll_seconds))
                    continue
                process = None
                # The child can persist between the receipt check and exit.
                # Recheck before spending another attempt on finished work.
                if (directory / "report.json").exists():
                    continue
                if (
                    state["attempts"] - state.get("wait_resumes", 0)
                    >= request["input"]["max_attempts"]
                ):
                    _set_state(
                        directory,
                        status="failed",
                        pid=None,
                        detail="task session exhausted bounded attempts without a valid report; "
                        + state.get("detail", ""),
                    )
                    _collect_codex_usage(directory)
                    return task_status(run_dir, round_id, task_id)
                _set_state(
                    directory,
                    status="running",
                    attempts=state["attempts"] + 1,
                    started_at=state.get("started_at") or utc_now(),
                    last_started_at=utc_now(),
                    pid=None,
                    detail="launching task session",
                )
                try:
                    process = _start_session(
                        Path(repo_root).resolve(), run_dir, directory, prompt
                    )
                except (OSError, ValueError) as error:
                    _set_state(directory, detail=f"task launcher failed: {error}")
                    continue
                _set_state(
                    directory,
                    pid=process.pid,
                    resume_after_wait=False,
                    detail="task session running",
                )
        except _UnsettledDependency as error:
            _set_state(directory, detail=str(error))
            return task_status(run_dir, round_id, task_id)
        except (OSError, ValueError, KeyError) as error:
            _set_state(directory, status="failed", detail=str(error))
            _collect_codex_usage(directory)
            return {
                "round_id": str(round_id),
                "task_id": task_id,
                "status": "failed",
                "detail": str(error),
                "task_dir": str(directory),
            }


def run_pending_tasks(
    run_dir: Path,
    repo_root: Path,
    *,
    round_id: str | int | None = None,
    concurrency: int = 2,
    timeout_seconds: float = 1800,
    poll_seconds: float = 2,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run only explicit tasks; settled failures release dependent analysis tasks."""
    _mutation_identity("run-pending", run_dir, round_id)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    visited = set()
    while True:
        statuses = list_tasks(run_dir, round_id)
        recovered_usage = False
        for status in statuses:
            if (
                status["status"] in TERMINAL_STATUSES
                and status.get("codex_usage") is None
            ):
                _collect_codex_usage(Path(status["task_dir"]))
                recovered_usage = True
        if recovered_usage:
            statuses = list_tasks(run_dir, round_id)
        indexed = {(item["round_id"], item["task_id"]): item for item in statuses}
        ready = [
            item
            for item in statuses
            if item["status"] not in TERMINAL_STATUSES
            and (item["round_id"], item["task_id"]) not in visited
            and all(
                indexed.get((item["round_id"], dep), {}).get("status")
                in TERMINAL_STATUSES
                for dep in item.get("dependencies", [])
            )
        ]
        if not ready:
            return statuses

        def execute(item: dict[str, Any]) -> dict[str, Any]:
            return run_task(
                run_dir,
                repo_root,
                item["round_id"],
                item["task_id"],
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(execute, ready))
        for outcome in outcomes:
            visited.add((outcome["round_id"], outcome["task_id"]))
            if log:
                log(
                    f"candidate task {outcome['task_id']} round {outcome['round_id']}: "
                    f"{outcome['status']} ({outcome.get('detail', '')})"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "roles", help="List task roles and the executor's dedicated delivery interface."
    )
    descriptions = {
        "request": "Freeze a root request. Input requires a non-empty objective and may include input_refs [{path, sha256}], "
        "dependencies [task_id], and max_attempts (1-3). Dependencies must already exist in this round. "
        "Set snapshot:true on mutable input references to freeze their content.",
        "persist": "Persist a report with task_id, round_id, role, input_digest, non-empty summary, "
        "and result_artifacts [{path, sha256}]. Digests are checked or filled when omitted; "
        "additional evidence fields are allowed.",
        "status": "List requested task outcomes, or inspect one task and its bound report.",
        "park": "Park an idle request with a reason; parked tasks do not block review or training.",
        "wait": "Suspend this task on its running gateway job; the harness resumes it when the job lock is released.",
        "run-task": "Run or recover one immutable request using only its own durable session.",
        "run-pending": "Run explicitly requested tasks, with ready tasks in parallel and bounded retries.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, description=description)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument(
            "--round-id", required=name not in {"status", "run-pending"}
        )
        if name != "run-pending":
            command.add_argument("--task-id", required=name != "status")
        if name in {"request", "persist"}:
            command.add_argument("--input", type=Path, required=True)
        if name == "request":
            command.add_argument("--role", choices=SEARCH_ROLES, required=True)
        if name == "park":
            command.add_argument("--reason", required=True)
        if name == "wait":
            command.add_argument("--job-lock", type=Path, required=True)
            command.add_argument("--job-label", required=True)
            command.add_argument("--job-progress", default="")
            command.add_argument(
                "--job-pid",
                type=int,
                help="detached runner pid, including before its lock is acquired",
            )
        if name in {"run-task", "run-pending"}:
            command.add_argument(
                "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
            )
            command.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.command == "roles":
            result = ROLE_CATALOG
        elif args.command == "request":
            result = request_task(
                args.run_dir,
                args.round_id,
                args.task_id,
                args.role,
                _read_json(args.input),
            )
        elif args.command == "persist":
            result = persist_report(
                args.run_dir, args.round_id, args.task_id, _read_json(args.input)
            )
        elif args.command == "status":
            if args.task_id and not args.round_id:
                raise ValueError("--task-id requires --round-id")
            result = (
                task_status(args.run_dir, args.round_id, args.task_id)
                if args.task_id
                else list_tasks(args.run_dir, args.round_id)
            )
        elif args.command == "park":
            result = park_task(args.run_dir, args.round_id, args.task_id, args.reason)
        elif args.command == "wait":
            result = wait_task(
                args.run_dir,
                args.round_id,
                args.task_id,
                job_lock=args.job_lock,
                job_label=args.job_label,
                job_progress=args.job_progress,
                job_pid=args.job_pid,
            )
        elif args.command == "run-task":
            result = run_task(
                args.run_dir,
                args.repo_root,
                args.round_id,
                args.task_id,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            result = run_pending_tasks(
                args.run_dir,
                args.repo_root,
                round_id=args.round_id,
                timeout_seconds=args.timeout_seconds,
            )
        print(json.dumps(result, sort_keys=True))
        if args.command == "run-task":
            return (
                0
                if result["status"] == "completed"
                else (75 if result["status"] in {"pending", "running"} else 1)
            )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
