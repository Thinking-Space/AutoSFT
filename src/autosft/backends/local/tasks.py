"""Durable same-host GPU jobs. No scheduler/client daemon or network is involved.

Submission returns after persisting an identity and starting a detached supervisor.
An independent subreaper keeps the machine-wide GPU lock until every workload
descendant exits, including torchrun ranks in separate sessions. Process start
ticks and boot ID prevent PID reuse from turning a stale task into a live one.
"""

from __future__ import annotations
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from autosft.core.common import REPO_ROOT, utc_now

TERMINAL = {"Succeeded", "Failed", "Stopped", "Killed", "Cancelled", "Canceled"}
CHILD_SUPERVISION = "linux-subreaper-v1"
STOP_GRACE_SECONDS = 15


def atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def process_token(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            + ":"
            + fields[19]
        )
    except (OSError, IndexError):
        return None


def _open_pidfd(pid: int) -> int:
    """Use Linux pidfds even when Python was built without their wrappers."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError as exc:
        raise RuntimeError("local supervision requires Linux pidfd support") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def _send_pidfd_signal(descriptor: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_send_signal
    except AttributeError as exc:
        raise RuntimeError("local supervision requires Linux pidfd signaling") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(descriptor, signum, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _signal_process(pid: int, token: str | None, signum: int) -> None:
    """Signal a recorded identity, with a pidfd to close the PID-reuse race."""
    if not token:
        return
    try:
        descriptor = _open_pidfd(pid)
    except ProcessLookupError:
        return
    try:
        if process_token(pid) == token:
            try:
                _send_pidfd_signal(descriptor, signum)
            except ProcessLookupError:
                pass
    finally:
        os.close(descriptor)


def task_path(run: Path, task: str) -> Path:
    if not re.fullmatch(r"tasks/local-[a-f0-9]{32}", task):
        raise ValueError(f"not a local task id: {task}")
    return run.resolve() / "local_tasks" / (task.split("/")[1] + ".json")


def _stop_requested(path: Path, data: dict) -> bool:
    try:
        request = json.loads(path.with_suffix(".stop").read_text())
    except FileNotFoundError:
        return False
    return all(
        request.get(key) == data.get(key) for key in ("task", "pid", "process_token")
    )


def status(run: Path, task: str) -> dict:
    path = task_path(run, task)
    # Submission publishes the receipt before it registers the supervisor.
    # Only recover that gap after the submitter has released its launch lock
    # (including on a crash). A run-wide lock cannot be used here: evaluation
    # submission already holds it when checking the preceding SFT task.
    with path.with_suffix(".launch.lock").open("a") as launch_lock:
        try:
            fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return json.loads(path.read_text())
        data = _recover_status(path)
    from autosft.backends.megatron.sft_records import finish_sft_accounting

    finish_sft_accounting(run, data)
    return data


def _recover_status(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data["status"] in TERMINAL:
        return data
    if data["host"] != socket.gethostname():
        raise RuntimeError(
            "local task belongs to another host; status cannot be inferred here"
        )
    pid = data.get("pid")
    if not pid:
        # A request persisted before a launcher crash must never be resubmitted
        # silently. It has no worker, and is explicitly recoverable as a failure.
        data.update(
            status="Failed", error="supervisor was not started", update_time=utc_now()
        )
        atomic(path, data)
        return data
    if process_token(pid) != data.get("process_token"):
        # The worker may have written its final receipt after our first read
        # and then exited. Once its identity is gone, re-read before recovery
        # so a completed job cannot be overwritten by a stale Running record.
        data = json.loads(path.read_text())
        if data["status"] in TERMINAL:
            return data
        child = data.get("child_pid")
        if (
            child
            and data.get("child_token")
            and process_token(child) == data["child_token"]
        ):
            if data.get("child_supervision") == CHILD_SUPERVISION:
                # The guard also detects the supervisor's pipe EOF. Never
                # kill it: it holds the GPU lock and adopts detached ranks.
                # Keep the receipt nonterminal until it has reaped them all.
                _signal_process(child, data["child_token"], signal.SIGTERM)
                message = "supervisor exited; waiting for workload process cleanup"
                if data.get("error") != message:
                    data.update(error=message, update_time=utc_now())
                    atomic(path, data)
                return data
            # Compatibility for receipts from before subreaper supervision.
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        data.update(
            status="Failed",
            error="supervisor exited without a terminal receipt",
            update_time=utc_now(),
        )
        atomic(path, data)
    return data


def recover_unregistered_sft(run: Path) -> list[dict]:
    """Close pre-manifest submit crashes without registering or executing them."""
    from autosft.core.round_ids import storage_round_id
    from autosft.orchestration.submission_registry import submission_records

    run = run.resolve()
    directory = run / "local_tasks"
    if directory.is_symlink():
        raise ValueError("local task receipts cannot use a symlink directory")
    recovered = []
    for path in sorted(directory.glob("local-*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        task = "tasks/" + path.stem
        try:
            task_path(run, task)
        except ValueError:
            continue
        with path.with_suffix(".launch.lock").open("a") as launch_lock:
            try:
                fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if any(
                item.get("task") == task
                for item in submission_records(run / "sft_submissions.jsonl")
            ):
                continue
            try:
                data = json.loads(path.read_text())
                name = storage_round_id(data.get("run_name"))
                output = run / "sft" / name
                dataset = run / "data" / "tokenized" / "final_data"
                if (
                    data.get("task") != task
                    or data.get("name") != task
                    or data.get("host") != socket.gethostname()
                    or data.get("status") != "Queued"
                    or "benchmark" in data
                    or any(
                        data.get(key) for key in ("pid", "child_pid", "process_token")
                    )
                    or data.get("output_dir") != str(output)
                    or data.get("dataset_path") != f"1.0 {dataset}"
                    or not output.resolve().is_relative_to(run)
                    or not dataset.resolve().is_relative_to(run)
                ):
                    continue
            except (OSError, ValueError, AttributeError):
                continue
            data.update(
                status="Failed",
                error="submission exited before manifest registration",
                update_time=utc_now(),
                submission_registered=False,
            )
            atomic(path, data)
            recovered.append(data)
    return recovered


def submit(
    run: Path,
    manifest: Path,
    record: dict,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Caller holds the run's submission lock (SFT/eval entrypoint)."""
    if dry_run:
        return {"dry_run": True, "command": command, "cwd": str(cwd), "record": record}
    task = "tasks/local-" + uuid.uuid4().hex
    path = task_path(run, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".launch.lock").open("a") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        return _start_submission(
            path, task, manifest, record, command, cwd=cwd, env=env
        )


def _start_submission(
    path: Path,
    task: str,
    manifest: Path,
    record: dict,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> dict:
    log = Path(record["output_dir"]) / "local_job.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        record,
        name=task,
        task=task,
        status="Queued",
        host=socket.gethostname(),
        create_time=utc_now(),
        submitted_at=utc_now(),
        command=command,
        cwd=str(cwd),
        log_path=str(log),
        child_pid=None,
    )
    atomic(path, data)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    **record,
                    "task": task,
                    "status": "Queued",
                    "submitted_at": data["submitted_at"],
                    "local_task_file": str(path),
                }
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    # Handshake pipe prevents the supervisor from reading an unregistered PID.
    read_fd, write_fd = os.pipe()
    merged = {**os.environ, **(env or {})}
    merged["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), merged.get("PYTHONPATH", "")]
    )
    try:
        with log.open("a") as stream:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "autosft.backends.local.tasks",
                    "worker",
                    str(path),
                    str(read_fd),
                ],
                cwd=REPO_ROOT,
                env=merged,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(read_fd,),
            )
        data.update(pid=child.pid, process_token=process_token(child.pid))
        atomic(path, data)
        os.write(write_fd, b"1")
    except Exception as exc:
        data.update(status="Failed", error=str(exc), update_time=utc_now())
        atomic(path, data)
        raise
    finally:
        os.close(read_fd)
        os.close(write_fd)
    return data


def _reap_workload(child: subprocess.Popen) -> bool:
    """Reap direct/adopted children; true means the entire workload is gone."""
    while True:
        try:
            pid, code = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            return False
        if pid == child.pid:
            child.returncode = os.waitstatus_to_exitcode(code)


def _drain_workload(child: subprocess.Popen) -> None:
    """Allow graceful shutdown, then reap even descendants with their own PGID.

    This process is a single-threaded Linux subreaper. Direct children cannot
    have their PIDs reused until we reap them; killing their parents reparents
    any remaining descendants here. Do not exit just because a timeout passed:
    a task stuck in uninterruptible I/O must retain its lock and live receipt.
    """
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    signalled: set[int] = set()
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        # Reap before reading children, and do not reap between reading and
        # signalling. No concurrent wait/SIGCHLD handler can recycle a PID.
        if _reap_workload(child):
            return
        children = {int(pid) for pid in children_path.read_text().split()}
        signalled.intersection_update(children)
        force = time.monotonic() >= deadline
        for pid in children:
            if force or pid not in signalled:
                try:
                    os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signalled.add(pid)
        time.sleep(0.05)


def exec_child(path: Path, ready: int, gpu_lock_fd: int) -> int:
    """Guard a registered workload and retain its GPU lock through cleanup.

    The supervisor exclusively owns the pipe writer. EOF before permission
    means no workload starts; EOF afterwards means the supervisor died. As a
    subreaper we adopt orphaned ranks even if they created their own sessions.
    """
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # The guard owns the lock. A workload must not inherit either descriptor.
    os.set_inheritable(gpu_lock_fd, False)
    os.set_inheritable(ready, False)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        go = os.read(ready, 1)
        if go != b"1" or stopping:
            return 1
        data = json.loads(path.read_text())
        if (
            data.get("status") != "Running"
            or data.get("host") != socket.gethostname()
            or data.get("child_pid") != os.getpid()
            or not data.get("child_token")
            or data["child_token"] != process_token(os.getpid())
        ):
            raise RuntimeError("workload guard has no matching durable receipt")
        child = subprocess.Popen(
            data["command"], cwd=data["cwd"], start_new_session=True
        )
        try:
            while not stopping:
                _reap_workload(child)
                if child.returncode is not None:
                    break
                readable, _, _ = select.select([ready], [], [], 0.05)
                if readable and not os.read(ready, 1):
                    stopping = True
        finally:
            _drain_workload(child)
        return child.returncode if child.returncode is not None and not stopping else 1
    finally:
        os.close(ready)


def worker(path: Path, ready: int) -> int:
    go = os.read(ready, 1)
    os.close(ready)
    if go != b"1":
        return 1
    data = json.loads(path.read_text())
    child = None
    stopping = False

    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        # Fail before starting a workload on kernels/containers that cannot
        # signal a durable process identity safely (Linux pidfd support).
        _signal_process(os.getpid(), process_token(os.getpid()), 0)
        # Serialize training and eval on this GPU host, including independent
        # benchmark tasks. The lock is held until all child processes exit.
        lock_path = Path(
            os.environ.get(
                "AUTO_SFT_LOCAL_GPU_LOCK",
                str(Path(tempfile.gettempdir()) / "autosft-local-gpu.lock"),
            )
        )
        with lock_path.open("a") as gpu_lock:
            while not stopping:
                if _stop_requested(path, data):
                    stopping = True
                    break
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            if stopping:
                raise InterruptedError("stopped while queued")
            data.update(status="Running", update_time=utc_now())
            atomic(path, data)
            # Do not let the workload execute between Popen and its receipt.
            # A supervisor crash in that interval leaves only a waiting helper
            # which exits on EOF; there is no untracked training/eval process.
            read_fd, write_fd = os.pipe()
            try:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "autosft.backends.local.tasks",
                        "_exec",
                        str(path),
                        str(read_fd),
                        str(gpu_lock.fileno()),
                    ],
                    cwd=REPO_ROOT,
                    start_new_session=True,
                    pass_fds=(read_fd, gpu_lock.fileno()),
                )
                child_token = process_token(child.pid)
                if child_token is None:
                    raise RuntimeError("workload helper exited before registration")
                data.update(
                    child_pid=child.pid,
                    child_token=child_token,
                    child_supervision=CHILD_SUPERVISION,
                )
                atomic(path, data)
                stopping = stopping or _stop_requested(path, data)
                if stopping:
                    raise InterruptedError("stopped before workload launch")
                os.write(write_fd, b"1")
                while child.poll() is None:
                    stopping = stopping or _stop_requested(path, data)
                    if stopping:
                        _signal_process(child.pid, child_token, signal.SIGTERM)
                    time.sleep(0.05)
                code = child.returncode
            finally:
                os.close(read_fd)
                # EOF also cleans up after exceptions before/after registration.
                # Wait inside the lock context; never forcibly kill the guard.
                os.close(write_fd)
                if child is not None:
                    child.wait()
        data.update(
            status="Stopped" if stopping else ("Succeeded" if code == 0 else "Failed"),
            returncode=code,
        )
        if stopping:
            data["error"] = "local task stopped"
    except BaseException as exc:
        data.update(status="Stopped" if stopping else "Failed", error=str(exc))
    finally:
        data["update_time"] = utc_now()
        atomic(path, data)
    if data["status"] == "Succeeded":
        from autosft.backends.megatron.sft_records import finish_sft_accounting

        # Preserve the successful task receipt if bookkeeping fails. A later
        # status call retries accounting without executing the workload again.
        finish_sft_accounting(path.parent.parent, data)
    return 0 if data["status"] == "Succeeded" else 1


def stop(run: Path, task: str) -> dict:
    data = status(run, task)
    if data["status"] not in TERMINAL and data.get("pid"):
        # A contended launch/recovery lock can return an unverified receipt.
        # Validate again here, including PID reuse since the status read.
        if data["host"] != socket.gethostname():
            raise RuntimeError(
                "local task belongs to another host; cannot stop it here"
            )
        if (
            data.get("process_token")
            and process_token(data["pid"]) == data["process_token"]
        ):
            # Submission can return before the supervisor installs its signal
            # handlers. A durable request also covers that startup interval.
            atomic(
                task_path(run, task).with_suffix(".stop"),
                {key: data[key] for key in ("task", "pid", "process_token")},
            )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["worker", "_exec", "status", "stop"])
    parser.add_argument("task")
    parser.add_argument("ready", nargs="?", type=int)
    parser.add_argument("gpu_lock_fd", nargs="?", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        raise SystemExit(worker(Path(args.task), args.ready))
    if args.action == "_exec":
        code = exec_child(Path(args.task), args.ready, args.gpu_lock_fd)
        if code < 0:
            # All descendants have been reaped. Mirror a workload's signal
            # exit so the supervisor retains the original negative returncode.
            if -code != signal.SIGKILL:
                signal.signal(-code, signal.SIG_DFL)
            os.kill(os.getpid(), -code)
        raise SystemExit(code)
    run = Path(os.environ["AUTO_SFT_RUN_DIR"])
    print(json.dumps((stop if args.action == "stop" else status)(run, args.task)))


if __name__ == "__main__":
    main()
