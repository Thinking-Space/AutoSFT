"""Drive the round-scoped executor lifecycle from the harness.

The executor is the only role this pump runs. The four analysis subagents
use task-specific identities and the search_tasks lifecycle, and the
role launcher accepts nothing else.

The pump runs scripts/agents/run_role.sh, waits out live sessions, and
resumes interrupted work through that launcher's own event log. A terminal
executor report completes only the task context that a session consumed;
the shared identity check also controls launcher idempotency and recovery.
The meta agent owns task selection and interpretation of results. Polling and
waiting stay in the harness without spending model context on those steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from autosft.agents.executor_reports import (
    executor_report_path,
    receipt_from_persisted_report,
)
from autosft.agents.roles import EXECUTOR_CONTEXT_DELIMITER

# run_role.sh exits 75 while a prior session for this (role, round) is still
# alive. That is the signal to wait, not a failure.
IN_PROGRESS_EXIT = 75


@dataclass
class RoleOutcome:
    """What became of one (role, round) delegation."""

    role: str
    round_id: str
    status: str  # "receipt" | "failed" | "timeout"
    receipt_path: str | None = None
    returncode: int | None = None
    detail: str = ""
    waited_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "receipt"


@dataclass
class PumpResult:
    outcomes: list[RoleOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(o.ok for o in self.outcomes)

    @property
    def failed(self) -> list[RoleOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def by_role(self) -> dict[str, RoleOutcome]:
        return {o.role: o for o in self.outcomes}


def _launcher(repo_root: Path) -> Path:
    return repo_root / "scripts" / "agents" / "run_role.sh"


def _invoke(
    repo_root: Path,
    role: str,
    round_id: str,
    task_context: Path,
    *,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(_launcher(repo_root)),
            role,
            str(round_id),
            str(task_context),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_role(
    repo_root: Path,
    role: str,
    round_id: str,
    task_context: Path,
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 0.0,
    call_timeout_seconds: float | None = 1800.0,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RoleOutcome:
    """Launch one role and block until its receipt exists.

    Re-invoking the launcher is the documented recovery move, so the wait loop
    is just that call repeated: it returns an existing receipt without starting
    a session, waits out a live one, and resumes an interrupted one. The
    launcher detaches the session from this process, so a call timeout here
    leaves the role running and the next iteration picks it back up.
    """
    say = log or (lambda _message: None)
    started = time.monotonic()
    while True:
        try:
            proc = _invoke(
                repo_root,
                role,
                round_id,
                task_context,
                timeout=call_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # The session outlives this call by design; treat the timeout as a
            # poll interval that already elapsed and ask again.
            proc = None

        if proc is not None:
            if proc.returncode == 0:
                receipt = (proc.stdout or "").strip().splitlines()
                path = receipt[-1] if receipt else None
                waited = time.monotonic() - started
                say(f"role {role} round {round_id}: receipt after {waited:.0f}s")
                return RoleOutcome(
                    role=role,
                    round_id=str(round_id),
                    status="receipt",
                    receipt_path=path,
                    returncode=0,
                    waited_seconds=waited,
                )
            if proc.returncode != IN_PROGRESS_EXIT:
                waited = time.monotonic() - started
                detail = (proc.stderr or proc.stdout or "").strip()
                say(
                    f"role {role} round {round_id}: exit {proc.returncode} "
                    f"after {waited:.0f}s"
                )
                return RoleOutcome(
                    role=role,
                    round_id=str(round_id),
                    status="failed",
                    returncode=proc.returncode,
                    detail=detail[-2000:],
                    waited_seconds=waited,
                )

        elapsed = time.monotonic() - started
        if timeout_seconds and elapsed >= timeout_seconds:
            say(f"role {role} round {round_id}: timed out after {elapsed:.0f}s")
            return RoleOutcome(
                role=role,
                round_id=str(round_id),
                status="timeout",
                detail=(
                    f"no receipt after {elapsed:.0f}s; the session may still be "
                    "running and can be picked up by re-invoking the launcher"
                ),
                waited_seconds=elapsed,
            )
        sleep(poll_seconds)


def run_roles(
    repo_root: Path,
    round_id: str,
    roles: Sequence[tuple[str, Path]],
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 0.0,
    call_timeout_seconds: float | None = 1800.0,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PumpResult:
    """Run each requested executor delegation in caller order.

    The launcher rejects every role but the executor, so anything else a
    caller passes here comes back as a failed outcome rather than a session.
    """
    result = PumpResult()
    for role, task_context in roles:
        result.outcomes.append(
            run_role(
                repo_root,
                role,
                round_id,
                task_context,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                call_timeout_seconds=call_timeout_seconds,
                log=log,
                sleep=sleep,
            )
        )
    return result


def pending_roles(
    run_dir: Path,
    round_id: str,
    roles: Iterable[str],
) -> list[str]:
    """Roles for this round whose delegation has not finished yet.

    This is the same completion test the launcher applies before deciding
    whether to start a session. A receipt only marks the end of a session,
    and a phase-split round ends one session per stage — so the executor is
    pending until its persisted report is terminal for the whole delegated
    job, receipt or not. Judging it by its receipt alone once parked a round
    at sft_poll for seven hours: the pump saw a receipt and never re-invoked
    the launcher, and nothing else was allowed to. A terminal report closes
    only its consumed task context. Changed content reopens success too;
    failed/blocked attempts also accept an identical context touch as the
    root's retry signal.

    Any other role is reported pending: the launcher rejects it, so it has no
    completion state this function could read.
    """
    return [
        role
        for role in roles
        if role != "executor" or not _executor_job_finished(Path(run_dir), round_id)
    ]


def _executor_job_finished(run_dir: Path, round_id: str) -> bool:
    """Use the launcher's task-bound report check, regardless of receipts."""
    return executor_task_state(run_dir, round_id)["receipt"] is not None


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _executor_input_path(run_dir: Path, round_id: str) -> Path:
    return Path(run_dir) / "role_io" / f"round_{round_id}" / "executor.context.json"


def record_executor_input(
    run_dir: Path,
    round_id: str,
    *,
    task_context: Path | None = None,
    resume: bool = False,
) -> None:
    """Bind a launch to the exact prompt bytes and exclude its predecessor report.

    An interrupted session retains its original identity and report baseline
    on resume. A fresh session must produce a new report, even if its input
    happens to be identical to a previous session's input.
    """
    destination = _executor_input_path(run_dir, round_id)
    prompt = destination.with_name("executor.prompt.md")
    if not prompt.is_file() and task_context is None:
        # The meta agent may record the delegation before run_role.sh has
        # assembled its combined prompt. Use the canonical task-context path
        # in that first call; the launcher writes the prompt with the same
        # context before starting Codex.
        task_context = task_context_path(run_dir, round_id, "executor")
    consumed_at_ns: int
    if prompt.is_file():
        _, separator, context = prompt.read_bytes().partition(
            EXECUTOR_CONTEXT_DELIMITER
        )
        if not separator:
            raise ValueError("executor prompt has no task-context delimiter")
        context_digest = _digest(context)
        consumed_at_ns = prompt.stat().st_mtime_ns
    elif task_context is not None and task_context.is_file():
        # The meta agent records the delegation before run_role.sh has built
        # its prompt copy. Bind that first call to the exact context bytes;
        # the launcher will later create a prompt with the same context and
        # the normal executor-state check will compare the two directly.
        context_bytes = task_context.read_bytes()
        context_digest = _digest(context_bytes)
        consumed_at_ns = task_context.stat().st_mtime_ns
    else:
        raise FileNotFoundError(
            f"executor prompt or task context not found for round {round_id}"
        )
    if resume and destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("context_sha256") == context_digest:
            return
    try:
        report_before = _digest(executor_report_path(run_dir, round_id).read_bytes())
    except FileNotFoundError:
        report_before = None
    payload = {
        "schema_version": 1,
        "context_sha256": context_digest,
        "consumed_at_ns": consumed_at_ns,
        "report_before_sha256": report_before,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def executor_task_state(
    run_dir: Path, round_id: str, task_context: Path | None = None
) -> dict:
    """Completion and restart decisions shared by the pump and shell launcher.

    Success acknowledges content, not timestamps: touching identical context
    is harmless, but a finalize request or correction needs its own report.
    Failed/blocked jobs also permit mtime-based rearming after an external
    precondition is repaired. A saved prelaunch report digest prevents old
    success or failure from answering a new invocation that never reported.
    """
    state = {"receipt": None, "needs_fresh_session": False, "phase_complete": False}
    context_path = task_context or task_context_path(run_dir, round_id, "executor")
    identity_path = _executor_input_path(run_dir, round_id)
    prompt_path = identity_path.with_name("executor.prompt.md")
    try:
        receipt = receipt_from_persisted_report(run_dir, round_id=str(round_id))
        context = context_path.read_bytes()
        context_mtime = context_path.stat().st_mtime_ns
        report_path = executor_report_path(run_dir, round_id)
        report_bytes = report_path.read_bytes() if report_path.exists() else None
        report_digest = _digest(report_bytes) if report_bytes is not None else None
        report = json.loads(report_bytes) if report_bytes is not None else {}
        terminal_failure = receipt is not None and receipt["status"] in {
            "failed",
            "blocked",
        }
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            same_context = identity["context_sha256"] == _digest(context)
            consumed_at = identity["consumed_at_ns"]
            fresh_report = (
                report_digest is not None
                and report_digest != identity["report_before_sha256"]
            )
            rearmed = terminal_failure and context_mtime > consumed_at
            state["needs_fresh_session"] = not same_context or rearmed
            bound = same_context and not rearmed and fresh_report
        else:
            # Read old runs without upgrading or rewriting their artifacts.
            # Legacy failures also retain the original mtime-only fallback.
            _, separator, consumed_context = prompt_path.read_bytes().partition(
                EXECUTOR_CONTEXT_DELIMITER
            )
            prompt_mtime = prompt_path.stat().st_mtime
            same_context = (
                context == consumed_context if separator else terminal_failure
            )
            bound = (
                same_context
                and report_bytes is not None
                and prompt_mtime <= report_path.stat().st_mtime
                and (
                    not terminal_failure or context_path.stat().st_mtime <= prompt_mtime
                )
            )
            state["needs_fresh_session"] = receipt is not None and not bound
        if bound:
            state["receipt"] = receipt
            state["phase_complete"] = (
                isinstance(report, dict)
                and report.get("status") == "phase_complete"
                and bool(report.get("persisted_at"))
            )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Missing or malformed evidence cannot complete a delegation.
        state["needs_fresh_session"] = True
    return state


def task_context_path(run_dir: Path, round_id: str, role: str) -> Path:
    return Path(run_dir) / "role_io" / f"round_{round_id}" / f"{role}.task_context.md"


def poll_seconds_from_env(default: float = 30.0) -> float:
    try:
        return float(os.environ.get("AUTO_SFT_ROLE_POLL_SECONDS", default))
    except (TypeError, ValueError):
        return default


def role_timeout_from_env(default: float = 0.0) -> float:
    try:
        return float(os.environ.get("AUTO_SFT_ROLE_TIMEOUT_SECONDS", default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executor task identity for the role launcher"
    )
    parser.add_argument("command", choices=("executor-state", "record-executor-input"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--task-context", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "record-executor-input":
        record_executor_input(
            args.run_dir,
            args.round_id,
            task_context=args.task_context,
            resume=args.resume,
        )
    else:
        print(
            json.dumps(
                executor_task_state(args.run_dir, args.round_id, args.task_context)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
