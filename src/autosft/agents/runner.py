from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from autosft.config.components import (
    composition_name,
    analysis_enabled,
    executor_enabled,
    root_prompt,
)
from autosft.orchestration.blocked_rounds import (
    acknowledged_rounds,
    describe as blocked_describe,
    unresolved_blocked_rounds,
)
from autosft.core.code_identity import (
    check_drift,
    describe_drift,
    drift_is_fatal,
    record_identity,
)
from autosft.agents.codex_home import prepare_run_codex_home, source_codex_home
from autosft.agents.executor_handoff import (
    build_executor_handoff,
    newest_recipe_round,
    refresh_executor_handoff,
)
from autosft.orchestration.experiment_memory import (
    format_learning_reminders,
    refresh_memory,
)
from autosft.orchestration.ledger import (
    best_scores,
    finalization_matches_target,
    read_iterations,
    target_checkpoint,
)
from autosft.orchestration.policy_guard import check_policy_or_exit
from autosft.agents.role_pump import (
    PumpResult,
    pending_roles,
    poll_seconds_from_env,
    role_timeout_from_env,
    run_roles,
    task_context_path,
)
from autosft.agents.search_tasks import list_tasks, run_pending_tasks
from autosft.agents.root_wait import acknowledge_wait, wait_status
from autosft.orchestration.round_gates import (
    evaluation_settings,
    max_rounds,
    round_budget_exhausted,
)


AGENT_COMMANDS = {
    "meta": "bash scripts/agents/run_codex.sh",
}

_CURRENT_AGENT_PROCESS: subprocess.Popen[object] | None = None
_RUN_LOCK_FD: int | None = None
_ATTEMPTS_LOG: Path | None = None
_CURRENT_ATTEMPT: int | None = None
_TERMINAL_STATUS: Path | None = None
_PROCESS_GROUP_CLEANUP_GRACE_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AutoSFT meta-agent agent task."
    )
    parser.add_argument("agent", choices=sorted(AGENT_COMMANDS))
    return parser.parse_args()


REPO_ROOT = Path(__file__).resolve().parents[3]


def _composition_name() -> str:
    try:
        return composition_name()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def default_prompt() -> str:
    return root_prompt(REPO_ROOT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_target_scores() -> dict[str, float]:
    targets: dict[str, float] = {}
    for part in os.environ.get("AUTO_SFT_TARGET_SCORES", "").split(","):
        part = part.strip()
        if not part:
            continue
        name, separator, value = part.partition("=")
        name = name.strip().lower()
        if not separator or not name:
            print(f"WARN: invalid target score entry: {part}", flush=True)
            continue
        try:
            targets[name] = float(value.strip())
        except ValueError:
            print(f"WARN: invalid target score value: {part}", flush=True)
    return targets


def _score_status_line(targets: dict[str, float], best: dict[str, float]) -> str:
    if not targets:
        return "(no score targets configured)"
    values = []
    for name, threshold in targets.items():
        score = best.get(name)
        if isinstance(score, (int, float)):
            state = "MET" if score >= threshold else "below"
            values.append(f"{name}={score:.4f}/{threshold:.2f} ({state})")
        else:
            values.append(f"{name}=n/a/{threshold:.2f} (no eval yet)")
    return ", ".join(values)


def _continuation_prompt(
    base_prompt: str,
    attempt: int,
    last_returncode: int,
    targets: dict[str, float],
    best: dict[str, float],
    finalize_only: bool,
    *,
    composition: str,
    executor_handoff: dict[str, object] | None = None,
    analysis_bootstrap: bool = False,
    search_task_status: list[dict[str, object]] | None = None,
    qualifying_checkpoint: dict[str, object] | None = None,
    learning_reminders: str = "",
    blocked_rounds: list[dict[str, object]] | None = None,
    wake_events: list[str] | None = None,
) -> str:
    if attempt == 1 and not any(
        (
            last_returncode,
            finalize_only,
            executor_handoff,
            analysis_bootstrap,
            search_task_status,
            learning_reminders,
            blocked_rounds,
            wake_events,
        )
    ):
        return base_prompt
    lines = [
        base_prompt,
        "",
        "# Resume the research run",
        "Read durable run state before acting. You own the data decision and result interpretation.",
        f"Targets and observed scores: {_score_status_line(targets, best)}. Only one completely evaluated checkpoint meeting every target qualifies.",
    ]
    if last_returncode == 3:
        lines.append(
            "Repair the run-local policy before finalizing or submitting again. Read ${AUTO_SFT_POLICY_FEEDBACK} and retry under the fixed run contract."
        )
        return "\n\n".join(lines)
    for event in wake_events or []:
        lines.append(f"Gateway wait event: {event}")
    if finalize_only:
        lines.append(
            "Targets are met. Finalize the qualifying checkpoint and its full evaluation receipts: "
            + json.dumps(qualifying_checkpoint or {})
        )
        if executor_enabled():
            lines.append(
                "Delegate only job_type=finalize for this qualifying checkpoint."
            )
        else:
            lines.append(
                "Publish the chosen model under ${AUTO_SFT_RUN_DIR}/final-model/ and record final_model_summary.json."
            )
        return "\n\n".join(lines)
    if executor_enabled() and executor_handoff:
        lines.append(
            "Resume the executor from executor_handoff.json. Recover existing task ids before resubmitting; the harness polls queue waits."
        )
    if not executor_enabled():
        lines.append(
            "Run the training, conversion and evaluation lifecycle yourself using the executor skills."
        )
    if analysis_enabled():
        lines.append(
            "Request independent proposals from the four analysis roles; inspect their evidence before selecting the next data."
        )
        if analysis_bootstrap:
            lines.append(
                "Bootstrap: only source evidence is available. Do not invent model failures or training signals."
            )
        if search_task_status:
            fields = ("round_id", "task_id", "role", "status", "report_path", "detail")
            lines.append(
                "Analysis task state: "
                + json.dumps(
                    [
                        {key: item[key] for key in fields if key in item}
                        for item in search_task_status
                    ]
                )
            )
    for item in blocked_rounds or []:
        lines.append(
            "UNRESOLVED BLOCKED ROUND "
            + str(item.get("round_id"))
            + ": recover it or record a digest-bound disposition with autosft.orchestration.blocked_rounds acknowledge: "
            + json.dumps(item)
        )
    if learning_reminders:
        lines.append(str(learning_reminders))
    if last_returncode:
        lines.append(
            f"The previous process exited with code {last_returncode}. Recover durable task state and preserve completed work before retrying."
        )
    return "\n\n".join(lines)


def _last_prompt_snapshot(run_dir: Path) -> str | None:
    """Content of the most recently written prompt snapshot.

    Retry snapshots are ordered by mtime, not attempt number: a restarted
    process resets its attempt counter, so the newest file on disk — not the
    numbering — identifies the last prompt actually recorded. Falls back to
    the launch prompt (agent_prompt.txt); unreadable or corrupt candidates
    are skipped; returns None when nothing readable is found.
    """

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    candidates = sorted(
        run_dir.glob("agent_prompt_retry_*.txt"), key=mtime, reverse=True
    )
    candidates.append(run_dir / "agent_prompt.txt")
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return None


def _parse_nonretryable_exit_codes() -> set[int]:
    codes: set[int] = set()
    for part in os.environ.get(
        "AUTO_SFT_AGENT_NONRETRYABLE_EXIT_CODES", "126,127"
    ).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            codes.add(int(part))
        except ValueError:
            print(f"WARN: invalid nonretryable exit code: {part}", flush=True)
    return codes


def _root_session_metadata(path: Path) -> dict[str, object] | None:
    """Read a root's identity without mistaking damaged history for no history."""
    try:
        with path.open(encoding="utf-8") as handle:
            meta = json.loads(handle.readline())
        if not isinstance(meta, dict) or meta.get("type") != "session_meta":
            raise ValueError("missing session metadata")
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("missing session metadata")
        source = payload.get("source")
        thread_source = payload.get("thread_source")
        if (
            isinstance(source, dict) and "subagent" in source
            or isinstance(thread_source, dict) and "subagent" in thread_source
            or payload.get("parent_thread_id")
            or payload.get("forked_from_id")
        ):
            return None
        session_id = payload.get("session_id") or payload.get("id")
        timestamp = payload.get("timestamp")
        if not _valid_session_id(session_id) or not isinstance(timestamp, str):
            raise ValueError("missing session ID or timestamp")
        return {**payload, "session_id": session_id}
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Cannot read root session metadata from {path}: {exc}. "
            "Refusing to start a replacement meta-agent session."
        ) from exc


def _valid_session_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and all(char.isalnum() or char in "-_" for char in value)
    )


def _logged_root_session_id(run_dir: Path) -> str | None:
    """Read only top-level CLI events, never JSON embedded in tool output."""
    path = run_dir / "logs" / "agent.log"
    session_ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The launcher and shell wrappers also write prose.
                if not isinstance(event, dict) or event.get("type") != "thread.started":
                    continue
                session_id = event.get("thread_id")
                if not _valid_session_id(session_id):
                    raise ValueError("thread.started has an invalid thread_id")
                session_ids.add(session_id)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read meta-agent identity from {path}: {exc}") from exc
    if len(session_ids) > 1:
        raise SystemExit(
            f"Multiple meta-agent sessions appear in {path}: {sorted(session_ids)}. "
            "Refusing to guess which session to resume."
        )
    return next(iter(session_ids), None)


def _legacy_root_session_id(run_dir: Path, codex_home: Path) -> str | None:
    """Recover a log-less old run only from unique, role-specific evidence.

    Executor and synthesis sessions may also be top-level exec sessions. A
    repository cwd alone does not identify meta; require its recorded prompt.
    """
    candidates = []
    for path in (codex_home / "sessions").rglob("rollout-*.jsonl"):
        identity = _root_session_metadata(path)
        if identity is not None:
            candidates.append((path, identity))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise SystemExit(
            "Multiple top-level sessions exist without a meta-agent thread.started "
            "event. Refusing to select one by timestamp."
        )
    path, identity = candidates[0]
    cwd = identity.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip() or Path(cwd).resolve() != REPO_ROOT.resolve():
        raise SystemExit("The only session is outside the meta-agent repository cwd.")
    try:
        prompts = [
            snapshot.read_text(encoding="utf-8").strip()
            for snapshot in [run_dir / "agent_prompt.txt", *run_dir.glob("agent_prompt_retry_*.txt")]
            if snapshot.is_file()
        ]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                payload = record.get("payload") if isinstance(record, dict) else None
                if not isinstance(payload, dict) or payload.get("role") != "user":
                    continue
                content = payload.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = part.get("text") if isinstance(part, dict) else None
                    if not isinstance(text, str):
                        continue
                    for prompt in prompts:
                        if prompt and (text.strip() == prompt or (
                            text.startswith(prompt)
                            and text[len(prompt):].startswith("\n\n")
                            and text[len(prompt):].lstrip("\n").startswith("# Resume the research run\n")
                        )):
                            return str(identity["session_id"])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot verify legacy meta-agent prompt in {path}: {exc}") from exc
    raise SystemExit(
        "Session history has no uniquely verified meta-agent prompt. "
        "Refusing to start a replacement session."
    )


_TOOL_CALL_ITEM_TYPES = frozenset(
    {
        "custom_tool_call",
        "function_call",
        "local_shell_call",
    }
)


def _session_rollout(codex_home: Path, session_id: str) -> Path | None:
    sessions = codex_home / "sessions"
    if not sessions.is_dir():
        return None
    matches = sorted(sessions.rglob(f"rollout-*{session_id}.jsonl"))
    if len(matches) > 1:
        raise SystemExit(f"Multiple rollout files claim session {session_id}; refusing ambiguous history.")
    return matches[0] if matches else None


def _last_turn_summary(rollout: Path) -> tuple[int, str] | None:
    """(tool calls, final assistant text) for the rollout's most recent turn.

    Returns None when the rollout has no turn to inspect.
    """
    calls = 0
    message = ""
    seen_turn = False
    try:
        with rollout.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("type")
                if kind == "task_started":
                    calls, message, seen_turn = 0, "", True
                elif kind in _TOOL_CALL_ITEM_TYPES:
                    calls += 1
                elif kind == "message" and payload.get("role") == "assistant":
                    parts = payload.get("content")
                    if isinstance(parts, list):
                        message = " ".join(
                            str(part.get("text", ""))
                            for part in parts
                            if isinstance(part, dict)
                        )
    except OSError:
        return None
    return (calls, message) if seen_turn else None


def _root_session_to_resume(
    run_dir: Path,
    codex_home: Path,
    *,
    require_existing: bool = False,
) -> str | None:
    """Bind once per run, then resume that exact root across retries/restarts.

    Discovery is only for the first binding: an existing run may predate the
    pin, or its first launcher may have died before persisting the new ID.
    A damaged pin or missing rollout must never silently create a new root.
    """
    pin = run_dir / "meta_agent_session.json"
    logged_session = _logged_root_session_id(run_dir)
    pinned_session = None
    try:
        payload = json.loads(pin.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Cannot read meta-agent session binding {pin}: {exc}. "
            "Refusing to start a replacement session."
        ) from exc
    else:
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not _valid_session_id(session_id):
            raise SystemExit(f"Invalid meta-agent session binding in {pin}.")
        pinned_session = session_id
        if logged_session is not None and logged_session != pinned_session:
            raise SystemExit(
                f"Meta-agent binding {pinned_session} conflicts with agent.log "
                f"thread.started {logged_session}. Refusing to resume either session."
            )

    session_id = logged_session or pinned_session or _legacy_root_session_id(run_dir, codex_home)
    if session_id is None:
        if (
            require_existing
            or (run_dir / "meta_agent_launch.json").exists()
            or (run_dir / "agent_attempts.jsonl").exists()
            or any((codex_home / "sessions").rglob("rollout-*.jsonl"))
        ):
            raise SystemExit(
                f"A meta-agent launch was attempted or session history exists in {run_dir}, "
                "but no root identity is recoverable. Inspect the launcher output; "
                "refusing to start a replacement meta-agent session."
            )
        return None
    rollout = _session_rollout(codex_home, session_id)
    identity = _root_session_metadata(rollout) if rollout is not None else None
    cwd = identity.get("cwd") if identity is not None else None
    if (
        identity is None or identity["session_id"] != session_id
        or not isinstance(cwd, str) or not cwd.strip()
        or Path(cwd).resolve() != REPO_ROOT.resolve()
    ):
        raise SystemExit(
            f"Meta-agent session {session_id} has no matching root rollout with the "
            f"repository cwd in {codex_home}. Refusing to start a replacement session."
        )
    if pinned_session is not None:
        return session_id
    # Publish a complete binding atomically, so interruption cannot leave a
    # partial JSON document that a later launcher would have to reinterpret.
    pending = pin.with_suffix(".json.tmp")
    try:
        _write_json(pending, {"session_id": session_id, "bound_at": _utc_now()})
        pending.replace(pin)
    except OSError as exc:
        raise SystemExit(
            f"Cannot persist meta-agent session binding {pin}: {exc}"
        ) from exc
    return session_id


def _is_interrupted_attempt(returncode: int, exit_reason: str) -> bool:
    if returncode == 0:
        return False
    return (
        exit_reason == "nonzero_exit"
        or exit_reason.endswith("_timeout")
        or exit_reason.startswith("signal_")
    )


def _parse_fast_fail_thresholds() -> tuple[int, float]:
    try:
        streak = int(os.environ.get("AUTO_SFT_AGENT_FAST_FAIL_STREAK", "10"))
    except ValueError:
        streak = 10
    try:
        seconds = float(os.environ.get("AUTO_SFT_AGENT_FAST_FAIL_SECONDS", "5"))
    except ValueError:
        seconds = 5.0
    return streak, seconds


def _terminate_process_group(
    process: subprocess.Popen[object], signum: int = signal.SIGTERM
) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass


def _process_group_has_live_members(pgid: int) -> bool:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                fields = stat[stat.rfind(")") + 2 :].split()
                state = fields[0]
                process_group = int(fields[2])
            except (
                FileNotFoundError,
                IndexError,
                PermissionError,
                ValueError,
            ):
                continue
            if process_group == pgid and state not in {"X", "Z"}:
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_process_group(
    process: subprocess.Popen[object],
    *,
    grace_seconds: float = _PROCESS_GROUP_CLEANUP_GRACE_SECONDS,
) -> None:
    if not _process_group_has_live_members(process.pid):
        return
    _terminate_process_group(process)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_has_live_members(process.pid):
            return
        time.sleep(0.05)
    if _process_group_has_live_members(process.pid):
        _terminate_process_group(process, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _run_agent_command(command: str, attempt_timeout: int) -> tuple[int, str]:
    global _CURRENT_AGENT_PROCESS
    # Keep an orphaned Codex process holding the run lock if the runner is
    # killed without cleanup. A restarted runner must not use its session
    # concurrently with the still-running process.
    process = subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        pass_fds=(_RUN_LOCK_FD,) if _RUN_LOCK_FD is not None else (),
    )
    _CURRENT_AGENT_PROCESS = process
    started = time.monotonic()
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                if returncode < 0:
                    return 128 + abs(returncode), f"signal_{abs(returncode)}"
                return returncode, "ok" if returncode == 0 else "nonzero_exit"
            if attempt_timeout > 0 and time.monotonic() - started >= attempt_timeout:
                _terminate_process_group(process)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process, signal.SIGKILL)
                    process.wait(timeout=30)
                return 124, "attempt_timeout"
            time.sleep(1)
    finally:
        _cleanup_process_group(process)
        _CURRENT_AGENT_PROCESS = None


def _signal_handler(signum: int, _frame: object) -> None:
    payload = {
        "status": "interrupted",
        "attempt": _CURRENT_ATTEMPT,
        "signal": signum,
        "finished_at": _utc_now(),
    }
    if _ATTEMPTS_LOG is not None:
        _append_jsonl(
            _ATTEMPTS_LOG,
            {
                "event": "agent_runner_signal",
                "attempt": _CURRENT_ATTEMPT,
                "signal": signum,
                "received_at": _utc_now(),
            },
        )
    if _TERMINAL_STATUS is not None:
        _write_json(_TERMINAL_STATUS, payload)
    if _CURRENT_AGENT_PROCESS is not None:
        _terminate_process_group(_CURRENT_AGENT_PROCESS)
    raise SystemExit(128 + signum)


def _check_final_policy(run_dir: Path) -> None:
    check_policy_or_exit(
        stage="agent-final",
        run_dir=run_dir,
        datasets_dir=Path(os.environ["AUTO_SFT_DATASETS_DIR"]),
        recipe_paths=[
            run_dir / "data-recipe.yaml",
        ],
    )


def _executor_handoff_for_resume(run_dir: Path) -> dict[str, object] | None:
    try:
        _, handoff = refresh_executor_handoff(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"WARN: could not refresh executor handoff: {exc}", flush=True)
        return None
    sft = handoff.get("sft") or {}
    evaluation = handoff.get("evaluation") or {}
    if (
        (isinstance(sft, dict) and bool(sft.get("task")))
        or (isinstance(evaluation, dict) and bool(evaluation.get("tasks")))
        or handoff.get("last_executor_failure")
    ):
        return handoff
    return None


PARKED_POLL_DEFAULT_SECONDS = 600


def _parked_poll_seconds() -> int:
    raw = os.environ.get("AUTO_SFT_PARKED_POLL_SECONDS", "")
    try:
        value = int(raw)
    except ValueError:
        return PARKED_POLL_DEFAULT_SECONDS
    return value if value > 0 else PARKED_POLL_DEFAULT_SECONDS


def _newest_polled_at(
    events_file: Path,
    *,
    tasks: set[str] | None = None,
) -> datetime | None:
    """Newest ``polled_at`` stamp in one status-events file, if any.

    With ``tasks`` given, only events for those tasks count: the run-root
    events file accumulates every round's SFT polls, and a fresh poll of an
    old round's task must not defer the parked round's own poll.
    """
    newest: datetime | None = None
    if not events_file.is_file():
        return None
    try:
        lines = events_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if tasks is not None and record.get("task") not in tasks:
            continue
        raw = record.get("polled_at")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if newest is None or stamp > newest:
            newest = stamp
    return newest


# The two stages whose only possible move is on the cluster's side. The
# harness owns their status checks; see `_parked_lifecycle_polls`.
_POLL_STAGES = frozenset({"sft_poll", "eval_poll"})


def _round_is_parked(run_dir: Path, handoff: dict[str, object]) -> bool:
    """True when a round is in flight with nobody but the harness to run it.

    The executor owns a round from submission to terminal report, but a
    Queued or Pending task blocks it: it persists a blocked (recoverable)
    report and stops, and the round then parks — no session is running it
    and only the cluster can move it. `_parked_lifecycle_polls` says when a
    status check is due; this says the round is in that shape at all.

    A parked round has no work for the meta agent: the harness owns the poll, and
    the mechanical stages belong to the executor, which `_pump_round_roles`
    delegates without a root turn. Waking the meta agent for either buys nothing
    and costs the run — every wake appends to one accumulating session, and
    a wake that finds the same wait appends again, until the context is
    large enough that the transcript is summarized. A run that polled a
    Queued task this way spent 26 percent of its agent time on wakes that
    changed nothing, and its final summary carried the round's bookkeeping
    but none of the data reasoning behind its rounds.

    The poll stages are not the whole wait. `_EXECUTOR_WORK_STAGES` —
    sft_collect, checkpoint_conversion, eval_submit, eval_collect — are
    stages the launcher delegates mechanically, so the meta agent has nothing to
    decide there either. Waking it at eval_collect is worse than wasteful:
    the round's scores reach iterations.jsonl only when that stage's collect
    script runs, so a root turn there sees a round whose evaluation looks
    absent and can open the next one over it. The next round's recipe then
    moves `current_round` forward (build_executor_handoff picks the newest
    recipe round), the in-flight round drops out of the handoff, and its
    finished evaluation is never collected. Sleeping through these stages
    keeps the meta agent's first turn after submission at `complete` or `blocked`,
    where the scores it reasons about are already durable.

    The executor's own delegation is the exception: while it is still
    pending, the role pump is about to run it and it does its own polling.

    A mechanical stage is only mechanical while something will actually
    advance it. When the executor has already answered this stage with a
    terminal report and `_author_executor_context` declines to re-delegate,
    nobody is left: the executor is finished, the launcher is silent, and
    sleeping here would strand the round forever. That is the shape an
    `unsupported_benchmark` round reaches — SFT and conversion succeeded,
    the eval submission created no task, and the executor reported blocked
    at `eval_submit`. Only the meta agent can decide what that round is worth, so
    it must be woken. The poll stages never take this exit: a park is a
    terminal report by design, and the cluster is still moving underneath it.
    """
    stage = str(handoff.get("stage") or "")
    if stage not in _HARNESS_OWNED_STAGES:
        return False
    current_round = str(handoff.get("current_round") or "")
    if current_round in acknowledged_rounds(run_dir):
        return False
    if current_round:
        context = task_context_path(run_dir, current_round, "executor")
        if context.is_file() and "executor" in pending_roles(
            run_dir, current_round, ("executor",)
        ):
            return False
        if stage not in _POLL_STAGES and not _stage_will_be_delegated(
            run_dir, current_round, stage
        ):
            return False
    return True


def _stage_will_be_delegated(run_dir: Path, round_id: str, stage: str) -> bool:
    """Whether the launcher will hand this mechanical stage to the executor.

    `_author_executor_context` is the only thing that re-delegates a round
    the root is asleep for, and it refuses when an existing context was not
    parked at a poll the cluster has since answered. Asking it directly
    keeps this check honest as that rule changes; it writes nothing, because
    a context it would write is exactly the delegation this predicts.
    """
    context = task_context_path(run_dir, round_id, "executor")
    if not context.is_file():
        # Nothing consumed yet: the launcher writes the delegation itself
        # for any stage on its whitelist.
        return stage in _EXECUTOR_WORK_STAGES
    return _parked_stage_of_round(run_dir, round_id) is not None


PARKED_IDLE_DEFAULT_SECONDS = 300


def _parked_idle_seconds() -> int:
    """How long the meta agent sleeps when the round is parked and no poll is due.

    The parked poll has its own interval; this is only the gap between
    checks, so it stays short enough that a due poll is not delayed much
    past its own schedule.
    """
    raw = os.environ.get("AUTO_SFT_PARKED_IDLE_SECONDS", "")
    try:
        value = int(raw)
    except ValueError:
        return PARKED_IDLE_DEFAULT_SECONDS
    return value if value > 0 else PARKED_IDLE_DEFAULT_SECONDS


def _parked_lifecycle_polls(
    run_dir: Path,
    handoff: dict[str, object],
    *,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Return status-script invocations due for a parked round.

    After an executor session concludes with a blocked report, the harness owns low-frequency checks of its submitted tasks. Status scripts refresh the handoff. Collection, conversion and submission still require an executor or meta-agent session.
    """
    stage = str(handoff.get("stage") or "")
    if stage not in _POLL_STAGES:
        return []
    current_round = str(handoff.get("current_round") or "")
    if current_round in acknowledged_rounds(run_dir):
        return []
    if current_round:
        context = task_context_path(run_dir, current_round, "executor")
        if context.is_file() and "executor" in pending_roles(
            run_dir, current_round, ("executor",)
        ):
            # A live delegation: the role pump will run the executor, which
            # does its own polling — a harness poll here would race it.
            return []
    interval = _parked_poll_seconds()
    now = now or datetime.now(timezone.utc)

    def due(events_file: Path, tasks: set[str] | None) -> bool:
        newest = _newest_polled_at(events_file, tasks=tasks)
        return newest is None or (now - newest).total_seconds() >= interval

    if stage == "sft_poll":
        sft = handoff.get("sft")
        sft_task = str(sft.get("task") or "") if isinstance(sft, dict) else ""
        if due(
            run_dir / "local_events.jsonl",
            {sft_task} if sft_task else None,
        ):
            return [{"kind": "sft"}]
        return []
    evaluation = handoff.get("evaluation")
    if not isinstance(evaluation, dict):
        return []
    manifests = evaluation.get("submission_manifests") or []
    # Only manifests that still own a live task are worth polling: the eval
    # status script exits without writing an event when nothing is pending,
    # so an all-terminal manifest would otherwise stay "due" forever.
    live_by_manifest: dict[str, set[str]] = {}
    for item in evaluation.get("tasks") or []:
        if not isinstance(item, dict) or item.get("terminal"):
            continue
        task = str(item.get("task") or "")
        manifest = str(item.get("submission_manifest") or "")
        if task and manifest:
            live_by_manifest.setdefault(manifest, set()).add(task)
    polls: list[dict[str, str]] = []
    for manifest in manifests:
        live = live_by_manifest.get(str(manifest))
        if not live:
            continue
        events_file = Path(str(manifest)).parent / "local_events.jsonl"
        if due(events_file, live):
            polls.append(
                {
                    "kind": "eval",
                    "manifest": str(manifest),
                }
            )
    return polls


def _poll_parked_lifecycle(
    run_dir: Path,
    repo_root: Path,
    handoff: dict[str, object],
) -> bool:
    """Run any due parked-round status polls; True when at least one ran."""
    polls = _parked_lifecycle_polls(run_dir, handoff)
    ran = False
    for poll in polls:
        env = os.environ.copy()
        if "manifest" in poll:
            env["AUTO_SFT_EVAL_MANIFEST"] = poll["manifest"]
        print(
            f"Parked round at {handoff.get('stage')}: running "
            f"autosft tools jobs {poll['kind']} status once so the wait can self-heal.",
            flush=True,
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "autosft", "tools", "jobs", poll["kind"], "status"],
                cwd=repo_root,
                env=env,
                timeout=900,
                check=False,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"WARN: parked lifecycle poll failed: {exc}", flush=True)
            continue
        if completed.returncode != 0:
            # A failing script writes no event, so the throttle never
            # engages; without this line the failure would repeat silently
            # on every attempt.
            tail = (completed.stderr or completed.stdout or "").strip()[-400:]
            print(
                f"WARN: parked lifecycle poll {poll['kind']} exited "
                f"{completed.returncode}: {tail}",
                flush=True,
            )
        else:
            ran = True
    return ran


def _role_pump_enabled() -> bool:
    return os.environ.get("AUTO_SFT_ROLE_PUMP", "1") != "0"


_EPOCH_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _executor_task_context(run_dir: Path, round_id: str) -> str:
    """The round-specific context that delegates a round to the executor.

    Every field is fixed by the run's configuration and the round's recipe.
    The harness uses this only to recover a submitted round. The meta agent must
    explicitly delegate the first submission; assembled data is not authority
    to begin training.
    """
    try:
        repeats = int(os.environ.get("AUTO_SFT_EVAL_REPEATS", "4"))
    except ValueError:
        repeats = 4
    benchmarks = os.environ.get("AUTO_SFT_BENCHMARK", "")
    datasets_dir = os.environ.get("AUTO_SFT_DATASETS_DIR", str(run_dir / "datasets"))
    manifest = run_dir / "run_manifest.json"
    base_model = "the fixed base initialization"
    # Spelled out, as the meta agent wrote it: the handoff's own prose says "three
    # epochs", and a context that differs only in that word would still churn
    # the durable prompt copy it is compared against.
    epochs = "three"
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        recorded = {}
    if isinstance(recorded, dict):
        if recorded.get("base_model"):
            base_model = str(recorded["base_model"])
        count = recorded.get("train_epochs")
        if isinstance(count, int):
            epochs = _EPOCH_WORDS.get(count, str(count))
    return (
        f"Run the normal AutoSFT round {round_id} pipeline through SFT, "
        f"checkpoint conversion, and evaluation.\n\n"
        f"Durable run directory: {run_dir}\n"
        f"Round id: {round_id}\n"
        f"Run name: round_{round_id}\n"
        f"Job type: round\n"
        f"Recipe: {run_dir}/data_rounds/round_{round_id}/data-recipe.yaml\n"
        f"Dataset directory: {datasets_dir}\n"
        f"Benchmarks: {benchmarks}\n"
        f"Evaluation repeats: {repeats} (all required repeats for the "
        f"benchmark)\n\n"
        f"Start from and obey the canonical state in "
        f"{run_dir}/executor_handoff.json. Use only documented entrypoints "
        f"and the fixed {base_model} initialization with exactly {epochs} "
        f"epochs. Do not edit or regenerate the recipe or final_data.jsonl. "
        f"Continue stage by stage until the round is terminal, persisting the "
        f"complete executor report under the run directory and returning only "
        f"its receipt.\n"
    )


# Recovery stages the executor advances for an already-submitted round:
# collect a terminal task, convert the checkpoint, submit and collect the
# evaluation. Initial SFT submission requires a meta-authored context.
# `sft_poll` and `eval_poll` wait on the cluster, so the harness polls them.
# `blocked` and `data_ready` name a precondition for the meta agent to repair.
_EXECUTOR_WORK_STAGES = frozenset(
    {
        "sft_collect",
        "checkpoint_conversion",
        "eval_submit",
        "eval_collect",
    }
)

# Every stage between an SFT submission and a terminal round: the harness
# polls the two waits and delegates the four mechanical stages, so none of
# them has a decision for the meta agent. `_round_is_parked` sleeps through all of
# them, which is what keeps the meta agent's next turn from landing on a round
# whose evaluation has finished on the cluster but not yet reached the
# ledger. `data_ready`, `blocked`, and `complete` are deliberately absent:
# each one needs the meta agent.
_HARNESS_OWNED_STAGES = _POLL_STAGES | _EXECUTOR_WORK_STAGES


def _parked_stage_of_round(run_dir: Path, round_id: str) -> str | None:
    """The poll stage a round's last executor report parked at, if it did.

    A park is a cluster wait: the executor ends its session so the harness
    can poll for it. The report's ``failure_details.stage`` names where it
    stopped. Only ``sft_poll`` and ``eval_poll`` are parks; every other
    stage names a precondition that needs a decision, not a wait.
    """
    report = run_dir / "executor_reports" / f"round_{round_id}.json"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    details = payload.get("failure_details")
    if not isinstance(details, dict):
        return None
    stage = str(details.get("stage") or "")
    return stage if stage in _POLL_STAGES else None


def _author_executor_context(run_dir: Path, round_id: str) -> bool:
    """Recover submitted executor work without authorizing initial training.

    The executor owns a round from submission to terminal report, and a park
    ends its session while the round is unfinished. Nothing runs it again
    until its task context is rewritten, because that rewrite is the
    re-delegation signal the pump reads (``_executor_job_finished`` closes a
    non-succeeded report only while ``context <= prompt <= report``). Only a
    rewritten context is therefore needed to resume collection after a
    queued task succeeds.

    The launcher writes the same delegation the meta agent would, but only where
    the round can be continued mechanically:

    - Its handoff names a post-submission stage the executor can act on by
      itself. Neither ``sft_submit`` nor ``data_ready`` authorizes an initial
      delegation: meta agent may still be preparing or reviewing the formal data.
    - A context that already exists is rewritten only for a park the cluster
      has since answered: while the handoff still asks for a poll, the wait
      belongs to the harness, and a ``blocked`` stage names a precondition
      for the meta agent to repair rather than a step to take.
    """
    if str(round_id) in acknowledged_rounds(run_dir):
        return False
    io_dir = run_dir / "role_io" / f"round_{round_id}"
    context = io_dir / "executor.task_context.md"
    try:
        handoff = build_executor_handoff(run_dir, round_id=str(round_id))
    except Exception as exc:  # pragma: no cover - must not end a run
        print(f"WARN: could not build the handoff: {exc}", flush=True)
        return False
    stage = str(handoff.get("stage") or "")
    if context.exists():
        # A round whose report parked it stops there; the launcher only
        # resumes work the executor can do on its own. While the handoff
        # still asks for a poll, the cluster has not answered yet.
        parked_at = _parked_stage_of_round(run_dir, round_id)
        if parked_at is None or stage == parked_at:
            return False
    if stage not in _EXECUTOR_WORK_STAGES:
        # Still parked, finished, or waiting on a repair only the meta agent can
        # make. Every one of those is a wait or a decision, not a delegation.
        return False
    try:
        io_dir.mkdir(parents=True, exist_ok=True)
        context.write_text(_executor_task_context(run_dir, round_id), encoding="utf-8")
    except OSError as exc:
        print(f"WARN: could not write {context}: {exc}", flush=True)
        return False
    print(
        f"Launcher delegated round {round_id} to the executor "
        f"(stage {stage}); no root turn needed.",
        flush=True,
    )
    return True


def _pump_round_roles(run_dir: Path, repo_root: Path) -> PumpResult | None:
    """Service all requested contributor tasks and the executor lifecycle.

    Proposal work is valid before the first training round. First submission
    requires root's explicit executor context; post-submission recovery remains
    automatic.
    """
    if analysis_enabled():
        run_pending_tasks(
            run_dir,
            repo_root,
            log=lambda message: print(f"Analysis tasks: {message}", flush=True),
        )
    if not executor_enabled():
        return None
    io_root = run_dir / "role_io"
    if not io_root.is_dir():
        return None
    try:
        round_dirs = sorted(
            (
                d
                for d in io_root.iterdir()
                if d.is_dir() and d.name.startswith("round_")
            ),
            key=lambda d: d.stat().st_mtime,
        )
    except OSError as exc:
        print(f"WARN: could not scan role_io: {exc}", flush=True)
        return None
    if not round_dirs:
        return None

    # Analysis proposals use task IDs above, even before any training.
    # Only execution uses the round-scoped handoff and its recovery ordering.
    ordered = ("executor",)
    combined: PumpResult | None = None
    for round_dir in round_dirs:
        round_id = round_dir.name[len("round_") :]
        # Root explicitly disposed of this exact failure. Reopening its old
        # delegation could create a new report and invalidate that decision.
        # Re-read the digest-bound set: changed failures still need attention.
        if round_id in acknowledged_rounds(run_dir):
            continue
        # Recover submitted executor work only. An assembled recipe or a
        # contributor's completion does not authorize the first submission.
        _author_executor_context(run_dir, round_id)
        waiting = [
            role
            for role in pending_roles(run_dir, round_id, ordered)
            if task_context_path(run_dir, round_id, role).is_file()
        ]
        if not waiting:
            continue

        print(
            f"Role pump: round {round_id} waiting on {', '.join(waiting)}.",
            flush=True,
        )
        result = run_roles(
            repo_root,
            round_id,
            [(role, task_context_path(run_dir, round_id, role)) for role in waiting],
            poll_seconds=poll_seconds_from_env(),
            timeout_seconds=role_timeout_from_env(),
            log=lambda message: print(f"Role pump: {message}", flush=True),
        )
        for outcome in result.failed:
            print(
                f"WARN: role {outcome.role} round {outcome.round_id} "
                f"{outcome.status}: {outcome.detail[:400]}",
                flush=True,
            )
        if combined is None:
            combined = result
        else:
            combined.outcomes.extend(result.outcomes)
    return combined


def _analysis_bootstrap_round(run_dir: Path) -> bool:
    """Whether the analysis component has no executed recipe yet.

    Source-grounded candidate exploration is already available at bootstrap;
    only trained-model evidence is absent.
    """
    if newest_recipe_round(run_dir) is not None:
        return False
    try:
        return not bool(read_iterations(run_dir))
    except OSError:
        return True


def main() -> None:
    global _RUN_LOCK_FD

    args = parse_args()
    composition = _composition_name()
    expected_agent = "meta"
    if args.agent != expected_agent:
        raise SystemExit(f"agent {args.agent} does not match composition {composition}")

    run_dir = Path(os.environ["AUTO_SFT_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # The run dir holds plaintext credentials (task.env) plus codex session
    # rollouts and shell snapshots that replicate them with modes we don't
    # control; deny non-owner traversal. Runs on every attempt, so resumed
    # permissions are checked on every resume.
    try:
        run_dir.chmod(0o700)
    except OSError:
        pass
    # Only one runner may select/create or resume this run's root at a time.
    # The root process inherits it, so even an orphan keeps a second runner out.
    with (run_dir / "meta_agent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"A meta-agent runner is already active for {run_dir}."
            ) from exc
        _RUN_LOCK_FD = lock.fileno()
        try:
            _run_task(args, composition, run_dir)
        finally:
            _RUN_LOCK_FD = None


def _stop_at_round_budget(
    run_dir: Path,
    terminal_status: Path,
    *,
    agent: str,
    composition: str,
    attempts: int,
    best: dict[str, float],
    policy_ok: bool,
) -> bool:
    if not round_budget_exhausted(run_dir):
        return False
    limit = max_rounds()
    print(f"Round budget exhausted ({limit} rounds); targets were not met.", flush=True)
    _write_json(
        terminal_status,
        {
            "status": "round_budget_exhausted",
            "agent": agent,
            "composition": composition,
            "attempts": attempts,
            "max_rounds": limit,
            "targets_met": False,
            "finalized": False,
            "policy_passed": policy_ok,
            "best_scores": best,
            "finished_at": _utc_now(),
        },
    )
    return True


def _run_task(args: argparse.Namespace, composition: str, run_dir: Path) -> None:
    global _ATTEMPTS_LOG, _CURRENT_ATTEMPT, _TERMINAL_STATUS

    max_rounds()
    evaluation_settings()
    source_home = source_codex_home()
    codex_home = prepare_run_codex_home(
        run_dir,
        source_home=source_home,
        warn=lambda message: print(f"WARN: {message}", flush=True),
    )
    os.environ["CODEX_HOME"] = str(codex_home)
    print(f"Per-run CODEX_HOME: {codex_home}", flush=True)

    # The prompt points at the benchmark's own items through this variable so
    # the path follows the launcher instead of a hardcoded benchmark name.
    # Runs created before it was written into task.env would otherwise render
    # a literal `${AUTO_SFT_EVAL_DATA_DIR}` into the prompt, so default it to
    # the checkout's tree when the launcher did not supply one.
    os.environ.setdefault(
        "AUTO_SFT_EVAL_DATA_DIR", str(REPO_ROOT / "assets" / "benchmarks")
    )

    # Pin the code version this run executes against, before any round
    # exists. Idempotent, so a relaunched runner compares against the
    # original baseline instead of blessing whatever is on disk now.
    try:
        record_identity(run_dir, REPO_ROOT)
    except OSError as exc:
        print(f"WARN: could not record the run's code identity: {exc}", flush=True)

    base_prompt = os.environ.get("AUTO_SFT_AGENT_PROMPT")
    prompt_snapshot = run_dir / "agent_prompt.txt"
    if not base_prompt and prompt_snapshot.is_file():
        base_prompt = prompt_snapshot.read_text(encoding="utf-8")
    if not base_prompt:
        base_prompt = os.path.expandvars(default_prompt())
    prompt_snapshot.write_text(base_prompt, encoding="utf-8")
    command = os.environ.get("AUTO_SFT_AGENT_COMMAND") or AGENT_COMMANDS[args.agent]

    retry_max = int(os.environ.get("AUTO_SFT_AGENT_RETRY_MAX", "0"))
    retry_delay = int(os.environ.get("AUTO_SFT_AGENT_RETRY_DELAY_SECONDS", "300"))
    success_retry_delay = int(
        os.environ.get("AUTO_SFT_AGENT_SUCCESS_RETRY_DELAY_SECONDS", "60")
    )
    policy_retry_delay = int(
        os.environ.get(
            "AUTO_SFT_AGENT_POLICY_RETRY_DELAY_SECONDS",
            str(success_retry_delay),
        )
    )
    interrupt_retry_delay = int(
        os.environ.get("AUTO_SFT_AGENT_INTERRUPT_RETRY_DELAY_SECONDS", "45")
    )
    # A round that ended without a result needs the meta agent to think, not a
    # fast retry. The 60-second success delay relaunching an unsettled round
    # is how the original failure scrolled past unnoticed.
    blocked_retry_delay = int(
        os.environ.get(
            "AUTO_SFT_AGENT_BLOCKED_RETRY_DELAY_SECONDS",
            str(max(success_retry_delay, 300)),
        )
    )
    attempt_timeout = int(os.environ.get("AUTO_SFT_AGENT_ATTEMPT_TIMEOUT_SECONDS", "0"))
    nonretryable_exit_codes = _parse_nonretryable_exit_codes()
    targets = _parse_target_scores()
    attempts_log = run_dir / "agent_attempts.jsonl"
    terminal_status = run_dir / "agent_terminal_status.json"
    max_attempts = retry_max if retry_max > 0 else 10**9
    attempts_label = str(retry_max) if retry_max > 0 else "unbounded"
    _ATTEMPTS_LOG = attempts_log
    _TERMINAL_STATUS = terminal_status
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    last_returncode = 0
    policy_ok = True
    initial_best = best_scores(run_dir)
    print(
        f"Stop condition — target scores: {_score_status_line(targets, initial_best)}.",
        flush=True,
    )
    if finalization_matches_target(run_dir, targets):
        try:
            _check_final_policy(run_dir)
        except SystemExit as exc:
            policy_ok = False
            last_returncode = exc.code if isinstance(exc.code, int) else 3
        if policy_ok:
            _write_json(
                terminal_status,
                {
                    "status": "succeeded",
                    "agent": args.agent,
                    "attempts": 0,
                    "targets_met": True,
                    "finalized": True,
                    "policy_passed": True,
                    "best_scores": initial_best,
                    "finished_at": _utc_now(),
                },
            )
            return

    candidate_tasks: list[dict[str, object]] = []
    wake_events: list[str] = []
    fast_fail_limit, fast_fail_seconds = _parse_fast_fail_thresholds()
    fast_fail_streak = 0
    fast_fail_reported = False

    for attempt in range(1, max_attempts + 1):
        # The checkout is shared and writable, so verify before spending a
        # round on it. A mid-run edit makes the rounds on either side of it
        # incomparable, which is a silent result-corrupting failure rather
        # than a crash — exactly what needs a check rather than a traceback.
        drift = check_drift(run_dir, REPO_ROOT)
        if drift["drifted"]:
            message = describe_drift(drift)
            _append_jsonl(
                attempts_log,
                {
                    "event": "code_drift_detected",
                    "agent": args.agent,
                    "composition": composition,
                    "attempt": attempt,
                    "fatal": drift_is_fatal(),
                    "detected_at": _utc_now(),
                    **{
                        key: drift.get(key)
                        for key in (
                            "baseline",
                            "baseline_commit",
                            "current_tree_sha256",
                            "current_commit",
                            "changed_paths",
                        )
                    },
                },
            )
            if drift_is_fatal():
                print(f"ERROR: {message}", flush=True)
                _write_json(
                    terminal_status,
                    {
                        "status": "code_drift",
                        "agent": args.agent,
                        "composition": composition,
                        "attempts": attempt - 1,
                        "detail": message,
                        "changed_paths": drift.get("changed_paths"),
                        "best_scores": best_scores(run_dir),
                        "finished_at": _utc_now(),
                    },
                )
                raise SystemExit(4)
            print(f"WARN: {message}", flush=True)

        try:
            prepare_run_codex_home(
                run_dir,
                source_home=source_home,
                warn=lambda message: print(f"WARN: {message}", flush=True),
            )
        except OSError as exc:
            print(f"WARN: could not refresh per-run CODEX_HOME: {exc}", flush=True)

        best = best_scores(run_dir)
        qualifying_checkpoint = target_checkpoint(run_dir, targets)
        finalize_only = (
            policy_ok
            and qualifying_checkpoint is not None
            and not finalization_matches_target(run_dir, targets)
        )
        if qualifying_checkpoint is None and _stop_at_round_budget(
            run_dir,
            terminal_status,
            agent=args.agent,
            composition=composition,
            attempts=attempt - 1,
            best=best,
            policy_ok=policy_ok,
        ):
            return
        executor_handoff = (
            _executor_handoff_for_resume(run_dir) if executor_enabled() else None
        )
        if executor_handoff is not None:
            # A parked poll refreshes the handoff from inside the status
            # script; re-read it so this attempt's prompt reflects any
            # movement the poll just discovered.
            if _poll_parked_lifecycle(run_dir, REPO_ROOT, executor_handoff):
                executor_handoff = (
                    _executor_handoff_for_resume(run_dir) or executor_handoff
                )
        # A root-owned detached gateway job has priority over parked executor
        # work: monitor it at the gateway cadence and resume the same root
        # session when it stops. Otherwise a parked executor interval could
        # delay the wake for several minutes.
        root_wait = wait_status(run_dir)
        # An in-flight round has no work for the meta agent: the executor owns it
        # from submission to terminal report, and the harness owns both the
        # cluster polls and the mechanical stages between them. Sleep instead
        # of starting a session, so the wait costs the meta agent's transcript
        # nothing — see _round_is_parked.
        if (
            policy_ok
            and root_wait["status"] == "idle"
            and executor_handoff is not None
            and _round_is_parked(run_dir, executor_handoff)
        ):
            # The pump runs before the sleep, and it is what advances this
            # round: at a mechanical stage `_author_executor_context` writes
            # the delegation and `run_roles` blocks until the executor's
            # receipt exists. Skipping it here would leave the round asleep at
            # a stage only the pump can move. It also services other rounds'
            # contexted roles, which would otherwise starve for as long as
            # this one is in flight.
            if (executor_enabled() or analysis_enabled()) and _role_pump_enabled():
                try:
                    _pump_round_roles(run_dir, REPO_ROOT)
                except Exception as exc:  # pragma: no cover
                    print(f"WARN: role pump failed: {exc}", flush=True)
            stage_before = str(executor_handoff.get("stage") or "")
            # The pump may have carried the round through a whole mechanical
            # stage while this attempt was inside it. Sleeping a full idle
            # interval on that outcome would stall a ready pipeline — collect,
            # convert, and eval submit are each seconds of work — and it would
            # delay the meta agent's turn at `complete` by one interval per stage.
            # Re-derive the stage and loop immediately when it moved.
            refreshed = _executor_handoff_for_resume(run_dir)
            if refreshed is not None:
                executor_handoff = refreshed
            stage_now = str(executor_handoff.get("stage") or "")
            _append_jsonl(
                attempts_log,
                {
                    "event": "parked_idle",
                    "agent": args.agent,
                    "composition": composition,
                    "attempt": attempt,
                    "stage": stage_before,
                    "stage_after_pump": stage_now,
                    "polled_at": _utc_now(),
                },
            )
            if stage_now != stage_before:
                print(
                    f"Round moved {stage_before} -> {stage_now}; continuing "
                    f"without an idle wait.",
                    flush=True,
                )
                continue
            idle = _parked_idle_seconds()
            print(
                f"Round in flight at {stage_before}: the harness owns this "
                f"stage, so no root session is needed; checking again in "
                f"{idle}s.",
                flush=True,
            )
            time.sleep(idle)
            continue

        # A root gateway wait is owned by the harness. While registered jobs
        # are alive, monitor their locks without starting another model turn.
        # The same root session resumes after a job stops so it can inspect
        # results; a stopped job is not itself a success receipt.
        waiting_generation_logged = None
        while (
            policy_ok
            and not finalize_only
            and root_wait["status"] == "waiting"
        ):
            if waiting_generation_logged != root_wait["generation"]:
                _append_jsonl(
                    attempts_log,
                    {
                        "event": "root_wait_started",
                        "agent": args.agent,
                        "composition": composition,
                        "generation": root_wait["generation"],
                        "started_at": _utc_now(),
                        "jobs": [
                            {"round_id": job["round_id"], "job_label": job["job_label"]}
                            for job in root_wait["jobs"]
                        ],
                    },
                )
                waiting_generation_logged = root_wait["generation"]
                print(
                    "Root is waiting on gateway work; monitoring without a model turn.",
                    flush=True,
                )
            time.sleep(30)
            if _role_pump_enabled():
                try:
                    _pump_round_roles(run_dir, REPO_ROOT)
                except Exception as exc:  # pragma: no cover
                    print(f"WARN: role pump failed: {exc}", flush=True)
            root_wait = wait_status(run_dir)
        wake_events = list(root_wait["events"])
        live_jobs = [
            {"round_id": job["round_id"], "job_label": job["job_label"]}
            for job in root_wait["jobs"]
            if job["status"] == "waiting"
        ]
        if live_jobs:
            wake_events.append(
                "Gateway jobs still running: "
                + json.dumps(live_jobs)
                + ". Register any continued wait before ending this turn."
            )

        analysis_bootstrap = (
            _analysis_bootstrap_round(run_dir) if analysis_enabled() else False
        )
        learning_reminders = ""
        if not finalize_only:
            try:
                memory_path = refresh_memory(run_dir)
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
                learning_reminders = format_learning_reminders(memory)
            except (OSError, ValueError, TypeError) as exc:
                print(f"WARN: experiment memory unavailable: {exc}", flush=True)
        if executor_enabled():
            candidate_tasks = list_tasks(run_dir)
        # A round the executor abandoned unrecoverably is the meta agent's to
        # settle. Surfacing it in the prompt is what turns the runner's old
        # silent relaunch into a turn that can actually see the failure.
        try:
            pending_blocked = unresolved_blocked_rounds(run_dir)
        except OSError as exc:
            print(f"WARN: could not check blocked rounds: {exc}", flush=True)
            pending_blocked = []
        prompt = os.path.expandvars(
            _continuation_prompt(
                base_prompt,
                attempt,
                last_returncode,
                targets,
                best,
                finalize_only,
                composition=composition,
                executor_handoff=executor_handoff,
                analysis_bootstrap=analysis_bootstrap,
                search_task_status=candidate_tasks,
                qualifying_checkpoint=qualifying_checkpoint,
                learning_reminders=learning_reminders,
                blocked_rounds=pending_blocked,
                wake_events=wake_events,
            )
        )
        os.environ["AUTO_SFT_AGENT_PROMPT"] = prompt

        resume_session = _root_session_to_resume(run_dir, codex_home)
        if resume_session is not None:
            os.environ["AUTO_SFT_CODEX_RESUME_SESSION"] = resume_session
            print(f"Resuming Codex session {resume_session}.", flush=True)
        else:
            os.environ.pop("AUTO_SFT_CODEX_RESUME_SESSION", None)
        if attempt > 1 or resume_session is not None or finalize_only:
            # agent_attempts.jsonl already records per-attempt metadata; the
            # snapshot only needs to capture prompts that differ from the one
            # already on disk.
            if prompt != _last_prompt_snapshot(run_dir):
                (run_dir / f"agent_prompt_retry_{attempt}.txt").write_text(
                    prompt,
                    encoding="utf-8",
                )

        _CURRENT_ATTEMPT = attempt
        started_at = _utc_now()
        attempt_started = time.monotonic()
        print(
            f"Starting agent attempt {attempt}/{attempts_label}; scores: "
            f"{_score_status_line(targets, best)}.",
            flush=True,
        )
        try:
            # Persist before invoking a custom shell command too: a crash before
            # thread.started must not make the next launcher assume a fresh run.
            launch_marker = run_dir / "meta_agent_launch.json"
            pending_marker = launch_marker.with_suffix(".json.tmp")
            _write_json(pending_marker, {"attempt": attempt, "started_at": started_at})
            pending_marker.replace(launch_marker)
            last_returncode, exit_reason = _run_agent_command(command, attempt_timeout)
        finally:
            # Persist the first session even on a failed or interrupted turn.
            # Later calls validate the binding; they never choose a newer root.
            attempt_session = _root_session_to_resume(run_dir, codex_home, require_existing=True)
        attempt_seconds = time.monotonic() - attempt_started
        agent_returncode = last_returncode
        if agent_returncode == 0 and root_wait["status"] != "idle":
            acknowledge_wait(run_dir, root_wait["generation"])
        # A successful root turn may have registered a new gateway wait just
        # before it ended. Consume that wait here, before the generic retry
        # backoff, so a long operator does not cause a second timed model turn.
        operator_wait_woke = False
        root_wait_after = wait_status(run_dir)
        if (
            not finalize_only
            and root_wait_after["status"] == "waiting"
        ):
            operator_wait_woke = True
            waiting_generation_logged = None
            while root_wait_after["status"] == "waiting":
                if waiting_generation_logged != root_wait_after["generation"]:
                    waiting_generation_logged = root_wait_after["generation"]
                    print(
                        "Root is waiting on gateway work; monitoring without a model turn.",
                        flush=True,
                    )
                time.sleep(30)
                if _role_pump_enabled():
                    try:
                        _pump_round_roles(run_dir, REPO_ROOT)
                    except Exception as exc:  # pragma: no cover
                        print(f"WARN: role pump failed: {exc}", flush=True)
                root_wait_after = wait_status(run_dir)
            wake_events = list(root_wait_after["events"])
        turn_summary: tuple[int, str] | None = None
        if attempt_session is not None:
            rollout = _session_rollout(codex_home, attempt_session)
            if rollout is not None:
                turn_summary = _last_turn_summary(rollout)
        turn_tool_calls = turn_summary[0] if turn_summary is not None else None

        # Root delegates through task requests or executor context, then stops. Waiting
        # for the roles it named is mechanical, so it happens here between
        # attempts: the meta agent pays nothing for the wait and resumes its own
        # session once the receipts exist.
        if (executor_enabled() or analysis_enabled()) and _role_pump_enabled():
            try:
                _pump_round_roles(run_dir, REPO_ROOT)
            except Exception as exc:  # pragma: no cover - pump must not end a run
                print(f"WARN: role pump failed: {exc}", flush=True)

        try:
            _check_final_policy(run_dir)
            policy_ok = True
        except SystemExit as exc:
            policy_ok = False
            last_returncode = exc.code if isinstance(exc.code, int) else 3

        best = best_scores(run_dir)
        qualifying_checkpoint = target_checkpoint(run_dir, targets)
        targets_met = qualifying_checkpoint is not None
        finalized = finalization_matches_target(run_dir, targets)
        # Re-read after the attempt: this turn may have created a blocked
        # round, or settled one that was already open.
        try:
            blocked_after = unresolved_blocked_rounds(run_dir)
        except OSError as exc:
            print(f"WARN: could not check blocked rounds: {exc}", flush=True)
            blocked_after = []
        if blocked_after:
            print(
                f"Unresolved blocked round(s): "
                f"{blocked_describe(blocked_after)}. A new round cannot be "
                f"submitted until the meta agent re-runs or acknowledges them.",
                flush=True,
            )
        if analysis_enabled():
            try:
                candidate_tasks = list_tasks(run_dir)
            except OSError as exc:
                print(f"WARN: could not check contributor tasks: {exc}", flush=True)

        # A completed round with unmet targets is resumable work for both
        # compositions; let the shared attempt loop relaunch the agent.
        _append_jsonl(
            attempts_log,
            {
                "event": "agent_attempt_finished",
                "agent": args.agent,
                "composition": composition,
                "attempt": attempt,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "returncode": last_returncode,
                "agent_returncode": agent_returncode,
                "duration_seconds": round(attempt_seconds, 3),
                "exit_reason": exit_reason,
                "resumed_session": resume_session,
                "session_id": attempt_session,
                "turn_tool_calls": turn_tool_calls,
                "best_scores": best,
                "targets_met": targets_met,
                "policy_passed": policy_ok,
                "contributor_tasks": candidate_tasks,
                "qualifying_checkpoint": qualifying_checkpoint,
                "unresolved_blocked_rounds": blocked_after,
            },
        )

        done = policy_ok and (
            targets_met and finalized if targets else last_returncode == 0
        )
        if done:
            _write_json(
                terminal_status,
                {
                    "status": "succeeded",
                    "agent": args.agent,
                    "composition": composition,
                    "attempts": attempt,
                    "targets_met": targets_met,
                    "finalized": finalized,
                    "policy_passed": policy_ok,
                    "best_scores": best,
                    "finished_at": _utc_now(),
                },
            )
            return

        if not targets_met and _stop_at_round_budget(
            run_dir,
            terminal_status,
            agent=args.agent,
            composition=composition,
            attempts=attempt,
            best=best,
            policy_ok=policy_ok,
        ):
            return

        if agent_returncode != 0 and attempt_seconds < fast_fail_seconds:
            fast_fail_streak += 1
        else:
            fast_fail_streak = 0
            fast_fail_reported = False
        if (
            fast_fail_limit > 0
            and fast_fail_streak >= fast_fail_limit
            and not fast_fail_reported
        ):
            fast_fail_reported = True
            print(
                f"WARN: {fast_fail_streak} consecutive attempts exited nonzero "
                f"in under {fast_fail_seconds:g}s; check launcher configuration.",
                flush=True,
            )

        if last_returncode in nonretryable_exit_codes:
            break
        if attempt >= max_attempts:
            break
        if not policy_ok:
            delay, reason = policy_retry_delay, "policy guard"
        elif targets_met:
            delay, reason = success_retry_delay, "targets met; finalizing"
        elif blocked_after:
            # Not "targets not yet reached": a round ended without a result
            # and the meta agent has not settled it. Relabelled because that wrong
            # label is what made this look routine in the logs, and slowed
            # because a fast relaunch of an unsettled round is what buried
            # the failure.
            delay, reason = (
                blocked_retry_delay,
                f"round {', '.join(item['round_id'] for item in blocked_after)} "
                f"blocked and unresolved",
            )
        elif operator_wait_woke:
            delay, reason = 0, "gateway wait completed"
        elif last_returncode == 0:
            delay, reason = success_retry_delay, "targets not yet reached"
        elif last_returncode == 3:
            delay, reason = policy_retry_delay, "policy guard"
        elif attempt_seconds >= fast_fail_seconds and _is_interrupted_attempt(
            last_returncode, exit_reason
        ):
            delay, reason = (
                interrupt_retry_delay,
                f"interrupted after {attempt_seconds:.0f}s",
            )
        else:
            delay, reason = retry_delay, f"exit {last_returncode}"
        print(
            f"Agent attempt {attempt} done ({reason}); relaunching after {delay}s.",
            flush=True,
        )
        if delay > 0:
            time.sleep(delay)

    best = best_scores(run_dir)
    targets_met = target_checkpoint(run_dir, targets) is not None
    finalized = finalization_matches_target(run_dir, targets)
    if targets_met and finalized:
        try:
            _check_final_policy(run_dir)
            policy_ok = True
        except SystemExit as exc:
            policy_ok = False
            last_returncode = exc.code if isinstance(exc.code, int) else 3
    _write_json(
        terminal_status,
        {
            "status": (
                "succeeded"
                if targets_met and finalized and policy_ok
                else "failed"
                if last_returncode != 0
                else "stopped"
            ),
            "agent": args.agent,
            "composition": composition,
            "last_returncode": last_returncode,
            "targets_met": targets_met,
            "finalized": finalized,
            "policy_passed": policy_ok,
            "best_scores": best,
            "finished_at": _utc_now(),
        },
    )
    if last_returncode != 0 and not (targets_met and finalized and policy_ok):
        raise SystemExit(last_returncode)


if __name__ == "__main__":
    main()
