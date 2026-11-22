"""Worker runtime for Synthesis: strict prompt rendering and Codex process control.

Placeholder rendering and worker incubation live together because they are two
halves of one hand-off: the render step decides exactly what a worker may see,
and the Codex runtime decides exactly how that rendered prompt runs.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autosft.synthesis.errors import SynPromptError
from autosft.synthesis._internal.batch_wait import acknowledge_wait, wait_status

# ---------------------------------------------------------------------------
# Worker model: the same one the meta agent runs with
# ---------------------------------------------------------------------------

# Top-level ``model = "..."`` / ``model_reasoning_effort = "..."`` lines of a
# codex config.toml. Only the part before the first ``[section]`` header is
# consulted, so a ``[profiles.*]`` entry never masquerades as the default.
_CONFIG_MODEL_RE = re.compile(r'(?m)^\s*model\s*=\s*"([^"]*)"')
_CONFIG_EFFORT_RE = re.compile(r'(?m)^\s*model_reasoning_effort\s*=\s*"([^"]*)"')


def _root_codex_config_text() -> str:
    """Text of the run-local codex config the meta agent session reads, or ""."""
    run_dir = os.environ.get("AUTO_SFT_RUN_DIR", "").strip()
    if not run_dir:
        return ""
    path = Path(run_dir).expanduser() / "codex_home" / "config.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    head, _, _ = text.partition("\n[")
    return head


def resolve_worker_model() -> tuple[str | None, str | None]:
    """Model + reasoning effort a synthesis worker is launched with.

    By default these are whatever the meta agent runs with: the top-level values of
    ``${AUTO_SFT_RUN_DIR}/codex_home/config.toml``, the per-run codex home that
    agent_task seeds for the meta agent. ``AUTO_SFT_RESEARCH_MODEL`` and
    ``AUTO_SFT_REASONING_EFFORT`` override either value for the
    worker alone. A value that neither source yields is returned as ``None``
    and its ``-c`` flag is omitted, so codex falls back to its own config.
    Read at call time so a per-run setting made before launch is observed.
    """
    config = _root_codex_config_text()
    model = os.environ.get("AUTO_SFT_RESEARCH_MODEL", "").strip() or None
    effort = os.environ.get("AUTO_SFT_REASONING_EFFORT", "").strip() or None
    if model is None:
        match = _CONFIG_MODEL_RE.search(config)
        model = match.group(1).strip() or None if match else None
    if effort is None:
        match = _CONFIG_EFFORT_RE.search(config)
        effort = match.group(1).strip() or None if match else None
    return model, effort


# ---------------------------------------------------------------------------
# Strict prompt rendering
# ---------------------------------------------------------------------------

_SYN_NAME = re.compile(r"^SYN_[A-Z0-9_]+$")
_PLACEHOLDER = re.compile(r"\$\{([^{}]+)\}")

EXPERIENCE_PLACEHOLDERS = frozenset(
    {
        "SYN_EXPERIENCE_UPDATE_ID",
        "SYN_TARGET_ID",
        "SYN_BATCH_ID",
        "SYN_DATA_SOURCE_DIR",
        "SYN_EVIDENCE_CURSOR",
        "SYN_EXPERIENCE_EVENTS_FILE",
        "SYN_EXPERIENCE_FILE",
        "SYN_EXPERIENCE_BASE_VERSION",
        "SYN_EXPERIENCE_BASE_CURSOR",
        "SYN_EXPERIENCE_NEXT_CURSOR",
        "SYN_EXPERIENCE_PROCESSED_EVENT_IDS",
        "SYN_NEW_EVIDENCE_DIR",
        "SYN_EXPERIENCE_OUTPUT_DIR",
    }
)


def _render(
    source_text: str,
    values: Mapping[str, object],
    *,
    allowed_placeholders: frozenset[str],
) -> str:
    invalid_allowed = sorted(
        name for name in allowed_placeholders if not _SYN_NAME.fullmatch(name)
    )
    if invalid_allowed:
        raise SynPromptError(
            "placeholder allowlist contains invalid names: "
            + ", ".join(invalid_allowed)
        )

    found = {match.group(1) for match in _PLACEHOLDER.finditer(source_text)}
    non_syn = sorted(name for name in found if not _SYN_NAME.fullmatch(name))
    if non_syn:
        raise SynPromptError(
            "prompt contains non-Synthesis placeholders: " + ", ".join(non_syn)
        )
    unknown = sorted(found - allowed_placeholders)
    if unknown:
        raise SynPromptError(
            "prompt contains unknown Synthesis placeholders: " + ", ".join(unknown)
        )

    supplied = set(values)
    unknown_values = sorted(supplied - allowed_placeholders)
    if unknown_values:
        raise SynPromptError(
            "render values contain unknown Synthesis placeholders: "
            + ", ".join(unknown_values)
        )
    missing = sorted(found - supplied)
    if missing:
        raise SynPromptError(
            "render values are missing Synthesis placeholders: " + ", ".join(missing)
        )
    unused = sorted(supplied - found)
    if unused:
        raise SynPromptError(
            "render values are unused by the prompt: " + ", ".join(unused)
        )

    rendered = source_text
    for name in sorted(found):
        value = values[name]
        if value is None:
            raise SynPromptError(f"render value {name} cannot be null")
        rendered = rendered.replace("${" + name + "}", str(value))

    residual = sorted({match.group(1) for match in _PLACEHOLDER.finditer(rendered)})
    if residual:
        raise SynPromptError(
            f"rendered prompt still contains placeholders: {', '.join(residual)}"
        )
    return rendered


def render_prompt(
    source: Path,
    values: Mapping[str, object],
    *,
    allowed_placeholders: frozenset[str],
) -> str:
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise SynPromptError(
            f"prompt source must be a regular non-symlink file: {path}"
        )
    try:
        source_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SynPromptError(f"could not read prompt source {path}: {exc}") from exc
    return _render(source_text, values, allowed_placeholders=allowed_placeholders)


# ---------------------------------------------------------------------------
# Runtime value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerPaths:
    """All paths deliberately exposed to one synthesis worker."""

    frozen_request_dir: Path
    target_file: Path
    materials_dir: Path
    data_source_snapshot_dir: Path
    experience_file: Path
    topic_stem_pool_file: Path
    output_dir: Path
    batch_skill_file: Path
    batch_runner: Path
    eval_runs_dir: Path | None
    log_path: Path
    data_pool_dir: Path | None = None
    row_check_tool: Path | None = None
    batch_reference_file: Path | None = None

    def allowed_read_roots(self) -> tuple[Path, ...]:
        roots = (
            self.frozen_request_dir,
            self.data_source_snapshot_dir,
            self.experience_file,
            self.output_dir,
            self.batch_skill_file,
            self.batch_runner,
        )
        # Topic seeds are optional. An absent pool must not become a required
        # read input when the worker starts or resumes.
        if self.topic_stem_pool_file.is_file():
            roots = roots + (self.topic_stem_pool_file,)
        if self.eval_runs_dir is not None:
            roots = roots + (self.eval_runs_dir,)
        if self.data_pool_dir is not None:
            roots = roots + (self.data_pool_dir,)
        if self.row_check_tool is not None:
            roots = roots + (self.row_check_tool,)
        if self.batch_reference_file is not None:
            roots = roots + (self.batch_reference_file,)
        return roots


@dataclass(frozen=True, slots=True)
class WorkerRunRequest:
    """Frozen inputs for one worker process and its bounded resumes."""

    attempt_id: str
    target_id: str
    rendered_prompt: str
    paths: WorkerPaths
    rows_required: int
    max_resumes: int
    resume_backoff_seconds: float = 0.0
    timeout_seconds: float = 0.0
    resume_prompt: str = (
        "The previous turn was interrupted. Continue the same Synthesis "
        "attempt from the existing output-directory state and finish the "
        "requested delivery without changing its target."
    )

    def __post_init__(self) -> None:
        if self.rows_required < 1:
            raise ValueError("rows_required must be positive")
        if self.max_resumes < 0:
            raise ValueError("max_resumes must be non-negative")
        if self.resume_backoff_seconds < 0 or self.timeout_seconds < 0:
            raise ValueError("runtime durations must be non-negative")


@dataclass(frozen=True, slots=True)
class WorkerRunResult:
    returncode: int | None
    thread_id: str | None
    last_event_type: str | None
    resume_count: int
    stopped: bool = False
    timed_out: bool = False
    reason: str = "worker_finished"

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "thread_id": self.thread_id,
            "last_event_type": self.last_event_type,
            "resume_count": self.resume_count,
            "stopped": self.stopped,
            "timed_out": self.timed_out,
            "reason": self.reason,
        }


StopPredicate = Callable[[], bool]
RuntimeEventCallback = Callable[[str, Mapping[str, Any]], None]


# ---------------------------------------------------------------------------
# Codex-backed worker execution
# ---------------------------------------------------------------------------


SYNTHESIS_WORKER_PLACEHOLDERS = frozenset(
    {
        "SYN_TARGET_ID",
        "SYN_TARGET_FILE",
        "SYN_MATERIALS_DIR",
        "SYN_DATA_SOURCE_DIR",
        "SYN_DATA_SOURCE_SNAPSHOT_ID",
        "SYN_EXPERIENCE_FILE",
        "SYN_EXPERIENCE_VERSION",
        "SYN_TOPIC_STEM_POOL_FILE",
        "SYN_OUTPUT_DIR",
        "SYN_ROWS_PER_TARGET",
        "SYN_BATCH_SKILL_FILE",
        "SYN_BATCH_RUNNER",
        "SYN_BATCH_REFERENCE_FILE",
        "SYN_EVAL_RUNS_DIR",
        "SYN_DATA_POOL_DIR",
        "SYN_ROW_CHECK_TOOL",
    }
)
_PROTECTED_ENV_NAMES = frozenset(
    {
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "PYTHONPATH",
    }
)


def worker_prompt_values(
    *,
    target_id: str,
    paths: WorkerPaths,
    data_source_snapshot_id: str,
    experience_version: str,
    rows_per_target: int,
) -> dict[str, str]:
    return {
        "SYN_TARGET_ID": target_id,
        "SYN_TARGET_FILE": str(paths.target_file.resolve()),
        "SYN_MATERIALS_DIR": str(paths.materials_dir.resolve()),
        "SYN_DATA_SOURCE_DIR": str(paths.data_source_snapshot_dir.resolve()),
        "SYN_DATA_SOURCE_SNAPSHOT_ID": data_source_snapshot_id,
        "SYN_EXPERIENCE_FILE": str(paths.experience_file.resolve()),
        "SYN_EXPERIENCE_VERSION": experience_version,
        "SYN_TOPIC_STEM_POOL_FILE": str(paths.topic_stem_pool_file.resolve())
        if paths.topic_stem_pool_file.exists()
        else "",
        "SYN_OUTPUT_DIR": str(paths.output_dir.resolve()),
        "SYN_ROWS_PER_TARGET": str(rows_per_target),
        "SYN_BATCH_SKILL_FILE": str(paths.batch_skill_file.resolve()),
        "SYN_BATCH_RUNNER": str(paths.batch_runner.resolve()),
        "SYN_BATCH_REFERENCE_FILE": str(paths.batch_reference_file.resolve())
        if paths.batch_reference_file is not None
        else "",
        "SYN_EVAL_RUNS_DIR": str(paths.eval_runs_dir.resolve())
        if paths.eval_runs_dir is not None
        else "",
        "SYN_DATA_POOL_DIR": str(paths.data_pool_dir.resolve())
        if paths.data_pool_dir is not None
        else "",
        "SYN_ROW_CHECK_TOOL": str(paths.row_check_tool.resolve())
        if paths.row_check_tool is not None
        else "",
    }


def read_thread_id(log_path: Path) -> str | None:
    thread_id: str | None = None
    for event in _json_events(log_path):
        if event.get("type") not in {"thread.started", "session.started"}:
            continue
        value = (
            event.get("thread_id") or event.get("threadId") or event.get("session_id")
        )
        if isinstance(value, str) and value.strip():
            thread_id = value.strip()
    return thread_id


def last_event_type(log_path: Path) -> str | None:
    event_type: str | None = None
    for event in _json_events(log_path):
        value = event.get("type")
        if isinstance(value, str):
            event_type = value
    return event_type


def _json_events(log_path: Path) -> list[dict[str, Any]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pgid}/stat").read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[-1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def codex_command_prefix(
    *,
    output_dir: Path,
    python_executable: str,
) -> list[str]:
    hook_command = f"{python_executable} -m autosft.synthesis._internal.worker_policy"
    hook_config = (
        'hooks.PreToolUse=[{matcher="^(shell_command|exec_command|Bash|shell|exec|'
        'local_shell_call|apply_patch|Edit|Write|spawn_agent|Task)$",'
        + 'hooks=[{type="command",command="'
        + hook_command
        + '",timeout=30}]}]'
    )
    model, effort = resolve_worker_model()
    model_flags: list[str] = []
    if model is not None:
        model_flags += ["-c", f'model="{model}"']
    if effort is not None:
        model_flags += ["-c", f'model_reasoning_effort="{effort}"']
    return [
        "codex",
        "-C",
        str(output_dir.resolve()),
        "--dangerously-bypass-approvals-and-sandbox",
        "--dangerously-bypass-hook-trust",
        "--disable",
        "skill_search",
        "--disable",
        "plugins",
        *model_flags,
        "-c",
        "features.multi_agent=true",
        "-c",
        "agents.interrupt_message=true",
        "-c",
        'web_search="disabled"',
        "-c",
        "shell_environment_policy.ignore_default_excludes=true",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        hook_config,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ignore-rules",
    ]


class CodexRuntime:
    """Execute a worker in a detached process group with bounded resume."""

    def __init__(
        self,
        *,
        repo_root: Path,
        python_executable: str,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.python_executable = python_executable
        self._base_environ = dict(os.environ)
        self.poll_seconds = 0.2
        self.terminate_grace_seconds = 2.0
        self._lock = threading.Lock()
        self._process: subprocess.Popen[Any] | None = None
        self._stop_requested = False

    def run(
        self,
        request: WorkerRunRequest,
        *,
        should_stop: StopPredicate | None = None,
        on_event: RuntimeEventCallback | None = None,
        work_complete: Callable[[], bool] | None = None,
    ) -> WorkerRunResult:
        should_stop = should_stop or (lambda: False)
        on_event = on_event or (lambda _type, _payload: None)
        paths = request.paths
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_paths(paths)
        with self._lock:
            if self._process is not None:
                raise RuntimeError("CodexRuntime already has an active worker")
            self._stop_requested = False

        started_at = time.monotonic()
        resumes = 0
        thread_id = read_thread_id(paths.log_path)
        pending_wait = wait_status(paths.output_dir)
        if thread_id and resumes >= request.max_resumes and pending_wait["status"] == "idle":
            return WorkerRunResult(
                returncode=None,
                thread_id=thread_id,
                last_event_type=last_event_type(paths.log_path),
                resume_count=resumes,
                reason="max_resumes_before_restart",
            )

        resume = thread_id is not None
        wait_resume = False
        recovering_failure = False
        wake_generation = ""
        while True:
            if self._should_stop(should_stop):
                return WorkerRunResult(
                    returncode=None,
                    thread_id=thread_id,
                    last_event_type=last_event_type(paths.log_path),
                    resume_count=resumes,
                    stopped=True,
                    reason="stop_requested",
                )
            # The marker survives launcher restarts. Wait in the harness, not
            # in a model turn; ordinary batch completion is not error recovery.
            batch_wait = (
                wait_status(paths.output_dir)
                if thread_id and not recovering_failure else {"status": "idle"}
            )
            if batch_wait["status"] != "idle":
                while batch_wait["status"] == "waiting":
                    if not self._interruptible_backoff(
                        self.poll_seconds, request.timeout_seconds, started_at, should_stop
                    ):
                        stopped_now = self._should_stop(should_stop)
                        return WorkerRunResult(
                            returncode=None, thread_id=thread_id,
                            last_event_type=last_event_type(paths.log_path),
                            resume_count=resumes, stopped=stopped_now,
                            timed_out=not stopped_now,
                            reason="stop_requested" if stopped_now else "worker_timeout",
                        )
                    batch_wait = wait_status(paths.output_dir)
                wait_resume = True
                resume = True
            if resume and not wait_resume:
                resumes += 1
            command = self._command(
                request=replace(request, resume_prompt=(
                    "A registered synthesis batch stopped. Inspect its --status, "
                    "aligned results and detach log; retry failed requests as needed. "
                    "Register any still-running batches again before ending the turn. "
                    "Continue this same synthesis attempt and complete the delivery."
                )) if wait_resume else request,
                resume_thread_id=thread_id if resume else None,
            )
            if wait_resume:
                wake_generation = batch_wait["generation"]
            wait_resume = False
            event_type = "worker_resuming" if resume else "worker_started"
            returncode, stopped, timed_out, observed_thread = self._run_once(
                command,
                request=request,
                should_stop=should_stop,
                on_event=on_event,
                event_type=event_type,
                resume_count=resumes,
                started_at=started_at,
            )
            if (
                wake_generation and returncode == 0 and not stopped and not timed_out
                and last_event_type(paths.log_path) != "turn.failed"
            ):
                acknowledge_wait(paths.output_dir, wake_generation)
                wake_generation = ""
            if observed_thread:
                thread_id = observed_thread
            final_event = last_event_type(paths.log_path)
            recovering_failure = returncode != 0 or final_event == "turn.failed"
            if stopped or timed_out:
                return WorkerRunResult(
                    returncode=returncode,
                    thread_id=thread_id,
                    last_event_type=final_event,
                    resume_count=resumes,
                    stopped=stopped,
                    timed_out=timed_out,
                    reason="stop_requested" if stopped else "worker_timeout",
                )
            if returncode == 0 and final_event != "turn.failed" and thread_id:
                if wait_status(paths.output_dir)["status"] != "idle":
                    resume = True
                    continue
            if final_event != "turn.failed":
                if (
                    returncode == 0
                    and work_complete is not None
                    and thread_id
                    and resumes < request.max_resumes
                    and not work_complete()
                ):
                    # A clean exit is not completion. Workers in long poll
                    # loops sometimes end the turn with a status summary
                    # instead of another tool call; the work product is the
                    # only success criterion, so resume the same session
                    # until it exists or the resume budget runs out.
                    if not self._interruptible_backoff(
                        request.resume_backoff_seconds,
                        request.timeout_seconds,
                        started_at,
                        should_stop,
                    ):
                        stopped_now = self._should_stop(should_stop)
                        return WorkerRunResult(
                            returncode=returncode,
                            thread_id=thread_id,
                            last_event_type=final_event,
                            resume_count=resumes,
                            stopped=stopped_now,
                            timed_out=not stopped_now,
                            reason=(
                                "stop_requested" if stopped_now else "worker_timeout"
                            ),
                        )
                    resume = True
                    continue
                return WorkerRunResult(
                    returncode=returncode,
                    thread_id=thread_id,
                    last_event_type=final_event,
                    resume_count=resumes,
                    reason=(
                        "worker_finished"
                        if returncode == 0
                        else "worker_process_failed"
                    ),
                )
            if not thread_id:
                return WorkerRunResult(
                    returncode=returncode,
                    thread_id=None,
                    last_event_type=final_event,
                    resume_count=resumes,
                    reason="turn_failed_without_thread_id",
                )
            if resumes >= request.max_resumes:
                return WorkerRunResult(
                    returncode=returncode,
                    thread_id=thread_id,
                    last_event_type=final_event,
                    resume_count=resumes,
                    reason="max_resumes",
                )
            if not self._interruptible_backoff(
                request.resume_backoff_seconds,
                request.timeout_seconds,
                started_at,
                should_stop,
            ):
                stopped_now = self._should_stop(should_stop)
                return WorkerRunResult(
                    returncode=returncode,
                    thread_id=thread_id,
                    last_event_type=final_event,
                    resume_count=resumes,
                    stopped=stopped_now,
                    timed_out=not stopped_now,
                    reason="stop_requested" if stopped_now else "worker_timeout",
                )
            resume = True

    def stop(self, signum: int = signal.SIGTERM) -> None:
        with self._lock:
            self._stop_requested = True
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return

    def _run_once(
        self,
        command: Sequence[str],
        *,
        request: WorkerRunRequest,
        should_stop: StopPredicate,
        on_event: RuntimeEventCallback,
        event_type: str,
        resume_count: int,
        started_at: float,
    ) -> tuple[int | None, bool, bool, str | None]:
        env = self._child_env(request)
        with request.paths.log_path.open("ab") as log_stream:
            process = subprocess.Popen(
                list(command),
                cwd=request.paths.output_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with self._lock:
            self._process = process
        on_event(
            event_type,
            {
                "pid": process.pid,
                "pgid": process.pid,
                "resume_count": resume_count,
            },
        )
        observed_thread: str | None = None
        stopped = False
        timed_out = False
        try:
            while process.poll() is None:
                current_thread = read_thread_id(request.paths.log_path)
                if current_thread and current_thread != observed_thread:
                    observed_thread = current_thread
                    on_event("thread_observed", {"thread_id": current_thread})
                if self._should_stop(should_stop):
                    stopped = True
                    self.stop()
                if (
                    request.timeout_seconds > 0
                    and time.monotonic() - started_at >= request.timeout_seconds
                ):
                    timed_out = True
                    self.stop()
                if stopped or timed_out:
                    deadline = time.monotonic() + self.terminate_grace_seconds
                    while process.poll() is None and time.monotonic() < deadline:
                        time.sleep(min(self.poll_seconds, 0.05))
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(self.poll_seconds)
            returncode = process.wait()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        observed_thread = read_thread_id(request.paths.log_path) or observed_thread
        return returncode, stopped, timed_out, observed_thread

    def _command(
        self,
        *,
        request: WorkerRunRequest,
        resume_thread_id: str | None,
    ) -> list[str]:
        prefix = codex_command_prefix(
            output_dir=request.paths.output_dir,
            python_executable=self.python_executable,
        )
        if resume_thread_id is None:
            return [*prefix, request.rendered_prompt]
        return [
            *prefix,
            "resume",
            resume_thread_id,
            request.resume_prompt,
        ]

    def _child_env(self, request: WorkerRunRequest) -> dict[str, str]:
        env = dict(self._base_environ)
        for name in list(env):
            if (
                name.startswith("AUTO_SFT_")
                or name.startswith("SYN_")
                or name.startswith("AUTO_SFT_SYN_")
                or name in _PROTECTED_ENV_NAMES
            ):
                env.pop(name, None)
        existing_pythonpath = self._base_environ.get("PYTHONPATH", "")
        for name in (
            "AUTO_SFT_LLM_BASE_URL",
            "AUTO_SFT_LLM_MODEL",
            "AUTO_SFT_LLM_API_KEY",
            "AUTO_SFT_LLM_ENABLE_THINKING",
        ):
            if name in self._base_environ:
                env[name] = self._base_environ[name]
        python_roots = [str(self.repo_root / "src")]
        if existing_pythonpath:
            python_roots.append(existing_pythonpath)
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(python_roots),
                "AUTO_SFT_SYN_WORKER": "1",
                "AUTO_SFT_SYN_REPO_ROOT": str(self.repo_root),
                "AUTO_SFT_SYN_WORKSPACE_ROOT": str(self.repo_root.parent),
                "AUTO_SFT_SYN_READ_ROOTS": os.pathsep.join(
                    str(path.resolve()) for path in request.paths.allowed_read_roots()
                ),
                "AUTO_SFT_SYN_WRITE_ROOT": str(request.paths.output_dir.resolve()),
                "AUTO_SFT_SYN_ALLOW_READONLY_SUBAGENTS": "1",
            }
        )
        return env

    def _validate_paths(self, paths: WorkerPaths) -> None:
        if paths.output_dir.is_symlink():
            raise ValueError("worker output directory cannot be a symlink")
        if paths.log_path.resolve() == paths.output_dir.resolve():
            raise ValueError("worker log path cannot be the output directory")
        for path in paths.allowed_read_roots():
            resolved = path.resolve()
            if resolved == paths.output_dir.resolve():
                continue
            if not resolved.exists():
                raise ValueError(f"worker read input does not exist: {resolved}")

    def _should_stop(self, predicate: StopPredicate) -> bool:
        with self._lock:
            requested = self._stop_requested
        return requested or predicate()

    def _interruptible_backoff(
        self,
        seconds: float,
        timeout_seconds: float,
        started_at: float,
        should_stop: StopPredicate,
    ) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._should_stop(should_stop):
                return False
            if timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds:
                return False
            time.sleep(min(self.poll_seconds, max(deadline - time.monotonic(), 0.0)))
        return not self._should_stop(should_stop)
