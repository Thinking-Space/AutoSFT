#!/usr/bin/env python3
"""Codex `PreToolUse` hook that enforces ablation hard rules in real time.

`scripts/agents/run_codex.sh` installs this hook through an inline
`hooks.PreToolUse` config for every component composition.

Contract (see https://developers.openai.com/codex/hooks):
- One JSON object arrives on stdin with `tool_name`, `tool_input`, `cwd`, ...
  Shell payloads use `tool_input.command` or `tool_input.cmd`.
- To DENY a tool call, print the hook-specific JSON below on stdout and exit 0:
    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "..."}}
  (exit code 2 with a stderr reason also blocks; we use the JSON shape.)

Two families are enforced here, both derived from AGENTS.md hard rules:
1. Forbidden training flows / fixed-config overrides / local job-submission &
   infrastructure changes (GPU resources, interpreters, model assets and
   evaluation settings) in shell commands —
   reused verbatim from policy_guard (single source of truth for the regexes).
2. File edits (`apply_patch`) whose target path is outside the writable
   allowlist (${AUTO_SFT_RUN_DIR} / ${AUTO_SFT_DATASETS_DIR} + the approved
   recipe files) — i.e. "do not edit repo source".

Defense in depth: the codex sandbox (network + writable roots) and the runner's
git-diff policy watchdog are independent backstops, so this hook fails OPEN on
unexpected internal errors (a hook crash must not brick an unattended run); the
error is written to stderr for visibility.

Every launcher passes the same agent settings. Analysis sessions come from search_tasks and executor sessions from the executor launcher. Built-in delegation attempts are recorded in run-local spawn_events.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `autosft.orchestration.policy_guard` importable regardless of cwd: this file
# lives at <repo>/scripts/agents/policy_hook.py, so the repo root is two up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from autosft.agents.roles import TASK_ROLES  # noqa: E402
from autosft.config.components import (  # noqa: E402
    composition_name,
    analysis_enabled,
    synthesis_enabled,
)
from autosft.agents.search_tasks import TASK_ID_RE  # noqa: E402

# Keep the legacy shell names and the current exec_command/cmd interface.
_SHELL_TOOLS = (
    "shell_command",
    "exec_command",
    "Bash",
    "shell",
    "exec",
    "local_shell_call",
)
_PATCH_TOOLS = ("apply_patch", "Edit", "Write")
_SPAWN_TOOLS = ("spawn_agent", "Task")
_LIFECYCLE_STATUS_RE = re.compile(
    r"autosft(?:\s+(?:tools\s+)?jobs|\.backends\.local\.lifecycle)\s+(?:sft|eval)\s+status\b"
)

# Executor sessions receive the run environment needed to launch the fixed
# training/evaluation pipeline, but they must never inspect credentials or
# evaluation outcomes.  Keep this list based on basenames so both absolute
# run paths and `${AUTO_SFT_RUN_DIR}/...` spellings are covered without
# expanding or printing any environment value.
_EXECUTOR_SECRET_BASENAMES = frozenset(
    {
        "task.env",
        ".env",
        ".env.local",
        ".env.development",
        ".env.test",
        ".env.production",
        ".netrc",
        "auth.json",
        "credentials.json",
        "secrets.json",
    }
)
_EXECUTOR_RESULT_BASENAME_RE = re.compile(
    r"^(?:scores?|metrics?|predictions?|failures?|failure_breakdown|"
    r"(?:eval|sft)_final_status|aggregate)\.(?:json|jsonl)$",
    re.IGNORECASE,
)
_EXECUTOR_READ_COMMANDS = frozenset(
    {
        # Content and metadata readers commonly used by Codex agents.
        "cat",
        "bat",
        "tac",
        "head",
        "tail",
        "less",
        "more",
        "sed",
        "awk",
        "gawk",
        "cut",
        "jq",
        "yq",
        "rg",
        "ripgrep",
        "grep",
        "egrep",
        "fgrep",
        "find",
        "fd",
        "fdfind",
        "ls",
        "ll",
        "tree",
        "stat",
        "file",
        "wc",
        "diff",
        "cmp",
        "comm",
        "git",
        "python",
        "python3",
        "bash",
        "sh",
        ".",
        "source",
    }
)
_EXECUTOR_COPY_COMMANDS = frozenset(
    {"cp", "mv", "install", "rsync", "tar", "zip", "unzip"}
)
_EXECUTOR_ENV_DUMP_COMMANDS = frozenset(
    {"env", "printenv", "set", "export", "declare", "typeset"}
)
_EXECUTOR_SECRET_ENV_RE = re.compile(
    r"\b(?:AUTO_SFT_LLM_API_KEY|HUGGINGFACE_HUB_TOKEN|HF_TOKEN|"
    r"[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|SECRET_KEY|PASSWORD))\b"
)


def _executor_protected_basename(value: str) -> str | None:
    """Return a protected artifact basename mentioned by *value*.

    The hook deliberately checks basenames after stripping shell punctuation.
    This catches absolute paths, ``../`` paths, glob arguments, and paths
    embedded in common Python expressions without expanding any secret-bearing
    environment variable.
    """
    cleaned = value.strip("\"'`;,()[]{}<>|&")
    if not cleaned:
        return None
    basename = Path(cleaned).name.lower()
    if basename in _EXECUTOR_SECRET_BASENAMES:
        return basename
    if _EXECUTOR_RESULT_BASENAME_RE.fullmatch(basename):
        return basename
    return None


def _executor_read_violation(
    command: str, cwd: Path, policy_guard: object | None = None
) -> str | None:
    """Deny executor reads of credentials and evaluation outcomes.

    Executor commands are still allowed to inspect handoff, manifests,
    training logs, and checkpoints.  This rule is intentionally scoped to the
    executor role; meta and analysis agents need to read evaluation evidence
    to make their decisions.
    """
    if os.environ.get("AUTO_SFT_ROLE") != "executor":
        return None
    run_dir_value = os.environ.get("AUTO_SFT_RUN_DIR")
    run_dir = None
    if run_dir_value:
        try:
            run_dir = Path(run_dir_value).expanduser().resolve()
        except OSError:
            run_dir = Path(os.path.normpath(run_dir_value))
    # Reading a credential directly from the inherited process environment is
    # equivalent to opening task.env.  Match variable names only; never expand
    # or include their values in the denial reason.
    if _EXECUTOR_SECRET_ENV_RE.search(command):
        return "executor cannot inspect credential environment variables"
    try:
        segments = (
            policy_guard._split_shell_segments(command)
            if policy_guard is not None
            else re.split(r"\s*(?:&&|\|\||[;|])\s*", command)
        )
    except Exception:
        segments = [command]
    for segment in segments or [command]:
        try:
            tokens = _shell_path_tokens(segment)
        except Exception:
            tokens = segment.split()
        if not tokens:
            continue
        # A recursive scan of the whole run can reach protected artifacts even
        # when no result filename appears in the command.  Executors should
        # use the handoff's explicit paths and stage directories instead.
        if run_dir is not None:
            reader_names = {Path(token).name.lower() for token in tokens}
            if reader_names & {"rg", "ripgrep", "grep", "find", "fd", "fdfind", "tree"}:
                # Variable-expanded paths cannot be resolved before the shell
                # runs.  Treat an explicit run-root expansion as a recursive
                # run scan too; otherwise `rg ... "$AUTO_SFT_RUN_DIR"` would
                # bypass the same check that catches its absolute spelling.
                if re.search(r"\$\{?AUTO_SFT_RUN_DIR\}?", segment):
                    return "executor cannot recursively scan the whole run directory"
                for token in tokens:
                    resolved = _resolve_shell_path(token, cwd)
                    if resolved is not None and (
                        resolved == run_dir or _is_relative_to(resolved, run_dir)
                    ):
                        return "executor cannot recursively scan the whole run directory"
        command_names = {Path(token).name.lower() for token in tokens[:3]}
        # Environment dumping is a credential read even when no .env path is
        # named explicitly.  ``env`` with assignments is still harmless here
        # only when it is part of a sanctioned command; a standalone dump is
        # denied below.
        if command_names & _EXECUTOR_ENV_DUMP_COMMANDS:
            first = Path(tokens[0]).name.lower()
            # ``env NAME=value command`` is a launcher form, not an
            # environment dump.  All other standalone env/set/printenv forms
            # can expose the inherited API or hub credentials.
            assignments = any("=" in token and not token.startswith("=") for token in tokens[1:])
            if not (first == "env" and assignments and len(tokens) > 2):
                return "executor cannot dump the process environment or credentials"

        protected: str | None = None
        for token in tokens:
            candidate = _executor_protected_basename(token)
            if candidate:
                protected = candidate
                break
        # Interpreter snippets often carry the filename inside one quoted
        # token (for example Path('scores.json').read_text()).  Inspect the
        # command text as well, but only for a basename-shaped occurrence.
        if protected is None:
            match = re.search(
                r"(?<![A-Za-z0-9_.-])((?:task\.env|\.env(?:\.[A-Za-z0-9_-]+)?|"
                r"scores?\.json|metrics?\.json|predictions?\.json|"
                r"failures?(?:\.jsonl?)?|failure_breakdown\.json|"
                r"(?:eval|sft)_final_status\.json|aggregate\.json))(?![A-Za-z0-9_.-])",
                segment,
                flags=re.IGNORECASE,
            )
            if match:
                protected = match.group(1).lower()
        if protected is None:
            continue
        # A bare filename in a prose argument is not enough to establish a
        # file read.  Limit the denial to commands that can inspect or copy
        # files, while allowing ordinary report text such as ``echo scores``.
        executable_names = {
            Path(token).name.lower()
            for token in tokens
            if token and not token.startswith("-")
        }
        if executable_names & (_EXECUTOR_READ_COMMANDS | _EXECUTOR_COPY_COMMANDS):
            return (
                "executor cannot read credential or evaluation-result artifact "
                f"'{protected}'"
            )
        # Python/Node-style file APIs and shell redirection do not always put
        # the protected path in a standalone argument.
        if re.search(
            r"(?:open\s*\(|read_text\s*\(|read_bytes\s*\(|load\s*\(|"
            r"json\.load\s*\(|(?:^|\s)[0-9]*<\s*)",
            segment,
            flags=re.IGNORECASE,
        ):
            return (
                "executor cannot read credential or evaluation-result artifact "
                f"'{protected}'"
            )
    return None


def _record_spawn_event(
    requested: object, architecture: str | None, *, allowed: bool, reason: str
) -> None:
    """Append one delegation attempt to `${AUTO_SFT_RUN_DIR}/spawn_events.jsonl`.

    This hook is the only place that observes every spawn attempt, so the
    run-local record is what tells us how each condition actually used codex
    built-in subagents.
    Best-effort by design: a recording failure must never deny a tool call.
    """
    run_dir = os.environ.get("AUTO_SFT_RUN_DIR")
    if not run_dir:
        return
    try:
        path = Path(run_dir) / "spawn_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "architecture": architecture,
                        "requested_agent": requested
                        if isinstance(requested, str)
                        else None,
                        "allowed": allowed,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError as exc:
        print(
            f"[autosft policy_hook] could not record spawn event: {exc}",
            file=sys.stderr,
        )


def _shell_path_tokens(command: str) -> list[str]:
    """Return shell tokens that may name files or directories."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    result: list[str] = []
    for token in tokens:
        token = token.strip("\"'`;|&()<>[],")
        if token:
            result.append(token)
    return result


def _resolve_shell_path(token: str, cwd: Path) -> Path | None:
    """Resolve concrete path tokens, ignoring shell variables and flags."""
    if not token or token.startswith("-") or "$" in token or "{" in token:
        return None
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve()
    except OSError:
        return Path(os.path.normpath(str(candidate)))


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or one of its descendants."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _recursive_scan_roots(command: str, cwd: Path) -> list[Path]:
    """Return directories a recursive read command may traverse.

    The hook sees a command before the shell expands it. In particular, a
    command such as 'rg pattern src runs' contains a bare directory
    name rather than a token with '/'. We parse the small set of recursive
    readers commonly used by agents so those roots are subject to the same
    run boundary as explicit absolute paths.
    """
    tokens = _shell_path_tokens(command)
    roots: list[Path] = []
    recursive_readers = {"rg", "ripgrep", "find", "grep", "ls", "tree", "du"}
    option_values = {
        "-e",
        "--regexp",
        "-g",
        "--glob",
        "-f",
        "--file",
        "--type",
        "--type-not",
        "--exclude",
        "--include",
    }

    for index, token in enumerate(tokens):
        name = Path(token).name
        if name not in recursive_readers:
            continue
        args = tokens[index + 1 :]
        positional: list[str] = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg in {";", "|", "&&", "||"}:
                break
            if arg in option_values:
                skip_next = True
                continue
            if arg.startswith("-") and arg != "-":
                continue
            positional.append(arg)

        if name in {"rg", "ripgrep", "grep"}:
            # rg --files PATH has no pattern. Ordinary rg/grep commands use
            # their first positional argument as the pattern.
            has_files_mode = "--files" in args
            if not has_files_mode and positional:
                positional = positional[1:]
        elif name == "find":
            # find consumes leading paths until its first expression option;
            # no path means the current working directory.
            positional = []
            for arg in args:
                if arg in {";", "|", "&&", "||"}:
                    break
                if arg.startswith("-"):
                    break
                positional.append(arg)

        if not positional:
            roots.append(cwd)
        else:
            for path_token in positional:
                resolved = _resolve_shell_path(path_token, cwd)
                if resolved is not None:
                    roots.append(resolved)
    return roots


def _condition_read_violation(command: str, cwd: Path) -> str | None:
    """Return a reason when a root references another condition's files."""
    run_dir_value = os.environ.get("AUTO_SFT_RUN_DIR")
    if not run_dir_value:
        return None
    run_dir = Path(run_dir_value).expanduser()
    try:
        run_dir = run_dir.resolve()
    except OSError:
        run_dir = Path(os.path.normpath(str(run_dir)))
    shared_runs = Path(
        os.environ.get(
            "AUTO_SFT_AGENT_RUNS_ROOT",
            str(_REPO_ROOT / "runs"),
        )
    ).expanduser()
    try:
        shared_runs = shared_runs.resolve()
    except OSError:
        shared_runs = Path(os.path.normpath(str(shared_runs)))

    for token in _shell_path_tokens(command):
        resolved = _resolve_shell_path(token, cwd)
        if resolved is None:
            continue
        try:
            resolved.relative_to(shared_runs)
        except ValueError:
            continue
        try:
            resolved.relative_to(run_dir)
        except ValueError:
            return (
                f"fixed {composition_name()} soft isolation: access to another "
                f"experiment run under '{shared_runs}' is denied"
            )

    # A recursive command can reach protected runs without naming one of them
    # directly: 'rg pattern src runs', 'rg pattern .', and 'find .'
    # can reach other experiments. Deny roots that contain the shared
    # run tree, as well as roots inside it that are not the current run.
    for root in _recursive_scan_roots(command, cwd):
        if _is_relative_to(root, shared_runs) and not _is_relative_to(root, run_dir):
            return (
                f"fixed {composition_name()} soft isolation: recursive access to "
                f"another experiment run under '{shared_runs}' is denied"
            )
        if _is_relative_to(shared_runs, root) and not _is_relative_to(root, run_dir):
            return (
                f"fixed {composition_name()} soft isolation: recursive access to "
                f"the shared experiment tree '{shared_runs}' is denied"
            )
    return None


def _manual_lifecycle_poll_loop(command: str) -> bool:
    """Detect status entrypoints embedded in agent-authored polling loops."""
    if not _LIFECYCLE_STATUS_RE.search(command):
        return False
    return bool(
        re.search(r"\b(?:while|until)\b", command)
        or re.search(r"\bwatch(?:\s|$)", command)
    )


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def _allow() -> None:
    sys.exit(0)


def _apply_patch_target_paths(patch_text: str) -> list[str]:
    """File paths an apply_patch payload would create / modify / delete / move."""
    prefixes = (
        "*** Add File: ",
        "*** Update File: ",
        "*** Delete File: ",
        "*** Move to: ",
    )
    paths: list[str] = []
    for line in patch_text.splitlines():
        stripped = line.strip()
        for prefix in prefixes:
            if stripped.startswith(prefix):
                candidate = stripped[len(prefix) :].strip()
                if candidate:
                    paths.append(candidate)
                break
    return paths


def _nested_apply_patch_target_paths(command: str) -> list[str]:
    """Targets from an ``apply_patch`` call wrapped by the Codex exec tool.

    Multi-agent Codex can emit JavaScript such as
    ``tools.apply_patch({patch: "..."})``. The PreToolUse hook sees that outer
    ``exec`` call, not a separate apply_patch event, so inspect its embedded
    patch as well. String literals may carry escaped newlines.
    """
    if not re.search(r"\btools\.apply_patch\s*\(", command):
        return []
    candidates = (command, command.replace("\\r\\n", "\n").replace("\\n", "\n"))
    paths: list[str] = []
    for candidate in candidates:
        for path in _apply_patch_target_paths(candidate):
            if path not in paths:
                paths.append(path)
    return paths


def _task_identity() -> tuple[str, Path, str, str] | None:
    roles = {
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {None, "", "meta", "executor"}
    task_root = os.environ.get("AUTO_SFT_TASK_DIR") or os.environ.get(
        "AUTO_SFT_SEARCH_TASK_DIR"
    )
    if not roles and not task_root:
        return None
    role = next(iter(roles), "")
    task_id = os.environ.get("AUTO_SFT_TASK_ID", "")
    round_id = os.environ.get("AUTO_SFT_TASK_ROUND_ID", "")
    run_dir = os.environ.get("AUTO_SFT_RUN_DIR", "")
    if (
        len(roles) != 1
        or role not in TASK_ROLES
        or not task_root
        or not run_dir
        or not Path(task_root).is_absolute()
        or not TASK_ID_RE.fullmatch(task_id)
        or not re.fullmatch(r"[1-9][0-9]*", round_id)
    ):
        _deny("AutoSFT task identity is invalid; recover the task through the harness.")
    task_dir = Path(task_root).resolve()
    expected = (
        Path(run_dir).resolve() / "role_io" / f"round_{round_id}" / "tasks" / task_id
    )
    if task_dir != expected or not task_dir.is_relative_to(Path(run_dir).resolve()):
        _deny("AutoSFT task directory does not match its round and task id.")
    return role, task_dir, round_id, task_id


def _writable_allowlist(cwd: Path) -> list[Path] | None:
    """Absolute dirs/files the agent may write, or None if it can't be determined."""
    run_dir = os.environ.get("AUTO_SFT_RUN_DIR")
    if not run_dir:
        return None  # cannot enforce paths without the run dir; fail open
    task = _task_identity()
    if task is not None:
        scratch = (task[1] / "scratch").resolve()
        if not scratch.is_relative_to(task[1]):
            _deny("AutoSFT task scratch must remain inside its own task directory.")
        return [scratch]
    datasets_dir = os.environ.get("AUTO_SFT_DATASETS_DIR") or str(
        Path(run_dir) / "datasets"
    )
    roots = [Path(run_dir), Path(datasets_dir)]
    # The generated recipe is owned by the run directory.
    roots.append(Path(run_dir) / "data-recipe.yaml")
    return [p if p.is_absolute() else (cwd / p) for p in roots]


def _task_control_violation(command: str, policy_guard: object, cwd: Path) -> bool:
    task = _task_identity()
    if task is None:
        return False
    role, _, round_id, task_id = task
    for segment in policy_guard._split_shell_segments(command):
        if policy_guard._segment_is_read_only(segment):
            continue
        try:
            tokens = [os.path.expandvars(token) for token in shlex.split(segment)]
        except ValueError:
            tokens = segment.split()
        if "--help" in tokens or "-h" in tokens:
            continue
        if re.search(
            r"(?:autosft\s+(?:(?:tools\s+)?(?:train|convert|evaluate|prepare)|run|round)|autosft\.(?:backends\.megatron\.(?:train|convert)|benchmarks\.submit)|src/autosft/(?:backends/megatron/(?:train|convert)|benchmarks/submit)\.py)",
            segment,
        ):
            return True
        if re.search(
            r"(?:\b(?:sft_train\.py|train\.py|autosft\s+(?:tools\s+)?train|autosft\.backends\.megatron\.train)\b|\bevaluate\.py\b|"
            r"\brun_(?:role|search_task)\.sh\b|"
            r"\bautosft\s+(?:tools\s+)?jobs\b|\bautosft\.backends\.local\.(?:submission|lifecycle|collection)\b)",
            segment,
        ):
            return True
        if re.search(
            r"\bautosft\.(?:agents\.(?:executor_handoff|runner)|data\.(?:preparation|pool_staging))\b",
            segment,
        ):
            return True
        if re.search(r"\bautosft\.data\.candidate_data\s+publish\b", segment):
            return True
        if re.search(
            r"\bautosft\.orchestration\.blocked_rounds\s+acknowledge\b", segment
        ):
            return True
        if re.search(
            r"\bautosft\.agents\.executor_reports\s+persist-executor\b", segment
        ):
            return True
        if re.search(
            r"\bautosft\.agents\.search_tasks\s+(?:request|run|run-pending|park)\b",
            segment,
        ):
            return True
        if (
            re.search(r"\bautosft\.orchestration\.experiment_memory\b", segment)
            and "--refresh" in tokens
        ):
            return True
        if re.search(
            r"\bautosft\.orchestration\.ledger\s+"
            r"(?:note|review|record-[\w-]+|hypothesis\s+(?:record|judge|update))\b",
            segment,
        ):
            return True
        if (
            re.search(r"\bautosft\.orchestration\.ledger\s+summarize\b", segment)
            and "--no-md" not in tokens
        ):
            return True
        if (
            any(Path(token).name == "batch.py" for token in tokens)
            or "autosft.backends.providers.batch" in segment
        ):
            if "--status" in tokens or "--validate-only" in tokens:
                continue
            options = {}
            for index, token in enumerate(tokens):
                if token.startswith("--"):
                    key, separator, value = token.partition("=")
                    options[key] = (
                        value
                        if separator
                        else (tokens[index + 1] if index + 1 < len(tokens) else "")
                    )
            label = options.get("--label", "")
            job_run = Path(options.get("--run-dir") or os.environ["AUTO_SFT_RUN_DIR"])
            job_run = job_run if job_run.is_absolute() else cwd / job_run
            if (
                options.get("--round-id") != round_id
                or not label.startswith(task_id + "__")
                or label == task_id + "__"
                or "/" in label
                or "\\" in label
                or job_run.resolve() != Path(os.environ["AUTO_SFT_RUN_DIR"]).resolve()
            ):
                return True
    return False


def _synthesis_violation(command: str, policy_guard: object) -> str | None:
    """Keep synthesis opt-in and owned by the meta agent."""
    for segment in policy_guard._split_shell_segments(command):
        if policy_guard._segment_is_read_only(segment):
            continue
        if not re.search(r"autosft\.synthesis\.job|synthesis_job\.py", segment):
            continue
        if not synthesis_enabled():
            return "Synthesis is disabled for this run"

        roles = {
            os.environ.get(name, "")
            for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
        } - {"", "meta"}
        if roles or _task_identity() is not None:
            return "Synthesis may only be launched by the meta agent"
    return None


def _analysis_violation(command: str, policy_guard: object) -> str | None:
    if analysis_enabled():
        return None
    for segment in policy_guard._split_shell_segments(command):
        if policy_guard._segment_is_read_only(segment):
            continue
        if re.search(
            r"autosft\.agents\.search_tasks\s+(?:request|run|run-pending)\b", segment
        ):
            return "Analysis tasks are disabled for this run"
    return None


def _path_allowed(target: str, cwd: Path, allowlist: list[Path]) -> bool:
    resolved = Path(target)
    resolved = resolved if resolved.is_absolute() else (cwd / resolved)
    try:
        resolved = resolved.resolve()
    except OSError:
        resolved = Path(os.path.normpath(str(resolved)))
    task = _task_identity()
    if task is not None:
        _, _, round_id, task_id = task
        gateway_dir = (
            Path(os.environ["AUTO_SFT_RUN_DIR"]).resolve()
            / "llm_calls"
            / f"round_{round_id}"
        )
        if (
            resolved.parent == gateway_dir
            and resolved.name.startswith(task_id + "__")
            and resolved.name != task_id + "__"
        ):
            return True
    for root in allowlist:
        try:
            root_r = root.resolve()
        except OSError:
            root_r = Path(os.path.normpath(str(root)))
        if resolved == root_r:
            return True
        try:
            resolved.relative_to(root_r)
            return True
        except ValueError:
            continue
    return False


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        _allow()
    event = json.loads(raw)
    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input") or {}
    command = ""
    if isinstance(tool_input, dict):
        command = str(tool_input.get("command") or tool_input.get("cmd") or "")
    cwd = Path(str(event.get("cwd") or _REPO_ROOT))

    import autosft.orchestration.policy_guard as policy_guard

    try:
        architecture = composition_name()
    except ValueError as exc:
        _deny("AutoSFT configuration: " + str(exc))

    if tool_name in _SPAWN_TOOLS:
        requested = None
        if isinstance(tool_input, dict):
            requested = (
                tool_input.get("agent_type")
                or tool_input.get("task_name")
                or tool_input.get("name")
            )
        # Codex built-in subagent delegation is shared by all compositions;
        # record it so each run keeps a durable trace of how it organized.
        _record_spawn_event(
            requested, architecture, allowed=True, reason="builtin_subagent"
        )

    # Family 1: forbidden training flows / fixed-config overrides / job-submission
    # & local runtime infrastructure changes.
    if tool_name in _SHELL_TOOLS and command:
        analysis_violation = _analysis_violation(command, policy_guard)
        if analysis_violation:
            _deny("AutoSFT policy: " + analysis_violation + ".")
        synthesis_violation = _synthesis_violation(command, policy_guard)
        if synthesis_violation:
            _deny("AutoSFT policy: " + synthesis_violation + ".")
        if _task_control_violation(command, policy_guard, cwd):
            _deny(
                "AutoSFT tasks write their own scratch and task-prefixed gateway jobs only; "
                "root owns shared state and publication; "
                "executor owns training/evaluation. Persist or wait through search_tasks for this task."
            )
        executor_read_violation = _executor_read_violation(
            command, cwd, policy_guard
        )
        if executor_read_violation:
            _deny("AutoSFT policy: " + executor_read_violation + ".")
        read_violation = _condition_read_violation(command, cwd)
        if read_violation:
            _deny(read_violation)
        if _manual_lifecycle_poll_loop(command):
            _deny(
                "AutoSFT lifecycle: run each Local status entrypoint once "
                "per shell call, then follow executor_handoff.json. Hand-written "
                "polling loops can read the wrong event stream and strand a "
                "completed task."
            )
        rules = policy_guard.forbidden_rules_for_command(command)
        if rules:
            _deny(
                "AutoSFT policy: blocked forbidden control/training flow "
                f"({', '.join(rules)}). SFT must go through a recipe YAML + "
                f"python {_REPO_ROOT}/src/autosft/backends/megatron/train.py; benchmark evaluation must "
                f"go through `python {_REPO_ROOT}/src/autosft/benchmarks/submit.py "
                "--run-name round_<n> ...` "
                f"(see {_REPO_ROOT}/skills/executor/evaluation-submission/SKILL.md); no "
                "DPO/RLHF/preference/reward-model flows and no direct "
                "internal task creation APIs. A policy denial means the approach "
                "is wrong, not the phrasing: use the sanctioned entrypoint or "
                "report blocked. Do not rewrite the command to evade this "
                "check — an evasion that submits will fail in the local task "
                "container, which does not share this session's environment."
            )
        overrides = policy_guard.training_param_overrides_for_command(command)
        if overrides:
            _deny("AutoSFT policy: blocked training-config override. " + overrides[0])
        infra = policy_guard.job_infra_overrides_for_command(command)
        if infra:
            _deny(
                "AutoSFT policy: blocked job-submission/infrastructure change "
                "(model / provider / interpreter / GPU / CPU / memory "
                "/ image / git source are fixed run defaults, outside the agent's "
                "allowed scope). " + infra[0]
            )

    # Family 2: file edits outside the writable allowlist (no repo-source edits).
    # Multi-agent custom exec calls can wrap tools.apply_patch without emitting
    # a second hook event, so inspect those embedded patch headers too.
    patch_targets: list[str] = []
    if tool_name in _PATCH_TOOLS and command:
        patch_targets = _apply_patch_target_paths(command)
    elif tool_name in _SHELL_TOOLS and command:
        patch_targets = _nested_apply_patch_target_paths(command)
    if tool_name in _PATCH_TOOLS and isinstance(tool_input, dict):
        # Edit/Write-shaped tools name their target directly instead of
        # embedding *** headers in a patch body.
        direct_target = tool_input.get("file_path") or tool_input.get("path")
        if isinstance(direct_target, str) and direct_target.strip():
            patch_targets.append(direct_target.strip())
    if patch_targets:
        allowlist = _writable_allowlist(cwd)
        if allowlist is not None:
            for target in patch_targets:
                if not _path_allowed(target, cwd, allowlist):
                    _deny(
                        f"AutoSFT policy: blocked edit to '{target}' outside the "
                        "writable areas (${AUTO_SFT_RUN_DIR} / "
                        "${AUTO_SFT_DATASETS_DIR}). Do not edit repo source."
                    )

    # Family 3: shell commands that write outside the allowlist. `apply_patch`
    # is not the only way to create a file -- a redirect, a `cp`, or an inline
    # `python - <<'PY'` heredoc all bypass Family 2 entirely. Only targets the
    # command names unambiguously are checked; see policy_guard for what that
    # deliberately misses.
    if tool_name in _SHELL_TOOLS and command:
        allowlist = _writable_allowlist(cwd)
        if allowlist is not None:
            for target in policy_guard.shell_write_targets(
                command, include_temporary=_task_identity() is not None
            ):
                if not _path_allowed(target, cwd, allowlist):
                    _deny(
                        f"AutoSFT policy: blocked write to '{target}' outside "
                        "the writable areas (${AUTO_SFT_RUN_DIR} / "
                        "${AUTO_SFT_DATASETS_DIR}). Run-local artifacts belong "
                        "under ${AUTO_SFT_RUN_DIR}; a bare relative path "
                        "resolves against the repository root, not the run "
                        "directory. Do not edit repo source."
                    )

    _allow()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # fail OPEN so a hook bug can't brick a run
        print(f"[autosft policy_hook] non-fatal hook error: {exc}", file=sys.stderr)
        sys.exit(0)
