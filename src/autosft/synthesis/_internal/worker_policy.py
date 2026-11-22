"""PreToolUse policy hook for a black-box Synthesis worker."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping


_SHELL_TOOLS = frozenset(
    {"shell_command", "exec_command", "Bash", "shell", "exec", "local_shell_call"}
)
_PATCH_TOOLS = frozenset({"apply_patch", "Edit", "Write"})
_SPAWN_TOOLS = frozenset({"spawn_agent", "Task"})
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9._+@=~-]+/)*[A-Za-z0-9._+@=~-]*"
)
_PARENT_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])\.\.(?:[/\\]|$|[\s'\";|&,)])")
_HOME_EXPANSION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:~(?:[/\\]|$)|\$(?:\{)?"
    r"(?:HOME|PWD|OLDPWD|CODEX_HOME)(?:\})?)"
)
# Worker paths are rendered into the prompt as frozen, absolute paths.  Do not
# allow shell variable expansion to defer path resolution until execution: an
# otherwise harmless-looking ``$PYTHONPATH``/``$AUTO_SFT_SYN_*`` can
# resolve to a repository or workspace path that the literal-path check never
# sees.  This is intentionally broad; the worker does not need shell variables
# for its documented generation contract.
_SHELL_VARIABLE_EXPANSION_RE = re.compile(
    r"(?<!\\)\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})"
)
_NETWORK_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:curl|wget|aria2c|nc|ncat|ssh|scp|rsync)\b"
    r"|\bgit\s+(?:clone|fetch|pull)\b"
    r"|\b(?:pip|pip3|uv)\s+install\b",
    re.IGNORECASE,
)
_SYN_SETTING_EDIT_RE = re.compile(
    r"\bAUTO_SFT_(?:SYN|LLM)_[A-Z0-9_]+\s*="
    r"|\b(?:export|unset)\s+AUTO_SFT_(?:SYN|LLM)_[A-Z0-9_]+"
    r"|\benviron\s*\[\s*['\"]AUTO_SFT_(?:SYN|LLM)_[A-Z0-9_]+['\"]\s*\]\s*="
    r"|\benviron\.(?:update|pop|setdefault)\s*\([^\n]*AUTO_SFT_(?:SYN|LLM)_"
    r"|\b(?:putenv|unsetenv)\s*\(\s*['\"]AUTO_SFT_(?:SYN|LLM)_"
)
_SUBAGENT_WRITE_RE = re.compile(
    r"(?<![0-9<>=!])>>?\s*[^=]"
    r"|(?:^|[;&|]\s*|\s)(?:touch|mkdir|install|tee|cp|mv|rsync|"
    r"truncate|chmod|chown|apply_patch)\b"
    r"|\b(?:write_text|write_bytes|json\.dump|yaml\.(?:safe_)?dump|"
    r"shutil\.(?:copy|copy2|copyfile|copytree|move|rmtree))\b"
    # Python file handles are a common way for a read-only subagent to evade
    # shell-oriented write matching.  Treat every mode containing a write,
    # append, exclusive-create, or update marker as a write intent while
    # leaving ordinary read modes (``r``/``rb``) untouched.
    r"|\b(?:io\.)?open\s*\([^)]*(?:,\s*|\bmode\s*=\s*)"
    r"['\"][^'\"]*[wax+][^'\"]*['\"]"
    # Mutating filesystem metadata or names is also a write, even when no
    # file contents are opened.
    r"|\b(?:os\.)?(?:chmod|chown|unlink|remove|rename|replace|rmdir)\s*\("
    r"|\bPath\s*\([^)]*\)\.(?:chmod|unlink|rename|replace|rmdir|mkdir)\s*\("
    r"|\b(?:os\.)?open\s*\([^)]*\b(?:O_WRONLY|O_RDWR|O_CREAT|O_APPEND)\b"
    r"|\b(?:tempfile\.)?(?:mkstemp|mkdtemp|NamedTemporaryFile|TemporaryFile)\s*\(",
    re.IGNORECASE,
)
# A delegated Synthesis subagent is an evidence reader, not an executor.  Keep this
# list intentionally small and command-oriented: a regex that only looks for
# obvious write verbs is easy to evade with an interpreter, a shell wrapper,
# or a less common file operation.  The top-level worker still has the
# separate output-directory write policy below; this applies only when an
# event is marked as coming from a nested agent.
_SUBAGENT_READONLY_COMMANDS = frozenset(
    {
        "cat",
        "cmp",
        "comm",
        "cut",
        "diff",
        "dirname",
        "du",
        "file",
        "find",
        "fold",
        "grep",
        "head",
        "ls",
        "md5sum",
        "nl",
        "paste",
        "printf",
        "pwd",
        "readlink",
        "realpath",
        "rev",
        "rg",
        "sha1sum",
        "sha256sum",
        "sort",
        "stat",
        "strings",
        "tail",
        "tac",
        "true",
        "false",
        "uniq",
        "wc",
        "which",
        "xxd",
        "echo",
        "jq",
        "yq",
        # Read-only git porcelain/plumbing useful for provenance inspection.
        "git",
    }
)
_SUBAGENT_GIT_READONLY_SUBCOMMANDS = frozenset(
    {
        "cat-file",
        "diff",
        "log",
        "ls-files",
        "rev-parse",
        "show",
        "status",
    }
)
_SUBAGENT_MUTATING_COMMANDS = frozenset(
    {
        "apply_patch",
        "chmod",
        "chown",
        "cp",
        "dd",
        "install",
        "mkdir",
        "mktemp",
        "mv",
        "rm",
        "rsync",
        "tee",
        "touch",
        "truncate",
        "unlink",
    }
)
_SUBAGENT_INTERPRETER_COMMANDS = frozenset(
    {
        "awk",
        "bash",
        "dash",
        "node",
        "perl",
        "python",
        "python2",
        "python3",
        "ruby",
        "sh",
        "zsh",
    }
)
_SUBAGENT_WRAPPER_COMMANDS = frozenset(
    {
        "command",
        "env",
        "eval",
        "exec",
        "nice",
        "nohup",
        "setsid",
        "sh -c",
        "sudo",
        "time",
        "timeout",
        "xargs",
    }
)
_SUBAGENT_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _split_nested_shell(command: str) -> tuple[list[str], str | None]:
    """Split a delegated shell command while retaining its safety signals.

    This is deliberately a small lexer rather than a shell interpreter.  Any
    construct that could execute another command or redirect a stream is
    rejected; ordinary pipelines/sequences are checked command-by-command.
    """

    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(command):
        char = command[i]
        if escaped:
            current.append(char)
            escaped = False
            i += 1
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            i += 1
            continue
        if quote is not None:
            # Double-quoted shell text still performs variable and command
            # substitution.  Single quotes are literal and can safely contain
            # characters such as ``$`` or backticks used in a search pattern.
            if quote == '"' and (
                char == "`" or command[i : i + 2] == "$(" or char == "$"
            ):
                return [], "shell expansion is not allowed"
            current.append(char)
            if char == quote:
                quote = None
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            i += 1
            continue
        if char == "`" or command[i : i + 2] in {"$(", "<(", ">("}:
            return [], "command substitution and process substitution are not allowed"
        if char == "$":
            return [], "shell variable expansion is not allowed"
        if char in {"<", ">"}:
            return [], "shell redirection is not allowed"
        if command[i : i + 2] in {"&&", "||"}:
            if "".join(current).strip():
                segments.append("".join(current).strip())
            current = []
            i += 2
            continue
        if char in {"|", ";", "\n"}:
            if "".join(current).strip():
                segments.append("".join(current).strip())
            current = []
            i += 1
            continue
        if char == "&":
            return [], "background shell commands are not allowed"
        current.append(char)
        i += 1
    if quote is not None:
        return [], "unterminated shell quote"
    if "".join(current).strip():
        segments.append("".join(current).strip())
    return segments, None


def _nested_command_name(tokens: list[str]) -> tuple[str | None, int]:
    """Return the executable token and its index, or ``(None, -1)``."""

    if not tokens:
        return None, -1
    # Assignments and wrappers are intentionally rejected by the caller.  This
    # helper still handles an assignment prefix so malformed input fails closed.
    index = 0
    while index < len(tokens) and _SUBAGENT_ASSIGNMENT_RE.match(tokens[index]):
        index += 1
    if index >= len(tokens):
        return None, -1
    return Path(tokens[index]).name.lower(), index


def _nested_readonly_command_reason(command: str) -> str | None:
    """Validate one nested-agent shell invocation against a read-only allowlist."""

    segments, syntax_error = _split_nested_shell(command)
    if syntax_error:
        return f"Synthesis read-only subagents cannot run file-writing commands: {syntax_error}"
    if not segments:
        return None
    for segment in segments:
        try:
            lexer = shlex.shlex(segment, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            return "Synthesis read-only subagents cannot run file-writing commands: invalid shell syntax"
        if not tokens:
            continue
        # Environment assignments and wrappers can redirect execution to an
        # arbitrary interpreter, so they are not part of the nested contract.
        if _SUBAGENT_ASSIGNMENT_RE.match(tokens[0]):
            return "Synthesis read-only subagents cannot run file-writing commands: environment assignments are not allowed"
        name, command_index = _nested_command_name(tokens)
        if name is None:
            return "Synthesis read-only subagents cannot run file-writing commands: missing command"
        raw_executable = tokens[command_index]
        if "/" in raw_executable and not raw_executable.startswith(
            ("/bin/", "/usr/bin/", "/usr/local/bin/", "/sbin/")
        ):
            return "Synthesis read-only subagents cannot run file-writing commands: executable path is not allowlisted"
        if (
            name in _SUBAGENT_INTERPRETER_COMMANDS
            or name in _SUBAGENT_MUTATING_COMMANDS
        ):
            return f"Synthesis read-only subagents cannot run file-writing commands: {name} is not read-only"
        if name in _SUBAGENT_WRAPPER_COMMANDS:
            return f"Synthesis read-only subagents cannot run file-writing commands: {name} is a command wrapper"
        if name not in _SUBAGENT_READONLY_COMMANDS and name != "cd":
            return f"Synthesis read-only subagents cannot run file-writing commands: {name} is not allowlisted"

        args = tokens[command_index + 1 :]
        if name == "cd":
            if len(args) != 1:
                return "Synthesis read-only subagents cannot run file-writing commands: cd requires one path"
            continue
        if name == "git":
            # Skip global read-only options before the subcommand.  A missing
            # or mutating subcommand (apply/checkout/restore/clean/etc.) fails.
            git_index = 0
            while git_index < len(args):
                value = args[git_index]
                if value in {
                    "--no-pager",
                    "--literal-pathspecs",
                    "--no-optional-locks",
                }:
                    git_index += 1
                    continue
                if value in {"-C", "--git-dir", "--work-tree", "--namespace"}:
                    git_index += 2
                    continue
                if value.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
                    git_index += 1
                    continue
                break
            if (
                git_index >= len(args)
                or args[git_index] not in _SUBAGENT_GIT_READONLY_SUBCOMMANDS
            ):
                subcommand = args[git_index] if git_index < len(args) else "(missing)"
                return f"Synthesis read-only subagents cannot run file-writing commands: git {subcommand} is not read-only"
            # ``git diff`` uses ``-o``/``--output`` while ``git show`` also
            # accepts ``-o``/``--output-directory``.  Both forms redirect
            # otherwise read-only output to the filesystem, including the
            # attached short form (``-opath``).
            if any(
                value in {"-o", "--output", "--output-directory"}
                or value.startswith(("-o", "--output=", "--output-directory="))
                for value in args
            ):
                return "Synthesis read-only subagents cannot run file-writing commands: git output file is not allowed"
            dangerous_git_exact = {
                "--ext-diff",
                "--textconv",
                "--filters",
                "--external-diff",
                "--paginate",
                "--no-pager=",
                "--exec-path",
                "--upload-pack",
                "--receive-pack",
                "--git-upload-pack",
                "--git-receive-pack",
                "-c",
                "--config-env",
                "--config-env=",
            }
            for value in args:
                normalized = value.lower()
                if normalized in dangerous_git_exact or normalized.startswith(
                    (
                        "--filter=",
                        "--filter-process=",
                        "--exec-path=",
                        "--upload-pack=",
                        "--receive-pack=",
                        "--git-upload-pack=",
                        "--git-receive-pack=",
                    )
                ):
                    return (
                        "Synthesis read-only subagents cannot run git external helpers"
                    )
        elif name == "rg":
            # ripgrep's preprocessor is an arbitrary executable invoked once
            # per matching input file.  It is not a read-only search operation.
            if any(
                value == "--pre"
                or value.startswith("--pre=")
                or value == "--pre-glob"
                or value.startswith("--pre-glob=")
                for value in args
            ):
                return "Synthesis read-only subagents cannot run rg preprocessors"
        elif name == "find":
            dangerous_find_flags = {
                "-delete",
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-fprint",
                "-fprintf",
                "-fls",
            }
            if any(
                value in dangerous_find_flags
                or value.startswith(
                    tuple(dangerous_find_flags - {"-exec", "-execdir", "-ok", "-okdir"})
                )
                for value in args
            ):
                return "Synthesis read-only subagents cannot run file-writing commands: find action can mutate files"
        elif name == "sort":
            if any(
                value == "-o"
                or value.startswith("-o")
                or value == "--output"
                or value.startswith("--output=")
                or value == "--compress-program"
                or value.startswith("--compress-program=")
                or value == "-T"
                or value.startswith("-T")
                or value == "--temporary-directory"
                or value.startswith("--temporary-directory=")
                for value in args
            ):
                return "Synthesis read-only subagents cannot run file-writing commands: sort output file is not allowed"
        elif name in {"jq", "yq"}:
            if any(
                value == "-i"
                or value.startswith("-i")
                or value in {"--in-place", "--inplace"}
                or value.startswith("--in-place=")
                or value.startswith("--inplace=")
                for value in args
            ):
                return "Synthesis read-only subagents cannot run file-writing commands: in-place query output is not allowed"
            if name == "yq" and any(
                value in {"-s", "--split-exp", "--split-file"}
                or value.startswith(("-s", "--split-exp=", "--split-file="))
                for value in args
            ):
                return "Synthesis read-only subagents cannot run file-writing commands: yq split output is not allowed"
    return None


_SYSTEM_READ_ROOTS = tuple(
    Path(value)
    for value in (
        "/bin",
        "/usr",
        "/lib",
        "/lib64",
        "/sbin",
        "/dev/null",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
    )
)


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve(path: str | Path, cwd: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve(strict=False)


def _roots(environ: Mapping[str, str], name: str) -> tuple[Path, ...]:
    values = []
    for raw in environ.get(name, "").split(os.pathsep):
        if raw.strip():
            values.append(Path(raw).expanduser().resolve(strict=False))
    return tuple(values)


def _command(event: Mapping[str, Any]) -> str:
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, Mapping):
        return ""
    for name in ("command", "cmd", "patch", "input"):
        value = tool_input.get(name)
        if isinstance(value, str):
            return value
    return ""


def _patch_targets(event: Mapping[str, Any], command: str) -> list[str]:
    targets: list[str] = []
    prefixes = (
        "*** Add File: ",
        "*** Update File: ",
        "*** Delete File: ",
        "*** Move to: ",
    )
    for line in command.replace("\\n", "\n").splitlines():
        stripped = line.strip()
        for prefix in prefixes:
            if stripped.startswith(prefix):
                value = stripped[len(prefix) :].strip()
                if value:
                    targets.append(value)
                break
    tool_input = event.get("tool_input")
    if isinstance(tool_input, Mapping):
        direct = tool_input.get("file_path") or tool_input.get("path")
        if isinstance(direct, str) and direct.strip():
            targets.append(direct.strip())
    return targets


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _changed_directories(command: str, cwd: Path) -> list[Path]:
    tokens = _shell_tokens(command)
    directories: list[Path] = []
    for index, token in enumerate(tokens[:-1]):
        if token == "cd":
            value = tokens[index + 1]
            if value not in {"&&", ";", "||", "|"} and "$" not in value:
                directories.append(_resolve(value, cwd))
    return directories


def _shell_write_targets(command: str) -> list[str]:
    direct = re.findall(
        r"(?:^|[;&|]\s*)(?:touch|mkdir|install|tee|truncate|chmod|chown)\b"
        r"[^;&|]*?\s((?:/[^\s;&|]+)+)",
        command,
        flags=re.IGNORECASE,
    )
    try:
        from autosft.orchestration.policy_guard import shell_write_targets

        return [*direct, *shell_write_targets(command)]
    except Exception:
        # The absolute redirection form is enough to fail closed if importing the
        # shared helper is unexpectedly unavailable.
        return [
            *direct,
            *re.findall(r"(?<![0-9<>=!])>>?\s*(/[^\s;|&<>()]+)", command),
        ]


def _forbidden_control_command(command: str) -> str | None:
    if _SYN_SETTING_EDIT_RE.search(command):
        return "Synthesis provider and runtime settings are fixed by the launcher"
    if _NETWORK_COMMAND_RE.search(command):
        return "Synthesis workers cannot use network-capable shell commands"
    try:
        import autosft.orchestration.policy_guard as policy_guard

        rules = policy_guard.forbidden_rules_for_command(command)
        overrides = policy_guard.training_param_overrides_for_command(command)
        infra = policy_guard.job_infra_overrides_for_command(command)
    except Exception:
        return None
    if rules:
        return (
            "Synthesis workers cannot launch training/evaluation control flows: "
            + ", ".join(rules)
        )
    if overrides or infra:
        return (
            "Synthesis workers cannot change fixed training or infrastructure settings"
        )
    return None


def _nested_agent(event: Mapping[str, Any]) -> bool:
    value = event.get("agent_path") or event.get("task_path") or event.get("agentPath")
    if isinstance(value, (list, tuple)):
        return len(value) > 1
    if isinstance(value, str) and value.strip():
        parts = [item for item in value.strip("/").split("/") if item]
        return len(parts) > 1
    return False


def denial_reason(
    event: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    if env.get("AUTO_SFT_SYN_WORKER") != "1":
        return None
    tool_name = str(event.get("tool_name") or "")
    cwd = Path(str(event.get("cwd") or env.get("AUTO_SFT_SYN_WRITE_ROOT") or "."))
    cwd = cwd.expanduser().resolve(strict=False)
    read_roots = _roots(env, "AUTO_SFT_SYN_READ_ROOTS")
    write_roots = _roots(env, "AUTO_SFT_SYN_WRITE_ROOT")
    if not read_roots or len(write_roots) != 1:
        return "Synthesis worker path boundary is not configured"
    write_root = write_roots[0]
    if not _under(cwd, write_root):
        return f"Synthesis worker cwd must stay under its output directory: {cwd}"

    if tool_name in _SPAWN_TOOLS:
        if env.get("AUTO_SFT_SYN_ALLOW_READONLY_SUBAGENTS") != "1":
            return "Synthesis worker subagents are disabled"
        if _nested_agent(event):
            return "Synthesis read-only subagents cannot spawn further agents"
        # Spawned workers inherit this hook and the same path roots. The worker
        # prompt limits them to analysis; they do not receive another write root.
        return None

    command = _command(event)
    nested_agent = _nested_agent(event)
    if nested_agent and tool_name in _PATCH_TOOLS:
        return "Synthesis read-only subagents cannot edit files"
    if nested_agent and tool_name in _SHELL_TOOLS:
        readonly_reason = _nested_readonly_command_reason(command)
        if readonly_reason:
            return readonly_reason
    if tool_name in _SHELL_TOOLS:
        if not command.strip():
            return None
        control = _forbidden_control_command(command)
        if control:
            return control
        if _PARENT_PATH_RE.search(command):
            return "Synthesis workers cannot use parent-directory paths"
        if _HOME_EXPANSION_RE.search(command):
            return "Synthesis workers cannot expand home or process-directory paths"
        if _SHELL_VARIABLE_EXPANSION_RE.search(command):
            return "Synthesis workers cannot use shell variable expansion"
        changed_directories = _changed_directories(command, cwd)
        for directory in changed_directories:
            if not any(_under(directory, root) for root in (*read_roots, write_root)):
                return (
                    "Synthesis workers cannot change cwd outside their declared roots"
                )
        if any(
            _under(directory, root)
            for directory in changed_directories
            for root in read_roots
        ) and _SUBAGENT_WRITE_RE.search(command):
            return "Synthesis workers cannot write after changing cwd into a read-only the meta agent"
        for raw in _ABSOLUTE_PATH_RE.findall(command):
            raw = raw.rstrip(".,:;)]}")
            if raw == "/":
                return "Synthesis workers cannot enumerate the filesystem root"
            resolved = Path(raw).resolve(strict=False)
            if any(
                _under(resolved, root) for root in (*read_roots, *_SYSTEM_READ_ROOTS)
            ):
                continue
            return f"Synthesis worker path is outside its frozen inputs: {raw}"
        for target in _shell_write_targets(command):
            resolved = _resolve(target.strip("'\""), cwd)
            if not _under(resolved, write_root):
                return (
                    f"Synthesis worker write is outside its output directory: {target}"
                )

    if tool_name in _PATCH_TOOLS:
        for target in _patch_targets(event, command):
            resolved = _resolve(target, cwd)
            if not _under(resolved, write_root):
                return (
                    f"Synthesis worker edit is outside its output directory: {target}"
                )
    return None


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "AutoSFT Synthesis policy: " + reason,
                }
            }
        )
    )


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        _deny(f"invalid hook input: {exc}")
        return 0
    if not isinstance(event, dict):
        _deny("hook input must be an object")
        return 0
    reason = denial_reason(event)
    if reason:
        _deny(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
