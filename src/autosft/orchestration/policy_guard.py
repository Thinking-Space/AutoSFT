from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from autosft.core.common import REPO_ROOT, load_structured, utc_now, write_json
from autosft.data.recipe import RecipeValidationError, validate_minimal_recipe


SOURCE_LIKE_SUFFIXES = {
    ".cfg",
    ".conf",
    ".dockerfile",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
SOURCE_LIKE_NAMES = {"Dockerfile", "Makefile"}
# These patterns intentionally match real *invocations* of forbidden training
# flows (training scripts, CLI flags, trainer classes, direct cluster
# submission), not bare keywords. Combined with scanning only executed commands
# (see agent_log_violations), this avoids flagging the agent for merely reading
# rules, searching the repo, or printing these words.
FORBIDDEN_COMMAND_PATTERNS = {
    "forbidden_dpo_flow": re.compile(
        r"""
        (?:
            \b[\w./-]*(?:dpo|ppo|grpo|orpo|kto|rloo|rlhf|reward[_-]?model|preference[_-]?optim)[\w./-]*\.py\b
          | --(?:dpo|ppo|grpo|orpo|kto|rloo|rlhf)\b
          | --reward[_-]?model\b
          | --preference[_-]?optim(?:ization)?\b
          | \b(?:DPO|PPO|GRPO|ORPO|KTO|RLOO|Reward)Trainer\b
          | \btrl\s+(?:dpo|ppo|grpo|orpo|kto|reward|rloo)\b
          | \b(?:trl|trlx|openrlhf)\.[A-Za-z_]
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    "direct_local_submission": re.compile(
        r"\bpython(?:3)?\s+-m\s+autosft\.backends\.local\.submission\b", re.IGNORECASE
    ),
}

# Leading executables that only read/search and therefore cannot launch a
# training flow. Segments led by one of these are skipped before matching.
READ_ONLY_COMMANDS = frozenset(
    {
        "rg",
        "grep",
        "egrep",
        "fgrep",
        "ag",
        "ack",
        "ack-grep",
        "cat",
        "bat",
        "tac",
        "nl",
        "head",
        "tail",
        "less",
        "more",
        "ls",
        "ll",
        "l",
        "dir",
        "tree",
        "find",
        "fd",
        "fdfind",
        "locate",
        "stat",
        "file",
        "wc",
        "du",
        "df",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "pwd",
        "echo",
        "printf",
        "true",
        "false",
        ":",
        "cd",
        "test",
        "[",
        "[[",
        "sort",
        "uniq",
        "cut",
        "column",
        "tr",
        "fold",
        "rev",
        "tee",
        "jq",
        "yq",
        "diff",
        "cmp",
        "comm",
        "cksum",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "b2sum",
        "date",
        "sleep",
        "hexdump",
        "xxd",
        "strings",
        "set",
        "export",
        "unset",
        "alias",
        "type",
        "which",
        "whereis",
    }
)

# Transparent prefixes skipped when resolving a segment's real executable
# (e.g. `sudo python ...` or `FOO=bar timeout 30 python ...`).
_COMMAND_WRAPPERS = frozenset(
    {
        "sudo",
        "nohup",
        "env",
        "time",
        "timeout",
        "xargs",
        "nice",
        "ionice",
        "stdbuf",
        "setsid",
        "command",
        "builtin",
        "exec",
        "then",
        "do",
        "else",
    }
)

# Training hyperparameters are FIXED run defaults in a guarded run: the agent's
# research variable is the DATA, never the training config. The only ways to
# change the config from an agent command are (a) setting one of these
# `AUTO_SFT_SFT_*` env vars (inline `VAR=... cmd` or `export VAR=...`), or
# (b) passing a training-length or checkpoint-initialization override flag to
# src/autosft/backends/megatron/train.py. These are flagged so guarded AutoSFT experiments always use the
# same Qwen3 Base initialization and three-epoch schedule.
TRAINING_PARAM_ENV_VARS = frozenset(
    {
        "AUTO_SFT_SFT_TRAIN_ITERS",
        "AUTO_SFT_SFT_LOAD_CKPT",
        "AUTO_SFT_SFT_SEQ_LENGTH",
        "AUTO_SFT_SFT_SAVE_INTERVAL",
        "AUTO_SFT_SFT_LR",
        "AUTO_SFT_SFT_MIN_LR",
        "AUTO_SFT_SFT_LR_WARMUP_ITERS",
        "AUTO_SFT_SFT_MICRO_BATCH_SIZE",
        "AUTO_SFT_SFT_GLOBAL_BATCH_SIZE",
        "AUTO_SFT_SFT_TP",
        "AUTO_SFT_SFT_PP",
        "AUTO_SFT_SFT_CP",
        "AUTO_SFT_SFT_CP_SIZE",
        "AUTO_SFT_SFT_DETERMINISTIC",
        "AUTO_SFT_SFT_RECOMPUTE_NUM_LAYERS",
        "AUTO_SFT_SFT_GPUS_PER_NODE",
        # The row-order seed decides the trainer's batch composition; an agent
        # that could pick it would be tuning the optimization path, not the data.
        "AUTO_SFT_SFT_SHUFFLE_SEED",
        # The run-level token budget is a launcher-owned cap. Letting the agent
        # change it would move the budget gate itself: an over-budget dataset must
        # be reworked to fit the fixed budget, never re-budgeted to fit the data.
        "AUTO_SFT_SFT_TOKEN_BUDGET",
    }
)
_SFT_ENV_ASSIGN_RE = re.compile(r"\b(AUTO_SFT_SFT_[A-Z0-9_]+)=")
_SFT_ENTRYPOINT_RE = re.compile(
    r"\b(?:train\.py|autosft\s+(?:tools\s+)?train|autosft\.backends\.megatron\.train)\b"
)
_SEQ_LENGTH_FLAG_RE = re.compile(r"(?<![\w-])--seq-length\b")
_TRAIN_ITERS_FLAG_RE = re.compile(r"(?<![\w-])--train-iters\b")
_LOAD_CKPT_FLAG_RE = re.compile(r"(?<![\w-])--load-ckpt\b")
_RESUME_FROM_FLAG_RE = re.compile(r"(?<![\w-])--resume-from\b")
# Leaders that assign env vars; kept when scanning for training-param overrides
# even though they would otherwise look read-only, so `export AUTO_SFT_SFT_LR=...`
# is still caught.
_ENV_ASSIGNMENT_LEADERS = frozenset(
    {"export", "set", "declare", "typeset", "local", "readonly"}
)

# Job submission / infrastructure settings are FIXED run defaults, NOT the agent's
# research variable. Cluster, project, resource pool, priority, node/GPU/CPU/memory
# placement, container image, and the training-code git source are all pinned by
# `scripts/agents/launch.sh` + the submit scripts. Changing any of them is a
# run-infrastructure change outside the agent's allowed scope (the agent iterates on
# the DATA/recipe only). The only way to override one from an agent command is to set
# the matching env var (inline `VAR=... cmd` or `export VAR=...`), so any such
# assignment is blocked on both the SFT and the eval submit path. Purely operational
# switches that do NOT change job identity/placement/resources (dry-run, repo dir, HF
# cache) are intentionally omitted so dry-run validation probes and path setup keep
# working.
JOB_INFRA_ENV_VARS = frozenset(
    {
        "AUTO_SFT_MAX_ROUNDS",
        "AUTO_SFT_TARGET_SCORES",
        "AUTO_SFT_BENCHMARKS",
        "AUTO_SFT_EVAL_BENCHMARK",
        "AUTO_SFT_BENCHMARK",
        "AUTO_SFT_COMPONENTS",
        "AUTO_SFT_LLM_BASE_URL",
        "AUTO_SFT_LLM_MODEL",
        "AUTO_SFT_LLM_API_KEY",
        "AUTO_SFT_LLM_ENABLE_THINKING",
        "AUTO_SFT_SYN_WORKER_CONCURRENCY",
        "AUTO_SFT_SYN_BATCH_RUNNER",
        "AUTO_SFT_SYN_BATCH_SKILL",
        "AUTO_SFT_SYN_BATCH_REFERENCE",
        "AUTO_SFT_SYN_WORKER_PROMPT",
        "AUTO_SFT_SYN_EXPERIENCE_PROMPT",
        "AUTO_SFT_SYN_TOPIC_STEM_POOL",
        "AUTO_SFT_SYN_ROW_CHECK_TOOL",
        "AUTO_SFT_SYN_DATA_SOURCE_DIR",
        "AUTO_SFT_SYN_ROWS_PER_TARGET",
        "AUTO_SFT_SYN_MAX_RESUMES",
        "AUTO_SFT_SYN_RESUME_BACKOFF_SECONDS",
        "AUTO_SFT_SYN_ATTEMPT_TIMEOUT_SECONDS",
        "AUTO_SFT_BASE_MODEL_PATH",
        "AUTO_SFT_BASE_CHECKPOINT",
        "AUTO_SFT_SFT_MODEL",
        "AUTO_SFT_SFT_TOKENIZER_MODEL",
        "AUTO_SFT_SFT_BASE_CKPT",
        "AUTO_SFT_TRAIN_PYTHON",
        "AUTO_SFT_EVAL_PYTHON",
        "AUTO_SFT_TOKENIZER_PYTHON",
        "AUTO_SFT_LOCAL_GPU_LOCK",
        "AUTO_SFT_SFT_GPUS_PER_NODE",
        "AUTO_SFT_EVAL_TP_SIZE",
        "AUTO_SFT_EVAL_GPU_MEMORY_UTILIZATION",
        "AUTO_SFT_EVAL_SEED",
        "AUTO_SFT_OPENCOMPASS_TEMPERATURE",
        "AUTO_SFT_OPENCOMPASS_MAX_SEQ_LEN",
        "AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN",
        "AUTO_SFT_OPENCOMPASS_SGLANG_CONTEXT_LENGTH",
        "AUTO_SFT_OPENCOMPASS_SGLANG_MEM_FRACTION",
        "AUTO_SFT_EVAL_GPUS_PER_NODE",
        "AUTO_SFT_EVAL_TENSOR_PARALLEL_SIZE",
        # Evaluation methodology is fixed for a run so scores remain comparable.
        "AUTO_SFT_EVAL_BACKEND",
        "AUTO_SFT_EVAL_REPEATS",
        "AUTO_SFT_EVAL_LIMIT",
        "AUTO_SFT_CODE_EVAL_PYTHON",
        "AUTO_SFT_CODE_INFERENCE_BACKEND",
        "AUTO_SFT_CODE_CONTEXT_LENGTH",
        "AUTO_SFT_CODE_GPU_MEMORY_UTILIZATION",
        "AUTO_SFT_EVALPLUS_DATA_DIR",
        "AUTO_SFT_EVAL_TOP_P",
        "AUTO_SFT_EVAL_TOP_K",
        "AUTO_SFT_EVAL_MIN_P",
        "AUTO_SFT_CONVERT_PYTHON",
        "AUTO_SFT_NLTK_DATA_DIR",
    }
)
_JOB_INFRA_ENV_ASSIGN_RE = re.compile(r"\b(AUTO_SFT_[A-Z0-9_]+)=")


def policy_enabled() -> bool:
    return os.environ.get("AUTO_SFT_POLICY_GUARD") == "1"


def default_run_dir() -> Path | None:
    value = os.environ.get("AUTO_SFT_RUN_DIR")
    return Path(value) if value else None


def default_datasets_dir() -> Path | None:
    value = os.environ.get("AUTO_SFT_DATASETS_DIR")
    return Path(value) if value else None


def default_baseline_path(run_dir: Path) -> Path:
    return Path(
        os.environ.get(
            "AUTO_SFT_POLICY_BASELINE", str(run_dir / "policy_baseline.json")
        )
    )


def default_feedback_path(run_dir: Path) -> Path:
    return Path(
        os.environ.get(
            "AUTO_SFT_POLICY_FEEDBACK", str(run_dir / "policy_feedback.json")
        )
    )


def default_violations_path(run_dir: Path) -> Path:
    return run_dir / "policy_violations.jsonl"


def _repo_rel(path: Path | str) -> str:
    path = Path(path).expanduser()
    try:
        if path.is_absolute():
            return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return path.as_posix()
    except ValueError:
        return str(path)


def _is_under(path: str, directory: str | None) -> bool:
    if not directory:
        return False
    directory = directory.rstrip("/")
    return path == directory or path.startswith(f"{directory}/")


def _is_source_like(path: str) -> bool:
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    return name in SOURCE_LIKE_NAMES or suffix in SOURCE_LIKE_SUFFIXES


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(
    args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _parse_git_status(raw: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parts = raw.split("\0")
    idx = 0
    while idx < len(parts):
        item = parts[idx]
        idx += 1
        if not item:
            continue
        status = item[:2]
        path = item[3:] if len(item) > 3 else ""
        original_path = None
        if status.startswith("R") or status.startswith("C"):
            if idx < len(parts):
                original_path = parts[idx] or None
                idx += 1
        if path:
            records.append(
                {"path": path, "status": status, "original_path": original_path}
            )
    return records


def git_status_records() -> list[dict[str, Any]]:
    result = _run_git(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    records = _parse_git_status(result.stdout)
    for record in records:
        repo_path = REPO_ROOT / record["path"]
        record["sha256"] = _sha256_file(repo_path)
    return records


def _status_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["path"]: record for record in records}


def _diff_excerpt(path: str, limit: int = 80) -> str:
    chunks: list[str] = []
    for args in (["diff", "--", path], ["diff", "--cached", "--", path]):
        result = _run_git(args, check=False)
        if result.stdout:
            chunks.append(result.stdout)
    if not chunks:
        repo_path = REPO_ROOT / path
        if repo_path.is_file():
            try:
                with repo_path.open(encoding="utf-8", errors="replace") as f:
                    chunks.append("".join(f.readlines()[:limit]))
            except OSError:
                return ""
    lines = "\n".join(chunks).splitlines()
    return "\n".join(lines[:limit])


def _violation(
    rule_id: str,
    message: str,
    *,
    severity: str = "error",
    path: str | None = None,
    git_status: str | None = None,
    diff_excerpt: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
    }
    if path is not None:
        payload["path"] = path
    if git_status is not None:
        payload["git_status"] = git_status
    if diff_excerpt:
        payload["diff_excerpt"] = diff_excerpt
    if detail:
        payload["detail"] = detail
    return payload


def _load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _allowed_recipe_paths(run_dir: Path, recipe_paths: list[Path]) -> set[str]:
    allowed = {_repo_rel(run_dir / "data-recipe.yaml")}
    for recipe_path in recipe_paths:
        allowed.add(_repo_rel(recipe_path))
    return allowed


def _is_allowed_change(
    path: str, run_dir: Path, datasets_dir: Path | None, recipe_paths: set[str]
) -> bool:
    run_rel = _repo_rel(run_dir)
    datasets_rel = _repo_rel(datasets_dir) if datasets_dir else None
    if _is_under(path, run_rel):
        return True
    if _is_under(path, datasets_rel):
        return True
    if path in recipe_paths:
        return True
    return False


def _is_new_or_changed_since_baseline(
    record: dict[str, Any], baseline_records: dict[str, dict[str, Any]]
) -> bool:
    baseline = baseline_records.get(record["path"])
    if baseline is None:
        return True
    return baseline.get("status") != record.get("status") or baseline.get(
        "sha256"
    ) != record.get("sha256")


def file_change_violations(
    run_dir: Path,
    datasets_dir: Path | None,
    recipe_paths: list[Path],
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    baseline_records = _status_map((baseline or {}).get("git_status", []))
    allowed_recipes = _allowed_recipe_paths(run_dir, recipe_paths)
    violations = []
    for record in git_status_records():
        path = record["path"]
        if _is_allowed_change(path, run_dir, datasets_dir, allowed_recipes):
            continue
        if not _is_new_or_changed_since_baseline(record, baseline_records):
            continue
        if record["status"] == "??" or _is_source_like(path):
            violations.append(
                _violation(
                    "unauthorized_file_change",
                    "Codex changed a file outside the policy allowlist.",
                    path=path,
                    git_status=record.get("status"),
                    diff_excerpt=_diff_excerpt(path),
                )
            )
    return violations


def _minimal_recipe_file_violations(
    recipe_path: Path, run_dir: Path
) -> list[dict[str, Any]]:
    """Parse one recipe file and check it against the minimal schema."""
    recipe_rel = _repo_rel(recipe_path)
    try:
        recipe = load_structured(recipe_path)
    except Exception as exc:
        return [
            _violation(
                "invalid_recipe",
                f"Could not parse SFT recipe YAML: {exc}",
                path=recipe_rel,
            )
        ]
    try:
        validate_minimal_recipe(recipe, run_dir=run_dir)
    except RecipeValidationError as exc:
        return [
            _violation(
                "invalid_recipe",
                f"SFT recipe failed the minimal-schema check: {exc}",
                path=recipe_rel,
            )
        ]
    return []


def recipe_violations(
    *,
    stage: str,
    recipe_path: Path | None,
    run_dir: Path,
) -> list[dict[str, Any]]:
    if stage != "sft":
        return []
    if recipe_path is None:
        return [
            _violation(
                "missing_recipe",
                "Guarded SFT submission must provide a recipe YAML path.",
            )
        ]
    recipe_rel = _repo_rel(recipe_path)
    if recipe_path.suffix.lower() not in {".yaml", ".yml"} or not _is_under(
        recipe_rel, _repo_rel(run_dir)
    ):
        return [
            _violation(
                "unapproved_recipe_path",
                "SFT recipe must be a YAML path under the run dir.",
                path=recipe_rel,
            )
        ]
    if not recipe_path.exists():
        return [
            _violation(
                "missing_recipe_file",
                "SFT recipe file does not exist.",
                path=recipe_rel,
            )
        ]
    return _minimal_recipe_file_violations(recipe_path, run_dir)


def run_recipe_violations(
    run_dir: Path, *, submitted_recipe: Path | None = None
) -> list[dict[str, Any]]:
    """Re-validate the run-root minimal recipe copy when present.

    ``prepare_recipe`` copies the accepted round recipe to
    ``run_dir/data-recipe.yaml``; the scan validates that copy too, so a
    post-hoc edit (an unknown key, an inline token_budget, a second dataset,
    or a dataset path escaping the run dir) cannot go unnoticed. When the
    submitted recipe IS the run-root copy, ``recipe_violations`` already
    covered it and the check is skipped.
    """
    recipe_path = run_dir / "data-recipe.yaml"
    if not recipe_path.is_file():
        return []
    if submitted_recipe is not None:
        try:
            if submitted_recipe.expanduser().resolve() == recipe_path.resolve():
                return []
        except OSError:
            pass
    return _minimal_recipe_file_violations(recipe_path, run_dir)


def _unwrap_shell_c(command: str) -> str:
    """Return the inner script of a ``bash -lc "<inner>"``-style invocation."""
    parts = command.strip().split(None, 2)
    if len(parts) == 3:
        exe = parts[0].rsplit("/", 1)[-1]
        if exe in {"bash", "sh", "zsh", "dash", "ksh"} and parts[1] in {
            "-c",
            "-lc",
            "-cl",
            "-lic",
            "-ic",
            "-li",
        }:
            inner = parts[2].strip()
            if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in ("'", '"'):
                inner = inner[1:-1]
            return inner
    return command


def _split_shell_segments(command: str) -> list[str]:
    """Split a (possibly compound) command into quote-aware segments.

    Heredoc bodies may be over-split, but that cannot create false positives:
    the tightened patterns only match real invocation tokens, never bare words.
    """
    inner = _unwrap_shell_c(command)
    segments: list[str] = []
    buf: list[str] = []
    quote = None
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(inner[i + 1])
                i += 2
            else:
                i += 1
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if inner[i : i + 2] in ("&&", "||"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "&", "\n"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        segments.append("".join(buf))
    return [seg.strip() for seg in segments if seg.strip()]


def _segment_is_read_only(segment: str) -> bool:
    for token in segment.split():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue  # leading VAR=value environment assignment
        name = token.rsplit("/", 1)[-1]
        if name in _COMMAND_WRAPPERS:
            continue
        return name in READ_ONLY_COMMANDS
    return False


def _forbidden_rules_for_command(command: str) -> list[str]:
    scannable = [
        seg for seg in _split_shell_segments(command) if not _segment_is_read_only(seg)
    ]
    if not scannable:
        return []
    haystack = "\n".join(scannable)
    return [
        rule_id
        for rule_id, pattern in FORBIDDEN_COMMAND_PATTERNS.items()
        if pattern.search(haystack)
    ]


def _training_param_scannable_segments(command: str) -> list[str]:
    """Segments that can change the training config.

    Keeps env-assignment leaders (``export``/``declare``/...) plus any
    non-read-only segment, but drops pure read-only searches (``grep``/``cat``/
    ``echo``/...). That way ``export AUTO_SFT_SFT_LR=...`` and
    ``AUTO_SFT_SFT_LR=... python src/autosft/backends/megatron/train.py`` are caught, while merely searching
    for or printing one of these var names (``grep AUTO_SFT_SFT_LR= ...``) is not.
    """
    kept: list[str] = []
    for seg in _split_shell_segments(command):
        tokens = seg.split()
        if tokens and tokens[0].rsplit("/", 1)[-1] in _ENV_ASSIGNMENT_LEADERS:
            kept.append(seg)
            continue
        if not _segment_is_read_only(seg):
            kept.append(seg)
    return kept


def _training_param_overrides_for_command(command: str) -> list[str]:
    """Messages for any attempt to change the FIXED training config in one command.

    Flags (a) setting a fixed ``AUTO_SFT_SFT_*`` setting and (b) passing a
    removed training-length or checkpoint-initialization override to
    src/autosft/backends/megatron/train.py.
    Returns one message per distinct override.
    """
    segments = _training_param_scannable_segments(command)
    if not segments:
        return []
    haystack = "\n".join(segments)
    messages: list[str] = []
    seen_vars: set[str] = set()
    for match in _SFT_ENV_ASSIGN_RE.finditer(haystack):
        var = match.group(1)
        if var in TRAINING_PARAM_ENV_VARS and var not in seen_vars:
            seen_vars.add(var)
            messages.append(
                f"Set fixed SFT setting env var {var}; guarded runs must use the default "
                f"training config (iterate on data/recipe, not training params)."
            )
    if _SFT_ENTRYPOINT_RE.search(haystack):
        if _SEQ_LENGTH_FLAG_RE.search(haystack):
            messages.append(
                "Passed --seq-length to src/autosft/backends/megatron/train.py; sequence length is a fixed run default — omit the flag."
            )
        if _TRAIN_ITERS_FLAG_RE.search(haystack):
            messages.append(
                "Passed --train-iters to src/autosft/backends/megatron/train.py; train-iters is never user-settable — a normal "
                "round computes one epoch from the selected data volume and trains for three "
                "epochs. Omit the flag."
            )
        if _LOAD_CKPT_FLAG_RE.search(haystack):
            messages.append(
                "Passed --load-ckpt to src/autosft/backends/megatron/train.py; guarded AutoSFT rounds always initialize "
                "from Qwen3 Base. Omit the flag."
            )
        if _RESUME_FROM_FLAG_RE.search(haystack):
            messages.append(
                "Passed --resume-from to src/autosft/backends/megatron/train.py; checkpoint resume is not part of the "
                "guarded AutoSFT workflow. Start the round from Qwen3 Base."
            )
    return messages


def _job_infra_overrides_for_command(command: str) -> list[str]:
    """Messages for any attempt to change FIXED job-submission / infrastructure
    settings in one command.

    Flags setting any pinned Local placement/resource/identity/image env var
    (cluster, project, resource pool, priority, node/GPU/CPU/memory, image, billing)
    or the SFT training-code git source, on either the SFT or the eval submit path.
    These are run infrastructure, outside the agent's allowed scope — the research
    variable is the data/recipe, and every round must submit with the fixed run
    defaults. Reuses the same env-assignment segment scan as the training-param
    check, so ``AUTO_SFT_LOCAL_RESOURCE_POOL=embodied python src/autosft/backends/megatron/train.py`` and
    ``export AUTO_SFT_LOCAL_PRIORITY=PREEMPTABLE`` are caught while merely
    printing/searching one of these names is not. Returns one message per distinct
    override.
    """
    segments = _training_param_scannable_segments(command)
    if not segments:
        return []
    haystack = "\n".join(segments)
    messages: list[str] = []
    seen_vars: set[str] = set()
    for match in _JOB_INFRA_ENV_ASSIGN_RE.finditer(haystack):
        var = match.group(1)
        if var in JOB_INFRA_ENV_VARS and var not in seen_vars:
            seen_vars.add(var)
            if var in ("AUTO_SFT_EVAL_BACKEND", "AUTO_SFT_EVAL_REPEATS"):
                messages.append(
                    f"Set fixed evaluation methodology env var {var}; the evaluation "
                    "backend and repeat count are selected once by the run owner and "
                    "must stay unchanged across rounds so scores remain comparable."
                )
            else:
                messages.append(
                    f"Set fixed job-submission/infrastructure env var {var}; cluster, "
                    f"project, resource pool, priority, node/GPU/CPU/memory placement, "
                    f"container image and training-code git source are FIXED run defaults "
                    f"outside the agent's allowed scope. Submit every round with the "
                    f"defaults (iterate on data/recipe, not job placement/infrastructure)."
                )
    return messages


# Public wrappers let the out-of-process policy hook reuse the exact command
# rules used by the after-the-fact agent-log scan.
def forbidden_rules_for_command(command: str) -> list[str]:
    """Rule ids for forbidden training flows invoked by ``command`` (empty if clean)."""
    return _forbidden_rules_for_command(command)


def training_param_overrides_for_command(command: str) -> list[str]:
    """Messages for any fixed-training-config override in ``command`` (empty if clean)."""
    return _training_param_overrides_for_command(command)


def job_infra_overrides_for_command(command: str) -> list[str]:
    """Messages for any fixed job-submission/infrastructure override in ``command`` (empty if clean)."""
    return _job_infra_overrides_for_command(command)


# Write targets a shell command would create or modify. The PreToolUse hook can
# only path-check `apply_patch`-style payloads, which name their targets; a
# shell command names them in arbitrary syntax, so this recovers the ones that
# appear in unambiguous positions. Deliberately partial: a heredoc that builds a
# path from variables is not recoverable by inspection, so the after-the-fact
# git scan stays the backstop for everything this misses. It must never guess --
# a false positive denies a legitimate command and strands an unattended run.
#
# Every pattern here therefore requires an ABSOLUTE path. A bare relative token
# is ambiguous (`>` inside a heredoc body is a Python comparison, not a
# redirect; `2:` is a dict key) and replaying the historical command log showed
# that accepting relative tokens misfires on ~3% of real commands.
_SHELL_REDIRECT_RE = re.compile(r"(?<![0-9<>=!])>>?\s*(/[^\s;|&<>()]*)")
_SHELL_WRITE_ARG_RE = re.compile(
    r"(?:^|[;|&]|\s)(cp|mv|install|tee|touch|mkdir|rsync)\b(?P<args>(?:\s+[^\s;|&<>()]+)+)"
)
# Quoted absolute paths inside an inline python/heredoc body: Path('/a/b'),
# open("/a/b", "w"). Only absolute paths, so a relative fragment (which we
# cannot resolve without knowing the interpreter's cwd) is never guessed at.
_EMBEDDED_ABS_PATH_RE = re.compile(r"""['"](/[^'"\s]+)['"]""")
# `cwd='/repo'` names where a subprocess runs, not a file it writes. Without
# this, any inline script that shells out from the repo root looks like a write
# to tracked source.
_CWD_KWARG_RE = re.compile(r"""\bcwd\s*=\s*['"](/[^'"\s]+)['"]""")
_WRITE_MODE_HINT_RE = re.compile(
    r"\b(?:write_text|write_bytes|writelines|to_json|to_csv|savefig|mkdir|makedirs|"
    r"open\s*\([^)]*['\"][wax]b?\+?['\"]|json\.dump|yaml\.(?:safe_)?dump|"
    r"shutil\.(?:copy2?|copyfile|copytree|move))"
)
# Sinks that are writes but never repo pollution; denying these only strands a
# run that was doing something ordinary.
_BENIGN_WRITE_TARGETS = (
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/tmp/",
    "/var/tmp/",
)


def shell_write_targets(command: str, *, include_temporary: bool = False) -> list[str]:
    """Absolute paths ``command`` plausibly writes, for real-time allowlist checks.

    Returns only targets recoverable with certainty from the command text.
    Relative targets are never reported: they cannot be resolved without knowing
    the writer's cwd, and guessing denies legitimate work. An empty list means
    "nothing detected", never "nothing written". Isolated tasks include
    temporary paths because their run or peer artifacts can also live there.
    """
    targets: list[str] = []
    for segment in _split_shell_segments(command):
        for match in _SHELL_REDIRECT_RE.finditer(segment):
            targets.append(match.group(1))
        for match in _SHELL_WRITE_ARG_RE.finditer(segment):
            args = [
                arg for arg in match.group("args").split() if not arg.startswith("-")
            ]
            # cp/mv/rsync/install write their LAST argument; tee/touch/mkdir
            # write all of theirs.
            if match.group(1) in {"cp", "mv", "install", "rsync"}:
                targets.extend(args[-1:])
            else:
                targets.extend(args)
    # An inline interpreter body (python - <<'PY' ...) never reaches the cases
    # above: its writes are library calls, not shell syntax, and a path assembled
    # from variables (`Path(root) / 'work_dir'`) cannot be paired with the
    # call that writes it. So when the body shows a write, every absolute path it
    # mentions is a candidate -- which over-reports, because the same body
    # usually reads paths too. Narrow that to the one thing this cannot be
    # allowed to miss: a write landing in tracked repository source. Run
    # directories under `scripts/setup/` are where these scripts legitimately
    # read and write, and anything outside the repo is not ours to police.
    if _WRITE_MODE_HINT_RE.search(command):
        repo = str(REPO_ROOT.resolve())
        runs_area = f"{repo}/runs"
        cwd_paths = set(_CWD_KWARG_RE.findall(command))
        for candidate in _EMBEDDED_ABS_PATH_RE.findall(command):
            if candidate in cwd_paths:
                continue
            normalized = os.path.normpath(candidate)
            if normalized == repo or (
                normalized.startswith(f"{repo}/")
                and not normalized.startswith(f"{runs_area}/")
                and normalized != runs_area
            ):
                targets.append(candidate)
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        target = target.strip("'\"")
        if not target.startswith("/"):
            continue  # relative: unresolvable without the writer's cwd
        benign = (
            _BENIGN_WRITE_TARGETS[:3] if include_temporary else _BENIGN_WRITE_TARGETS
        )
        if target.startswith(benign):
            continue
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    return ordered


def _iter_logged_commands(lines: list[str]):
    """Yield ``(lineno, command)`` for each executed command in the codex JSON stream.

    Only ``command_execution`` items are inspected, so reasoning text, agent
    messages, and runner status lines are never scanned. Each command is yielded
    once, deduped by item id.
    """
    seen: set[str] = set()
    for lineno, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw or raw[0] != "{":
            continue
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        key = item.get("id") or f"__line_{lineno}"
        if key in seen:
            continue
        seen.add(key)
        yield lineno, command


def agent_log_violations(run_dir: Path) -> list[dict[str, Any]]:
    log_path = run_dir / "logs" / "agent.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [
            _violation(
                "agent_log_unreadable",
                f"Could not read agent log for policy scan: {exc}",
                path=_repo_rel(log_path),
            )
        ]
    violations = []
    for lineno, command in _iter_logged_commands(lines):
        for rule_id in _forbidden_rules_for_command(command):
            violations.append(
                _violation(
                    rule_id,
                    "Agent log shows an executed command matching a forbidden training flow.",
                    path=_repo_rel(log_path),
                    detail={"line": lineno, "command": command[:500]},
                )
            )
        for message in _training_param_overrides_for_command(command):
            violations.append(
                _violation(
                    "training_param_override",
                    f"Agent log shows a training-parameter override: {message}",
                    path=_repo_rel(log_path),
                    detail={"line": lineno, "command": command[:500]},
                )
            )
        for message in _job_infra_overrides_for_command(command):
            violations.append(
                _violation(
                    "job_infra_override",
                    f"Agent log shows a job-submission/infrastructure override: {message}",
                    path=_repo_rel(log_path),
                    detail={"line": lineno, "command": command[:500]},
                )
            )
    return violations


def _error_violations(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Violations that fail the attempt (warnings stay in the report only)."""
    return [v for v in violations if v.get("severity", "error") != "warning"]


def build_feedback(
    stage: str, run_dir: Path, violations: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "policy_status": "violation" if _error_violations(violations) else "ok",
        "stage": stage,
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "violations": violations,
        "policy_feedback_path": str(default_feedback_path(run_dir)),
        "policy_violations_path": str(default_violations_path(run_dir)),
    }


def write_policy_feedback(
    stage: str, run_dir: Path, violations: list[dict[str, Any]]
) -> dict[str, Any]:
    feedback = build_feedback(stage, run_dir, violations)
    feedback_path = default_feedback_path(run_dir)
    violations_path = default_violations_path(run_dir)
    write_json(feedback_path, feedback)
    violations_path.parent.mkdir(parents=True, exist_ok=True)
    with violations_path.open("w", encoding="utf-8") as f:
        for violation in violations:
            f.write(json.dumps(violation, ensure_ascii=False) + "\n")
    return feedback


def record_policy_violation_and_exit(
    *,
    stage: str,
    run_dir: Path,
    rule_id: str,
    message: str,
    path: str | None = None,
) -> None:
    """Persist an early guard failure through the normal policy interface."""
    violation = _violation(rule_id, message, path=path)
    write_policy_feedback(stage, Path(run_dir), [violation])
    print(
        f"AutoSFT policy violation at {default_feedback_path(Path(run_dir))}",
        file=sys.stderr,
    )
    print(f"- {rule_id}: {message}", file=sys.stderr)
    raise SystemExit(3)


def policy_summary(run_dir: Path) -> dict[str, Any] | None:
    feedback_path = default_feedback_path(run_dir)
    if not feedback_path.exists():
        return None
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    return {
        "policy_status": feedback.get("policy_status"),
        "policy_feedback_path": str(feedback_path),
        "policy_violations": [
            {
                "rule_id": item.get("rule_id"),
                "severity": item.get("severity"),
                "path": item.get("path"),
                "message": item.get("message"),
            }
            for item in feedback.get("violations", [])[:10]
        ],
    }


def check_policy(
    *,
    stage: str,
    run_dir: Path,
    datasets_dir: Path | None,
    recipe_paths: list[Path],
    recipe_path: Path | None,
    exit_on_violation: bool = True,
) -> dict[str, Any]:
    baseline_path = default_baseline_path(run_dir)
    baseline = _load_baseline(baseline_path)
    violations = []
    if baseline is None:
        violations.append(
            _violation(
                "missing_policy_baseline",
                "Policy baseline is missing; start the run through scripts/run.sh so the controller records its baseline.",
                path=_repo_rel(baseline_path),
            )
        )
    violations.extend(
        file_change_violations(run_dir, datasets_dir, recipe_paths, baseline)
    )
    violations.extend(
        recipe_violations(
            stage=stage,
            recipe_path=recipe_path,
            run_dir=run_dir,
        )
    )
    violations.extend(run_recipe_violations(run_dir, submitted_recipe=recipe_path))
    violations.extend(agent_log_violations(run_dir))
    feedback = write_policy_feedback(stage, run_dir, violations)
    errors = _error_violations(violations)
    if errors:
        print(
            f"AutoSFT policy violation at {default_feedback_path(run_dir)}",
            file=sys.stderr,
        )
        for violation in errors:
            path = f" [{violation['path']}]" if violation.get("path") else ""
            print(
                f"- {violation['rule_id']}{path}: {violation['message']}",
                file=sys.stderr,
            )
        if exit_on_violation:
            raise SystemExit(3)
    return feedback


def check_policy_or_exit(
    *,
    stage: str,
    run_dir: Path | None = None,
    datasets_dir: Path | None = None,
    recipe_path: Path | None = None,
    recipe_paths: list[Path] | None = None,
) -> dict[str, Any] | None:
    if not policy_enabled():
        return None
    run_dir = run_dir or default_run_dir()
    if run_dir is None:
        raise SystemExit("AUTO_SFT_POLICY_GUARD=1 requires AUTO_SFT_RUN_DIR")
    datasets_dir = datasets_dir or default_datasets_dir()
    all_recipe_paths = list(recipe_paths or [])
    if recipe_path is not None:
        all_recipe_paths.append(recipe_path)
    return check_policy(
        stage=stage,
        run_dir=run_dir,
        datasets_dir=datasets_dir,
        recipe_paths=all_recipe_paths,
        recipe_path=recipe_path,
    )


def snapshot_policy(
    run_dir: Path, datasets_dir: Path | None, recipe_paths: list[Path]
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = {
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "datasets_dir": str(datasets_dir) if datasets_dir else None,
        "allowed_recipe_paths": sorted(_allowed_recipe_paths(run_dir, recipe_paths)),
        "git_status": git_status_records(),
    }
    write_json(default_baseline_path(run_dir), baseline)
    write_policy_feedback("snapshot", run_dir, [])
    return baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based policy guard for AutoSFT agent runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--run-dir", type=Path, default=default_run_dir())
    snapshot.add_argument("--datasets-dir", type=Path, default=default_datasets_dir())
    snapshot.add_argument("--recipe", action="append", type=Path, default=[])

    check = subparsers.add_parser("check")
    check.add_argument("--stage", default="manual")
    check.add_argument("--run-dir", type=Path, default=default_run_dir())
    check.add_argument("--datasets-dir", type=Path, default=default_datasets_dir())
    check.add_argument("--recipe", type=Path, default=None)
    check.add_argument("--allowed-recipe", action="append", type=Path, default=[])
    check.add_argument("--no-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None:
        raise SystemExit("--run-dir is required")
    if args.command == "snapshot":
        snapshot_policy(args.run_dir, args.datasets_dir, args.recipe)
        return
    if args.command == "check":
        check_policy(
            stage=args.stage,
            run_dir=args.run_dir,
            datasets_dir=args.datasets_dir,
            recipe_paths=args.allowed_recipe,
            recipe_path=args.recipe,
            exit_on_violation=not args.no_exit,
        )


if __name__ == "__main__":
    main()
