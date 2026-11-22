"""Runtime configuration and canonical path layout for Synthesis artifacts.

Configuration is evaluated explicitly for every invocation, and the layout
dataclass turns validated configuration into stable paths without embedding
lifecycle policy.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from autosft.backends.megatron.tokenization_runtime import missing_tokenizer_files
from autosft.synthesis.errors import SynConfigError
from autosft.synthesis._internal.filesystem import require_within, validate_safe_id


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynLayout:
    """Construct all stable paths without embedding lifecycle policy."""

    run_dir: Path
    data_source_dir: Path

    def __post_init__(self) -> None:
        run_dir = Path(self.run_dir).expanduser().resolve(strict=False)
        data_source_dir = require_within(
            Path(self.data_source_dir),
            run_dir,
            "Synthesis Data Source",
            allow_root=False,
        )
        object.__setattr__(self, "run_dir", run_dir)
        object.__setattr__(self, "data_source_dir", data_source_dir)

    @property
    def synthesis_dir(self) -> Path:
        return self.run_dir / "synthesis"

    def round_dir(self, round_id: str) -> Path:
        return self.synthesis_dir / validate_safe_id(round_id, "round_id")

    def round_manifest(self, round_id: str) -> Path:
        return self.round_dir(round_id) / "round.json"

    def target_dir(self, round_id: str, target_id: str) -> Path:
        return (
            self.round_dir(round_id)
            / "targets"
            / validate_safe_id(target_id, "target_id")
        )

    def target_input_dir(self, round_id: str, target_id: str) -> Path:
        return self.target_dir(round_id, target_id) / "input"

    def attempts_dir(self, round_id: str, target_id: str) -> Path:
        return self.target_dir(round_id, target_id) / "attempts"

    def attempt_dir(self, round_id: str, target_id: str, attempt_id: str) -> Path:
        return self.attempts_dir(round_id, target_id) / validate_safe_id(
            attempt_id, "attempt_id"
        )

    def attempt_output_dir(
        self, round_id: str, target_id: str, attempt_id: str
    ) -> Path:
        return self.attempt_dir(round_id, target_id, attempt_id) / "output"


# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------


def _resolved_path(value: str, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve(strict=False)


def _env_path(
    environ: Mapping[str, str],
    name: str,
    *,
    default: Path | None = None,
    base: Path | None = None,
    required: bool = False,
) -> Path:
    raw = environ.get(name, "").strip()
    if raw:
        return _resolved_path(raw, base=base)
    if default is not None:
        return Path(default).expanduser().resolve(strict=False)
    if required:
        raise SynConfigError(f"{name} is required")
    raise SynConfigError(f"{name} has no configured path")


def _env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise SynConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise SynConfigError(f"{name} must be >= {minimum}")
    return value


def _env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw = environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise SynConfigError(f"{name} must be numeric") from exc
    if value < minimum or not value < float("inf"):
        raise SynConfigError(f"{name} must be finite and >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class SynConfig:
    repo_root: Path
    run_dir: Path
    data_source_dir: Path
    tokenizer_code_dir: Path
    tokenizer_python: str
    worker_prompt_path: Path
    experience_prompt_path: Path
    batch_skill_path: Path
    batch_runner_path: Path
    batch_reference_path: Path
    topic_stem_pool_path: Path
    row_check_tool_path: Path
    rows_per_target: int
    max_resumes: int
    resume_backoff_seconds: float
    attempt_timeout_seconds: float
    max_concurrency: int = 10

    def validate_tokenizer_runtime(self) -> None:
        missing = missing_tokenizer_files(self.tokenizer_code_dir)
        if missing:
            raise SynConfigError(
                "local SSTable runtime is incomplete; missing: "
                + ", ".join(str(path) for path in missing)
            )

    @classmethod
    def from_env(cls) -> "SynConfig":
        # Read os.environ at call time, never at import time: resumed runs and
        # launchers must observe the environment of the invoking process.
        env = os.environ
        root = Path(__file__).resolve().parents[4]
        run_dir = _env_path(env, "AUTO_SFT_RUN_DIR", required=True)
        tokenizer_code_dir = _env_path(
            env,
            "AUTO_SFT_TOKENIZER_CODE_DIR",
            default=root / "src" / "autosft" / "data",
            base=root,
        )
        tokenizer_python = (
            env.get("AUTO_SFT_TOKENIZER_PYTHON", "").strip() or sys.executable
        )
        data_source_dir = _env_path(
            env,
            "AUTO_SFT_SYN_DATA_SOURCE_DIR",
            default=run_dir / "syn_data_source",
            base=run_dir,
        )
        try:
            data_source_dir = require_within(
                data_source_dir,
                run_dir,
                "AUTO_SFT_SYN_DATA_SOURCE_DIR",
                allow_root=False,
            )
        except Exception as exc:
            raise SynConfigError(str(exc)) from exc
        pool_dir = _env_path(
            env,
            "AUTO_SFT_DATA_POOL_DIR",
            default=run_dir / "data_pool",
            base=run_dir,
        )
        for label, directory in (
            ("AUTO_SFT_SYN_DATA_SOURCE_DIR", data_source_dir),
        ):
            if (
                directory == pool_dir
                or pool_dir in directory.parents
                or directory in pool_dir.parents
            ):
                raise SynConfigError(
                    f"{label} must be separate from the staged data pool"
                )
        internal_prompts = (
            root / "src" / "autosft" / "synthesis" / "_internal" / "prompts"
        )
        worker_prompt_path = _env_path(
            env,
            "AUTO_SFT_SYN_WORKER_PROMPT",
            default=internal_prompts / "synthesis_worker.md",
            base=root,
        )
        experience_prompt_path = _env_path(
            env,
            "AUTO_SFT_SYN_EXPERIENCE_PROMPT",
            default=internal_prompts / "experience_maintainer.md",
            base=root,
        )
        batch_skill_path = _env_path(
            env,
            "AUTO_SFT_SYN_BATCH_SKILL",
            default=root / "skills" / "synthesis" / "batch-generation" / "SKILL.md",
            base=root,
        )
        batch_runner_path = _env_path(
            env,
            "AUTO_SFT_SYN_BATCH_RUNNER",
            default=(
                root
                / "src"
                / "autosft"
                / "backends"
                / "providers"
                / "batch.py"
            ),
            base=root,
        )
        batch_reference_path = _env_path(
            env,
            "AUTO_SFT_SYN_BATCH_REFERENCE",
            default=(
                root
                / "skills"
                / "shared"
                / "batch-inference"
                / "references"
                / "recorded-calls.md"
            ),
            base=root,
        )
        topic_stem_pool_path = _env_path(
            env,
            "AUTO_SFT_SYN_TOPIC_STEM_POOL",
            default=(
                root
                / "skills"
                / "shared"
                / "batch-inference"
                / "references"
                / "topic_stem_pool.json"
            ),
            base=root,
        )
        row_check_tool_path = _env_path(
            env,
            "AUTO_SFT_SYN_ROW_CHECK_TOOL",
            default=(
                root
                / "src"
                / "autosft"
                / "synthesis"
                / "_internal"
                / "tools"
                / "check_training_rows.py"
            ),
            base=root,
        )
        for label, path in (
            ("AUTO_SFT_SYN_WORKER_PROMPT", worker_prompt_path),
            ("AUTO_SFT_SYN_EXPERIENCE_PROMPT", experience_prompt_path),
            ("AUTO_SFT_SYN_BATCH_SKILL", batch_skill_path),
            ("AUTO_SFT_SYN_BATCH_RUNNER", batch_runner_path),
            ("AUTO_SFT_SYN_BATCH_REFERENCE", batch_reference_path),
            ("AUTO_SFT_SYN_ROW_CHECK_TOOL", row_check_tool_path),
        ):
            try:
                require_within(path, root, label)
            except Exception as exc:
                raise SynConfigError(str(exc)) from exc
            if not path.is_file():
                raise SynConfigError(
                    f"{label} is missing or is not a regular file: {path}"
                )

        # An omitted topic pool is optional; an explicit path must exist so
        # a typo cannot silently alter the worker's source materials.
        if (
            env.get("AUTO_SFT_SYN_TOPIC_STEM_POOL", "").strip()
            and not topic_stem_pool_path.is_file()
        ):
            raise SynConfigError(
                "AUTO_SFT_SYN_TOPIC_STEM_POOL must name an existing file"
            )
        if topic_stem_pool_path.exists():
            try:
                require_within(
                    topic_stem_pool_path, root, "AUTO_SFT_SYN_TOPIC_STEM_POOL"
                )
            except Exception as exc:
                raise SynConfigError(str(exc)) from exc

        return cls(
            repo_root=root,
            run_dir=run_dir,
            data_source_dir=data_source_dir,
            tokenizer_code_dir=tokenizer_code_dir,
            tokenizer_python=tokenizer_python,
            worker_prompt_path=worker_prompt_path,
            experience_prompt_path=experience_prompt_path,
            batch_skill_path=batch_skill_path,
            batch_runner_path=batch_runner_path,
            batch_reference_path=batch_reference_path,
            topic_stem_pool_path=topic_stem_pool_path,
            row_check_tool_path=row_check_tool_path,
            rows_per_target=_env_int(
                env, "AUTO_SFT_SYN_ROWS_PER_TARGET", 1000, minimum=1
            ),
            max_resumes=_env_int(env, "AUTO_SFT_SYN_MAX_RESUMES", 8, minimum=0),
            resume_backoff_seconds=_env_float(
                env,
                "AUTO_SFT_SYN_RESUME_BACKOFF_SECONDS",
                120.0,
                minimum=0.0,
            ),
            attempt_timeout_seconds=_env_float(
                env,
                "AUTO_SFT_SYN_ATTEMPT_TIMEOUT_SECONDS",
                0.0,
                minimum=0.0,
            ),
            max_concurrency=_env_int(
                env, "AUTO_SFT_SYN_WORKER_CONCURRENCY", 10, minimum=1
            ),
        )
