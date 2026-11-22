"""Round boundaries shared by preparation, recovery and the agent runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from autosft.benchmarks.catalog import supported_benchmarks
from autosft.backends.local.status import TERMINAL_STATUSES
from autosft.core.round_ids import logical_round_id
from autosft.orchestration.submission_registry import submission_records


def positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def max_rounds() -> int:
    return positive_env_int("AUTO_SFT_MAX_ROUNDS", 5)


def evaluation_settings() -> tuple[list[str], int]:
    names = ("AUTO_SFT_BENCHMARKS", "AUTO_SFT_EVAL_BENCHMARK", "AUTO_SFT_BENCHMARK")
    # Older launchers exported empty aliases alongside the populated one.
    # Preserve that form, but reject an explicitly empty configuration.
    raw = next(
        (os.environ[name] for name in names if os.environ.get(name, "").strip()),
        "" if any(name in os.environ for name in names) else "ifeval",
    )
    benchmarks = sorted(set(raw.replace(",", " ").lower().split()))
    if not benchmarks or set(benchmarks) - set(supported_benchmarks()):
        raise ValueError(
            f"configured benchmarks must name supported benchmarks; got {raw!r}"
        )
    return benchmarks, positive_env_int("AUTO_SFT_EVAL_REPEATS", 4)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _artifact(path: str, run_dir: Path) -> Path:
    candidate = Path(path)
    candidate = (
        candidate if candidate.is_absolute() else run_dir / candidate
    ).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        raise ValueError("evaluation artifacts must remain inside this run")
    return candidate


def if_coverage_problem(summary: dict[str, Any], benchmark: str) -> str | None:
    """Reject explicitly partial IF results; preserve legacy score readability.

    New local evaluations record actual and full prompt counts. Legacy scores
    without coverage metadata are not retroactively claimed to be verified.
    """
    expected = {"ifeval": 541, "ifbench": 300}.get(benchmark)
    if expected is None:
        return None
    if any(
        key in summary
        for key in ("num_prompts", "full_num_prompts", "is_full_benchmark")
    ):
        if (
            type(summary.get("num_prompts")) is not int
            or type(summary.get("full_num_prompts")) is not int
            or summary["num_prompts"] != expected
            or summary["full_num_prompts"] != expected
            or summary.get("is_full_benchmark") is not True
        ):
            return "evaluation must cover the full benchmark"
    if "limit" in summary:
        limit = summary["limit"]
        if type(limit) is not int or (limit != -1 and limit < expected):
            return "evaluation limit does not cover the full benchmark"
    return None


def evaluation_problems(
    record: dict[str, Any],
    run_dir: Path,
    *,
    benchmarks: list[str] | None = None,
    repeats: int | None = None,
) -> list[str]:
    """A nonempty score block is insufficient: every configured repeat must land."""
    configured, expected = evaluation_settings()
    if benchmarks is None:
        benchmarks = configured
    if repeats is not None:
        if type(repeats) is not int or repeats < 1:
            raise ValueError("evaluation repeats must be a positive integer")
        expected = repeats
    evaluations = record.get("eval")
    evaluations = evaluations if isinstance(evaluations, dict) else {}
    problems: list[str] = []
    for benchmark in benchmarks:
        summary = evaluations.get(benchmark)
        if not isinstance(summary, dict):
            problems.append(f"{benchmark}: no recorded evaluation")
            continue
        scores, repeats = summary.get("primary_scores"), summary.get("repeats")
        valid_scores = (
            isinstance(scores, list)
            and len(scores) == expected
            and all(_finite(score) for score in scores)
        )
        if (
            type(summary.get("n_repeats")) is not int
            or summary["n_repeats"] != expected
        ):
            problems.append(f"{benchmark}: n_repeats must equal {expected}")
        if not valid_scores:
            problems.append(
                f"{benchmark}: primary_scores must contain {expected} finite repeats"
            )
        elif not _finite(summary.get("primary_score")) or not math.isclose(
            summary["primary_score"],
            sum(scores) / expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            problems.append(f"{benchmark}: primary_score disagrees with repeat mean")
        indices = (
            [item.get("repeat") for item in repeats if isinstance(item, dict)]
            if isinstance(repeats, list)
            else []
        )
        if (
            not isinstance(repeats, list)
            or len(repeats) != expected
            or len(indices) != expected
            or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(1, expected + 1))
        ):
            problems.append(
                f"{benchmark}: repeat indices must cover 1..{expected} exactly"
            )
        from autosft.benchmarks.evalplus import (
            CODE_BENCHMARKS,
            EVALPLUS_EXPECTED_DETAILS,
        )

        if benchmark in CODE_BENCHMARKS and (
            summary.get("primary_metric") != "plus_pass_at_1"
            or summary.get("is_full_benchmark") is not True
            or type(summary.get("num_problems")) is not int
            or summary["num_problems"] != EVALPLUS_EXPECTED_DETAILS[benchmark]
        ):
            problems.append(f"{benchmark}: evaluation must cover the full benchmark")
        coverage = if_coverage_problem(summary, benchmark)
        if coverage:
            problems.append(f"{benchmark}: {coverage}")
        if isinstance(repeats, list):
            for item in repeats:
                if isinstance(item, dict) and (
                    coverage := if_coverage_problem(item, benchmark)
                ):
                    problems.append(
                        f"{benchmark}: repeat {item.get('repeat')}: {coverage}"
                    )
        artifacts = summary.get("result_artifacts")
        aggregate = artifacts.get("aggregate") if isinstance(artifacts, dict) else None
        try:
            if not isinstance(aggregate, str) or not aggregate:
                raise ValueError("aggregate.json is missing")
            path = _artifact(aggregate, run_dir)
            if not path.is_file():
                raise ValueError("aggregate.json is missing")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or type(payload.get("n_repeats")) is not int
                or payload["n_repeats"] != expected
                or payload.get("primary_scores") != scores
                or not valid_scores
                or not _finite(payload.get("primary_score_mean"))
                or not math.isclose(
                    payload["primary_score_mean"],
                    sum(scores) / expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("aggregate.json disagrees with the recorded repeats")
        except (OSError, ValueError) as exc:
            problems.append(f"{benchmark}: {exc}")
    return problems


def _round_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    from autosft.orchestration.ledger import read_iterations, record_round_id

    records: dict[str, dict[str, Any]] = {}
    for record in read_iterations(run_dir):
        round_id = record_round_id(record)
        if round_id is None or round_id in records:
            raise ValueError("ledger has missing or duplicate round identities")
        records[round_id] = record
    return records


def verify_previous_round_complete(recipe: dict[str, Any], run_dir: Path) -> None:
    """Require all previous rounds evaluated, or explicitly disposed of as failed."""
    from autosft.orchestration.blocked_rounds import acknowledged_rounds

    round_id = int(logical_round_id(recipe.get("round_id")))
    limit = max_rounds()
    evaluation_settings()  # Invalid configuration fails even at round one.
    if round_id > limit:
        raise ValueError(
            f"round {round_id} exceeds AUTO_SFT_MAX_ROUNDS={limit}; "
            "this run must not start another training round"
        )
    records = _round_records(run_dir)
    disposed = acknowledged_rounds(run_dir)
    for prior in range(1, round_id):
        if str(prior) in disposed:
            continue
        record = records.get(str(prior))
        if record is None:
            raise ValueError(
                f"round {round_id} is blocked: prior round round_{prior} "
                "has no ledger iteration or acknowledged failure"
            )
        problems = evaluation_problems(record, run_dir)
        if problems:
            raise ValueError(
                f"round {round_id} is blocked until round_{prior} evaluation "
                "is complete: " + "; ".join(problems)
            )


def assert_data_preparation_safe(run_dir: Path) -> None:
    """Protect shared training arrays while any local SFT may still read them.

    The caller holds the same submission lock as SFT submission. Read durable
    receipts directly; an unknown/stale status requires lifecycle recovery,
    never an optimistic overwrite. Include orphan receipts from a submit crash.
    """
    from autosft.backends.local.tasks import task_path

    receipts = {
        task_path(run_dir, record["task"])
        for record in submission_records(run_dir / "sft_submissions.jsonl")
    }
    for path in (run_dir / "local_tasks").glob("local-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot verify local task receipt before preparation: {path}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"invalid local task receipt: {path}")
        if "dataset_path" in record and "benchmark" not in record:
            receipts.add(path)
    for path in sorted(receipts):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot verify SFT receipt before preparation: {path}"
            ) from exc
        if (
            not isinstance(record, dict)
            or record.get("status") not in TERMINAL_STATUSES
        ):
            raise ValueError(
                "cannot prepare data while local SFT is queued, running, "
                f"or its status is unknown; recover the task first: {path}"
            )


def round_budget_exhausted(run_dir: Path) -> bool:
    """Stop retries only after all budgeted rounds are evaluated or acknowledged."""
    from autosft.orchestration.blocked_rounds import acknowledged_rounds

    limit = max_rounds()
    records = _round_records(run_dir)
    disposed = acknowledged_rounds(run_dir)
    for round_id in map(str, range(1, limit + 1)):
        if round_id in disposed:
            continue
        if round_id not in records or evaluation_problems(records[round_id], run_dir):
            return False
    try:
        assert_data_preparation_safe(run_dir)
    except ValueError:
        return False
    return True
