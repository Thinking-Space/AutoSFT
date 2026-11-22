"""Aggregate repeat evaluations of one benchmark into a single artifact.

With ``AUTO_SFT_EVAL_REPEATS`` > 1 each benchmark of an eval run is evaluated
by one Local task writing independent repeats to ``eval/<benchmark>/rep_<k>/``.
This module joins those repeat artifacts into one deterministic
``eval/<benchmark>/aggregate.json`` per benchmark:

- ``primary_scores``: the raw per-repeat primary scores, in repeat order
- ``primary_score_mean``: their arithmetic mean (the round's benchmark score)
- ``primary_score_sd``: their sample standard deviation (``null`` with one
  repeat) — the eval-side noise of a single repeat on this checkpoint
- ``repeats``: per-repeat compact summaries (score accuracies, num_failures)
- ``failure_persistence``: for every sample that failed in any repeat, how many
  repeats it failed in (``failed_repeats`` out of ``n_repeats``) and the union
  of failed instruction ids, or base/plus failure counts for code benchmarks

Aggregation is refused while any repeat's artifacts are missing: a round is
evaluated only when every repeat of every benchmark has completed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from autosft.benchmarks.evalplus import (
    CODE_BENCHMARKS,
    EVALPLUS_EXPECTED_DETAILS,
    EVALPLUS_PROBLEM_IDS,
)

from autosft.core.common import utc_now, write_json
from autosft.orchestration.submission_registry import submission_records


AGGREGATE_FILENAME = "aggregate.json"
_ACCURACY_KEYS = (
    "strict_prompt_accuracy",
    "loose_prompt_accuracy",
    "strict_instruction_accuracy",
    "loose_instruction_accuracy",
)


class EvalAggregateError(RuntimeError):
    """Raised when repeat artifacts are missing or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalAggregateError(f"unreadable eval artifact: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalAggregateError(f"eval artifact must be an object: {path}")
    return payload


def code_score_summary(
    scores: dict[str, Any],
    *,
    benchmark: str | None = None,
    repeat: int | None = None,
) -> dict[str, Any] | None:
    """Validate a code repeat before aggregation or ledger recording."""
    metrics = scores.get("metrics") if isinstance(scores.get("metrics"), dict) else {}
    benchmark = benchmark or scores.get("benchmark") or metrics.get("benchmark")
    if benchmark not in CODE_BENCHMARKS:
        if (
            scores.get("primary_metric", metrics.get("primary_metric"))
            == "plus_pass_at_1"
        ):
            raise EvalAggregateError("code scores lack a supported benchmark identity")
        return None

    def field(key: str) -> Any:
        value = scores.get(key, metrics.get(key))
        if key in scores and key in metrics and scores[key] != metrics[key]:
            raise EvalAggregateError(
                f"code scores disagree with nested metrics for {key}"
            )
        return value

    for payload in (scores, metrics):
        if "benchmark" in payload and payload["benchmark"] != benchmark:
            raise EvalAggregateError(
                "code score artifact belongs to a different benchmark"
            )
        if "repeat" in payload and (
            isinstance(payload["repeat"], bool)
            or not isinstance(payload["repeat"], int)
            or payload["repeat"] < 1
            or (repeat is not None and payload["repeat"] != repeat)
        ):
            raise EvalAggregateError(
                "code score artifact belongs to a different repeat"
            )
    if field("primary_metric") != "plus_pass_at_1":
        raise EvalAggregateError("code primary_metric must be plus_pass_at_1")
    summary: dict[str, Any] = {
        "benchmark": benchmark,
        "primary_metric": "plus_pass_at_1",
    }
    for key in ("num_problems", "num_base_pass", "num_plus_pass", "num_failures"):
        value = field(key)
        values = [payload[key] for payload in (scores, metrics) if key in payload]
        if not values or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ):
            raise EvalAggregateError(f"code {key} must be a non-negative integer")
        summary[key] = value
    count = summary["num_problems"]
    if not 1 <= count <= EVALPLUS_EXPECTED_DETAILS[benchmark]:
        raise EvalAggregateError("code num_problems is outside the benchmark split")
    if not 0 <= summary["num_plus_pass"] <= summary["num_base_pass"] <= count:
        raise EvalAggregateError("code pass counts are inconsistent")
    if summary["num_failures"] != count - summary["num_plus_pass"]:
        raise EvalAggregateError("code failure count disagrees with plus passes")
    for key, expected in (
        ("base_pass_at_1", summary["num_base_pass"] / count),
        ("plus_pass_at_1", summary["num_plus_pass"] / count),
        ("primary_score", summary["num_plus_pass"] / count),
    ):
        value = field(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
        ):
            raise EvalAggregateError(
                f"code {key} disagrees with complete problem counts"
            )
        summary[key] = float(value)
    for key, expected in (
        ("score", summary["primary_score"]),
        ("overall_accuracy", summary["primary_score"]),
        ("base_pass@1", summary["base_pass_at_1"]),
        ("plus_pass@1", summary["plus_pass_at_1"]),
    ):
        if key in scores or key in metrics:
            value = field(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
            ):
                raise EvalAggregateError(
                    f"code {key} disagrees with the primary metric"
                )
    full = count == EVALPLUS_EXPECTED_DETAILS[benchmark]
    if "is_full_benchmark" in scores or "is_full_benchmark" in metrics:
        if field("is_full_benchmark") is not full:
            raise EvalAggregateError(
                "code is_full_benchmark disagrees with problem count"
            )
    summary["is_full_benchmark"] = full
    return summary


def _repeat_summary(
    repeat: int, rep_dir: Path, benchmark: str | None = None
) -> dict[str, Any]:
    scores = _read_json(rep_dir / "scores.json")
    primary = scores.get("primary_score")
    if (
        not isinstance(primary, (int, float))
        or isinstance(primary, bool)
        or not math.isfinite(primary)
    ):
        raise EvalAggregateError(
            f"repeat {repeat} scores.json has no finite primary_score: "
            f"{rep_dir / 'scores.json'}"
        )
    code = code_score_summary(scores, benchmark=benchmark, repeat=repeat)
    if code is not None:
        return {**code, "repeat": repeat, "output_dir": str(rep_dir)}
    metrics = scores.get("metrics") if isinstance(scores.get("metrics"), dict) else {}
    summary: dict[str, Any] = {
        "repeat": repeat,
        "output_dir": str(rep_dir),
        "primary_score": float(primary),
        "num_failures": scores.get("num_failures"),
    }
    for key in _ACCURACY_KEYS:
        value = scores.get(key)
        if not isinstance(value, (int, float)):
            value = metrics.get(key)
        summary[key] = float(value) if isinstance(value, (int, float)) else None
    if "limit" in scores:
        summary["limit"] = scores["limit"]
    if "num_prompts" in metrics:
        count, full = metrics["num_prompts"], metrics.get("full_num_prompts")
        if (
            type(count) is not int
            or type(full) is not int
            or not 0 < count <= full
            or metrics.get("is_full_benchmark") is not (count == full)
        ):
            raise EvalAggregateError(
                "IF coverage metadata disagrees with prompt counts"
            )
        summary.update(
            num_prompts=count, full_num_prompts=full, is_full_benchmark=count == full
        )
    return summary


def _failed_samples(
    rep_dir: Path,
    *,
    benchmark: str | None = None,
    repeat: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Read one repeat's failures without treating code failures as instructions."""
    failures_path = rep_dir / "failures.jsonl"
    if not failures_path.is_file():
        raise EvalAggregateError(f"repeat is missing failures.jsonl: {failures_path}")
    failed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        failures_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalAggregateError(
                f"invalid JSON in {failures_path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise EvalAggregateError(
                f"failure row must be an object: {failures_path}:{line_number}"
            )
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise EvalAggregateError(
                f"failure row without sample_id: {failures_path}:{line_number}"
            )
        if benchmark in CODE_BENCHMARKS:
            if sample_id in failed:
                raise EvalAggregateError(
                    f"duplicate code failure {sample_id}: {failures_path}"
                )
            if (
                sample_id not in EVALPLUS_PROBLEM_IDS[benchmark]
                or row.get("problem_id") != sample_id
                or row.get("benchmark") != benchmark
                or isinstance(row.get("repeat"), bool)
                or not isinstance(row.get("repeat"), int)
                or row.get("repeat") != repeat
            ):
                raise EvalAggregateError(
                    f"code failure has mismatched problem or repeat identity: {failures_path}"
                )
            base, plus = row.get("base_result"), row.get("plus_result")
            category = (
                "plus_only_failure" if base == "pass" else "base_and_plus_failure"
            )
            if (
                base not in ("pass", "fail")
                or plus != "fail"
                or row.get("failure_class") != category
            ):
                raise EvalAggregateError(
                    f"invalid code failure verdicts: {failures_path}:{line_number}"
                )
            failed[sample_id] = {
                "base_result": base,
                "plus_result": plus,
                "failure_class": category,
            }
        else:
            failed[sample_id] = {
                "strict": [
                    str(i) for i in (row.get("failed_instructions_strict") or [])
                ],
                "loose": [str(i) for i in (row.get("failed_instructions_loose") or [])],
            }
    return failed


def repeat_dirs(bench_dir: Path, n_repeats: int) -> list[Path]:
    if n_repeats <= 1:
        return [bench_dir]
    return [bench_dir / f"rep_{k}" for k in range(1, n_repeats + 1)]


def aggregate_benchmark(bench_dir: Path, n_repeats: int) -> dict[str, Any]:
    """Build the aggregate payload for one benchmark directory."""
    bench_dir = Path(bench_dir)
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise EvalAggregateError("n_repeats must be a positive integer")
    dirs = repeat_dirs(bench_dir, n_repeats)
    missing = [str(d) for d in dirs if not (d / "scores.json").is_file()]
    if missing:
        raise EvalAggregateError(
            "aggregation requires every repeat's scores.json; missing: "
            + ", ".join(missing)
        )

    repeats = [
        _repeat_summary(
            index,
            rep_dir,
            bench_dir.name if bench_dir.name in CODE_BENCHMARKS else None,
        )
        for index, rep_dir in enumerate(dirs, start=1)
    ]
    code = repeats[0].get("primary_metric") == "plus_pass_at_1"
    if any(
        (entry.get("primary_metric") == "plus_pass_at_1") != code for entry in repeats
    ):
        raise EvalAggregateError("repeat artifacts mix code and instruction metrics")
    benchmark = repeats[0].get("benchmark") if code else None
    if code and any(
        entry.get("benchmark") != benchmark
        or entry["num_problems"] != repeats[0]["num_problems"]
        for entry in repeats
    ):
        raise EvalAggregateError(
            "code repeats use different benchmarks or problem counts"
        )
    if not code and any("num_prompts" in entry for entry in repeats):
        if any(
            (entry.get("num_prompts"), entry.get("full_num_prompts"))
            != (repeats[0].get("num_prompts"), repeats[0].get("full_num_prompts"))
            for entry in repeats
        ):
            raise EvalAggregateError("IF repeats use different prompt counts")
    per_repeat_failures = [
        _failed_samples(rep_dir, benchmark=benchmark, repeat=index)
        for index, rep_dir in enumerate(dirs, start=1)
    ]
    if code:
        for entry, failures in zip(repeats, per_repeat_failures):
            base_failures = sum(
                row["base_result"] == "fail" for row in failures.values()
            )
            if (
                len(failures) != entry["num_failures"]
                or base_failures != entry["num_problems"] - entry["num_base_pass"]
            ):
                raise EvalAggregateError(
                    "code failure evidence disagrees with repeat score counts"
                )

    persistence: dict[str, dict[str, Any]] = {}
    for sample_id in sorted({s for failed in per_repeat_failures for s in failed}):
        if code:
            plus_hits = [
                k + 1
                for k, failed in enumerate(per_repeat_failures)
                if sample_id in failed
            ]
            base_hits = [
                k + 1
                for k, failed in enumerate(per_repeat_failures)
                if sample_id in failed and failed[sample_id]["base_result"] == "fail"
            ]
            persistence[sample_id] = {
                "failed_repeats": len(plus_hits),
                "failed_repeats_plus": len(plus_hits),
                "failed_repeats_base": len(base_hits),
                "n_repeats": n_repeats,
                "repeats_failed_in": plus_hits,
                "repeats_failed_base_in": base_hits,
                "failure_classes": sorted(
                    {
                        failed[sample_id]["failure_class"]
                        for failed in per_repeat_failures
                        if sample_id in failed
                    }
                ),
            }
            continue
        strict_hits = [
            k + 1
            for k, failed in enumerate(per_repeat_failures)
            if sample_id in failed and failed[sample_id]["strict"]
        ]
        any_hits = [
            k + 1 for k, failed in enumerate(per_repeat_failures) if sample_id in failed
        ]
        strict_union = sorted(
            {
                instruction
                for failed in per_repeat_failures
                for instruction in failed.get(sample_id, {}).get("strict", [])
            }
        )
        loose_union = sorted(
            {
                instruction
                for failed in per_repeat_failures
                for instruction in failed.get(sample_id, {}).get("loose", [])
            }
        )
        persistence[sample_id] = {
            "failed_repeats_strict": len(strict_hits),
            "failed_repeats": len(any_hits),
            "n_repeats": n_repeats,
            "repeats_failed_in": any_hits,
            "failed_instructions_strict": strict_union,
            "failed_instructions_loose": loose_union,
        }

    primary_scores = [entry["primary_score"] for entry in repeats]
    return {
        "generated_at": utc_now(),
        "benchmark_dir": str(bench_dir),
        "n_repeats": n_repeats,
        "primary_scores": primary_scores,
        "primary_score_mean": sum(primary_scores) / len(primary_scores),
        "primary_score_sd": sample_sd(primary_scores),
        "repeats": repeats,
        "failure_persistence": persistence,
    }


def sample_sd(scores: list[float]) -> float | None:
    """Sample standard deviation of the per-repeat scores; None below two."""
    if len(scores) < 2:
        return None
    return statistics.stdev(scores)


def aggregate_eval_run(
    eval_run_dir: Path,
    n_repeats: int,
    *,
    manifest: Path | None = None,
) -> list[Path]:
    """Write aggregate.json for every benchmark dir under <eval_run>/eval/."""
    eval_run_dir = Path(eval_run_dir)
    eval_root = eval_run_dir / "eval"
    if not eval_root.is_dir():
        raise EvalAggregateError(f"eval run has no eval/ directory: {eval_run_dir}")
    written: list[Path] = []
    if manifest is None:
        benchmark_dirs = sorted(path for path in eval_root.iterdir() if path.is_dir())
    else:
        selected_dirs: set[Path] = set()
        for record in submission_records(manifest):
            output = Path(record["output_dir"]).resolve()
            if "repeats" not in record and record.get("repeat") and n_repeats > 1:
                output = output.parent
            selected_dirs.add(output)
        benchmark_dirs = sorted(selected_dirs)
        if any(path.parent != eval_root.resolve() for path in benchmark_dirs):
            raise EvalAggregateError(
                "manifest benchmark output must belong to its eval run"
            )
    for bench_dir in benchmark_dirs:
        payload = aggregate_benchmark(bench_dir, n_repeats)
        target = bench_dir / AGGREGATE_FILENAME
        write_json(target, payload)
        written.append(target)
    if not written:
        raise EvalAggregateError(f"no benchmark directories under {eval_root}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate repeat eval artifacts into aggregate.json"
    )
    parser.add_argument("--eval-run-dir", required=True, type=Path)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be a positive integer")
    try:
        written = aggregate_eval_run(
            args.eval_run_dir, args.repeats, manifest=args.manifest
        )
    except EvalAggregateError as exc:
        raise SystemExit(f"eval-aggregate: {exc}") from exc
    for path in written:
        print(f"eval-aggregate: wrote {path}")


if __name__ == "__main__":
    main()
