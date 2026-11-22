"""Validate and normalize one completion per problem from public EvalPlus.

This module does not execute generated code. Local reports explicitly identify
their combined base-plus verdicts; imported OpenCompass reports can provide the
separate base and extra-test verdicts used by public EvalPlus.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


CODE_BENCHMARKS = frozenset({"humaneval_plus", "mbpp_plus"})
EVALPLUS_EXPECTED_DETAILS = {"humaneval_plus": 164, "mbpp_plus": 378}


def _expand_ranges(spec: str) -> frozenset[int]:
    values: set[int] = set()
    for part in spec.split(","):
        bounds = part.split("-", 1)
        values.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return frozenset(values)


# EvalPlus's public HumanEval+ and sanitized MBPP+ test splits.
_MBPP_NUMBERS = _expand_ranges(
    "2-4,6-9,11-12,14,16-20,56-59,61-72,74-75,77,79-80,82,84-106,"
    "108-109,111,113,116,118-120,123-133,135,137-142,145,160-162,"
    "165-168,170-172,222-224,226-227,230,232-235,237-240,242,244-245,"
    "247,250-253,255-257,259-262,264-274,276-287,290,292-294,296-297,"
    "299-301,305-306,308-312,388-392,394-395,397-398,404-406,409-410,"
    "412-415,418-422,424-430,432-433,435-437,439-441,445-448,450-451,"
    "453-460,462-463,465,468,470-479,554-560,562-569,572-573,576-581,"
    "583,585-594,596-600,602-608,610-612,614-616,618-620,622-624,626,"
    "628-633,635,637-639,641,643-644,720-726,728,730-737,739-745,"
    "748-755,757-760,762-767,769-773,775,777-778,780-782,784-788,"
    "790-794,796-801,803-809"
)
EVALPLUS_PROBLEM_IDS = {
    "humaneval_plus": frozenset(f"HumanEval/{number}" for number in range(164)),
    "mbpp_plus": frozenset(f"Mbpp/{number}" for number in _MBPP_NUMBERS),
}


def _unwrap(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _problem_id(detail: dict[str, Any], benchmark: str) -> str:
    identifiers: list[str] = []
    for field in ("task_id", "problem_id", "sample_id", "reference", "example_abbr"):
        if field not in detail or detail[field] is None:
            continue
        value = _unwrap(detail[field])
        # OpenCompass's example_abbr (and some sample_id fields) identify a
        # dataset row, e.g. mbpp_plus_0, not its canonical EvalPlus task. Never
        # infer a task number from that index: MBPP's task ids are sparse.
        if field in {"sample_id", "example_abbr"} and (
            not isinstance(value, str) or value not in EVALPLUS_PROBLEM_IDS[benchmark]
        ):
            if isinstance(value, str) and value.startswith(("HumanEval/", "Mbpp/")):
                raise ValueError(
                    f"invalid {benchmark} problem identity in {field}: {value!r}"
                )
            continue
        if not isinstance(value, str) or value not in EVALPLUS_PROBLEM_IDS[benchmark]:
            raise ValueError(
                f"invalid {benchmark} problem identity in {field}: {value!r}"
            )
        identifiers.append(value)
    if not identifiers:
        raise ValueError(f"EvalPlus {benchmark} detail has no problem identity")
    if len(set(identifiers)) != 1:
        raise ValueError(
            f"EvalPlus detail has conflicting problem identities: {identifiers}"
        )
    return identifiers[0]


def _verdict(value: Any, *, problem_id: str, key: str) -> str:
    value = _unwrap(value)
    # A candidate timing out is an evaluated failure. Missing or unknown
    # execution statuses indicate an incomplete report and must fail collection.
    if value not in ("pass", "fail", "timeout"):
        raise ValueError(
            f"EvalPlus {problem_id} has missing or invalid {key}: {value!r}"
        )
    return value


def _percentage(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise ValueError(f"EvalPlus {key} must be a finite percentage in [0, 100]")
    return float(value)


def build_evalplus_artifacts(
    report: dict[str, Any],
    *,
    benchmark: str,
    source_path: Path,
    repeat: int = 1,
    expected_count: int | None = None,
    expected_problem_ids: Iterable[str] | None = None,
    eval_backend: str = "evalplus",
    prediction_context: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Normalize a complete EvalPlus repeat, rejecting partial or mixed results.

    ``expected_count`` may explicitly select a smaller smoke-test split. Full
    runs require every official problem exactly once. If a caller selected a
    subset, ``expected_problem_ids`` also verifies that exact selection. Scores
    must agree with the per-problem verdicts; missing results never score zero.
    """
    benchmark = benchmark.strip().lower()
    if benchmark not in CODE_BENCHMARKS:
        raise ValueError(f"unsupported EvalPlus benchmark: {benchmark}")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("EvalPlus repeat must be a positive integer")
    if report.get("benchmark", benchmark) != benchmark:
        raise ValueError(
            f"EvalPlus report belongs to a different benchmark: {report.get('benchmark')}"
        )
    if "repeat" in report and (
        isinstance(report["repeat"], bool)
        or not isinstance(report["repeat"], int)
        or report["repeat"] != repeat
    ):
        raise ValueError(
            f"EvalPlus report repeat does not match expected repeat {repeat}"
        )
    semantics = report.get(
        "plus_result_semantics",
        "combined" if eval_backend == "evalplus" else "extra_tests",
    )
    if semantics not in ("combined", "extra_tests"):
        raise ValueError(f"unknown EvalPlus plus-result semantics: {semantics!r}")
    full_count = EVALPLUS_EXPECTED_DETAILS[benchmark]
    expected_count = full_count if expected_count is None else expected_count
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or not 1 <= expected_count <= full_count
    ):
        raise ValueError(f"EvalPlus expected_count must be in [1, {full_count}]")
    expected_ids = None
    if expected_problem_ids is not None:
        expected_values = list(expected_problem_ids)
        if not all(isinstance(value, str) for value in expected_values):
            raise ValueError("EvalPlus expected problem ids must be strings")
        expected_ids = set(expected_values)
        if (
            len(expected_ids) != len(expected_values)
            or len(expected_ids) != expected_count
            or not expected_ids <= EVALPLUS_PROBLEM_IDS[benchmark]
        ):
            raise ValueError(
                "EvalPlus expected problem ids are duplicate, invalid, or incomplete"
            )
    elif expected_count == full_count:
        expected_ids = EVALPLUS_PROBLEM_IDS[benchmark]

    raw_details = report.get("details")
    if not isinstance(raw_details, (list, dict)):
        raise ValueError("EvalPlus report details must be a list or object")
    details = (
        list(raw_details.values()) if isinstance(raw_details, dict) else raw_details
    )
    if len(details) != expected_count:
        raise ValueError(
            f"EvalPlus {benchmark} has {len(details)} problem details; expected {expected_count}"
        )
    reported = {key: _percentage(report, key) for key in ("base_pass@1", "plus_pass@1")}
    seen: set[str] = set()
    passes: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for detail in details:
        if not isinstance(detail, dict):
            raise ValueError("EvalPlus report details must contain objects")
        problem_id = _problem_id(detail, benchmark)
        if problem_id in seen:
            raise ValueError(f"EvalPlus report repeats problem {problem_id}")
        seen.add(problem_id)
        if "repeat" in detail and (
            isinstance(detail["repeat"], bool)
            or not isinstance(detail["repeat"], int)
            or detail["repeat"] != repeat
        ):
            raise ValueError(
                f"EvalPlus {problem_id} detail belongs to a different repeat"
            )
        raw_base = _verdict(
            detail.get("base_result"), problem_id=problem_id, key="base_result"
        )
        raw_plus = _verdict(
            detail.get("plus_result"), problem_id=problem_id, key="plus_result"
        )
        base = "pass" if raw_base == "pass" else "fail"
        plus = "pass" if raw_plus == "pass" else "fail"
        if semantics == "combined" and base == "fail" and plus == "pass":
            raise ValueError(
                f"EvalPlus {problem_id} passes the combined suite but fails base tests"
            )
        if semantics == "extra_tests" and base == "fail":
            plus = "fail"
        passes["base_result"] += base == "pass"
        passes["plus_result"] += plus == "pass"
        if plus == "pass":
            continue
        failure_class = (
            "plus_only_failure" if base == "pass" else "base_and_plus_failure"
        )
        classes[failure_class] += 1
        context = (prediction_context or {}).get(problem_id, {})
        prompt = _unwrap(detail.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = context.get("prompt")
        prediction = _unwrap(detail.get("prediction", detail.get("pred")))
        if prediction is None:
            prediction = context.get("prediction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"EvalPlus failure {problem_id} lacks prompt context")
        if not isinstance(prediction, str):
            raise ValueError(f"EvalPlus failure {problem_id} lacks prediction text")
        failures.append(
            {
                "evidence_schema_version": "1.0",
                "sample_id": problem_id,
                "problem_id": problem_id,
                "benchmark": benchmark,
                "syn_target_id": f"{benchmark}.problem_{problem_id.rsplit('/', 1)[1]}",
                "repeat": repeat,
                "base_result": base,
                "plus_result": plus,
                "source_base_result": raw_base,
                "source_plus_result": raw_plus,
                "source_plus_result_semantics": semantics,
                "plus_result_semantics": "combined",
                "failure_class": failure_class,
                "prompt": prompt,
                "prediction": prediction,
                "reference": problem_id,
                "opencompass_result_ref": str(source_path),
                "prediction_source_ref": context.get("prediction_source_ref"),
            }
        )

    if expected_ids is not None and seen != expected_ids:
        raise ValueError(
            f"EvalPlus {benchmark} problem identities do not match the selected split; "
            f"missing={sorted(expected_ids - seen)}, unexpected={sorted(seen - expected_ids)}"
        )
    for kind in ("base", "plus"):
        detail_percent = 100 * passes[f"{kind}_result"] / expected_count
        # A percentage rounded to two decimal places is harmless. A missing
        # or differently scored task is not: even 1/378 is much larger.
        if not math.isclose(
            reported[f"{kind}_pass@1"], detail_percent, rel_tol=0, abs_tol=0.005000001
        ):
            raise ValueError(
                f"EvalPlus reported {kind} score disagrees with problem verdicts"
            )

    breakdown = {
        name: {"count": count, "total": expected_count, "rate": count / expected_count}
        for name, count in sorted(classes.items())
    }
    base_score = passes["base_result"] / expected_count
    plus_score = passes["plus_result"] / expected_count
    metrics = {
        # Keep the raw metric spelling alongside the normalized artifact keys;
        # OpenCompass reports these names and downstream ledgers sometimes
        # address metrics by their report key.
        "base_pass@1": base_score,
        "plus_pass@1": plus_score,
        "base_pass_at_1": base_score,
        "plus_pass_at_1": plus_score,
        "overall_accuracy": plus_score,
        "score": plus_score,
        "primary_metric": "plus_pass_at_1",
        "benchmark": benchmark,
        "repeat": repeat,
        "plus_result_semantics": "combined",
        "source_plus_result_semantics": semantics,
        "num_problems": expected_count,
        "num_failures": len(failures),
        "num_base_pass": passes["base_result"],
        "num_plus_pass": passes["plus_result"],
        "is_full_benchmark": expected_count == full_count,
        "top_failure_categories": [
            {"category": name, **breakdown[name]}
            for name in sorted(classes, key=lambda name: (-classes[name], name))
        ],
        "failures_path": "failures.jsonl",
        "failure_breakdown_path": "failure_breakdown.json",
        "eval_backend": eval_backend,
        "opencompass_result_path": str(source_path),
        "opencompass_reported_metrics": reported,
    }
    return metrics, failures, breakdown


def write_evalplus_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    **kwargs: Any,
) -> None:
    """Write the standard metric and failure artifacts after full validation."""
    metrics, failures, breakdown = build_evalplus_artifacts(report, **kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("metrics.json", metrics),
        ("failure_breakdown.json", breakdown),
    ):
        (output_dir / filename).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "failures.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in failures), encoding="utf-8"
    )
