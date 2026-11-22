"""Machine-readable Synthesis target catalog and code-evidence validation.

Instruction targets keep the original catalog contract.  EvalPlus targets are
an optional extension and require a small, public-artifact evidence record so
that a worker can diagnose a failing problem without receiving benchmark
solutions or private evaluator details.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from autosft.benchmarks.evalplus import CODE_BENCHMARKS, EVALPLUS_PROBLEM_IDS
from autosft.synthesis.errors import SynRequestError


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = (
    REPO_ROOT
    / "skills"
    / "meta"
    / "synthesis-handbook"
    / "references"
    / "target_catalog.md"
)
SUPPORTED_BENCHMARKS = frozenset({"ifeval", "ifbench"}) | CODE_BENCHMARKS
EVALPLUS_EVIDENCE_NAME = "evalplus_failures.jsonl"
EVALPLUS_EVIDENCE_SCHEMA_VERSION = "1.0"

_CATALOG_ROW_RE = re.compile(r"^\|\s*`(?P<target>(?:ifeval|ifbench)\.[^`]+)`\s*\|")
_CODE_TARGET_RE = re.compile(
    r"^(?P<benchmark>humaneval_plus|mbpp_plus)\.problem_(?P<number>0|[1-9][0-9]*)$"
)
_PROBLEM_PREFIX = {
    "humaneval_plus": "HumanEval",
    "mbpp_plus": "Mbpp",
}


# Evaluation and synthesis share one canonical public problem catalog.
_EVALPLUS_PROBLEM_NUMBERS = {
    benchmark: frozenset(int(problem_id.rsplit("/", 1)[1]) for problem_id in ids)
    for benchmark, ids in EVALPLUS_PROBLEM_IDS.items()
}


@lru_cache(maxsize=1)
def instruction_target_ids() -> frozenset[str]:
    try:
        lines = CATALOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SynRequestError(
            f"could not read Synthesis target catalog: {CATALOG_PATH}"
        ) from exc
    targets = {
        match.group("target")
        for line in lines
        if (match := _CATALOG_ROW_RE.match(line)) is not None
    }
    if not targets:
        raise SynRequestError(
            f"Synthesis target catalog contains no instruction targets: {CATALOG_PATH}"
        )
    return frozenset(targets)


def evalplus_target_id(benchmark: str, problem_id: str) -> str:
    name = benchmark.strip().lower()
    prefix = _PROBLEM_PREFIX.get(name)
    if prefix is None:
        raise SynRequestError(f"{benchmark!r} is not an EvalPlus benchmark")
    match = re.fullmatch(rf"{re.escape(prefix)}/(0|[1-9][0-9]*)", problem_id)
    if match is None:
        raise SynRequestError(
            f"problem_id {problem_id!r} does not belong to benchmark {name!r}"
        )
    number = int(match.group(1))
    if number not in _EVALPLUS_PROBLEM_NUMBERS[name]:
        raise SynRequestError(
            f"problem_id {problem_id!r} is not in the checked-in {name} catalog"
        )
    return f"{name}.problem_{number}"


def target_benchmark(target_id: str) -> str:
    benchmark = target_id.split(".", 1)[0]
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise SynRequestError(f"Synthesis target is not catalog-backed: {target_id}")
    return benchmark


def validate_target_id(target_id: str) -> str:
    benchmark = target_benchmark(target_id)
    if benchmark not in CODE_BENCHMARKS:
        if target_id not in instruction_target_ids():
            raise SynRequestError(
                f"Synthesis target is not catalog-backed: {target_id}"
            )
        return target_id

    match = _CODE_TARGET_RE.fullmatch(target_id)
    if match is None:
        raise SynRequestError(
            f"EvalPlus Synthesis targets must use '<benchmark>.problem_<number>': {target_id}"
        )
    number = int(match.group("number"))
    if number not in _EVALPLUS_PROBLEM_NUMBERS[benchmark]:
        raise SynRequestError(f"Synthesis target is not catalog-backed: {target_id}")
    return target_id


def _required_string(
    payload: Mapping[str, Any], key: str, *, allow_empty: bool = False
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise SynRequestError(f"EvalPlus evidence field {key!r} must be {qualifier}")
    return value


def _expected_failure_class(base_result: str, plus_result: str) -> str:
    if base_result == "pass" and plus_result == "fail":
        return "plus_only_failure"
    if base_result == "fail" and plus_result == "fail":
        return "base_and_plus_failure"
    raise SynRequestError(
        "EvalPlus synthesis evidence must describe a plus-suite failure; got "
        f"base_result={base_result!r}, plus_result={plus_result!r}"
    )


def validate_evalplus_evidence(target_id: str, materials_dir: Path) -> Path:
    validate_target_id(target_id)
    match = _CODE_TARGET_RE.fullmatch(target_id)
    if match is None:
        raise SynRequestError(f"target is not an EvalPlus target: {target_id}")
    benchmark = match.group("benchmark")
    number = int(match.group("number"))
    problem_id = f"{_PROBLEM_PREFIX[benchmark]}/{number}"
    evidence_path = Path(materials_dir) / EVALPLUS_EVIDENCE_NAME
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise SynRequestError(
            f"EvalPlus target {target_id} requires materials/{EVALPLUS_EVIDENCE_NAME}"
        )

    rows = 0
    try:
        stream = evidence_path.open(encoding="utf-8")
    except OSError as exc:
        raise SynRequestError(
            f"could not read EvalPlus evidence: {evidence_path}"
        ) from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SynRequestError(
                    f"invalid JSON in EvalPlus evidence at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} must be a JSON object"
                )
            schema = _required_string(payload, "evidence_schema_version")
            if schema != EVALPLUS_EVIDENCE_SCHEMA_VERSION:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} has unsupported schema {schema!r}"
                )
            if _required_string(payload, "benchmark").lower() != benchmark:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} benchmark does not match {target_id}"
                )
            if _required_string(payload, "problem_id") != problem_id:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} problem_id does not match {target_id}"
                )
            if _required_string(payload, "sample_id") != problem_id:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} sample_id must equal problem_id"
                )
            if _required_string(payload, "syn_target_id") != target_id:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} syn_target_id does not match directory"
                )
            _required_string(payload, "prompt")
            _required_string(payload, "prediction", allow_empty=True)
            # ``opencompass_result_ref`` is retained for compatibility with
            # existing evaluation exports.  Public local evaluators may use
            # the source-neutral alias ``evaluation_result_ref`` instead.
            if not (
                isinstance(payload.get("opencompass_result_ref"), str)
                and payload["opencompass_result_ref"].strip()
            ) and not (
                isinstance(payload.get("evaluation_result_ref"), str)
                and payload["evaluation_result_ref"].strip()
            ):
                raise SynRequestError(
                    "EvalPlus evidence requires a non-empty evaluation result reference"
                )
            repeat = payload.get("repeat")
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} repeat must be a positive integer"
                )
            base_result = _required_string(payload, "base_result").lower()
            plus_result = _required_string(payload, "plus_result").lower()
            expected_class = _expected_failure_class(base_result, plus_result)
            if _required_string(payload, "failure_class") != expected_class:
                raise SynRequestError(
                    f"EvalPlus evidence line {line_number} failure_class does not match verdicts"
                )
            rows += 1
    if rows == 0:
        raise SynRequestError(f"EvalPlus evidence is empty: {evidence_path}")
    return evidence_path


def validate_request_target(target_id: str, materials_dir: Path) -> str:
    target = validate_target_id(target_id)
    if target_benchmark(target) in CODE_BENCHMARKS:
        validate_evalplus_evidence(target, materials_dir)
    return target


def validate_benchmarks(raw: str) -> tuple[str, ...]:
    benchmarks = tuple(
        dict.fromkeys(token for token in re.split(r"[,\s]+", raw) if token)
    )
    if not benchmarks:
        raise SynRequestError("at least one Synthesis benchmark is required")
    unsupported = [name for name in benchmarks if name not in SUPPORTED_BENCHMARKS]
    if unsupported:
        raise SynRequestError(
            "Synthesis target catalog does not support benchmark(s): "
            + ", ".join(unsupported)
        )
    return benchmarks


def validate_target_benchmarks(target_ids: list[str], benchmarks: str) -> None:
    configured = set(validate_benchmarks(benchmarks))
    mismatched = [
        target_id
        for target_id in target_ids
        if target_benchmark(target_id) not in configured
    ]
    if mismatched:
        raise SynRequestError(
            "Synthesis targets are outside the run's configured benchmarks: "
            + ", ".join(mismatched)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the fixed Synthesis target catalog"
    )
    parser.add_argument("--benchmarks", required=True)
    args = parser.parse_args()
    try:
        validate_benchmarks(args.benchmarks)
    except SynRequestError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
