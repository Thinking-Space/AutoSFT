#!/usr/bin/env python3
"""Bridge OpenCompass inference/results into AutoSFT's evaluation contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CONFIGS = {
    "ifeval": ("IFEvalDataset", "IFEvaluator", "IFEval", "opencompass/IFEval"),
    "ifbench": (
        "IFBenchDataset",
        "IFBenchEvaluator",
        "IFBench",
        "allenai/IFBench_test",
    ),
}
METRIC_KEYS = {
    "strict_prompt_accuracy": "Prompt-level-strict-accuracy",
    "loose_prompt_accuracy": "Prompt-level-loose-accuracy",
    "strict_instruction_accuracy": "Inst-level-strict-accuracy",
    "loose_instruction_accuracy": "Inst-level-loose-accuracy",
}
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# OpenCompass NUM_TRIALS=N (expand_trial_datasets) keeps the base abbr for
# trial 1 and suffixes trials 2..N as <abbr>_trial<k>; older checkouts wrote
# <abbr>_repeat<k>. The suffix is 1-indexed and the base file is trial 1.
_TRIAL_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?:trial|repeat)(?P<index>\d+)$")


def write_runtime_config(
    output: Path,
    *,
    benchmark: str,
    infer_type: str,
    limit: int,
) -> Path:
    """Write a small OpenCompass config with an optional dataset limit."""
    benchmark = benchmark.lower()
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"unsupported OpenCompass benchmark: {benchmark}")
    if not _MODULE_NAME_RE.fullmatch(infer_type):
        raise ValueError(
            f"OpenCompass infer_type must be a dotted Python module name; got {infer_type!r}"
        )
    if limit < -1:
        raise ValueError(f"limit must be -1 or non-negative, got {limit}")

    dataset_type, evaluator_type, dataset_abbr, dataset_path = BENCHMARK_CONFIGS[
        benchmark
    ]
    test_range = f", test_range='[:{limit}]'" if limit >= 0 else ""
    text = (
        f"_model_module = __import__({infer_type!r}, fromlist=['models'])\n"
        "models = _model_module.models\n"
        "del _model_module\n"
        f"from opencompass.datasets import {dataset_type}, {evaluator_type}\n"
        "from opencompass.openicl.icl_inferencer import GenInferencer\n"
        "from opencompass.openicl.icl_prompt_template import PromptTemplate\n"
        "from opencompass.openicl.icl_retriever import ZeroRetriever\n\n"
        "datasets = [\n"
        "    dict(\n"
        f"        abbr={dataset_abbr!r},\n"
        f"        type={dataset_type},\n"
        f"        path={dataset_path!r},\n"
        "        reader_cfg=dict(\n"
        "            input_columns=['prompt'],\n"
        f"            output_column='reference'{test_range}),\n"
        "        infer_cfg=dict(\n"
        "            prompt_template=dict(\n"
        "                type=PromptTemplate,\n"
        "                template=dict(round=[\n"
        "                    dict(role='HUMAN', prompt='{prompt}'),\n"
        "                ])),\n"
        "            retriever=dict(type=ZeroRetriever),\n"
        "            inferencer=dict(type=GenInferencer)),\n"
        "        eval_cfg=dict(\n"
        f"            evaluator=dict(type={evaluator_type}),\n"
        "            pred_role='BOT')),\n"
        "]\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _result_candidates(
    result_root: Path, benchmark: str
) -> list[tuple[int, int, Path, dict[str, Any]]]:
    """Every valid result JSON below ``result_root`` as (name_match, mtime, path, payload)."""
    benchmark_lower = benchmark.lower()
    candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
    for path in result_root.rglob("*.json"):
        if "results" not in path.parts:
            continue
        payload = _load_json(path)
        if payload is None or not isinstance(payload.get("details"), (dict, list)):
            continue
        if not all(source_key in payload for source_key in METRIC_KEYS.values()):
            continue
        name_match = int(benchmark_lower in path.name.lower())
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        candidates.append((name_match, modified_ns, path, payload))
    return candidates


def _trial_index(path: Path) -> int:
    match = _TRIAL_SUFFIX_RE.match(path.stem)
    return int(match.group("index")) if match else 1


def find_result_file(result_root: Path, benchmark: str) -> tuple[Path, dict[str, Any]]:
    """Find the newest OpenCompass result JSON for one benchmark."""
    candidates = _result_candidates(result_root, benchmark)
    if not candidates:
        raise FileNotFoundError(
            f"no OpenCompass {benchmark} result JSON found below {result_root}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, path, payload = candidates[0]
    return path, payload


def find_trial_result_files(
    result_root: Path, benchmark: str, *, repeats: int
) -> list[tuple[Path, dict[str, Any]]]:
    """Return exactly ``repeats`` result JSONs in trial order (1..N).

    Trial 1 is the un-suffixed file; trial k>1 is ``<abbr>_trial<k>.json``.
    A missing or duplicated trial is an error: a partial set must never be
    turned into a "complete" repeat set.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be a positive integer, got {repeats}")
    candidates = _result_candidates(result_root, benchmark)
    matching = [item for item in candidates if item[0]] or candidates
    by_trial: dict[int, list[tuple[int, Path, dict[str, Any]]]] = {}
    for _, modified_ns, path, payload in matching:
        by_trial.setdefault(_trial_index(path), []).append((modified_ns, path, payload))

    ordered: list[tuple[Path, dict[str, Any]]] = []
    missing: list[int] = []
    for trial in range(1, repeats + 1):
        entries = by_trial.get(trial)
        if not entries:
            missing.append(trial)
            continue
        if len(entries) > 1:
            names = ", ".join(str(entry[1]) for entry in entries)
            raise FileNotFoundError(
                f"OpenCompass {benchmark} trial {trial} is ambiguous under "
                f"{result_root}: {names}"
            )
        _, path, payload = entries[0]
        ordered.append((path, payload))
    if missing:
        found = sorted(by_trial)
        raise FileNotFoundError(
            f"OpenCompass {benchmark} produced trials {found} under {result_root}; "
            f"expected 1..{repeats} (missing {missing})"
        )
    extra = sorted(trial for trial in by_trial if trial > repeats)
    if extra:
        raise FileNotFoundError(
            f"OpenCompass {benchmark} produced unexpected trials {extra} under "
            f"{result_root}; expected exactly 1..{repeats}"
        )
    return ordered


def _details_in_order(details: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(details, list):
        values = details
    else:

        def sort_key(item: tuple[str, Any]) -> tuple[int, int | str]:
            key = str(item[0])
            return (0, int(key)) if key.isdigit() else (1, key)

        values = [value for _, value in sorted(details.items(), key=sort_key)]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("OpenCompass result details must contain JSON objects")
    return values


@lru_cache(maxsize=2)
def _load_evaluation_lib(benchmark: str) -> Any:
    if benchmark == "ifeval":
        sys.path.insert(
            0, str(REPO_ROOT / "third_party" / "instruction_following_eval")
        )
        from instruction_following_eval import evaluation_lib

        return evaluation_lib

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "ifbench"))
    import evaluation_lib

    return evaluation_lib


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _score_detail(benchmark: str, detail: dict[str, Any]) -> dict[str, Any]:
    reference = _unwrap_singleton(detail.get("refer") or detail.get("reference"))
    if not isinstance(reference, dict):
        raise ValueError("OpenCompass result detail is missing a reference object")
    prompt = str(reference.get("prompt") or detail.get("prompt") or "")
    completion = str(
        _unwrap_singleton(
            detail.get("pred")
            if detail.get("pred") is not None
            else detail.get("prediction", "")
        )
    )
    instruction_ids = list(reference.get("instruction_id_list") or [])
    kwargs = list(reference.get("kwargs") or [])
    if not prompt or not instruction_ids:
        raise ValueError("OpenCompass result detail lacks prompt/instruction metadata")

    evaluation_lib = _load_evaluation_lib(benchmark)
    example = evaluation_lib.InputExample(
        key=reference.get("key"),
        instruction_id_list=instruction_ids,
        prompt=prompt,
        kwargs=kwargs,
    )
    prompt_to_response = {prompt: completion}
    strict = evaluation_lib.test_instruction_following_strict(
        example, prompt_to_response
    )
    loose = evaluation_lib.test_instruction_following_loose(example, prompt_to_response)
    return {
        "sample_id": str(reference.get("key", "")),
        "prompt": prompt,
        "completion": completion,
        "instruction_id_list": instruction_ids,
        "kwargs": kwargs,
        "follow_instruction_list_strict": list(strict.follow_instruction_list),
        "follow_instruction_list_loose": list(loose.follow_instruction_list),
    }


def _coarse_category(instruction_id: str) -> str:
    return instruction_id.split(":", 1)[0] if ":" in instruction_id else instruction_id


def _subcategory(instruction_id: str) -> str:
    return instruction_id.split(":", 1)[1] if ":" in instruction_id else ""


def build_artifacts(
    report: dict[str, Any],
    *,
    benchmark: str,
    source_path: Path,
    score_detail: Callable[[str, dict[str, Any]], dict[str, Any]] = _score_detail,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Re-score OpenCompass predictions with AutoSFT's vendored checkers."""
    benchmark = benchmark.lower()
    details = _details_in_order(report["details"])
    scored = [score_detail(benchmark, detail) for detail in details]
    if not scored:
        raise ValueError("OpenCompass result contains no scored samples")

    fine_total: Counter[str] = Counter()
    fine_strict_fail: Counter[str] = Counter()
    fine_loose_fail: Counter[str] = Counter()
    coarse_total: Counter[str] = Counter()
    coarse_strict_fail: Counter[str] = Counter()
    coarse_loose_fail: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    strict_prompt_correct = 0
    loose_prompt_correct = 0
    instruction_total = 0
    strict_instruction_correct = 0
    loose_instruction_correct = 0

    for record in scored:
        instruction_ids = record["instruction_id_list"]
        strict_follows = record["follow_instruction_list_strict"]
        loose_follows = record["follow_instruction_list_loose"]
        if not (len(instruction_ids) == len(strict_follows) == len(loose_follows)):
            raise ValueError(
                f"checker result length mismatch for sample {record['sample_id']}"
            )

        strict_prompt_ok = all(strict_follows)
        loose_prompt_ok = all(loose_follows)
        strict_prompt_correct += int(strict_prompt_ok)
        loose_prompt_correct += int(loose_prompt_ok)
        instruction_total += len(instruction_ids)
        strict_instruction_correct += sum(bool(value) for value in strict_follows)
        loose_instruction_correct += sum(bool(value) for value in loose_follows)

        for instruction_id, strict_ok, loose_ok in zip(
            instruction_ids, strict_follows, loose_follows
        ):
            category = _coarse_category(instruction_id)
            fine_total[instruction_id] += 1
            coarse_total[category] += 1
            if not strict_ok:
                fine_strict_fail[instruction_id] += 1
                coarse_strict_fail[category] += 1
            if not loose_ok:
                fine_loose_fail[instruction_id] += 1
                coarse_loose_fail[category] += 1

        if not strict_prompt_ok:
            failure = dict(record)
            failure["failed_instructions_strict"] = [
                instruction_id
                for instruction_id, follows in zip(instruction_ids, strict_follows)
                if not follows
            ]
            failure["failed_instructions_loose"] = [
                instruction_id
                for instruction_id, follows in zip(instruction_ids, loose_follows)
                if not follows
            ]
            failure["scores"] = {
                f"{benchmark}_strict_prompt": float(strict_prompt_ok),
                f"{benchmark}_loose_prompt": float(loose_prompt_ok),
                f"{benchmark}_strict_instruction": (
                    sum(bool(value) for value in strict_follows) / len(strict_follows)
                ),
                f"{benchmark}_loose_instruction": (
                    sum(bool(value) for value in loose_follows) / len(loose_follows)
                ),
            }
            failure["opencompass_result_ref"] = str(source_path)
            failures.append(failure)

    prompt_total = len(scored)
    strict_prompt_accuracy = strict_prompt_correct / prompt_total
    loose_prompt_accuracy = loose_prompt_correct / prompt_total
    strict_instruction_accuracy = strict_instruction_correct / instruction_total
    loose_instruction_accuracy = loose_instruction_correct / instruction_total
    overall_accuracy = (
        strict_prompt_accuracy
        + loose_prompt_accuracy
        + strict_instruction_accuracy
        + loose_instruction_accuracy
    ) / 4

    breakdown: dict[str, dict[str, Any]] = {
        instruction_id: {
            "category": _coarse_category(instruction_id),
            "subcategory": _subcategory(instruction_id),
            "total": fine_total[instruction_id],
            "strict_fail": fine_strict_fail[instruction_id],
            "loose_fail": fine_loose_fail[instruction_id],
            "strict_fail_rate": (
                fine_strict_fail[instruction_id] / fine_total[instruction_id]
            ),
            "loose_fail_rate": (
                fine_loose_fail[instruction_id] / fine_total[instruction_id]
            ),
        }
        for instruction_id in sorted(fine_total)
    }
    top_categories = [
        {
            "category": category,
            "strict_fail": coarse_strict_fail[category],
            "loose_fail": coarse_loose_fail[category],
            "total": coarse_total[category],
            "strict_fail_rate": (coarse_strict_fail[category] / coarse_total[category]),
            "loose_fail_rate": (coarse_loose_fail[category] / coarse_total[category]),
        }
        for category in coarse_total
    ]
    top_categories.sort(
        key=lambda row: (
            -row["strict_fail"],
            -row["strict_fail_rate"],
            row["category"],
        )
    )
    top_subcategories = [
        {"instruction_id": instruction_id, **values}
        for instruction_id, values in breakdown.items()
    ]
    top_subcategories.sort(
        key=lambda row: (
            -row["strict_fail"],
            -row["strict_fail_rate"],
            row["instruction_id"],
        )
    )

    reported_metrics = {
        key: value for key, value in report.items() if isinstance(value, (int, float))
    }
    metrics = {
        "strict_prompt_accuracy": strict_prompt_accuracy,
        "loose_prompt_accuracy": loose_prompt_accuracy,
        "strict_instruction_accuracy": strict_instruction_accuracy,
        "loose_instruction_accuracy": loose_instruction_accuracy,
        "overall_accuracy": overall_accuracy,
        "score": overall_accuracy,
        "num_failures": len(failures),
        "top_failure_categories": top_categories,
        "top_failure_subcategories": top_subcategories,
        "failures_path": "failures.jsonl",
        "failure_breakdown_path": "failure_breakdown.json",
        "inspect_log_dir": "opencompass",
        "eval_backend": "opencompass",
        "opencompass_result_path": str(source_path),
        "opencompass_reported_metrics": reported_metrics,
    }
    return metrics, failures, breakdown


def _write_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    *,
    benchmark: str,
    source_path: Path,
) -> None:
    metrics, failures, breakdown = build_artifacts(
        report,
        benchmark=benchmark,
        source_path=source_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
    (output_dir / "failure_breakdown.json").write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def repeat_output_dirs(output_dir: Path, repeats: int) -> list[Path]:
    """Where repeat k of a benchmark lands: ``rep_<k>/`` below the benchmark dir.

    Mirrors ``autosft.benchmarks.aggregate.repeat_dirs`` (the bridge runs
    inside the eval container, which does not import the control plane).
    """
    output_dir = Path(output_dir)
    if repeats <= 1:
        return [output_dir]
    return [output_dir / f"rep_{k}" for k in range(1, repeats + 1)]


def collect_results(
    result_root: Path,
    output_dir: Path,
    *,
    benchmark: str,
    repeats: int = 1,
) -> list[Path]:
    """Normalize one OpenCompass run into AutoSFT artifacts, one dir per repeat.

    With ``repeats == 1`` the artifacts land directly in ``output_dir`` (the
    single-repeat layout). With ``repeats > 1`` the run must contain exactly N trial
    result files, and trial k is written to ``output_dir/rep_<k>/`` so
    ``eval_aggregate`` can join them. Returns the source paths in trial order.
    """
    if repeats <= 1:
        source_path, report = find_result_file(result_root, benchmark)
        _write_artifacts(
            Path(output_dir), report, benchmark=benchmark, source_path=source_path
        )
        return [source_path]

    trials = find_trial_result_files(result_root, benchmark, repeats=repeats)
    sources: list[Path] = []
    for rep_dir, (source_path, report) in zip(
        repeat_output_dirs(output_dir, repeats), trials
    ):
        _write_artifacts(rep_dir, report, benchmark=benchmark, source_path=source_path)
        sources.append(source_path)
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenCompass configs and normalize its results for AutoSFT."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--benchmark", choices=sorted(BENCHMARK_CONFIGS), required=True
    )
    prepare.add_argument("--infer-type", required=True)
    prepare.add_argument("--limit", type=int, default=-1)
    prepare.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument(
        "--benchmark", choices=sorted(BENCHMARK_CONFIGS), required=True
    )
    collect.add_argument("--result-root", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of OpenCompass trials (NUM_TRIALS) the run was submitted "
            "with; trial k is normalized into <output-dir>/rep_<k>/ when > 1."
        ),
    )
    args = parser.parse_args()
    if args.command == "collect" and args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        output = write_runtime_config(
            args.output,
            benchmark=args.benchmark,
            infer_type=args.infer_type,
            limit=args.limit,
        )
        print(f"OpenCompass runtime config: {output}")
        return

    sources = collect_results(
        args.result_root,
        args.output_dir,
        benchmark=args.benchmark,
        repeats=args.repeats,
    )
    for index, source in enumerate(sources, start=1):
        label = f" (repeat {index})" if len(sources) > 1 else ""
        print(f"Normalized OpenCompass result{label}: {source}")


if __name__ == "__main__":
    main()
