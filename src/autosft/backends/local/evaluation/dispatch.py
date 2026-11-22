#!/usr/bin/env python3
"""Fixed local evaluation adapters; all inference stays on the local GPU host."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from autosft.core.common import REPO_ROOT as ROOT

sys.path.insert(0, str(ROOT / "third_party"))
from autosft.benchmarks.opencompass_bridge import repeat_output_dirs
from autosft.core.common import write_json


def scores(record: dict, output: Path, repeat: int) -> None:
    metrics = json.loads((output / "metrics.json").read_text())
    write_json(
        output / "scores.json",
        {
            **record,
            "repeat": repeat,
            "metrics": metrics,
            "primary_score": metrics["score"],
            "num_failures": metrics.get("num_failures"),
            "top_failure_categories": metrics.get("top_failure_categories"),
            "top_failure_subcategories": metrics.get("top_failure_subcategories"),
            "failures_path": str(output / "failures.jsonl"),
            "failure_breakdown_path": str(output / "failure_breakdown.json"),
            "stdout": record.get(
                "log_path", str(Path(record["output_dir"]) / "local_job.log")
            ),
            "stderr": record.get(
                "log_path", str(Path(record["output_dir"]) / "local_job.log")
            ),
        },
    )


def run_opencompass(record: dict) -> None:
    # One model load per benchmark, every repeat consumes independent samples.
    benchmark = record["benchmark"]
    if benchmark in {"humaneval_plus", "mbpp_plus"}:
        run_evalplus(record)
        return
    from autosft.backends.local.evaluation.instruction_following import run_if

    run_if(record, root=ROOT, score_writer=scores)


def run_evalplus(record: dict) -> None:
    """Evaluate staged HumanEval+/MBPP+ repeats through the public EvalPlus guard."""
    # ``src/autosft/benchmarks/submit.py`` is launched as a script, so its directory is sys.path[0].
    # Keep the adapter importable as ``autosft.backends.local.evaluation.code_eval`` while the adapter hides
    # that directory before importing the external EvalPlus package.
    from autosft.backends.local.evaluation.code_eval import (
        LocalScorer,
        create_generator,
        prepare_record,
        run_record,
    )

    prepared = prepare_record(record)
    rows, cache_key = prepared
    with LocalScorer() as scorer:
        scorer.prepare(rows, record["benchmark"], cache_key)
        with create_generator(record, rows=rows) as generator:
            for repeat, output in enumerate(
                repeat_output_dirs(Path(record["output_dir"]), record["repeats"]), 1
            ):
                repeat_record = dict(record)
                repeat_record["output_dir"] = str(output)
                repeat_record["repeat"] = repeat
                report = run_record(
                    repeat_record,
                    generator=generator,
                    repeat=repeat,
                    prepared=prepared,
                    scorer=scorer,
                )
                scores(repeat_record, output, repeat)
                write_json(
                    output / "generation_config.json",
                    {
                        **generator.metadata(repeat),
                        "scoring_protocol": report.get("scoring_protocol"),
                        "dataset_version": report.get("dataset_version"),
                        "dataset_sha256": cache_key.split(":", 1)[0],
                    },
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", required=True)
    record = json.loads(p.parse_args().record)
    if record.get("eval_backend") not in {"opencompass", "evalplus"}:
        raise SystemExit(
            "unsupported local evaluation adapter; use opencompass or evalplus"
        )
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        os.environ[key] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    nltk_paths = [
        os.environ.get("AUTO_SFT_NLTK_DATA_DIR", ""),
        str(ROOT / "third_party/ifbench/.nltk_data"),
        os.environ.get("NLTK_DATA", ""),
    ]
    os.environ["NLTK_DATA"] = os.pathsep.join(path for path in nltk_paths if path)
    run_opencompass(record)


if __name__ == "__main__":
    main()
