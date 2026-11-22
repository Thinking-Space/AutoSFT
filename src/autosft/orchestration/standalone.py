#!/usr/bin/env python3
"""Run one local Qwen3 SFT round and OpenCompass evaluation without agents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from autosft.core.common import REPO_ROOT as ROOT

from autosft.backends.megatron.convert import validate_hf_template
from autosft.core.common import utc_now, write_json
from autosft.data.preparation import validate_final_data
from autosft.benchmarks.catalog import preflight_code_runtime, unsupported
from autosft.benchmarks.aggregate import aggregate_benchmark
from autosft.orchestration.preflight import asset_checks, preflight_evaluation_runtime


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="Source ChatML JSONL; never modified."
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Local Qwen3-1.7B-Base HF directory."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for this round."
    )
    parser.add_argument("--token-budget", required=True, type=positive_int)
    parser.add_argument("--gpus", type=positive_int, default=1)
    parser.add_argument("--context-parallel", type=positive_int, default=1)
    parser.add_argument("--global-batch-size", type=positive_int, default=16)
    parser.add_argument("--micro-batch-size", type=positive_int, default=1)
    parser.add_argument("--seq-length", type=positive_int, default=8192)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("ifeval", "ifbench", "humaneval_plus", "mbpp_plus"),
        default=["ifeval"],
    )
    parser.add_argument("--eval-repeats", type=positive_int, default=4)
    parser.add_argument(
        "--eval-limit", type=positive_int, help="Limit prompts for a smoke test only."
    )
    parser.add_argument("--eval-max-out-len", type=positive_int)
    parser.add_argument("--eval-max-seq-len", type=positive_int)
    # Respect the same stage-specific environment overrides used by agent
    # runs.  Without these defaults a standalone smoke round silently falls
    # back to the controller interpreter, which is especially surprising when
    # evaluation needs a separate vLLM/PyTorch environment.
    parser.add_argument(
        "--train-python",
        default=os.environ.get("AUTO_SFT_TRAIN_PYTHON") or sys.executable,
    )
    parser.add_argument(
        "--convert-python",
        default=os.environ.get("AUTO_SFT_CONVERT_PYTHON"),
    )
    parser.add_argument(
        "--eval-python",
        default=os.environ.get("AUTO_SFT_EVAL_PYTHON"),
    )
    parser.add_argument(
        "--code-eval-python",
        default=os.environ.get("AUTO_SFT_CODE_EVAL_PYTHON"),
        help="Separate SGLang environment for Code benchmarks.",
    )
    args = parser.parse_args()
    # Keep whether conversion was explicitly pinned.  The converter itself
    # can then fall back from the active interpreter to the validated local
    # conversion environment when no override was requested.
    args._convert_python_override = args.convert_python
    args.convert_python = args.convert_python or args.train_python
    args.eval_python = args.eval_python or args.train_python
    args.code_eval_python = args.code_eval_python or args.eval_python
    args.data, args.model, args.output = (
        path.expanduser().resolve() for path in (args.data, args.model, args.output)
    )
    if args.gpus % args.context_parallel:
        parser.error("GPU count must be divisible by context parallel size")
    data_parallel = args.gpus // args.context_parallel
    if args.global_batch_size % (args.micro_batch_size * data_parallel):
        parser.error(
            "Global batch must be divisible by micro batch times data parallel size"
        )
    if args.context_parallel > 1 and args.seq_length % (2 * args.context_parallel):
        parser.error(
            "Sequence length must be divisible by twice the context parallel size"
        )
    if args.output.exists():
        parser.error(
            "Output must be a new directory; previous results are never overwritten"
        )
    if os.environ.get("AUTO_SFT_RUN_DIR"):
        parser.error(
            "Use a standalone shell without AUTO_SFT_RUN_DIR for this agent-free command"
        )
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Benchmarks must not be repeated")
    missing = unsupported(args.benchmarks)
    if missing:
        parser.error(
            "benchmark assets are not prepared: "
            + ", ".join(missing)
            + ". Run src/autosft/benchmarks/prepare_evalplus.py for code benchmarks."
        )
    return args


def run_round(args):
    failed_assets = [check for check in asset_checks() if not check["ok"]]
    if failed_assets:
        raise ValueError(
            "Public runtime assets are incomplete: " + json.dumps(failed_assets)
        )
    validate_hf_template(args.model)
    validate_final_data(args.data)
    # Construct the exact environment used by every later stage before
    # checking code evaluation.  This keeps preflight settings (especially
    # tensor parallelism and the selected EvalPlus data directory) identical
    # to the eventual src/autosft/benchmarks/submit.py process.
    environment = dict(os.environ)
    environment.update(
        {
            "AUTO_SFT_REPO_ROOT": str(ROOT),
            "AUTO_SFT_SFT_TOKEN_BUDGET": str(args.token_budget),
            "AUTO_SFT_SFT_GPUS_PER_NODE": str(args.gpus),
            "AUTO_SFT_SFT_CP_SIZE": str(args.context_parallel),
            "AUTO_SFT_SFT_CP": str(args.context_parallel),
            "AUTO_SFT_SFT_TP": "1",
            "AUTO_SFT_SFT_PP": "1",
            "AUTO_SFT_SFT_GLOBAL_BATCH_SIZE": str(args.global_batch_size),
            "AUTO_SFT_SFT_MICRO_BATCH_SIZE": str(args.micro_batch_size),
            "AUTO_SFT_SFT_SEQ_LENGTH": str(args.seq_length),
            "AUTO_SFT_SFT_TOKENIZER_MODEL": str(args.model),
            "AUTO_SFT_SFT_LOAD_CKPT": str(args.output / "base_checkpoint"),
            "AUTO_SFT_SFT_OUTPUT_DIR": str(args.output / "sft"),
            "AUTO_SFT_TRAIN_PYTHON": args.train_python,
            "AUTO_SFT_EVAL_PYTHON": args.eval_python,
            "AUTO_SFT_CODE_EVAL_PYTHON": getattr(args, "code_eval_python", None)
            or args.eval_python,
            "AUTO_SFT_BENCHMARKS": ",".join(args.benchmarks),
            "AUTO_SFT_EVAL_REPEATS": str(args.eval_repeats),
            "AUTO_SFT_MAX_ROUNDS": "1",
            "AUTO_SFT_COMPONENTS": "none",
            "AUTO_SFT_EVAL_TP_SIZE": "1",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT / "src"),
                    str(ROOT / "third_party"),
                    environment.get("PYTHONPATH", ""),
                ]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    if args.eval_max_out_len:
        environment["AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN"] = str(args.eval_max_out_len)
    if args.eval_max_seq_len:
        environment["AUTO_SFT_OPENCOMPASS_MAX_SEQ_LEN"] = str(args.eval_max_seq_len)
        environment["AUTO_SFT_CODE_CONTEXT_LENGTH"] = str(args.eval_max_seq_len)
    eval_options = {
        "limit": args.eval_limit or -1,
        "repeats": args.eval_repeats,
        "tensor_parallel_size": 1,
    }
    if args.eval_max_out_len:
        eval_options["max_out_len"] = args.eval_max_out_len
    if args.eval_max_seq_len:
        eval_options["max_model_len"] = args.eval_max_seq_len
    preflight_code_runtime(
        args.benchmarks,
        python=environment["AUTO_SFT_CODE_EVAL_PYTHON"],
        model_path=args.model,
        options=eval_options,
        env=environment,
    )
    if set(args.benchmarks) & {"ifeval", "ifbench"}:
        preflight_evaluation_runtime(args.eval_python, env=environment)
    args.output.mkdir(parents=True)
    logs = args.output / "logs"
    logs.mkdir()
    receipt = {
        "status": "running",
        "started_at": utc_now(),
        "stages": {},
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "epochs": 3,
        "scores": {},
    }

    def run_stage(name, command):
        log = logs / f"{name}.log"
        receipt["stages"][name] = {
            "status": "running",
            "started_at": utc_now(),
            "log": str(log.relative_to(args.output)),
        }
        write_json(args.output / "round.json", receipt)
        print(f"{name}: running; log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                [str(part) for part in command],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        receipt["stages"][name].update(status="complete", finished_at=utc_now())
        write_json(args.output / "round.json", receipt)

    try:
        data_dir = args.output / "data_rounds/round_1"
        data_dir.mkdir(parents=True)
        dataset = data_dir / "final_data.jsonl"
        shutil.copyfile(args.data, dataset)
        recipe = data_dir / "data-recipe.json"
        write_json(
            recipe,
            {
                "recipe_id": "standalone-round-1",
                "round_id": 1,
                "datasets": [{"path": str(dataset)}],
                "rewrote_pool": False,
                "execution_summary": "Use the supplied ChatML data unchanged to validate one training and evaluation round.",
            },
        )
        prepared_dir = args.output
        run_stage(
            "prepare",
            [
                args.train_python,
                "-m",
                "autosft.data.preparation",
                "--recipe",
                recipe,
                "--run-dir",
                prepared_dir,
                "--tokenizer-code-dir",
                ROOT / "src/autosft/data",
                "--tokenizer-python",
                args.train_python,
            ],
        )
        prepared = json.loads((prepared_dir / "data_manifest.json").read_text())
        total_tokens = prepared["total_tokens"]
        tokens_per_step = args.global_batch_size * args.seq_length
        train_iters = 3 * ((total_tokens + tokens_per_step - 1) // tokens_per_step)
        receipt.update(
            total_tokens=total_tokens,
            train_iters=train_iters,
            data_sha256=prepared["datasets"][0]["sha256"],
        )
        environment["AUTO_SFT_SFT_DATASET_PATH"] = prepared["dataset_path"]
        run_stage(
            "initialize",
            [
                args.convert_python,
                ROOT / "src/autosft/backends/megatron/import_model.py",
                "--model",
                args.model,
                "--output",
                args.output / "base_checkpoint",
            ],
        )
        run_stage(
            "train",
            [
                "bash",
                ROOT / "scripts/local/train_megatron.sh",
                train_iters,
                args.seq_length,
            ],
        )
        hf_model = args.output / "hf_model"
        export_command = [
            args.convert_python,
            ROOT / "src/autosft/backends/megatron/convert.py",
            "--checkpoint",
            args.output / "sft/checkpoints",
            "--ckpt-step",
            train_iters,
            "--output",
            hf_model,
            "--hf-template",
            args.model,
        ]
        if args._convert_python_override:
            export_command.extend(("--python", args.convert_python))
        run_stage("export", export_command)
        for benchmark in args.benchmarks:
            output = args.output / "eval" / benchmark
            record = dict(
                benchmark=benchmark,
                model_path=str(hf_model),
                limit=args.eval_limit or -1,
                repeats=args.eval_repeats,
                output_dir=str(output),
                eval_backend="opencompass",
                run_name=args.output.name,
                log_path=str(logs / f"eval_{benchmark}.log"),
            )
            eval_python = (
                environment["AUTO_SFT_CODE_EVAL_PYTHON"]
                if benchmark in {"humaneval_plus", "mbpp_plus"}
                else args.eval_python
            )
            run_stage(
                f"eval_{benchmark}",
                [
                    eval_python,
                    ROOT / "src/autosft/backends/local/evaluation/dispatch.py",
                    "--record",
                    json.dumps(record),
                ],
            )
            aggregate = aggregate_benchmark(output, args.eval_repeats)
            write_json(output / "aggregate.json", aggregate)
            receipt["scores"][benchmark] = {
                "mean": aggregate["primary_score_mean"],
                "repeats": aggregate["primary_scores"],
            }
        receipt.update(status="complete", finished_at=utc_now())
        write_json(args.output / "round.json", receipt)
        print(
            json.dumps({"status": receipt["status"], "scores": receipt["scores"]}),
            flush=True,
        )
    except BaseException as error:
        receipt.update(status="failed", error=str(error), finished_at=utc_now())
        for stage in receipt["stages"].values():
            if stage["status"] == "running":
                stage.update(status="failed", finished_at=utc_now())
        write_json(args.output / "round.json", receipt)
        raise


if __name__ == "__main__":
    run_round(parse_args())
