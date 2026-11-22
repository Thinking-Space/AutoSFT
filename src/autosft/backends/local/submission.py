"""Launch the fixed SFT/eval entrypoints as durable local subprocesses."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from autosft.core.common import REPO_ROOT
from autosft.backends.local.tasks import submit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["sft", "eval"])
    p.add_argument("run_name")
    p.add_argument("value")
    p.add_argument("extra")
    a = p.parse_args()
    from autosft.core.round_ids import storage_round_id

    if a.kind == "sft":
        try:
            storage_round_id(a.run_name)
        except ValueError as exc:
            p.error(str(exc))
    run = Path(os.environ["AUTO_SFT_RUN_DIR"]).resolve()
    dry = os.environ.get("AUTO_SFT_LOCAL_DRY_RUN") == "1"
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(REPO_ROOT / "third_party")
            + ":"
            + str(REPO_ROOT)
            + ":"
            + os.environ.get("PYTHONPATH", ""),
        }
    )
    if a.kind == "sft":
        from autosft.backends.megatron.model_profiles import get_model_profile

        try:
            profile = get_model_profile()
        except ValueError as exc:
            p.error(str(exc))
        environment["AUTO_SFT_SFT_MODEL"] = profile.model_id
        train_iters, seq = int(a.value), int(a.extra)
        if train_iters < 1 or seq < 1:
            raise SystemExit("train iterations/sequence length must be positive")
        output = Path(os.environ["AUTO_SFT_SFT_OUTPUT_DIR"]).resolve()
        record = dict(
            run_name=a.run_name,
            output_dir=str(output),
            dataset_path=os.environ["AUTO_SFT_SFT_DATASET_PATH"],
            train_iters=train_iters,
            seq_length=seq,
            base_model=profile.model_id,
        )
        manifest = run / "sft_submissions.jsonl"
        if output != run / "sft" / a.run_name:
            raise SystemExit("non-canonical SFT output")
        command = [
            "bash",
            str(REPO_ROOT / "scripts/local/train_megatron.sh"),
            str(train_iters),
            str(seq),
        ]
        if not dry:
            for key in ["AUTO_SFT_SFT_TOKENIZER_MODEL", "AUTO_SFT_SFT_LOAD_CKPT"]:
                if not Path(os.environ[key]).is_dir():
                    raise SystemExit(
                        f"missing local model asset: {key}={os.environ[key]}"
                    )
    else:
        backend = os.environ.get("AUTO_SFT_EVAL_BACKEND", "opencompass")
        if int(os.environ.get("AUTO_SFT_EVAL_TP_SIZE", "1")) < 1:
            p.error("eval tensor parallel size must be positive")
        if backend != "opencompass":
            raise SystemExit("unsupported local evaluation adapter; use opencompass")
        from autosft.benchmarks.catalog import supported_benchmarks

        if a.extra not in supported_benchmarks():
            p.error(
                f"unsupported benchmark: {a.extra}; supported: "
                + ", ".join(supported_benchmarks())
            )
        model = Path(a.value).resolve()
        if not dry and not model.is_dir():
            raise SystemExit(f"model path missing: {model}")
        output = Path(os.environ["AUTO_SFT_EVAL_OUTPUT_DIR"]).resolve()
        manifest = Path(os.environ["AUTO_SFT_EVAL_MANIFEST"]).resolve()
        if (
            manifest.parent.parent != run / "eval_runs"
            or manifest.name != "eval_submissions.jsonl"
        ):
            raise SystemExit("non-canonical eval manifest")
        if output != manifest.parent / "eval" / a.extra:
            raise SystemExit("non-canonical eval output")
        repeats = int(os.environ.get("AUTO_SFT_EVAL_REPEATS", "1"))
        if repeats < 1:
            raise SystemExit("repeats must be positive")
        limit = int(os.environ.get("AUTO_SFT_EVAL_LIMIT", "-1"))
        if limit != -1 and limit < 1:
            p.error("eval limit must be -1 or a positive integer")
        record = dict(
            run_name=a.run_name,
            output_dir=str(output),
            model_path=str(model),
            model_identity=os.environ["AUTO_SFT_EVAL_MODEL_IDENTITY"],
            benchmark=a.extra,
            eval_backend=backend,
            limit=limit,
            repeats=repeats,
        )
        eval_python = os.environ.get("AUTO_SFT_EVAL_PYTHON") or sys.executable
        if a.extra in {"humaneval_plus", "mbpp_plus"}:
            eval_python = os.environ.get("AUTO_SFT_CODE_EVAL_PYTHON") or eval_python
        command = [
            eval_python,
            str(REPO_ROOT / "src/autosft/backends/local/evaluation/dispatch.py"),
            "--record",
            json.dumps(record),
        ]
        # Status is checked, not inferred from a checkpoint marker alone.
        from autosft.orchestration.submission_registry import submission_records
        from autosft.backends.local.tasks import status, TERMINAL

        for sft in submission_records(run / "sft_submissions.jsonl"):
            if status(run, sft["task"])["status"] not in TERMINAL:
                raise SystemExit("cannot evaluate while local SFT is queued or running")
        if not dry:
            from autosft.benchmarks.catalog import preflight_code_runtime

            preflight_code_runtime(
                [a.extra],
                python=command[0],
                model_path=model,
                options=record,
                env=environment,
            )
    print(
        json.dumps(
            submit(
                run,
                manifest,
                record,
                command,
                cwd=REPO_ROOT,
                env=environment,
                dry_run=dry,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
