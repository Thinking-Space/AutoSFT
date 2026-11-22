"""Convert and evaluate one completed SFT round's final checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from autosft.core.common import REPO_ROOT
from autosft.core.round_ids import storage_round_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", type=Path, help="The round's Megatron checkpoints directory"
    )
    args = parser.parse_args(argv)
    if not os.environ.get("AUTO_SFT_RUN_DIR"):
        parser.error("AUTO_SFT_RUN_DIR is required")
    checkpoint = args.checkpoints.resolve()
    if not checkpoint.is_dir():
        parser.error(f"checkpoints dir not found: {checkpoint}")
    try:
        round_name = storage_round_id(checkpoint.parent.name)
    except ValueError:
        parser.error(
            "checkpoints must live under sft/round_<positive integer>/checkpoints"
        )
    marker = checkpoint / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        parser.error(f"no latest_checkpointed_iteration.txt under {checkpoint}")
    step = marker.read_text().strip()
    if not re.fullmatch(r"[1-9][0-9]*", step):
        parser.error(f"unsupported latest checkpoint marker: {step}")
    dry_run = os.environ.get("AUTO_SFT_LOCAL_DRY_RUN") == "1"
    if not dry_run and not (checkpoint / f"iter_{int(step):07d}").is_dir():
        parser.error(f"final checkpoint directory is missing: iter_{int(step):07d}")
    repeats = os.environ.get("AUTO_SFT_EVAL_REPEATS", "1")
    if not re.fullmatch(r"[1-9][0-9]*", repeats):
        parser.error("AUTO_SFT_EVAL_REPEATS must be a positive integer")
    run = Path(os.environ["AUTO_SFT_RUN_DIR"]).resolve()
    model = run / "ckpt_eval" / round_name / f"step_{step}" / "hf_model"
    template = (
        os.environ.get("AUTO_SFT_SFT_TOKENIZER_MODEL")
        or os.environ.get("AUTO_SFT_BASE_MODEL_PATH")
        or str(REPO_ROOT / "models/Qwen3-1.7B-Base")
    )
    benchmarks = (
        os.environ.get("AUTO_SFT_BENCHMARKS")
        or os.environ.get("AUTO_SFT_EVAL_BENCHMARK")
        or os.environ.get("AUTO_SFT_BENCHMARK")
        or "ifeval,ifbench"
    )
    backend = os.environ.get("AUTO_SFT_EVAL_BACKEND", "opencompass")
    environment = dict(os.environ)
    environment["AUTO_SFT_EVAL_TP_SIZE"] = environment.get("AUTO_SFT_EVAL_TP_SIZE", "1")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    convert = [
        os.environ.get("AUTO_SFT_CONVERT_PYTHON") or sys.executable,
        "-m",
        "autosft.backends.megatron.convert",
        "--checkpoint",
        str(checkpoint),
        "--ckpt-step",
        step,
        "--output",
        str(model),
        "--hf-template",
        template,
    ]
    evaluate = [
        os.environ.get("AUTO_SFT_EVAL_PYTHON") or sys.executable,
        "-m",
        "autosft.benchmarks.submit",
        "--model-path",
        str(model),
        "--benchmark",
        benchmarks,
        "--eval-backend",
        backend,
        "--run-name",
        round_name,
        "--repeats",
        repeats,
    ]
    if dry_run:
        convert.append("--dry-run")
        evaluate.append("--dry-run")
    print(f"Round checkpoints dir : {checkpoint}", flush=True)
    print(f"Final step            : {step}", flush=True)
    print(f"Benchmarks            : {benchmarks}", flush=True)
    print(f"Repeats per benchmark : {repeats}", flush=True)
    print(f"Eval run name         : {round_name}", flush=True)
    for command in (convert, evaluate):
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment)
        if result.returncode:
            return result.returncode
    print(f"Eval run dir: {run / 'eval_runs' / round_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
