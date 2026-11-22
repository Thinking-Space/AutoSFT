"""Read-only checks for public source assets and optional local runtimes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
CORE_FILES = (
    "src/autosft/data/training_data.py",
    "src/autosft/backends/megatron/data.py",
    "src/autosft/data/training_data_audit.py",
    "src/autosft/backends/megatron/model_profiles.py",
    "src/autosft/data/preparation.py",
    "src/autosft/backends/megatron/train.py",
    "src/autosft/backends/megatron/convert.py",
    "src/autosft/benchmarks/evaluate_round.py",
    "src/autosft/backends/local/lifecycle.py",
    "src/autosft/backends/local/collection.py",
    "src/autosft/orchestration/standalone.py",
    "scripts/local/train_megatron.sh",
    "src/autosft/backends/megatron/import_model.py",
    "src/autosft/backends/megatron/checkpoint/dist_ckpt_to_hf_qwen3_cpu.py",
    "src/autosft/backends/megatron/sft.py",
    "scripts/setup/fetch_megatron.py",
    "src/autosft/data/prepare_tokens.py",
    "src/autosft/backends/local/evaluation/dispatch.py",
    "src/autosft/backends/local/evaluation/code_eval.py",
    "src/autosft/backends/local/evaluation/instruction_following.py",
    "src/autosft/backends/local/evaluation/sglang_generator.py",
    "src/autosft/orchestration/round_gates.py",
    "src/autosft/synthesis/usage.py",
    "src/autosft/benchmarks/evalplus.py",
    "src/autosft/benchmarks/evalplus_data.py",
    "src/autosft/benchmarks/opencompass_bridge.py",
    "third_party/opencompass_local/IFEval/ifeval.py",
    "third_party/opencompass_local/IFBench/ifbench.py",
    "third_party/instruction_following_eval/instruction_following_eval/evaluation_lib.py",
    "third_party/ifbench/evaluation_lib.py",
    "assets/benchmarks/ifeval/input_data.jsonl",
    "assets/benchmarks/ifbench/input_data.jsonl",
)
AGENT_FILES = (
    "src/autosft/agents/runner.py",
    "src/autosft/agents/roles.py",
    "src/autosft/agents/role_pump.py",
    "src/autosft/agents/executor_handoff.py",
    "src/autosft/backends/local/submission.py",
    "src/autosft/backends/local/tasks.py",
    "scripts/agents/launch.sh",
    "src/autosft/agents/prompts/meta.md",
    "src/autosft/agents/prompts/meta_executor.md",
    "src/autosft/agents/prompts/meta_analysis.md",
    "src/autosft/agents/prompts/meta_synthesis.md",
    "src/autosft/agents/prompts/executor.md",
    "src/autosft/agents/prompts/analysis.md",
    "src/autosft/agents/prompts/task_report_contract.md",
    "skills/analysis/training-data/SKILL.md",
    "skills/analysis/failure-case/SKILL.md",
    "skills/analysis/learning-dynamics/SKILL.md",
    "skills/analysis/history/SKILL.md",
    "skills/shared/row-analysis/SKILL.md",
    "skills/shared/pool-rewrite/SKILL.md",
    "skills/meta/recipe-authoring/SKILL.md",
    "skills/shared/batch-inference/references/recorded-calls.md",
    "src/autosft/backends/providers/batch.py",
    "scripts/agents/run_codex.sh",
    "scripts/agents/run_role.sh",
)
SYNTHESIS_FILES = (
    "src/autosft/synthesis/job.py",
    "src/autosft/synthesis/api.py",
    "src/autosft/synthesis/_internal/runtime.py",
    "src/autosft/synthesis/_internal/batch_wait.py",
    "src/autosft/synthesis/_internal/config.py",
    "src/autosft/synthesis/_internal/prompts/synthesis_worker.md",
    "skills/meta/synthesis-handbook/SKILL.md",
    "skills/synthesis/batch-generation/SKILL.md",
    "skills/shared/batch-inference/references/recorded-calls.md",
)
RUNTIME_MODULES = {
    "training": (
        "numpy",
        "torch",
        "transformers",
        "transformer_engine.pytorch",
        "flash_attn",
        "safetensors",
        "yaml",
        "autosft.backends.megatron.data",
        "megatron.core",
    ),
    "conversion": ("numpy", "torch", "safetensors", "torch.distributed.checkpoint"),
    "evaluation": (
        "vllm",
        "transformers",
        "opencompass_local.IFEval.ifeval",
        "opencompass_local.IFBench.ifbench",
    ),
}
TOKENIZER_PROBE = "import transformers, json; print(json.dumps({'ok': True}))"


def asset_checks(
    root: Path = ROOT, *, agents: bool = False, synthesis: bool = False
) -> list[dict]:
    """Validate the checkout without importing GPU frameworks or requiring weights."""
    checks = []
    groups = {"core": CORE_FILES}
    if agents:
        groups["agents"] = AGENT_FILES
    if synthesis:
        groups["synthesis"] = SYNTHESIS_FILES
    for group, paths in groups.items():
        missing = [path for path in paths if not (root / path).is_file()]
        checks.append(
            {"check": f"assets.{group}", "ok": not missing, "missing": missing}
        )
    data_dir = Path(
        os.environ.get("AUTO_SFT_EVALPLUS_DATA_DIR")
        or root / "assets/benchmarks/evalplus"
    )
    try:
        from autosft.benchmarks.evalplus_data import load_staged_rows

        counts = {
            name: len(load_staged_rows(data_dir, name))
            for name in ("humaneval_plus", "mbpp_plus")
        }
        checks.append({"check": "assets.evalplus", "ok": True, "counts": counts})
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        checks.append({"check": "assets.evalplus", "ok": False, "error": str(exc)})
    return checks


_RUNTIME_PROBE = r"""
import importlib, importlib.metadata, json, sys
stage, modules = sys.argv[1], json.loads(sys.argv[2])
errors = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append({"module": name, "error": str(exc)})
if not errors and stage in ("training", "conversion"):
    try:
        import numpy as np, torch
        source = np.array([1., 2.], dtype=np.float32)
        assert np.array_equal(torch.from_numpy(source).numpy(), source)
    except Exception as exc:
        errors.append({"module": "numpy_torch_interop", "error": str(exc)})
if not errors and stage == "evaluation":
    # vLLM lazily loads its compiled CUDA extension.  Importing only the
    # top-level package can therefore report success even when the symbols
    # used by the local evaluator are ABI-incompatible with PyTorch.
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as exc:
        errors.append({"module": "vllm.LLM", "error": str(exc)})
if not errors and stage == "code_evaluation":
    version = importlib.metadata.version("evalplus")
    if version != "0.3.1":
        errors.append({"module": "evalplus", "error": "requires 0.3.1, found " + version})
print(json.dumps({"ok": not errors, "errors": errors}))
raise SystemExit(bool(errors))
"""
_GPU_PROBE = r"""
import json, torch
count = torch.cuda.device_count()
available = torch.cuda.is_available() and count > 0
print(json.dumps({"ok": available, "device_count": count,
                  "devices": [torch.cuda.get_device_name(i) for i in range(count)],
                  "error": None if available else "no CUDA device is available"}))
raise SystemExit(not available)
"""


def probe(
    python: str,
    script: str,
    *arguments: str,
    root: Path = ROOT,
    extra_pythonpath: tuple[Path, ...] = (),
    env: dict[str, str] | None = None,
) -> dict:
    base_environment = dict(os.environ if env is None else env)
    python_paths = [
        str(root / "src"),
        str(root / "third_party"),
        *(str(path) for path in extra_pythonpath),
    ]
    if base_environment.get("PYTHONPATH"):
        python_paths.append(base_environment["PYTHONPATH"])
    env = {
        **base_environment,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(python_paths),
    }
    try:
        completed = subprocess.run(
            [python, "-B", "-c", script, *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("ok"), bool):
                return {
                    **payload,
                    "ok": payload["ok"] and completed.returncode == 0,
                    "python": python,
                    "exit_code": completed.returncode,
                }
        return {
            "ok": False,
            "python": python,
            "exit_code": completed.returncode,
            "error": (
                completed.stderr or completed.stdout or "probe produced no report"
            )[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "python": python, "error": str(exc)}


def preflight_evaluation_runtime(
    python: str, *, env: dict[str, str] | None = None
) -> None:
    """Fail before a workload when vLLM cannot load in the eval interpreter."""
    result = probe(
        python,
        _RUNTIME_PROBE,
        "evaluation",
        json.dumps(RUNTIME_MODULES["evaluation"]),
        # Preserve the stage environment (offline flags, repository paths and
        # any caller-selected interpreter settings) used by the workload.
        root=ROOT,
        env=env,
    )
    if not result.get("ok"):
        detail = result.get("error") or json.dumps(result.get("errors", []))
        raise ValueError(
            f"evaluation runtime preflight failed with {python}: {detail}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents", action="store_true", help="Also require local agent workflow files"
    )
    parser.add_argument(
        "--synthesis", action="store_true", help="Also require optional synthesis files"
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Import dependencies in their selected local interpreters",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Check visible CUDA devices without loading model weights",
    )
    parser.add_argument(
        "--model", type=Path, help="Validate a local Qwen3-1.7B-Base HF template"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Validate the matching Megatron base checkpoint; requires --model",
    )
    parser.add_argument(
        "--train-python",
        default=os.environ.get("AUTO_SFT_TRAIN_PYTHON") or sys.executable,
    )
    parser.add_argument(
        "--eval-python",
        default=os.environ.get("AUTO_SFT_EVAL_PYTHON") or sys.executable,
    )
    parser.add_argument(
        "--code-eval-python", default=os.environ.get("AUTO_SFT_CODE_EVAL_PYTHON")
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("ifeval", "ifbench", "humaneval_plus", "mbpp_plus"),
        default=[
            name.strip()
            for name in os.environ.get("AUTO_SFT_BENCHMARKS", "ifeval").split(",")
            if name.strip()
        ],
    )
    parser.add_argument(
        "--convert-python",
        default=(
            os.environ.get("AUTO_SFT_CONVERT_PYTHON")
            or os.environ.get("AUTO_SFT_TRAIN_PYTHON")
            or sys.executable
        ),
    )
    parser.add_argument(
        "--tokenizer-python",
        default=(
            os.environ.get("AUTO_SFT_TOKENIZER_PYTHON")
            or os.environ.get("AUTO_SFT_TRAIN_PYTHON")
            or sys.executable
        ),
    )
    args = parser.parse_args(argv)
    checks = asset_checks(agents=args.agents, synthesis=args.synthesis)
    if args.checkpoint and not args.model:
        checks.append(
            {"check": "model", "ok": False, "error": "--checkpoint requires --model"}
        )
    elif args.model:
        try:
            from autosft.backends.megatron.convert import validate_hf_template
            from autosft.backends.megatron.model_profiles import (
                get_model_profile,
                validate_model_assets,
            )

            profile = get_model_profile()
            validate_hf_template(args.model, profile)
            if args.checkpoint:
                validate_model_assets(
                    profile, tokenizer_model=args.model, checkpoint_path=args.checkpoint
                )
            checks.append({"check": "model", "ok": True, "model_id": profile.model_id})
        except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
            checks.append({"check": "model", "ok": False, "error": str(exc)})
    if args.runtime:
        stages = [("training", args.train_python), ("conversion", args.convert_python)]
        if set(args.benchmarks) & {"ifeval", "ifbench"}:
            stages.append(("evaluation", args.eval_python))
        if set(args.benchmarks) & {"humaneval_plus", "mbpp_plus"}:
            stages.append(
                ("code_evaluation", args.code_eval_python or args.eval_python)
            )
        for stage, python in stages:
            modules = RUNTIME_MODULES.get(stage)
            if stage == "code_evaluation":
                backend = os.environ.get("AUTO_SFT_CODE_INFERENCE_BACKEND", "sglang")
                if backend not in {"sglang", "vllm"}:
                    checks.append(
                        {
                            "check": "runtime.code_evaluation",
                            "ok": False,
                            "error": "AUTO_SFT_CODE_INFERENCE_BACKEND must be sglang or vllm",
                        }
                    )
                    continue
                modules = (backend, "transformers", "evalplus")
            extra = (
                (
                    Path(
                        os.environ.get("AUTO_SFT_MEGATRON_DIR")
                        or ROOT / "third_party/megatron"
                    ),
                )
                if stage == "training"
                else ()
            )
            checks.append(
                {
                    "check": f"runtime.{stage}",
                    **probe(
                        python,
                        _RUNTIME_PROBE,
                        stage,
                        json.dumps(modules),
                        extra_pythonpath=extra,
                    ),
                }
            )
        if args.agents:
            checks.append(
                {
                    "check": "runtime.agents",
                    **probe(
                        sys.executable,
                        _RUNTIME_PROBE,
                        "agents",
                        json.dumps(
                            (
                                "yaml",
                                "httpx",
                                "aiohttp",
                                "tensorboard.compat.proto.event_pb2",
                            )
                        ),
                    ),
                }
            )
        if args.synthesis:
            checks.append(
                {
                    "check": "runtime.synthesis",
                    **probe(
                        sys.executable,
                        _RUNTIME_PROBE,
                        "synthesis",
                        json.dumps(("yaml", "httpx", "aiohttp")),
                    ),
                }
            )
        checks.append(
            {
                "check": "runtime.tokenizer",
                **probe(
                    args.tokenizer_python,
                    TOKENIZER_PROBE,
                    extra_pythonpath=(ROOT / "src/autosft/data",),
                ),
            }
        )
    if args.gpu:
        checks.append({"check": "gpu", **probe(args.train_python, _GPU_PROBE)})
    ok = all(check["ok"] for check in checks)
    print(
        json.dumps(
            {
                "status": "passed" if ok else "failed",
                "checks": checks,
                "training_executed": False,
                "evaluation_executed": False,
                "scope": "asset validation and explicitly requested runtime/device checks",
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
