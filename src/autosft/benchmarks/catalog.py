"""Discover benchmarks from local checkers, data and runtime entrypoints.

Reject unavailable targets before a run is created. Support is derived from
the complete execution path, rather than a benchmark name alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
# Keep direct ``python src/autosft/benchmarks/catalog.py`` invocation equivalent
# to the module form used by the launchers.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
CODE_BENCHMARKS = frozenset({"humaneval_plus", "mbpp_plus"})


def _evalplus_data_dir() -> Path:
    """Return the offline EvalPlus staging directory selected by the launcher."""
    # Keep the launcher and the evaluator on exactly the same empty-variable
    # semantics.  ``dataset_root`` treats an unset or empty override as the
    # repository's checked-in public-data directory.
    from autosft.benchmarks.evalplus_data import dataset_root

    return dataset_root()


def _opencompass_dataset_map() -> dict[str, str]:
    """The bridge's benchmark -> OpenCompass dataset-config mapping.

    Load the standard-library-only bridge without importing GPU backends.
    """
    bridge = REPO_ROOT / "src" / "autosft" / "benchmarks" / "opencompass_bridge.py"
    spec = importlib.util.spec_from_file_location("_autosft_opencompass_bridge", bridge)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - a broken bridge means no benchmarks
        return {}
    configs = getattr(module, "BENCHMARK_CONFIGS", {})
    if not isinstance(configs, dict):
        return {}
    # The bridge names the dataset family (e.g. "IFEval"); the submission
    # script needs the concrete config id it passes to OpenCompass.
    return {
        str(name): f"{str(entry[2])}_gen_353ae7"
        for name, entry in configs.items()
        if isinstance(entry, (tuple, list)) and len(entry) >= 3
    }


def _has_assets(benchmark: str) -> bool:
    """Whether every local asset this benchmark's evaluation needs exists."""
    if benchmark in CODE_BENCHMARKS:
        # Code benchmarks are scored by the public EvalPlus package and use a
        # staged, hashed release dataset.  Keep them out of the supported list
        # until the operator has prepared that offline directory.
        try:
            from autosft.benchmarks.evalplus_data import load_staged_rows

            load_staged_rows(_evalplus_data_dir(), benchmark)
            return (
                REPO_ROOT
                / "src"
                / "autosft"
                / "backends"
                / "local"
                / "evaluation"
                / "code_eval.py"
            ).is_file() and (
                REPO_ROOT / "src" / "autosft" / "benchmarks" / "evalplus.py"
            ).is_file()
        except (OSError, json.JSONDecodeError, ValueError, ImportError, AttributeError):
            return False
    family = {"ifeval": "IFEval", "ifbench": "IFBench"}.get(benchmark)
    if family is None:
        return False
    checker = (
        REPO_ROOT
        / "third_party"
        / "instruction_following_eval"
        / "instruction_following_eval"
        / "evaluation_lib.py"
        if benchmark == "ifeval"
        else REPO_ROOT / "third_party" / "ifbench" / "evaluation_lib.py"
    )
    return all(
        path.is_file()
        for path in (
            REPO_ROOT / "src/autosft/backends/local/evaluation/dispatch.py",
            REPO_ROOT
            / "third_party"
            / "opencompass_local"
            / family
            / f"{benchmark}.py",
            checker,
            REPO_ROOT / "assets" / "benchmarks" / benchmark / "input_data.jsonl",
        )
    )


def supported_benchmarks() -> list[str]:
    """Benchmark names this checkout can actually evaluate, sorted."""
    names = set(_opencompass_dataset_map()) | CODE_BENCHMARKS
    return sorted(name for name in names if _has_assets(name))


def opencompass_dataset(benchmark: str) -> str | None:
    """The OpenCompass dataset config id for a supported benchmark."""
    if benchmark in CODE_BENCHMARKS:
        return None
    if not _has_assets(benchmark):
        return None
    return _opencompass_dataset_map().get(benchmark)


def unsupported(names: list[str]) -> list[str]:
    """Which of ``names`` this checkout cannot evaluate, in input order."""
    available = set(supported_benchmarks())
    missing: list[str] = []
    for name in names:
        normalized = name.strip().lower()
        if normalized and normalized not in available and normalized not in missing:
            missing.append(normalized)
    return missing


def preflight_code_runtime(
    names: list[str],
    *,
    python: str | None = None,
    model_path: str | Path | None = None,
    options: dict | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Validate code evaluation in its selected interpreter before training."""
    environment = dict(os.environ if env is None else env)
    executable = (
        python
        or environment.get("AUTO_SFT_CODE_EVAL_PYTHON")
        or environment.get("AUTO_SFT_EVAL_PYTHON")
        or sys.executable
    )
    model = (
        model_path
        or environment.get("AUTO_SFT_SFT_TOKENIZER_MODEL")
        or environment.get(
            "AUTO_SFT_BASE_MODEL_PATH", str(REPO_ROOT / "models" / "Qwen3-1.7B-Base")
        )
    )
    for name in sorted({item.strip().lower() for item in names} & CODE_BENCHMARKS):
        record = {
            "benchmark": name,
            "model_path": str(model),
            "repeats": environment.get("AUTO_SFT_EVAL_REPEATS", "1"),
            "limit": environment.get("AUTO_SFT_EVAL_LIMIT", "-1"),
            **(options or {}),
        }
        try:
            result = subprocess.run(
                [
                    executable,
                    str(
                        REPO_ROOT / "src/autosft/backends/local/evaluation/code_eval.py"
                    ),
                    "--preflight",
                    "--record",
                    json.dumps(record),
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(
                f"{name} evaluation preflight could not run with {executable}: {exc}"
            ) from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise ValueError(
                f"{name} evaluation preflight failed with {executable}:\n{detail}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Print supported benchmarks, one per line.")
    check = commands.add_parser(
        "check", help="Exit nonzero when any named benchmark is unsupported."
    )
    check.add_argument("names", nargs="*", help="benchmark names to validate")
    check.add_argument(
        "--runtime",
        action="store_true",
        help="Also validate code evaluator dependencies and settings",
    )
    check.add_argument(
        "--python",
        help="Code interpreter (defaults to AUTO_SFT_CODE_EVAL_PYTHON, then AUTO_SFT_EVAL_PYTHON)",
    )
    check.add_argument(
        "--model",
        type=Path,
        help="Local initial or converted model for code prompt validation",
    )
    dataset = commands.add_parser(
        "dataset", help="Print one benchmark's OpenCompass dataset config id."
    )
    dataset.add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in supported_benchmarks():
            print(name)
        return 0
    if args.command == "dataset":
        config = opencompass_dataset(args.name.strip().lower())
        if config is None:
            print(
                f"ERROR: unsupported benchmark: {args.name}; this checkout "
                f"evaluates {', '.join(supported_benchmarks()) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        print(config)
        return 0

    missing = unsupported(args.names)
    if missing:
        print(
            f"ERROR: unsupported benchmark(s): {', '.join(missing)}\n"
            f"  This checkout evaluates: "
            f"{', '.join(supported_benchmarks()) or '(none)'}\n"
            "  IFEval/IFBench need their local scorer, checker, and input file; "
            "HumanEval+/MBPP+ additionally need a prepared offline EvalPlus "
            "manifest (src/autosft/benchmarks/prepare_evalplus.py).",
            file=sys.stderr,
        )
        return 2
    if args.runtime:
        try:
            preflight_code_runtime(
                args.names, python=args.python, model_path=args.model
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
