#!/usr/bin/env python3
"""Resolve and validate the Python interpreter used for checkpoint conversion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_CONVERT_PYTHON = REPO_ROOT / "scripts/setup" / ".convert_venv" / "bin" / "python"
BOOTSTRAP_COMMAND = "bash scripts/setup/bootstrap_convert_env.sh"

_COMPATIBILITY_CHECK = r"""
import json
import sys


def fail(message):
    print(json.dumps({"conversion_compatibility_error": message}))
    raise SystemExit(1)


try:
    import numpy
except Exception as exc:
    fail(f"cannot import NumPy: {exc}")

try:
    numpy_major = int(numpy.__version__.split(".", 1)[0])
except (TypeError, ValueError):
    fail(f"cannot parse NumPy version: {numpy.__version__!r}")
if numpy_major != 1:
    fail(
        f"NumPy {numpy.__version__} is incompatible; "
        "checkpoint conversion requires NumPy 1.x"
    )

try:
    import torch
except Exception as exc:
    fail(f"cannot import PyTorch: {exc}")

try:
    source = numpy.array([[1.25, -2.5], [0.0, 4.0]], dtype=numpy.float32)
    tensor = torch.from_numpy(source)
    if tuple(tensor.shape) != source.shape or tensor.dtype != torch.float32:
        fail("NumPy-to-PyTorch check returned an invalid shape or dtype")
    if tensor.tolist() != source.tolist():
        fail("NumPy-to-PyTorch check changed array values")
except Exception as exc:
    fail(f"NumPy-to-PyTorch interoperability check failed: {exc}")

try:
    restored = tensor.numpy()
    if restored.dtype != source.dtype or not numpy.array_equal(restored, source):
        fail("PyTorch-to-NumPy check changed array values, shape, or dtype")
except Exception as exc:
    fail(f"PyTorch-to-NumPy interoperability check failed: {exc}")

print(f"numpy={numpy.__version__} torch={torch.__version__}")
"""


class ConversionEnvironmentError(RuntimeError):
    """The checkpoint conversion interpreter is missing or incompatible."""


def conversion_subprocess_env(
    *,
    megatron_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the same isolated backend settings for probing and conversion."""
    result = dict(os.environ if env is None else env)
    result.update(
        {
            "PYTHONNOUSERSITE": "1",
            "USE_TORCH": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
        }
    )
    if megatron_dir is not None:
        pythonpath = result.get("PYTHONPATH")
        paths = [str(Path(megatron_dir).expanduser().resolve()), str(REPO_ROOT / "src")]
        if pythonpath:
            paths.append(pythonpath)
        result["PYTHONPATH"] = os.pathsep.join(paths)
    return result


def _resolve_executable(python_bin: str | Path) -> str:
    raw = os.path.expanduser(str(python_bin).strip())
    if not raw:
        raise ConversionEnvironmentError("conversion Python path is empty")

    if os.sep in raw or (os.altsep and os.altsep in raw):
        # abspath intentionally preserves the final venv symlink. Resolving that
        # symlink would bypass pyvenv.cfg and select the host site-packages.
        executable = os.path.abspath(raw)
    else:
        executable = shutil.which(raw) or raw
    if not Path(executable).is_file():
        raise ConversionEnvironmentError(
            f"conversion Python does not exist: {executable}"
        )
    if not os.access(executable, os.X_OK):
        raise ConversionEnvironmentError(
            f"conversion Python is not executable: {executable}"
        )
    return executable


def check_convert_python(
    python_bin: str | Path,
    *,
    timeout_seconds: float = 60,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str:
    """Validate NumPy 1.x and bidirectional interop in the execution environment."""
    executable = _resolve_executable(python_bin)
    probe_env = conversion_subprocess_env(env=env)
    try:
        result = subprocess.run(
            [executable, "-c", _COMPATIBILITY_CHECK],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=probe_env,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionEnvironmentError(
            f"failed to inspect conversion Python {executable}: "
            f"compatibility check timed out after {timeout_seconds:g}s"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionEnvironmentError(
            f"failed to inspect conversion Python {executable}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = _compatibility_failure_detail(result)
        raise ConversionEnvironmentError(
            f"conversion Python is incompatible: {executable}: {detail}"
        )
    return executable


def _compatibility_failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    # Native imports can emit long ABI tracebacks before the controlled error.
    # Prefer the probe's structured diagnostic over those incidental logs.
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("conversion_compatibility_error"), str
        ):
            return " ".join(payload["conversion_compatibility_error"].splitlines())[
                :500
            ]
    lines = (result.stderr or result.stdout).strip().splitlines()
    detail = lines[-1] if lines else f"compatibility check exited {result.returncode}"
    return detail[:500]


def resolve_convert_python(
    *,
    check_compatibility: bool = True,
    shared_python: str | Path | None = None,
    current_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str:
    """Resolve an override or the first usable current interpreter."""
    configured = os.environ.get("AUTO_SFT_CONVERT_PYTHON", "").strip()
    if configured:
        if not check_compatibility:
            return configured
        try:
            return check_convert_python(configured, env=env, cwd=cwd)
        except ConversionEnvironmentError as exc:
            raise ConversionEnvironmentError(
                f"AUTO_SFT_CONVERT_PYTHON is unusable: {exc}\n"
                f"Rebuild the local environment with: {BOOTSTRAP_COMMAND}"
            ) from exc

    # The launch shell runs under the controller interpreter, while training
    # may intentionally be pinned to AUTO_SFT_TRAIN_PYTHON.  Consider that
    # stage override before the controller Python so launch-time validation
    # matches the interpreter that will actually execute conversion.
    candidates = (
        current_python
        or os.environ.get("AUTO_SFT_TRAIN_PYTHON")
        or sys.executable,
        shared_python or SHARED_CONVERT_PYTHON,
    )
    if not check_compatibility:
        for candidate in candidates:
            try:
                return _resolve_executable(candidate)
            except ConversionEnvironmentError:
                continue
        return str(current_python or sys.executable)

    failures: list[str] = []
    for candidate in candidates:
        try:
            return check_convert_python(candidate, env=env, cwd=cwd)
        except ConversionEnvironmentError as exc:
            failures.append(str(exc))
    details = "\n  - ".join(failures)
    raise ConversionEnvironmentError(
        "no compatible checkpoint conversion Python was found:\n"
        f"  - {details}\n"
        f"Rebuild the local environment with: {BOOTSTRAP_COMMAND}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve or check the checkpoint conversion Python environment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "resolve",
        help="print the first compatible configured/default interpreter",
    )
    check_parser = subparsers.add_parser(
        "check",
        help="validate one interpreter and print its executable path",
    )
    check_parser.add_argument("python")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "check":
            python_bin = check_convert_python(args.python)
        else:
            python_bin = resolve_convert_python()
    except ConversionEnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(python_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
