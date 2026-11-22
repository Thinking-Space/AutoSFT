from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_structured(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to read recipe YAML files"
                ) from exc
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                # yaml.YAMLError derives from Exception, not ValueError, so
                # without this it escapes every `except ValueError` reader as
                # a traceback instead of a rejection with a fix path.
                raise ValueError(f"{path} is not parseable YAML: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return data
    raise ValueError(f"unsupported structured file type: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print("+ " + " ".join(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        env=merged_env,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
