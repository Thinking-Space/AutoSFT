#!/usr/bin/env python3
"""Stage the source datasets directory into a run's open data pool.

One staging per experiment run: the launcher copies the entire source
datasets tree (arbitrary files and subdirectories, in any format) into
``<run_dir>/data_pool`` and writes ``staging_manifest.json`` describing the
frozen baseline: per-file bytes and sha256, non-blank line counts for
``*.jsonl`` files, and their totals for accounting. The staged
copy is fully writable by the agent -- it is a workspace, not an archive --
while the source directory outside the run is never written.

The copy lands in a temporary sibling directory first and is renamed into
place, so a crash mid-copy never leaves a half-staged pool that looks
complete. Staging is idempotent: an existing ``data_pool/staging_manifest.json``
means the run already has its baseline and the call is a no-op (recovery
path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from autosft.core.common import utc_now, write_json


POOL_DIR_NAME = "data_pool"
STAGING_MANIFEST_NAME = "staging_manifest.json"
SCHEMA_VERSION = "1.0"


class PoolStagingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> int:
    """Count non-blank lines; no JSON parsing (the format is not trusted)."""

    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                rows += 1
    return rows


def _make_writable(root: Path) -> None:
    """Do not carry read-only source modes into the agent's workspace."""

    for path in sorted(root.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _build_manifest(pool_dir: Path, *, source_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    known_rows_total = 0
    unknown_format_files = 0
    for path in sorted(pool_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(pool_dir).as_posix()
        if relative == STAGING_MANIFEST_NAME:
            continue
        entry: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".jsonl":
            rows = _jsonl_rows(path)
            entry["jsonl_rows"] = rows
            known_rows_total += rows
        else:
            entry["jsonl_rows"] = None
            unknown_format_files += 1
        files.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "staged_at": utc_now(),
        "source_dir": str(source_dir),
        "files": files,
        "known_rows_total": known_rows_total,
        "unknown_format_files": unknown_format_files,
    }


def stage_pool(
    *,
    source_dir: Path,
    run_dir: Path,
    pool_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the source datasets tree into the run and write its manifest."""

    source_dir = Path(source_dir).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    pool_dir = (
        Path(pool_dir).expanduser().resolve()
        if pool_dir is not None
        else run_dir / POOL_DIR_NAME
    )
    if not source_dir.is_dir():
        raise PoolStagingError(f"source datasets dir not found: {source_dir}")
    if source_dir == pool_dir:
        raise PoolStagingError("staging destination equals the source dir")

    manifest_path = pool_dir / STAGING_MANIFEST_NAME
    if manifest_path.is_file():
        print(
            f"[pool_staging] {manifest_path} already exists; "
            "staging skipped (recovery path).",
            file=sys.stderr,
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PoolStagingError(
                f"existing staging manifest is not valid JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise PoolStagingError(
                f"existing staging manifest must be an object: {manifest_path}"
            )
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage into a temporary sibling and rename into place so an interrupted
    # copy never leaves a directory that looks like a finished pool. The
    # manifest is written inside the temp tree, so its presence in data_pool/
    # always implies a complete copy. Sibling placement keeps the rename on
    # one filesystem.
    temp_root = Path(tempfile.mkdtemp(prefix=f".{pool_dir.name}.", dir=pool_dir.parent))
    try:
        staged = temp_root / POOL_DIR_NAME
        shutil.copytree(source_dir, staged)
        _make_writable(staged)
        manifest = _build_manifest(staged, source_dir=source_dir)
        write_json(staged / STAGING_MANIFEST_NAME, manifest)
        if pool_dir.exists():
            # A partial pool dir without a manifest is an aborted earlier
            # attempt; replace it with the complete copy.
            shutil.rmtree(pool_dir)
        os.replace(staged, pool_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pool-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        manifest = stage_pool(
            source_dir=args.source_dir,
            run_dir=args.run_dir,
            pool_dir=args.pool_dir,
        )
    except PoolStagingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
