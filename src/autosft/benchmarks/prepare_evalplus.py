#!/usr/bin/env python3
"""Stage pinned public EvalPlus releases for offline local evaluation.

Download the release assets, or supply the same JSONL/JSONL.gz files explicitly
for an offline host. No evaluator installation or implicit package cache is used.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.request import urlopen

from autosft.core.common import REPO_ROOT as ROOT
from autosft.benchmarks.evalplus_data import (
    DATASET_SPECS,
    EVALPLUS_VERSION,
    load_staged_rows,
    validate_dataset_bytes,
)


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".prepare-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def stage(
    output_dir: Path,
    benchmarks: list[str],
    *,
    force: bool = False,
    sources: dict[str, Path] | None = None,
) -> dict[str, Any]:
    sources = sources or {}
    unknown = sorted((set(benchmarks) | set(sources)) - set(DATASET_SPECS))
    if unknown or not benchmarks:
        raise ValueError(
            "select one or more supported EvalPlus benchmarks: "
            + ", ".join(DATASET_SPECS)
        )
    if set(sources) - set(benchmarks):
        raise ValueError("each supplied source must be included in --benchmarks")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent staging commands must not overwrite each other's manifests.
    with (output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = output_dir / "manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "evaluator": f"evalplus=={EVALPLUS_VERSION}",
            "datasets": {},
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != "1.0"
                or manifest.get("evaluator") != f"evalplus=={EVALPLUS_VERSION}"
                or not isinstance(manifest.get("datasets"), dict)
                or set(manifest["datasets"]) - set(DATASET_SPECS)
            ):
                raise ValueError("existing EvalPlus manifest has an unsupported schema")
        datasets = manifest["datasets"]
        for name in datasets:
            if not (force and name in benchmarks):
                load_staged_rows(output_dir, name)
        pending: dict[str, bytes] = {}
        for benchmark in dict.fromkeys(benchmarks):
            if benchmark in datasets and not force:
                if benchmark in sources:
                    raise FileExistsError(
                        f"{benchmark} is already staged; use --force to replace it"
                    )
                continue
            spec = DATASET_SPECS[benchmark]
            source = sources.get(benchmark)
            if source is None:
                try:
                    with urlopen(spec["url"], timeout=60) as response:
                        raw_source = response.read()
                except OSError as exc:
                    raise RuntimeError(
                        f"could not download {spec['url']}; download this public release on a connected "
                        f"host and pass --{benchmark.split('_')[0]}-source /path/to/release.jsonl.gz"
                    ) from exc
            else:
                raw_source = Path(source).expanduser().read_bytes()
            raw = (
                gzip.decompress(raw_source)
                if raw_source.startswith(b"\x1f\x8b")
                else raw_source
            )
            validate_dataset_bytes(raw, benchmark)
            pending[benchmark] = raw
            datasets[benchmark] = {
                "file": spec["file"],
                "version": spec["version"],
                "count": spec["count"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_sha256": hashlib.sha256(raw_source).hexdigest(),
                "source_kind": "public_release" if source is None else "supplied_file",
                "release_url": spec["url"],
            }
        # Validate all inputs before touching staged data. Each file is replaced
        # atomically and the manifest is published last; readers reject any hash
        # mismatch if a process stops between those replacements.
        for benchmark, raw in pending.items():
            _atomic_write(output_dir / DATASET_SPECS[benchmark]["file"], raw)
        _atomic_write(
            manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        )
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", default="humaneval_plus,mbpp_plus")
    parser.add_argument(
        "--humaneval-source",
        type=Path,
        help="Offline HumanEval+ v0.1.10 release JSONL[.gz]",
    )
    parser.add_argument(
        "--mbpp-source", type=Path, help="Offline MBPP+ v0.2.0 release JSONL[.gz]"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace selected datasets; retain other staged benchmarks",
    )
    args = parser.parse_args()
    selected = [
        item.strip().lower() for item in args.benchmarks.split(",") if item.strip()
    ]
    sources = {
        name: path
        for name, path in (
            ("humaneval_plus", args.humaneval_source),
            ("mbpp_plus", args.mbpp_source),
        )
        if path is not None
    }
    print(
        json.dumps(
            stage(args.output_dir, selected, force=args.force, sources=sources),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
