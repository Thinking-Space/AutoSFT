"""Freeze source-grounded candidate data and assemble reproducible experiments.

Candidate origins are JSONL records with candidate_row, source_path, and an
optional source_line (physical, one-based). Omitting source_line anchors an
arbitrary-format source file and preserves its complete bytes. Optional
source_sha256 pins the expected source file version. Quality is caller-supplied
pass/fail/unknown evidence bound to content_sha256; it never filters data.

An assembly plan contains base_rows ("all" or physical line numbers),
replacements [{base_row, manifest, candidate_row}], additions [{manifest,
candidate_row}], and an optional deduplicate boolean (default true).
Candidates and assemblies are immutable. Before training, a round can select
a revised assembly while preserving its previous publication and review evidence.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from autosft.core.common import utc_now, write_json
from autosft.data.intervention_audit import _messages_fingerprint, _rows
from autosft.agents.roles import ANALYSIS_ROLES
from autosft.data.preparation import _sha256_file, tokenizer_subprocess_env
from autosft.orchestration.submission_registry import (
    find_sft_submission,
    submission_lock,
)


QUALITY_STATUSES = {"pass", "fail", "unknown"}
FROZEN_KINDS = {"candidate_data", "candidate_assembly"}
ANALYSIS_TASK_ROLES = set(ANALYSIS_ROLES)


def content_sha256(row: dict[str, Any]) -> str:
    """Identity of complete messages and tools, excluding row-level provenance."""
    return _messages_fingerprint(row)


def _json_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, value


def _write_rows(path: Path, rows: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _under_run(run_dir: Path, raw: Path | str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw):
        raise ValueError("run-local path must be a non-empty string or Path")
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else run_dir / path).resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError(f"path must stay under run directory {run_dir}: {path}")
    return path


def _output_path(run_dir: Path, path: Path | str) -> Path:
    result = _under_run(run_dir, path)
    if result == run_dir or any(
        result.is_relative_to(run_dir / name)
        for name in ("data_pool", "data_rounds", "data", "llm_calls")
    ):
        raise ValueError("candidate output must be a fresh scratch/candidate directory")
    roles = {
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {None, "", "meta"}
    role = next(iter(roles), os.environ.get("AUTO_SFT_TASK_ROLE"))
    task_root = os.environ.get("AUTO_SFT_TASK_DIR") or os.environ.get(
        "AUTO_SFT_SEARCH_TASK_DIR"
    )
    if role not in {None, "", "meta"} or task_root:
        if (
            len(roles) != 1
            or role not in ANALYSIS_TASK_ROLES
            or not task_root
            or not Path(task_root).is_absolute()
        ):
            raise ValueError(
                "candidate task output requires a valid task role and absolute AUTO_SFT_TASK_DIR"
            )
        task_dir = _under_run(run_dir, task_root)
        if (
            task_dir == run_dir
            or result == task_dir
            or not result.is_relative_to(task_dir)
        ):
            raise ValueError(
                "candidate task output must stay inside its own task directory; use --output-dir"
            )
    return result


@contextmanager
def _new_directory(destination: Path):
    """Build beside the destination and publish it under a cooperative lock."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".candidate_data.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.lexists(destination):
            raise ValueError(
                f"output already exists; choose a new version: {destination}"
            )
        with tempfile.TemporaryDirectory(
            prefix=".candidate-data-", dir=destination.parent
        ) as temp:
            stage = Path(temp) / "complete"
            stage.mkdir()
            yield stage
            if os.path.lexists(destination):
                raise ValueError(f"output appeared during construction: {destination}")
            os.rename(stage, destination)


def _file_ref(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _artifact(stage: Path, name: str, *, rows: int | None = None) -> dict[str, Any]:
    ref = _file_ref(stage / name, relative_to=stage)
    if rows is not None:
        ref["rows"] = rows
    return ref


def _artifact_path(manifest_path: Path, ref: dict[str, Any]) -> Path:
    raw = ref.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact path must be a non-empty string")
    path = (manifest_path.parent / raw).resolve()
    if Path(raw).is_absolute() or not path.is_relative_to(
        manifest_path.parent.resolve()
    ):
        raise ValueError(f"artifact escapes its frozen directory: {raw}")
    return path


def _verify_versions(run_dir: Path, refs: list[dict[str, Any]]) -> None:
    verified: dict[Path, str] = {}
    for ref in refs:
        path = _under_run(run_dir, ref["path"])
        if path in verified:
            if verified[path] != ref["sha256"]:
                raise ValueError(f"conflicting input versions: {path}")
            continue
        if not path.is_file() or _sha256_file(path) != ref["sha256"]:
            raise ValueError(f"input version changed or is stale: {path}")
        verified[path] = ref["sha256"]


def _load_frozen(
    run_dir: Path, path: Path, input_versions: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_ref = _file_ref(path, relative_to=run_dir)
    manifest = inspect_candidate(path)
    if Path(manifest["run_dir"]).resolve() != run_dir:
        raise ValueError("frozen manifest belongs to a different run")
    input_versions.append(manifest_ref)
    for ref in manifest["artifacts"].values():
        input_versions.append(
            {**ref, "path": str(_artifact_path(path, ref).relative_to(run_dir))}
        )
    return manifest


def _inspect_origins(
    manifest_path: Path, manifest: dict[str, Any], rows: dict[int, dict[str, Any]]
) -> None:
    artifacts = manifest["artifacts"]
    origins = list(_json_rows(_artifact_path(manifest_path, artifacts["origins"])))
    snapshots = {}
    run_dir = Path(manifest["run_dir"]).resolve()
    source_paths: set[str] = set()
    for _, snapshot in _json_rows(
        _artifact_path(manifest_path, artifacts["source_rows"])
    ):
        source_path = snapshot.get("source_path")
        if not isinstance(source_path, str) or source_path not in source_paths:
            _under_run(run_dir, source_path)
            source_paths.add(source_path)
        source_line = snapshot.get("source_line")
        key = (source_path, source_line)
        if key in snapshots:
            raise ValueError("duplicate source snapshot")
        if source_line is None:
            name = snapshot.get("snapshot_artifact")
            if name not in artifacts or artifacts[name]["sha256"] != snapshot.get(
                "source_sha256"
            ):
                raise ValueError("whole-file source snapshot mismatch")
        else:
            _positive(source_line, "snapshot.source_line")
            try:
                raw = base64.b64decode(snapshot.get("bytes_base64", ""), validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid source snapshot bytes") from exc
            if not raw.strip() or hashlib.sha256(raw).hexdigest() != snapshot.get(
                "record_sha256"
            ):
                raise ValueError("source snapshot record digest mismatch")
        snapshots[key] = snapshot
    seen = set()
    quality_counts: Counter = Counter()
    for _, origin in origins:
        row_id = _positive(origin.get("candidate_row"), "origin.candidate_row")
        if row_id in seen or row_id not in rows:
            raise ValueError("origin has duplicate or unknown candidate_row")
        seen.add(row_id)
        fingerprint = content_sha256(rows[row_id])
        quality = origin.get("quality", {})
        if (
            origin.get("content_sha256") != fingerprint
            or not isinstance(quality, dict)
            or quality.get("status") not in QUALITY_STATUSES
            or quality.get("content_sha256") != fingerprint
        ):
            raise ValueError("origin content/quality binding mismatch")
        quality_counts[quality["status"]] += 1
        snapshot = snapshots.get((origin.get("source_path"), origin.get("source_line")))
        if snapshot is None or snapshot.get("source_sha256") != origin.get(
            "source_sha256"
        ):
            raise ValueError("origin source snapshot is missing or mismatched")
        if origin.get("source_line") is not None and (
            origin.get("source_row_sha256") != snapshot.get("record_sha256")
        ):
            raise ValueError("origin source record digest mismatch")
    if seen != set(rows) or dict(quality_counts) != manifest.get("quality_counts"):
        raise ValueError("origin coverage or quality counts mismatch")
    if len(origins) != artifacts["origins"].get("rows") or len(snapshots) != artifacts[
        "source_rows"
    ].get("rows"):
        raise ValueError("origin/source snapshot row count mismatch")


def _inspect_exposure(
    manifest_path: Path, manifest: dict[str, Any], rows: dict[int, dict[str, Any]]
) -> None:
    artifacts = manifest["artifacts"]
    exposure = json.loads(
        _artifact_path(manifest_path, artifacts["exposure"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(exposure, dict)
        or exposure.get("kind") != "candidate_exposure"
        or exposure.get("data_sha256") != artifacts["data"]["sha256"]
        or exposure.get("total_rows") != len(rows)
    ):
        raise ValueError("assembly exposure does not describe its data")
    origins = list(_json_rows(_artifact_path(manifest_path, artifacts["row_origins"])))
    seen = set()
    selected: dict[str, list[int]] = {}
    for _, origin in origins:
        row_id = _positive(origin.get("output_row"), "origin.output_row")
        if (
            row_id in seen
            or row_id not in rows
            or origin.get("content_sha256") != content_sha256(rows[row_id])
        ):
            raise ValueError("assembly row origin coverage/content mismatch")
        seen.add(row_id)
        if origin.get("kind") == "candidate":
            selected.setdefault(origin.get("key"), []).append(row_id)
        elif origin.get("kind") != "base":
            raise ValueError("unknown assembly row origin kind")
    if seen != set(rows) or len(origins) != artifacts["row_origins"].get("rows"):
        raise ValueError("assembly row origin coverage mismatch")
    outcomes = exposure.get("candidate_outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("assembly exposure must contain candidate outcomes")
    keys = set()
    totals: Counter = Counter()
    qualities: Counter = Counter()
    for record in outcomes:
        key = f"{record.get('manifest')}:{record.get('manifest_sha256')}:{record.get('candidate_row')}"
        if key in keys:
            raise ValueError("duplicate candidate exposure outcome")
        keys.add(key)
        lines = selected.get(key, [])
        if record.get("selected_lines") != lines or record.get(
            "emitted_occurrences"
        ) != len(lines):
            raise ValueError("candidate exposure does not match selected output rows")
        for field in (
            "requested_occurrences",
            "emitted_occurrences",
            "deduplicated_occurrences",
        ):
            value = record.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    "candidate exposure counts must be nonnegative integers"
                )
            totals[field] += value
        if (
            record["requested_occurrences"]
            != record["emitted_occurrences"] + record["deduplicated_occurrences"]
        ):
            raise ValueError("candidate exposure occurrence counts do not reconcile")
        quality = record.get("quality", {})
        if (
            not isinstance(quality, dict)
            or quality.get("status") not in QUALITY_STATUSES
            or quality.get("content_sha256") != record.get("content_sha256")
            or any(
                content_sha256(rows[line]) != record.get("content_sha256")
                for line in lines
            )
        ):
            raise ValueError("candidate exposure content/quality binding mismatch")
        qualities[quality["status"]] += len(lines)
    if set(selected) - keys or dict(qualities) != exposure.get(
        "candidate_quality_rows"
    ):
        raise ValueError(
            "candidate exposure quality totals or origin references mismatch"
        )
    for field in ("requested", "emitted", "deduplicated"):
        if totals[f"{field}_occurrences"] != exposure.get(
            f"{field}_candidate_occurrences"
        ):
            raise ValueError("candidate exposure totals mismatch")
    base_rows = sum(origin["kind"] == "base" for _, origin in origins)
    retained = sum("base_row" in origin for _, origin in origins)
    if (
        exposure.get("noncandidate_base_rows") != base_rows
        or exposure.get("base_retained_occurrences") != retained
        or base_rows + totals["emitted_occurrences"] != len(rows)
    ):
        raise ValueError("base/candidate exposure totals mismatch")
    base_counts = [
        exposure.get(f"base_{field}_occurrences")
        for field in ("selected", "replaced", "retained", "deduplicated")
    ]
    if any(type(value) is not int or value < 0 for value in base_counts) or base_counts[
        0
    ] != sum(base_counts[1:]):
        raise ValueError("base exposure occurrence counts do not reconcile")


def inspect_candidate(
    manifest_path: Path, *, check_inputs: bool = False
) -> dict[str, Any]:
    """Verify frozen artifacts; optionally check that live input versions remain current."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") not in FROZEN_KINDS:
        raise ValueError(f"not a candidate or assembly manifest: {manifest_path}")
    if manifest.get("state") != "frozen" or manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported or unfinished candidate manifest")
    declared_run = manifest.get("run_dir")
    if not isinstance(declared_run, str) or not Path(declared_run).is_absolute():
        raise ValueError("manifest must declare an absolute run_dir")
    run_dir = Path(declared_run).resolve()
    _under_run(run_dir, manifest_path)
    artifacts = manifest.get("artifacts")
    required = (
        {"data", "origins", "source_rows"}
        if manifest["kind"] == "candidate_data"
        else {"data", "row_origins", "plan", "exposure"}
    )
    if not isinstance(artifacts, dict) or required - set(artifacts):
        raise ValueError("manifest is missing required frozen artifacts")
    for name, ref in artifacts.items():
        if not isinstance(ref, dict):
            raise ValueError(f"invalid artifact entry: {name}")
        path = _artifact_path(manifest_path, ref)
        if not path.is_file() or _sha256_file(path) != ref.get("sha256"):
            raise ValueError(f"frozen artifact digest mismatch: {name}: {path}")
        if path.stat().st_size != ref.get("bytes"):
            raise ValueError(f"frozen artifact size mismatch: {name}: {path}")
    rows = dict(_rows(_artifact_path(manifest_path, artifacts["data"])))
    if not rows or len(rows) != artifacts["data"].get("rows"):
        raise ValueError("frozen data row count mismatch")
    if manifest["kind"] == "candidate_data":
        _inspect_origins(manifest_path, manifest, rows)
    else:
        _inspect_exposure(manifest_path, manifest, rows)
    if check_inputs:
        _verify_versions(run_dir, manifest.get("input_versions", []))
    return manifest


def _quality(origin: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    declared = origin.get("quality")
    if declared is None:
        return {
            "status": "unknown",
            "content_sha256": fingerprint,
            "reason": "no quality measurement supplied",
        }
    if not isinstance(declared, dict) or declared.get("status") not in QUALITY_STATUSES:
        raise ValueError("quality must declare status pass, fail, or unknown")
    if declared.get("content_sha256") != fingerprint:
        return {
            "status": "unknown",
            "content_sha256": fingerprint,
            "reason": "quality evidence is unbound or stale",
            "previous": declared,
        }
    return dict(declared)


def _source_allowed(run_dir: Path, path: Path, parent_data: Path | None) -> bool:
    return path == parent_data or any(
        path.is_relative_to(run_dir / name)
        for name in ("data_pool", "data_rounds", "data/jsonl")
    )


def _count_tokens(
    stage: Path, data_path: Path, converter: Path | None, python_bin: str
) -> tuple[dict[str, Any] | None, dict[int, int]]:
    if converter is None:
        return None, {}
    converter = Path(converter).resolve()
    count_input = stage / "count_input.jsonl"
    expected = {line for line, _ in _rows(data_path)}
    _write_rows(
        count_input, ({**row, "pool_row_id": line} for line, row in _rows(data_path))
    )
    lengths = stage / "token_lengths.tsv"
    completed = subprocess.run(
        [
            python_bin,
            str(converter),
            "-i",
            str(count_input),
            "-v",
            "qwen3",
            "--count-only",
            "--count-output",
            str(lengths),
            "--workers",
            "1",
        ],
        cwd=converter.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=tokenizer_subprocess_env(),
        check=False,
    )
    if completed.returncode:
        raise ValueError(
            f"converter token counting failed ({completed.returncode}): "
            + completed.stderr[-1500:]
        )
    counts: dict[int, int] = {}
    loss_tokens = 0
    with lengths.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    "converter count output must contain id/tokens/loss_tokens"
                )
            row_id, tokens, loss = map(int, parts)
            if row_id in counts or row_id not in expected or not 0 <= loss <= tokens:
                raise ValueError(
                    "converter count output has duplicate/unknown ids or invalid counts"
                )
            counts[row_id] = tokens
            loss_tokens += loss
    if set(counts) != expected:
        raise ValueError("converter skipped rows; token coverage is incomplete")
    return {
        "method": "tokenizer_count_only",
        "converter": str(converter),
        "converter_sha256": _sha256_file(converter),
        "rows": len(counts),
        "total_tokens": sum(counts.values()),
        "loss_tokens": loss_tokens,
        "scope": "before downstream packing or truncation",
    }, counts


def _token_artifacts(stage: Path, artifacts: dict[str, Any], accounting: Any) -> None:
    if accounting is not None:
        artifacts["count_input"] = _artifact(stage, "count_input.jsonl")
        artifacts["token_lengths"] = _artifact(stage, "token_lengths.tsv")


def freeze_candidate(
    run_dir: Path,
    *,
    candidate_id: str,
    version: str,
    data_path: Path,
    origins_path: Path,
    output_dir: Path | None = None,
    parent_manifest: Path | None = None,
    description: str = "",
    converter: Path | None = None,
    python_bin: str = sys.executable,
) -> Path:
    run_dir = Path(run_dir).resolve()
    for value in (candidate_id, version):
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
        ):
            raise ValueError("candidate_id and version must be plain non-empty names")
    destination = _output_path(
        run_dir,
        output_dir or run_dir / "candidates" / candidate_id / "versions" / version,
    )
    data_path, origins_path = (
        _under_run(run_dir, path) for path in (data_path, origins_path)
    )
    input_versions = [
        _file_ref(path, relative_to=run_dir) for path in (data_path, origins_path)
    ]
    rows = dict(_rows(data_path))
    if not rows:
        raise ValueError("candidate data must contain at least one ChatML row")
    parent_ref, parent_data = None, None
    if parent_manifest is not None:
        parent_manifest = _under_run(run_dir, parent_manifest)
        parent_index = len(input_versions)
        parent = _load_frozen(run_dir, parent_manifest, input_versions)
        parent_data = _artifact_path(parent_manifest, parent["artifacts"]["data"])
        parent_ref = input_versions[parent_index]
    origins: dict[int, dict[str, Any]] = {}
    source_refs: dict[Path, dict[str, Any]] = {}
    requested_lines: dict[Path, set[int]] = {}
    for _, origin in _json_rows(origins_path):
        row_id = _positive(origin.get("candidate_row"), "candidate_row")
        if row_id not in rows or row_id in origins:
            raise ValueError(f"origin has duplicate or unknown candidate_row: {row_id}")
        source = _under_run(run_dir, origin.get("source_path"))
        if not _source_allowed(run_dir, source, parent_data):
            raise ValueError(
                f"source must be pool, round data, or a frozen parent: {source}"
            )
        if source not in source_refs:
            source_refs[source] = _file_ref(source, relative_to=run_dir)
        source_ref = source_refs[source]
        if origin.get("source_sha256", source_ref["sha256"]) != source_ref["sha256"]:
            raise ValueError(f"source version is stale: {source}")
        source_line = origin.get("source_line")
        if source_line is not None:
            _positive(source_line, "source_line")
            requested_lines.setdefault(source, set()).add(source_line)
        fingerprint = content_sha256(rows[row_id])
        origins[row_id] = {
            **origin,
            "source_path": source_ref["path"],
            "source_sha256": source_ref["sha256"],
            "content_sha256": fingerprint,
            "quality": _quality(origin, fingerprint),
        }
    if set(origins) != set(rows):
        raise ValueError("every candidate data row must have exactly one origin")
    input_versions.extend(source_refs.values())
    snapshots: dict[tuple[Path, int | None], dict[str, Any]] = {}
    for source, selected_lines in requested_lines.items():
        final_requested_line = max(selected_lines)
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if line_number in selected_lines:
                    if not raw.strip():
                        raise ValueError(
                            f"source line is blank: {source}:{line_number}"
                        )
                    snapshots[source, line_number] = {
                        "source_path": source_refs[source]["path"],
                        "source_line": line_number,
                        "source_sha256": source_refs[source]["sha256"],
                        "record_sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes_base64": base64.b64encode(raw).decode("ascii"),
                    }
                if line_number >= final_requested_line:
                    break
        if any((source, line) not in snapshots for line in selected_lines):
            raise ValueError(f"source_line is outside the source file: {source}")
    with _new_directory(destination) as stage:
        shutil.copyfile(data_path, stage / "data.jsonl")
        artifacts: dict[str, Any] = {}
        for row_id, origin in origins.items():
            source = _under_run(run_dir, origin["source_path"])
            if origin.get("source_line") is not None:
                snapshot = snapshots[source, origin["source_line"]]
                expected = origin.get("source_row_sha256", snapshot["record_sha256"])
                if expected != snapshot["record_sha256"]:
                    raise ValueError(f"source row version is stale: {source}")
                origin["source_row_sha256"] = snapshot["record_sha256"]
            elif (source, None) not in snapshots:
                name = f"source_{len(snapshots)}.bin"
                shutil.copyfile(source, stage / name)
                artifacts[name] = _artifact(stage, name)
                snapshots[source, None] = {
                    "source_path": source_refs[source]["path"],
                    "source_sha256": source_refs[source]["sha256"],
                    "scope": "whole_file",
                    "snapshot_artifact": name,
                }
                if artifacts[name]["sha256"] != source_refs[source]["sha256"]:
                    raise ValueError("source input changed during freeze")
        _write_rows(stage / "origins.jsonl", (origins[row] for row in sorted(origins)))
        _write_rows(stage / "source_rows.jsonl", snapshots.values())
        for name in ("data", "origins", "source_rows"):
            artifacts[name] = _artifact(
                stage,
                f"{name}.jsonl",
                rows=len(rows) if name != "source_rows" else len(snapshots),
            )
        if artifacts["data"]["sha256"] != input_versions[0]["sha256"]:
            raise ValueError("candidate input changed during freeze")
        accounting, _ = _count_tokens(
            stage, stage / "data.jsonl", converter, python_bin
        )
        _token_artifacts(stage, artifacts, accounting)
        _verify_versions(run_dir, input_versions)
        manifest = {
            "schema_version": "1.0",
            "kind": "candidate_data",
            "state": "frozen",
            "candidate_id": candidate_id,
            "version": version,
            "run_dir": str(run_dir),
            "created_at": utc_now(),
            "description": description,
            "parent": parent_ref,
            "input_versions": input_versions,
            "artifacts": artifacts,
            "quality_counts": dict(
                Counter(o["quality"]["status"] for o in origins.values())
            ),
            "token_accounting": accounting,
            "limitations": [
                "Source snapshots prove referenced bytes, not semantic grounding.",
                "Quality is caller-supplied evidence, not an admission gate.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
    return destination / "manifest.json"


def assemble_candidate_data(
    run_dir: Path,
    *,
    base_path: Path,
    plan_path: Path,
    output_dir: Path,
    converter: Path | None = None,
    python_bin: str = sys.executable,
) -> Path:
    run_dir = Path(run_dir).resolve()
    destination = _output_path(run_dir, output_dir)
    base_path, plan_path = (
        _under_run(run_dir, path) for path in (base_path, plan_path)
    )
    input_versions = [_file_ref(plan_path, relative_to=run_dir)]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) - {
        "base_rows",
        "base_sha256",
        "replacements",
        "additions",
        "deduplicate",
    }:
        raise ValueError("assembly plan has unknown keys or is not an object")
    deduplicate = plan.get("deduplicate", True)
    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate must be a boolean")
    base_manifest_path = None
    inherited_candidates: dict[int, dict[str, Any]] = {}
    if base_path.suffix == ".json":
        base_manifest_path = base_path
        base_manifest = _load_frozen(run_dir, base_path, input_versions)
        if base_manifest["kind"] == "candidate_assembly":
            base_artifacts = base_manifest["artifacts"]
            exposure = json.loads(
                _artifact_path(base_path, base_artifacts["exposure"]).read_text(
                    encoding="utf-8"
                )
            )
            outcomes = {
                f"{r['manifest']}:{r['manifest_sha256']}:{r['candidate_row']}": r
                for r in exposure["candidate_outcomes"]
            }
            for _, origin in _json_rows(
                _artifact_path(base_path, base_artifacts["row_origins"])
            ):
                if origin["kind"] == "candidate":
                    record = outcomes[origin["key"]]
                    inherited_candidates[origin["output_row"]] = {
                        key: record[key]
                        for key in ("manifest", "manifest_sha256", "candidate_row")
                    }
        base_path = _artifact_path(base_path, base_manifest["artifacts"]["data"])
    elif not _source_allowed(run_dir, base_path, None):
        raise ValueError(
            "base must be an existing pool/round dataset or a frozen manifest"
        )
    base_ref = _file_ref(base_path, relative_to=run_dir)
    if plan.get("base_sha256", base_ref["sha256"]) != base_ref["sha256"]:
        raise ValueError("base version is stale")
    input_versions.append(base_ref)
    base_rows = dict(_rows(base_path))
    selected = plan.get("base_rows", "all")
    if selected == "all":
        selected = list(base_rows)
    elif not isinstance(selected, list):
        raise ValueError("base_rows must be 'all' or a list of physical row numbers")
    for row_id in selected:
        if _positive(row_id, "base_row") not in base_rows:
            raise ValueError(f"unknown base_row: {row_id}")
    selected_ids = set(selected)
    replacements = plan.get("replacements", [])
    additions = plan.get("additions", [])
    if not isinstance(replacements, list) or not isinstance(additions, list):
        raise ValueError("replacements and additions must be lists")
    by_base: dict[int, dict[str, Any]] = {}
    for replacement in replacements:
        if not isinstance(replacement, dict):
            raise ValueError("replacement must be an object")
        row_id = _positive(replacement.get("base_row"), "replacement.base_row")
        if row_id in by_base or row_id not in selected_ids:
            raise ValueError(
                f"replacement base_row is duplicate or not selected: {row_id}"
            )
        by_base[row_id] = replacement

    loaded: dict[Path, tuple[dict[int, dict], dict[int, dict], dict[str, Any]]] = {}
    requested: dict[str, dict[str, Any]] = {}

    def candidate(ref: dict[str, Any]) -> tuple[dict, dict]:
        if not isinstance(ref, dict) or set(ref) - {
            "manifest",
            "manifest_sha256",
            "candidate_row",
            "base_row",
        }:
            raise ValueError("candidate reference must name manifest and candidate_row")
        path = _under_run(run_dir, ref.get("manifest"))
        if path not in loaded:
            manifest_index = len(input_versions)
            manifest = _load_frozen(run_dir, path, input_versions)
            if manifest["kind"] != "candidate_data":
                raise ValueError(
                    "replacement/addition must reference a frozen candidate"
                )
            manifest_ref = input_versions[manifest_index]
            data = dict(_rows(_artifact_path(path, manifest["artifacts"]["data"])))
            origins = {
                value["candidate_row"]: value
                for _, value in _json_rows(
                    _artifact_path(path, manifest["artifacts"]["origins"])
                )
            }
            loaded[path] = data, origins, manifest_ref
        data, origins, manifest_ref = loaded[path]
        if ref.get("manifest_sha256", manifest_ref["sha256"]) != manifest_ref["sha256"]:
            raise ValueError(f"candidate manifest version is stale: {path}")
        row_id = _positive(ref.get("candidate_row"), "candidate_row")
        if row_id not in data:
            raise ValueError(f"unknown candidate_row {row_id} in {path}")
        key = f"{manifest_ref['path']}:{manifest_ref['sha256']}:{row_id}"
        record = requested.setdefault(
            key,
            {
                "manifest": manifest_ref["path"],
                "manifest_sha256": manifest_ref["sha256"],
                "candidate_row": row_id,
                "content_sha256": content_sha256(data[row_id]),
                "quality": origins[row_id]["quality"],
                "requested_occurrences": 0,
                "emitted_occurrences": 0,
                "deduplicated_occurrences": 0,
                "selected_lines": [],
            },
        )
        record["requested_occurrences"] += 1
        return data[row_id], {"kind": "candidate", "key": key}

    materialized: list[dict] = []
    row_origins: list[dict] = []
    fingerprints: dict[str, int] = {}
    base_retained = 0
    base_deduplicated = 0

    def emit(row: dict, origin: dict) -> None:
        nonlocal base_retained, base_deduplicated
        fingerprint = content_sha256(row)
        if deduplicate and fingerprint in fingerprints:
            if origin["kind"] == "candidate":
                record = requested[origin["key"]]
                record["deduplicated_occurrences"] += 1
                record.setdefault("duplicate_of_lines", []).append(
                    fingerprints[fingerprint]
                )
            if "base_row" in origin:
                base_deduplicated += 1
            return
        line = len(materialized) + 1
        materialized.append(row)
        fingerprints.setdefault(fingerprint, line)
        row_origins.append(
            {"output_row": line, "content_sha256": fingerprint, **origin}
        )
        if origin["kind"] == "candidate":
            record = requested[origin["key"]]
            record["emitted_occurrences"] += 1
            record["selected_lines"].append(line)
        if "base_row" in origin:
            base_retained += 1

    for row_id in selected:
        if row_id in by_base:
            row, origin = candidate(by_base[row_id])
            emit(row, {**origin, "replaced_base_row": row_id})
        elif (
            base_manifest_path is not None and base_manifest["kind"] == "candidate_data"
        ):
            row, origin = candidate(
                {"manifest": str(base_manifest_path), "candidate_row": row_id}
            )
            emit(row, {**origin, "base_row": row_id})
        elif row_id in inherited_candidates:
            row, origin = candidate(inherited_candidates[row_id])
            if content_sha256(row) != content_sha256(base_rows[row_id]):
                raise ValueError(
                    "inherited candidate content differs from assembly base"
                )
            emit(row, {**origin, "base_row": row_id})
        else:
            emit(
                base_rows[row_id],
                {"kind": "base", "base_row": row_id, "base_sha256": base_ref["sha256"]},
            )
    for ref in additions:
        row, origin = candidate(ref)
        emit(row, origin)
    if not materialized:
        raise ValueError("assembly must retain at least one ChatML row")
    with _new_directory(destination) as stage:
        _write_rows(stage / "data.jsonl", materialized)
        _write_rows(stage / "row_origins.jsonl", row_origins)
        shutil.copyfile(plan_path, stage / "plan.json")
        accounting, token_counts = _count_tokens(
            stage, stage / "data.jsonl", converter, python_bin
        )
        records = list(requested.values())
        for record in records:
            record["selected_tokens"] = (
                sum(token_counts[line] for line in record["selected_lines"])
                if accounting is not None
                else None
            )
        quality_rows = Counter()
        for record in records:
            quality_rows[record["quality"]["status"]] += record["emitted_occurrences"]
        exposure = {
            "schema_version": "1.0",
            "kind": "candidate_exposure",
            "data_sha256": _sha256_file(stage / "data.jsonl"),
            "base": base_ref,
            "base_selected_occurrences": len(selected),
            "base_replaced_occurrences": sum(row in by_base for row in selected),
            "base_retained_occurrences": base_retained,
            "base_deduplicated_occurrences": base_deduplicated,
            "noncandidate_base_rows": sum(o["kind"] == "base" for o in row_origins),
            "total_rows": len(materialized),
            "deduplicate": deduplicate,
            "requested_candidate_occurrences": sum(
                r["requested_occurrences"] for r in records
            ),
            "emitted_candidate_occurrences": sum(
                r["emitted_occurrences"] for r in records
            ),
            "deduplicated_candidate_occurrences": sum(
                r["deduplicated_occurrences"] for r in records
            ),
            "candidate_quality_rows": dict(quality_rows),
            "candidate_outcomes": records,
            "token_accounting": accounting,
            "limitations": [
                "Exact content equality does not establish causal contribution.",
                "Quality statuses are supplied evidence and never admission gates.",
            ],
        }
        write_json(stage / "exposure.json", exposure)
        artifacts = {
            "data": _artifact(stage, "data.jsonl", rows=len(materialized)),
            "row_origins": _artifact(stage, "row_origins.jsonl", rows=len(row_origins)),
            "plan": _artifact(stage, "plan.json"),
            "exposure": _artifact(stage, "exposure.json"),
        }
        _token_artifacts(stage, artifacts, accounting)
        if _sha256_file(stage / "plan.json") != input_versions[0]["sha256"]:
            raise ValueError("assembly plan changed during materialization")
        _verify_versions(run_dir, input_versions)
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "1.0",
                "kind": "candidate_assembly",
                "state": "frozen",
                "run_dir": str(run_dir),
                "created_at": utc_now(),
                "input_versions": input_versions,
                "artifacts": artifacts,
                "token_accounting": accounting,
            },
        )
    return destination / "manifest.json"


@contextmanager
def _publication_lock(run_dir: Path, round_id: str) -> Iterator[None]:
    # The executor launcher holds this separate lock for its session lifetime.
    # Never wait for, or interrupt, an executor in order to change its inputs.
    io_dir = _under_run(run_dir, run_dir / "role_io" / f"round_{round_id}")
    io_dir.mkdir(parents=True, exist_ok=True)
    with (io_dir / ".executor.lifecycle.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "executor is active; cannot publish changed round data"
            ) from exc
        with submission_lock(run_dir):
            yield


def _assert_round_unsubmitted(run_dir: Path, round_id: str) -> None:
    if find_sft_submission(
        run_dir / "sft_submissions.jsonl", run_name=f"round_{round_id}"
    ):
        raise ValueError(
            "round already has a durable SFT submission; its data is immutable"
        )
    for path in (
        run_dir / "sft" / f"round_{round_id}",
        run_dir / "eval_runs" / f"round_{round_id}",
    ):
        if path.exists():
            raise ValueError(f"round already has execution evidence: {path}")
    report_path = run_dir / "executor_reports" / f"round_{round_id}.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("tasks") != []
            or report.get("status") not in {"failed", "blocked"}
        ):
            raise ValueError(f"round already has execution evidence: {report_path}")
    history = run_dir / "iterations.jsonl"
    if history.exists():
        for _, record in _json_rows(history):
            recipe = record.get("recipe") or {}
            ids = [record.get("round_id"), record.get("run_name")]
            if isinstance(recipe, dict):
                ids.append(recipe.get("round_id"))
            if any(str(value) in {round_id, f"round_{round_id}"} for value in ids):
                raise ValueError("round already appears in the experiment ledger")


def _superseded_executor_files(run_dir: Path, round_id: str) -> list[Path]:
    io_dir = run_dir / "role_io" / f"round_{round_id}"
    paths = [
        io_dir / name
        for name in (
            "executor.task_context.md",
            "executor.context.json",
            "executor.prompt.md",
            "executor.receipt.json",
            ".executor.codex.pid",
        )
    ]
    paths.append(run_dir / "executor_reports" / f"round_{round_id}.json")
    handoff = run_dir / "executor_handoff.json"
    if handoff.exists():
        payload = json.loads(handoff.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("executor handoff must be a JSON object")
        if str(payload.get("current_round")) == round_id:
            paths.append(handoff)
    existing = []
    for path in paths:
        if path.exists():
            if _under_run(run_dir, path) != path or path.is_symlink():
                raise ValueError(f"executor artifact must not be a symlink: {path}")
            existing.append(path)
    return existing


def publish_candidate_assembly(
    run_dir: Path, *, assembly_manifest: Path, round_id: int | str, revise: bool = False
) -> Path:
    """Publish a selected assembly, optionally revising the same unsubmitted round."""
    if os.environ.get("AUTO_SFT_TASK_DIR") or os.environ.get(
        "AUTO_SFT_SEARCH_TASK_DIR"
    ):
        raise ValueError("only the meta agent may publish an experiment round")
    for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE"):
        role = os.environ.get(name)
        if role and role not in {"meta"}:
            raise ValueError("only the meta agent may publish an experiment round")
    run_dir = Path(run_dir).resolve()
    if isinstance(round_id, bool) or not re.fullmatch(r"[1-9][0-9]*", str(round_id)):
        raise ValueError("round_id must be a positive integer")
    round_id = str(round_id)
    destination = run_dir / "data_rounds" / f"round_{round_id}"
    if _under_run(run_dir, destination) != destination:
        raise ValueError("round publication destination must not be a symlink")
    assembly_manifest = _under_run(run_dir, assembly_manifest)
    with _publication_lock(run_dir, round_id):
        _assert_round_unsubmitted(run_dir, round_id)
        if destination.exists() and not revise:
            raise ValueError(
                "round already exists; use publish --revise with the same round_id "
                "for a pre-training correction, not a new round number"
            )
        if revise and not destination.is_dir():
            raise ValueError(
                "--revise requires an existing unsubmitted round directory"
            )
        input_versions: list[dict[str, Any]] = []
        manifest = _load_frozen(run_dir, assembly_manifest, input_versions)
        if manifest["kind"] != "candidate_assembly":
            raise ValueError("publish requires a frozen assembly manifest")
        manifest_ref = input_versions[0]
        archive_root = _under_run(
            run_dir, run_dir / "data_rounds" / "revisions" / f"round_{round_id}"
        )
        archived_revisions = [
            int(path.name[1:])
            for path in archive_root.glob("v*")
            if re.fullmatch(r"v[1-9][0-9]*", path.name)
        ]
        previous_revision = max(archived_revisions, default=0) + 1
        archive = archive_root / (f"v{previous_revision}" if revise else "unpublished")
        retired = _superseded_executor_files(run_dir, round_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".candidate-data-", dir=destination.parent
        ) as temp:
            stage = Path(temp) / "complete"
            stage.mkdir()
            shutil.copyfile(
                _artifact_path(assembly_manifest, manifest["artifacts"]["data"]),
                stage / "final_data.jsonl",
            )
            exposure_bytes = _artifact_path(
                assembly_manifest, manifest["artifacts"]["exposure"]
            ).read_bytes()
            if (
                hashlib.sha256(exposure_bytes).hexdigest()
                != manifest["artifacts"]["exposure"]["sha256"]
            ):
                raise ValueError("assembly exposure changed during publication")
            exposure = json.loads(exposure_bytes)
            if (
                _sha256_file(stage / "final_data.jsonl")
                != manifest["artifacts"]["data"]["sha256"]
                or exposure["data_sha256"] != manifest["artifacts"]["data"]["sha256"]
            ):
                raise ValueError(
                    "assembly exposure does not describe the published data"
                )
            exposure["published_data"] = str(
                (destination / "final_data.jsonl").relative_to(run_dir)
            )
            exposure["assembly_manifest"] = manifest_ref
            write_json(stage / "exposure.json", exposure)
            write_json(stage / "candidate_assembly.json", manifest_ref)
            write_json(
                stage / "publication.json",
                {
                    "round_id": round_id,
                    "revision": previous_revision + int(revise),
                    "published_at": utc_now(),
                    "assembly_manifest": manifest_ref,
                    "previous_publication": str((archive / "data").relative_to(run_dir))
                    if revise
                    else None,
                },
            )
            _verify_versions(run_dir, input_versions)
            # Prepare the complete new version before withdrawing execution authority.
            # A failed switch restores both the previous data and its authorizations.
            moved: list[tuple[Path, Path]] = []
            if revise or retired:
                archive.mkdir(parents=True, exist_ok=False)
            try:
                for path in retired:
                    saved = archive / "execution" / path.relative_to(run_dir)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(saved)
                    moved.append((path, saved))
                if revise:
                    destination.rename(archive / "data")
                    moved.append((destination, archive / "data"))
                stage.rename(destination)
            except BaseException:
                for original, saved in reversed(moved):
                    saved.rename(original)
                if archive.exists():
                    shutil.rmtree(archive)
                raise
    return destination / "final_data.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze",
        help="freeze a candidate and its source references",
        description="Freeze ChatML JSONL, exact source references, and optional quality evidence. "
        "Quality never filters rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Origins JSONL: one object per nonblank candidate physical line (one-based):
  {"candidate_row":1,"source_path":"data_pool/raw.jsonl","source_line":37,
   "quality":{"status":"unknown","content_sha256":"<digest>","reasons":["not measured"]}}
Write each complete object on one line. Omit source_line for a whole-file snapshot.
Optional source_sha256 and source_row_sha256 pin expected source versions.
Source paths must be pool/round data or the --parent manifest's frozen data.

Use the public helper on the COMPLETE parsed ChatML row, not only answer text:
  from autosft.data.candidate_data import content_sha256
  origin["quality"]["content_sha256"] = content_sha256(row)
It includes all messages fields and top-level tools, ignoring row metadata.
Quality status is pass, fail, or unknown; reasons are optional caller evidence.
Omitted, unbound, or stale quality becomes unknown; no status is an admission gate.

Root default: RUN/candidates/ID/versions/V/manifest.json (new version only).
Task sessions must specify --output-dir inside their own AUTO_SFT_TASK_DIR.
Example:
  python -m autosft.data.candidate_data freeze --run-dir RUN \\
    --candidate-id repair --version v1 --data scratch/replacements.jsonl \\
    --origins scratch/origins.jsonl --output-dir TASK_DIR/artifacts/repair-v1""",
    )
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--data", type=Path, required=True)
    freeze.add_argument("--origins", type=Path, required=True)
    freeze.add_argument("--parent", type=Path)
    freeze.add_argument("--output-dir", type=Path)
    freeze.add_argument("--description", default="")
    assemble = commands.add_parser(
        "assemble",
        help="assemble explicit base/candidate row references",
        description="Assemble explicit base selections, replacements, and additions into a fresh directory. "
        "Candidate quality does not restrict selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Plan JSON example:
  {"base_rows":"all",
   "replacements":[{"base_row":2,"manifest":"candidates/repair/versions/v1/manifest.json","candidate_row":1}],
   "additions":[{"manifest":"candidates/repair/versions/v1/manifest.json","candidate_row":3}],
   "deduplicate":true}
All row numbers are one-based physical lines. References use run-relative or
absolute paths. --base accepts existing pool/round ChatML or a frozen manifest.
base_rows defaults to "all"; an explicit list selects/reorders base rows.
Example weighting plan: {"base_rows":[1,1,3],"deduplicate":false}
Use base_rows: [] for candidates only. Replacements name selected base rows;
additions append in order. Optional base_sha256 and reference manifest_sha256
pin expected versions. deduplicate defaults to true and uses full content identity.

Output: manifest.json, data.jsonl, row_origins.jsonl, plan.json, exposure.json.
Exposure records requested/emitted/deduplicated occurrences and selected_lines.
--converter adds real token counts before downstream packing/truncation.
Task sessions must put --output-dir inside their own AUTO_SFT_TASK_DIR.
Example:
  python -m autosft.data.candidate_data assemble --run-dir RUN \\
    --base data_pool/base.jsonl --plan scratch/plan.json --output-dir scratch/assembly-v1""",
    )
    assemble.add_argument("--run-dir", type=Path, required=True)
    assemble.add_argument("--base", type=Path, required=True)
    assemble.add_argument("--plan", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    for command in (freeze, assemble):
        command.add_argument(
            "--converter",
            type=Path,
            help="optional actual SSTable converter for token counts",
        )
        command.add_argument("--python", default=sys.executable, dest="python_bin")
    inspect = commands.add_parser(
        "inspect",
        help="verify frozen artifact digests",
        description="Read-only verification of frozen artifact hashes, source snapshots, "
        "row/content bindings, and exposure consistency; prints the manifest as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python -m autosft.data.candidate_data inspect \\
    --manifest RUN/candidates/repair/versions/v1/manifest.json
Add --check-inputs to require live source/input files to match their pinned versions.
Without it, self-contained frozen snapshots remain valid after live inputs change.
Verification establishes byte/content consistency, not semantic quality.""",
    )
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--check-inputs", action="store_true")
    publish = commands.add_parser(
        "publish",
        help="root-only publication or revision of unsubmitted round data",
        description="Publish exact assembly data and exposure; --revise replaces an unsubmitted "
        "round's selected data without increasing its round_id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python -m autosft.data.candidate_data publish --run-dir RUN \\
    --assembly scratch/assembly-v1/manifest.json --round-id 2
Creates data_rounds/round_2/final_data.jsonl, exposure.json, candidate_assembly.json.
Does not create a recipe or submit training. Root authors the recipe afterward.
For a correction after review, publish the new assembly with the SAME --round-id
and --revise. The old publication and executor authorization are archived under
data_rounds/revisions/round_<id>/v<N>/; review snapshots are unchanged. Write a
new recipe ID and explicitly reauthorize executor only after handling the review.
An active executor or execution/submission/ledger evidence causes refusal.
Candidate task sessions cannot publish.""",
    )
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--assembly", type=Path, required=True)
    publish.add_argument("--round-id", required=True)
    publish.add_argument(
        "--revise",
        action="store_true",
        help="archive and replace this unsubmitted round",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_candidate(
                args.run_dir,
                candidate_id=args.candidate_id,
                version=args.version,
                data_path=args.data,
                origins_path=args.origins,
                parent_manifest=args.parent,
                output_dir=args.output_dir,
                description=args.description,
                converter=args.converter,
                python_bin=args.python_bin,
            )
        elif args.command == "assemble":
            result = assemble_candidate_data(
                args.run_dir,
                base_path=args.base,
                plan_path=args.plan,
                output_dir=args.output_dir,
                converter=args.converter,
                python_bin=args.python_bin,
            )
        elif args.command == "publish":
            result = publish_candidate_assembly(
                args.run_dir,
                assembly_manifest=args.assembly,
                round_id=args.round_id,
                revise=args.revise,
            )
        else:
            result = inspect_candidate(args.manifest, check_inputs=args.check_inputs)
        print(
            json.dumps(
                result if isinstance(result, dict) else {"path": str(result)}, indent=2
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"candidate_data: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
