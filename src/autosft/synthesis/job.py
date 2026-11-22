"""Lifecycle facade for the modular Synthesis system.

The command intentionally exposes only target-level lifecycle and public
delivery references.  Worker prompts, retries, validation, and publication are
implemented behind the facade and may evolve without changing the handbook.
"""

from __future__ import annotations
from autosft.synthesis.inspection import inspect_delivery
from autosft.config.components import synthesis_enabled

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autosft.synthesis.errors import (
    SynConflictError as _SynConflictError,
    SynContractError as _SynContractError,
    SynError as _SynError,
)
from autosft.synthesis._internal.attempts import (
    TERMINAL_STATES as _TERMINAL,
    AttemptStore as _AttemptStore,
    AttemptStoreError as _AttemptStoreError,
    AttemptSupervisor as _AttemptSupervisor,
)
from autosft.synthesis._internal.config import (
    SynConfig as _SynConfig,
    SynLayout as _SynLayout,
)
from autosft.synthesis._internal.contracts import (
    DELIVERY_DATA_NAME as _DELIVERY_DATA_NAME,
    DELIVERY_REPORT_NAME as _DELIVERY_REPORT_NAME,
    EXPERIENCE_DELTA_NAME as _EXPERIENCE_DELTA_NAME,
    FrozenTarget as _FrozenTarget,
    discover_requests as _discover_requests,
    freeze_request as _freeze_request,
    validate_delivery as _validate_delivery,
)
from autosft.synthesis._internal.experience import (
    ExperienceMaintainer as _ExperienceMaintainer,
    ExperienceStore as _ExperienceStore,
)
from autosft.synthesis._internal.filesystem import (
    atomic_write_json as _atomic_write_json,
    exclusive_file_lock as _exclusive_file_lock,
    read_json_object as _load_json,
    utc_now as _utc_now,
    sha256_file as _sha256_file,
    validate_safe_id as _validate_safe_id,
)
from autosft.synthesis._internal.runtime import (
    SYNTHESIS_WORKER_PLACEHOLDERS as _WORKER_PLACEHOLDERS,
    CodexRuntime as _CodexRuntime,
    WorkerPaths as _WorkerPaths,
    WorkerRunRequest as _WorkerRunRequest,
    process_group_alive as _process_group_alive,
    render_prompt as _render_prompt,
    worker_prompt_values as _worker_prompt_values,
)
from autosft.synthesis._internal.storage import (
    DataSource as _DataSource,
    Publisher as _Publisher,
)
from autosft.synthesis.catalog import (
    target_benchmark as _target_benchmark,
    validate_benchmarks as _validate_syn_benchmarks,
)


_SCHEMA_VERSION = "1.0"
_MAX_TARGETS_PER_LAUNCH = 10
_ACTIVE_ATTEMPT_STATES = frozenset(
    {"starting", "running", "resuming", "validating", "publishing"}
)
# A launcher claim with no PID is a narrow crash window between ``Popen`` and
# the durable ``launcher_started`` event.  Give a surviving parent enough time
# to publish that event, then fail closed if the claim is still orphaned.
_ORPHAN_CLAIM_GRACE_SECONDS = 30.0

__all__ = [
    "abandon",
    "cancel_pending",
    "dispatch",
    "launch",
    "retry",
    "main",
    "parse_args",
    "status",
]


def _config() -> _SynConfig:
    return _SynConfig.from_env()


def _layout(config: _SynConfig) -> _SynLayout:
    return _SynLayout(config.run_dir, config.data_source_dir)


def _round_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise _SynContractError(
            "round_id must be a canonical positive integer, for example 2"
        )
    return _validate_safe_id(value, "round_id")


def _write_round_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(path, dict(payload))


def _round_manifest(layout: _SynLayout, round_id: str) -> dict[str, Any]:
    return _load_json(layout.round_manifest(round_id), "round manifest")


def _round_lock(layout: _SynLayout, round_id: str) -> Path:
    return layout.round_dir(round_id) / ".round.lock"


def _intent_path(layout: _SynLayout, round_id: str) -> Path:
    return layout.round_dir(round_id) / "round_intent.json"


def _recover_round_intent(layout: _SynLayout, round_id: str) -> None:
    """Finish an interrupted manifest/attempt commit under the round lock."""
    path = _intent_path(layout, round_id)
    if not path.is_file():
        return
    intent = _load_json(path, "round transaction intent")
    if intent.get("state") == "committed":
        return
    if (
        intent.get("state") != "prepared"
        or intent.get("round_id") != round_id
        or not isinstance(intent.get("manifest"), dict)
        or not isinstance(intent.get("attempts"), list)
    ):
        raise _SynContractError("invalid round transaction intent")
    if intent["manifest"].get("round_id") != round_id or intent["manifest"].get(
        "transaction_id"
    ) != intent.get("transaction_id"):
        raise _SynContractError("round transaction identity mismatch")
    manifest_path = layout.round_manifest(round_id)
    current = _round_manifest(layout, round_id) if manifest_path.is_file() else None
    if current is not None and current.get("transaction_id") == intent.get(
        "transaction_id"
    ):
        intent["state"] = "committed"
        _atomic_write_json(path, intent)
        return
    actual_base = _sha256_file(manifest_path) if manifest_path.is_file() else None
    if actual_base != intent.get("base_sha256"):
        raise _SynConflictError("round changed outside its pending transaction")
    for attempt in intent["attempts"]:
        _attempt_store(layout, round_id, attempt["target_id"]).ensure_created(
            attempt["attempt_id"],
            target_id=attempt["target_id"],
            round_id=round_id,
            metadata=attempt["metadata"],
        )
    _write_round_manifest(manifest_path, intent["manifest"])
    intent["state"] = "committed"
    _atomic_write_json(path, intent)


def _commit_round_intent(
    layout: _SynLayout,
    round_id: str,
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    operation: str,
) -> None:
    """Persist recovery information before the first attempt event is written."""
    base = layout.round_manifest(round_id)
    transaction_id = hashlib.sha256(
        json.dumps(
            {
                "manifest": manifest,
                "attempts": attempts,
                "operation": operation,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest["transaction_id"] = transaction_id
    _atomic_write_json(
        _intent_path(layout, round_id),
        {
            "schema_version": _SCHEMA_VERSION,
            "round_id": round_id,
            "operation": operation,
            "state": "prepared",
            "transaction_id": transaction_id,
            "base_sha256": _sha256_file(base) if base.is_file() else None,
            "manifest": manifest,
            "attempts": attempts,
        },
    )
    _recover_round_intent(layout, round_id)


def _require_open_round(manifest: Mapping[str, Any]) -> None:
    if manifest.get("disposition") is not None:
        raise _SynConflictError(
            "synthesis was abandoned for this training round; use a future round for new Synthesis"
        )


def _lifecycle_process_matches(
    pid: int,
    *,
    round_id: str,
    target_id: str,
    attempt_id: str,
) -> bool:
    """Return whether ``pid`` is still our detached attempt process.

    A PID in a manifest is only a hint after the process exits.  Before
    signalling a running attempt, require both a live process group and the
    immutable `_attempt` command arguments.  If procfs is unavailable, fail
    closed instead of risking a signal to a reused PID.
    """

    try:
        alive = _process_group_alive(pid)
    except (OverflowError, ValueError, OSError):
        alive = False
    if not alive:
        return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return False
    argv = [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]
    required = (
        "autosft.synthesis.job",
        "_attempt",
        "--round-id",
        round_id,
        "--target-id",
        target_id,
        "--attempt-id",
        attempt_id,
    )
    return all(item in argv for item in required)


def _attempt_store(layout: _SynLayout, round_id: str, target_id: str) -> _AttemptStore:
    return _AttemptStore(layout.attempts_dir(round_id, target_id))


def _attempt_dir(
    layout: _SynLayout, round_id: str, target_id: str, attempt_id: str
) -> Path:
    return layout.attempt_dir(round_id, target_id, attempt_id)


def _job(manifest: Mapping[str, Any], target_id: str) -> dict[str, Any]:
    for item in manifest.get("targets", []):
        if isinstance(item, Mapping) and item.get("target_id") == target_id:
            return dict(item)
    raise _SynContractError(f"round manifest has no target: {target_id}")


def _latest_attempt(job: Mapping[str, Any]) -> dict[str, Any]:
    attempts = job.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise _SynContractError(f"target {job.get('target_id')} has no attempts")
    item = attempts[-1]
    if not isinstance(item, Mapping):
        raise _SynContractError("attempt manifest entry must be an object")
    return dict(item)


def _stable_batch_id(round_id: str, target_id: str, attempt_id: str) -> str:
    # All components have already passed the safe-id check.  A new attempt is
    # therefore a new immutable batch even when a previous attempt failed.
    return _validate_safe_id(f"syn-{round_id}-{target_id}-{attempt_id}", "batch_id")


def _launch_claim_id(round_id: str, target_id: str, attempt_id: str) -> str:
    """Return the deterministic identity of a launcher claim."""

    digest = hashlib.sha256(
        f"{round_id}\0{target_id}\0{attempt_id}".encode("utf-8")
    ).hexdigest()
    return f"claim-{digest[:32]}"


def _experience_versions(
    source: _DataSource,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve the immutable experience version consumed by one attempt."""

    experience = _ExperienceStore(source).ensure()
    return (
        experience.experience_path,
        experience.version_id,
        {"experience": experience.to_dict()},
    )


def _frozen_experience_versions(
    source: _DataSource, job: Mapping[str, Any]
) -> tuple[Path, str]:
    """Use the experience version frozen at launch time."""

    recorded = job.get("experience_versions")
    if not isinstance(recorded, Mapping):
        raise _SynContractError("frozen experience version is missing")
    entry = recorded.get("experience")
    if not isinstance(entry, Mapping):
        raise _SynContractError("frozen experience version is missing")
    raw_path = entry.get("experience_path")
    version_id = entry.get("version_id")
    if not isinstance(raw_path, str) or not isinstance(version_id, str):
        raise _SynContractError("frozen experience version is invalid")
    path = Path(raw_path).expanduser().resolve(strict=False)
    try:
        path.relative_to(source.root.resolve())
    except ValueError as exc:
        raise _SynContractError("frozen experience path escapes Data Source") from exc
    if path.is_symlink() or not path.is_file():
        raise _SynContractError("frozen experience file is missing")
    return path, version_id


def _maintain_published_experience(
    config: _SynConfig,
    *,
    target_id: str,
    batch_id: str,
) -> dict[str, Any]:
    """Run the private maintainer after a complete immutable batch exists."""

    maintainer = _ExperienceMaintainer.from_config(
        config, data_source=_DataSource(config.data_source_dir)
    )
    return maintainer.maintain(target_id, batch_id=batch_id)


def _snapshot_for_round(
    source: _DataSource, layout: _SynLayout, round_id: str
) -> dict[str, Any]:
    destination = layout.round_dir(round_id) / "data_source_snapshot"
    if destination.exists():
        # A crash can install the immutable snapshot before round_intent.json.
        # Recover that pinned view even if other rounds advanced the source.
        pinned = _load_json(destination / "manifest.json", "round Data Source snapshot")
        source._validate_manifest(pinned)
        source._validate_snapshot_tree(destination, pinned)
        return {
            "schema_version": pinned["schema_version"],
            "snapshot_id": pinned["snapshot_id"],
            "revision": pinned["revision"],
            "data_source_dir": str(destination),
            "source_data_source_dir": str(source.root),
            "snapshot_dir": str(destination),
            "manifest_path": str(destination / "manifest.json"),
            "manifest_sha256": _sha256_file(destination / "manifest.json"),
        }
    return source.snapshot(destination)


def _configured_benchmarks() -> frozenset[str]:
    """Benchmarks selected for this run, including the instruction-following defaults."""
    raw = os.environ.get("AUTO_SFT_BENCHMARKS", "ifeval,ifbench")
    try:
        return frozenset(_validate_syn_benchmarks(raw))
    except Exception as exc:
        raise _SynContractError(str(exc)) from exc


def _freeze_and_manifest(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    requests_dir: Path,
) -> dict[str, Any]:
    round_id = _round_id(round_id)
    round_dir = layout.round_dir(round_id)
    round_dir.mkdir(parents=True, exist_ok=True)
    requests = _discover_requests(
        requests_dir,
        allowed_root=config.run_dir,
        minimum_targets=1,
    )
    configured = _configured_benchmarks()
    unsupported = sorted(
        {
            _target_benchmark(request.target_id)
            for request in requests
            if _target_benchmark(request.target_id) not in configured
        }
    )
    if unsupported:
        raise _SynContractError(
            "Synthesis targets are outside AUTO_SFT_BENCHMARKS: "
            + ", ".join(unsupported)
        )
    source = _DataSource(config.data_source_dir)
    source.ensure()
    manifest_path = layout.round_manifest(round_id)

    with _exclusive_file_lock(_round_lock(layout, round_id)):
        _recover_round_intent(layout, round_id)
        existing = (
            _load_json(manifest_path, "round manifest")
            if manifest_path.exists()
            else None
        )
        if existing is not None:
            _require_open_round(existing)
            old = [
                (item.get("target_id"), item.get("input_fingerprint"))
                for item in existing.get("targets", [])
                if isinstance(item, Mapping)
            ]
            current = [(item.target_id, item.input_fingerprint) for item in requests]
            if old != current:
                raise _SynConflictError(
                    "request group changed after launch; preserve the frozen target inputs"
                )
            return existing

        # Old releases could leave attempts before recording any round intent.
        # Refuse to invent a replacement history for such an unbound partial run.
        if any((round_dir / "targets").glob("*/attempts/*/events.jsonl")):
            raise _SynConflictError(
                "orphan attempt evidence has no round transaction intent"
            )

        # Snapshot only when installing a new round.  A repeated launch must
        # keep using the round's frozen Data Source view even if the live Data
        # Source has advanced since the first invocation.
        snapshot = _snapshot_for_round(source, layout, round_id)
        targets: list[dict[str, Any]] = []
        planned_attempts: list[dict[str, Any]] = []
        for request in requests:
            target_dir = layout.target_input_dir(round_id, request.target_id)
            frozen = _freeze_request(
                request,
                target_dir,
                rows_required=config.rows_per_target,
            )
            attempt_id = "attempt-001"
            experience_versions = _experience_versions(source)[2]
            planned_attempts.append(
                {
                    "attempt_id": attempt_id,
                    "target_id": request.target_id,
                    "metadata": {
                        "frozen_target": frozen.to_dict(),
                        "experience_versions": experience_versions,
                    },
                }
            )
            targets.append(
                {
                    "target_id": request.target_id,
                    "input_fingerprint": request.input_fingerprint,
                    "frozen_target": frozen.to_dict(),
                    "experience_versions": experience_versions,
                    "attempts": [
                        {
                            "attempt_id": attempt_id,
                            "attempt_number": 1,
                            "state": "created",
                            "pid": None,
                        }
                    ],
                }
            )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "round_id": round_id,
            "implementation": "modular_syn",
            "state": "created",
            "max_concurrency": max(
                1,
                int(getattr(config, "max_concurrency", _MAX_TARGETS_PER_LAUNCH)),
            ),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "requests_dir": str(Path(requests_dir).resolve()),
            "data_source_dir": str(config.data_source_dir),
            "data_source_snapshot": snapshot,
            "targets": targets,
        }
        _commit_round_intent(
            layout, round_id, manifest, planned_attempts, operation="launch"
        )
        return manifest


def _start_attempt(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    target_id: str,
    attempt_id: str,
) -> int | None:
    attempt_dir = _attempt_dir(layout, round_id, target_id, attempt_id)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "process.log"
    command = [
        sys.executable,
        "-m",
        "autosft.synthesis.job",
        "_attempt",
        "--round-id",
        round_id,
        "--target-id",
        target_id,
        "--attempt-id",
        attempt_id,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(config.repo_root / "src"))
    manifest_path = layout.round_manifest(round_id)
    store = _attempt_store(layout, round_id, target_id)
    claim_id = _launch_claim_id(round_id, target_id, attempt_id)

    # The durable claim and Popen are serialized by the round lock.  Another
    # launcher can therefore only observe either an unclaimed created attempt
    # or the starting state, never a window in which both launch it.
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        _recover_round_intent(layout, round_id)
        manifest = _round_manifest(layout, round_id)
        _require_open_round(manifest)
        job = _job(manifest, target_id)
        attempt = _latest_attempt(job)
        if attempt.get("attempt_id") != attempt_id:
            raise _SynConflictError("attempt changed while starting worker")
        current = store.current(attempt_id)
        if str(current.get("state")) != "created":
            return None
        if str(current.get("desired_state")) != "running":
            return None
        for previous_attempt in job["attempts"]:
            _require_attempt_quiescent(layout, round_id, target_id, previous_attempt)

        store.append_event(
            attempt_id,
            "launch_claimed",
            {
                "claim_id": claim_id,
                "launcher_parent_pid": os.getpid(),
            },
        )
        try:
            with log_path.open("ab") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=config.repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as exc:
            store.append_event(
                attempt_id,
                "attempt_failed",
                {
                    "reason": "launcher_spawn_failed",
                    "error": repr(exc),
                    "retryable": True,
                    "claim_id": claim_id,
                },
            )
            for item in manifest["targets"]:
                if item.get("target_id") == target_id:
                    item["state"] = "failed"
                    summary = item["attempts"][-1]
                    summary["state"] = "failed"
                    summary["pid"] = None
                    summary["pgid"] = None
                    summary["launch_claim_id"] = claim_id
            manifest["updated_at"] = _utc_now()
            _write_round_manifest(manifest_path, manifest)
            return None

        for item in manifest["targets"]:
            if item.get("target_id") == target_id:
                item["state"] = "starting"
                summary = item["attempts"][-1]
                summary["pid"] = process.pid
                summary["pgid"] = process.pid
                summary["state"] = "starting"
                summary["launch_claim_id"] = claim_id
        manifest["state"] = "running"
        manifest["updated_at"] = _utc_now()
        _write_round_manifest(manifest_path, manifest)
    return process.pid


def _attempt_worker(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    target_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    store = _attempt_store(layout, round_id, target_id)
    claim_id = _launch_claim_id(round_id, target_id, attempt_id)
    current = store.current(attempt_id)
    if str(current.get("state")) in _TERMINAL:
        return current
    if current.get("launch_claim_id") != claim_id:
        raise _SynConflictError(
            "detached attempt process does not hold the durable launch claim"
        )
    launcher_pid = os.getpid()
    launcher_pgid = os.getpgrp()
    recorded_launcher = current.get("launcher_pid")
    if recorded_launcher is None:
        store.append_event(
            attempt_id,
            "launcher_started",
            {
                "claim_id": claim_id,
                "pid": launcher_pid,
                "pgid": launcher_pgid,
            },
        )
    elif recorded_launcher != launcher_pid:
        raise _SynConflictError("attempt already belongs to another launcher process")

    manifest = _round_manifest(layout, round_id)
    job = _job(manifest, target_id)
    attempt = next(
        item for item in job["attempts"] if item.get("attempt_id") == attempt_id
    )
    frozen = _FrozenTarget.from_dict(job["frozen_target"])
    snapshot = manifest["data_source_snapshot"]
    snapshot_dir = Path(snapshot["snapshot_dir"]).resolve()
    source = _DataSource(config.data_source_dir)
    experience_path, experience_version = _frozen_experience_versions(source, job)
    paths = _WorkerPaths(
        frozen_request_dir=frozen.frozen_dir,
        target_file=frozen.target_file,
        materials_dir=frozen.materials_dir,
        data_source_snapshot_dir=snapshot_dir,
        experience_file=experience_path,
        topic_stem_pool_file=config.topic_stem_pool_path,
        output_dir=layout.attempt_output_dir(round_id, target_id, attempt_id),
        batch_skill_file=config.batch_skill_path,
        batch_runner=config.batch_runner_path,
        batch_reference_file=config.batch_reference_path,
        eval_runs_dir=config.run_dir / "eval_runs"
        if (config.run_dir / "eval_runs").is_dir()
        else None,
        log_path=_attempt_dir(layout, round_id, target_id, attempt_id) / "worker.jsonl",
        data_pool_dir=config.run_dir / "data_pool"
        if (config.run_dir / "data_pool").is_dir()
        else None,
        row_check_tool=config.row_check_tool_path,
    )
    rendered = _render_prompt(
        config.worker_prompt_path,
        _worker_prompt_values(
            target_id=target_id,
            paths=paths,
            data_source_snapshot_id=snapshot["snapshot_id"],
            experience_version=experience_version,
            rows_per_target=config.rows_per_target,
        ),
        allowed_placeholders=_WORKER_PLACEHOLDERS,
    )
    request = _WorkerRunRequest(
        attempt_id=attempt_id,
        target_id=target_id,
        rendered_prompt=rendered,
        paths=paths,
        rows_required=config.rows_per_target,
        max_resumes=config.max_resumes,
        resume_backoff_seconds=config.resume_backoff_seconds,
        timeout_seconds=config.attempt_timeout_seconds,
    )

    def publish(delivery: Any, _request: _WorkerRunRequest) -> dict[str, Any]:
        from autosft.synthesis._internal.storage import count_jsonl_tokens

        counts = count_jsonl_tokens(
            delivery.data_path,
            tokenizer_code_dir=config.tokenizer_code_dir,
            python_executable=config.tokenizer_python,
        )
        result = _Publisher(source).publish(
            delivery,
            batch_id=_stable_batch_id(round_id, target_id, attempt_id),
            frozen_target=frozen,
            token_counts=counts.total_tokens,
            loss_counts=counts.loss_tokens,
            tokenizer={"name": "qwen3", "counter": "prepare_tokens.encode_row"},
            round_id=round_id,
            attempt_id=attempt_id,
        )
        return result

    runtime = _CodexRuntime(
        repo_root=config.repo_root,
        python_executable=sys.executable,
    )
    _record_target_attempt(
        source,
        target_id,
        round_id=round_id,
        attempt_id=attempt_id,
        state="running",
        attempt_dir=_attempt_dir(layout, round_id, target_id, attempt_id),
    )
    result = _AttemptSupervisor(
        store=store,
        runtime=runtime,
        publisher=publish,
        validator=lambda req: inspect_delivery(
            _validate_delivery(
                req.paths.output_dir,
                target_id=req.target_id,
                rows_required=req.rows_required,
            ),
            config.run_dir,
            str(round_id),
        ),
        delivery_complete=lambda req: _delivery_files_present(req.paths.output_dir),
    ).run(attempt_id, request, install_signal_handlers=True)
    # Experience maintenance is a derived, private projection.  A maintainer
    # failure must never retract a complete published batch or turn a
    # successful delivery into a failed attempt; its durable event keeps the
    # update retryable from the immutable batch evidence.
    if result.get("state") == "succeeded" and result.get("batch_id"):
        try:
            maintenance = _maintain_published_experience(
                config,
                target_id=target_id,
                batch_id=str(result["batch_id"]),
            )
            store.append_event(
                attempt_id,
                "experience_maintenance_succeeded",
                {"batch_id": result["batch_id"], "maintenance": maintenance},
            )
        except Exception as exc:
            # Keep the attempt terminally successful while preserving the
            # failure as append-only evidence for an operator/retry pass.
            store.append_event(
                attempt_id,
                "experience_maintenance_failed",
                {
                    "batch_id": result.get("batch_id"),
                    "error": repr(exc),
                    "retryable": True,
                },
            )
    _record_target_attempt(
        source,
        target_id,
        round_id=round_id,
        attempt_id=attempt_id,
        state=str(result.get("state")),
        attempt_dir=_attempt_dir(layout, round_id, target_id, attempt_id),
        resume_count=result.get("resume_count"),
    )
    _refresh_round_state(layout, round_id)
    return result


def _delivery_files_present(output_dir: Path) -> bool:
    """File-level completion check; content quality stays the validator's job."""
    try:
        return all(
            (output_dir / name).is_file()
            for name in (
                _DELIVERY_DATA_NAME,
                _DELIVERY_REPORT_NAME,
                _EXPERIENCE_DELTA_NAME,
            )
        )
    except OSError:
        return False


def _record_target_attempt(
    source: _DataSource,
    target_id: str,
    *,
    round_id: str,
    attempt_id: str,
    state: str,
    attempt_dir: Path,
    resume_count: object = None,
) -> None:
    # Visibility projection only: a failure here must never affect the
    # attempt itself. The attempt event stream remains the source of truth.
    try:
        projection: dict[str, Any] = {
            "round_id": round_id,
            "attempt_id": attempt_id,
            "state": state,
            "resume_count": resume_count,
            "recorded_at": _utc_now(),
            "attempt_dir": str(attempt_dir),
        }
        source.record_target_attempt(target_id, projection)
    except Exception:
        pass


def _launch_claim_is_within_grace(current: Mapping[str, Any]) -> bool:
    """Return whether a claimed attempt may still be between claim and spawn."""

    raw = current.get("launch_claimed_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        claimed_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if claimed_at.tzinfo is None:
        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
    age = (
        datetime.now(timezone.utc) - claimed_at.astimezone(timezone.utc)
    ).total_seconds()
    return -5.0 <= age < _ORPHAN_CLAIM_GRACE_SECONDS


def _published_batch_recovery(
    source: _DataSource,
    *,
    round_id: str,
    target_id: str,
    attempt_id: str,
) -> dict[str, Any] | None:
    """Rebuild the supervisor receipt from a fully admitted stable batch."""

    batch_id = _stable_batch_id(round_id, target_id, attempt_id)
    manifest = source.manifest()
    batches = manifest.get("batches")
    batch = batches.get(batch_id) if isinstance(batches, Mapping) else None
    if not isinstance(batch, Mapping):
        return None
    if (
        batch.get("batch_id") != batch_id
        or batch.get("round_id") != round_id
        or batch.get("target_id") != target_id
        or batch.get("attempt_id") != attempt_id
        or batch.get("status") != "available"
    ):
        return None

    publication = {
        "batch": copy.deepcopy(dict(batch)),
        "idempotent": True,
        "revision": manifest["revision"],
        "snapshot_id": manifest["snapshot_id"],
        "manifest_path": str(source.manifest_path),
        "reconciled": True,
    }
    return {
        "batch_id": batch_id,
        "manifest_ref": {
            "manifest_path": str(source.manifest_path),
            "snapshot_id": manifest["snapshot_id"],
            "revision": manifest["revision"],
        },
        "publication": publication,
        "reconciled": True,
    }


def _append_reconciled_terminal_event(
    store: _AttemptStore,
    attempt_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    event_id: str,
) -> dict[str, Any]:
    """Append one terminal fact while tolerating a supervisor completion race."""

    try:
        store.append_event(
            attempt_id,
            event_type,
            payload,
            event_id=event_id,
        )
    except _AttemptStoreError:
        current = store.current(attempt_id)
        if str(current.get("state")) in _TERMINAL:
            return current
        raise
    return store.current(attempt_id)


def _reconcile_orphaned_attempt(
    layout: _SynLayout,
    round_id: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Close a non-terminal attempt whose detached launcher no longer exists."""

    target_id = str(job["target_id"])
    latest = _latest_attempt(job)
    attempt_id = str(latest["attempt_id"])
    store = _attempt_store(layout, round_id, target_id)
    current = store.current(attempt_id)
    current_state = str(current.get("state", latest.get("state", "created")))
    if current_state not in _ACTIVE_ATTEMPT_STATES:
        return current

    raw_launcher_pid = latest.get("pid") or current.get("launcher_pid")
    launcher_pid = (
        raw_launcher_pid
        if isinstance(raw_launcher_pid, int)
        and not isinstance(raw_launcher_pid, bool)
        and raw_launcher_pid > 0
        else None
    )
    if launcher_pid is not None and _lifecycle_process_matches(
        launcher_pid,
        round_id=round_id,
        target_id=target_id,
        attempt_id=attempt_id,
    ):
        return current
    if launcher_pid is None and _launch_claim_is_within_grace(current):
        return current

    # The launcher can exit after Publisher has atomically admitted the batch
    # but before the supervisor appends delivery_published.  The stable batch
    # identity is sufficient to recover that exact commit without publishing
    # a second time.
    recovered = _published_batch_recovery(
        _DataSource(layout.data_source_dir),
        round_id=round_id,
        target_id=target_id,
        attempt_id=attempt_id,
    )
    current = store.current(attempt_id)
    if str(current.get("state")) in _TERMINAL:
        return current
    claim_id = str(
        current.get("launch_claim_id")
        or _launch_claim_id(round_id, target_id, attempt_id)
    )
    if recovered is not None:
        return _append_reconciled_terminal_event(
            store,
            attempt_id,
            "delivery_published",
            recovered,
            event_id=f"reconciled-publication-{claim_id}",
        )
    if str(current.get("desired_state")) == "stopped":
        return _append_reconciled_terminal_event(
            store,
            attempt_id,
            "attempt_stopped",
            {
                "reason": "orphaned_after_stop_request",
                "launcher_pid": launcher_pid,
            },
            event_id=f"reconciled-stop-{claim_id}",
        )
    return _append_reconciled_terminal_event(
        store,
        attempt_id,
        "attempt_failed",
        {
            "reason": "orphaned_launcher",
            "retryable": True,
            "last_state": current_state,
            "launcher_pid": launcher_pid,
        },
        event_id=f"reconciled-failure-{claim_id}",
    )


def _attempt_state(
    layout: _SynLayout, round_id: str, job: Mapping[str, Any]
) -> dict[str, Any]:
    target_id = str(job["target_id"])
    latest = _latest_attempt(job)
    attempt_id = str(latest["attempt_id"])
    store = _attempt_store(layout, round_id, target_id)
    try:
        current = store.current(attempt_id)
    except _AttemptStoreError:
        current = {"state": latest.get("state", "created"), "attempt_id": attempt_id}
    current_state = str(current.get("state", latest.get("state", "created")))
    # The lifecycle PID is the detached `_attempt` process-group leader.  The
    # attempt store may also contain the nested Codex PID; keep both explicit
    # and never expose the nested PID as a kill target for the lifecycle CLI.
    launcher_pid = latest.get("pid") or current.get("launcher_pid")
    launcher_pgid = latest.get("pgid") or current.get("launcher_pgid") or launcher_pid
    if current_state in _TERMINAL:
        # A terminal attempt can leave a stale PID in round.json.  Returning
        # no lifecycle PID keeps callers from signalling a reused PID.
        launcher_pid = None
        launcher_pgid = None

    experience_deltas: dict[str, Any] | None = None
    delivery = current.get("delivery")
    publication = delivery.get("publication") if isinstance(delivery, Mapping) else None
    batch = publication.get("batch") if isinstance(publication, Mapping) else None
    if isinstance(batch, Mapping):
        delta_path = batch.get("experience_delta_path")
        delta_sha = batch.get("experience_delta_sha256")
        if all(isinstance(value, str) and value for value in (delta_path, delta_sha)):
            # This is an immutable Data Source reference.  The lifecycle
            # facade only links the evidence; a future experience-maintainer
            # pass is responsible for merging candidates.
            experience_deltas = {
                "status": "pending_maintenance",
                "experience_delta": {"path": delta_path, "sha256": delta_sha},
            }
    return {
        "target_id": target_id,
        "attempt_id": attempt_id,
        "attempt_number": latest.get("attempt_number"),
        "state": current_state,
        "pid": launcher_pid,
        "pgid": launcher_pgid,
        "worker_pid": current.get("pid"),
        "worker_pgid": current.get("pgid"),
        "batch_id": current.get("batch_id"),
        "manifest_ref": current.get("manifest_ref"),
        "experience_deltas": experience_deltas,
        "experience_maintenance": current.get("experience_maintenance"),
        "current": current,
    }


def _refresh_round_state(layout: _SynLayout, round_id: str) -> dict[str, Any]:
    path = layout.round_manifest(round_id)
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        _recover_round_intent(layout, round_id)
        manifest = _round_manifest(layout, round_id)
        states: list[str] = []
        for item in manifest.get("targets", []):
            _reconcile_orphaned_attempt(layout, round_id, item)
            state = _attempt_state(layout, round_id, item)
            states.append(state["state"])
            item["state"] = state["state"]
            attempt_summary = item["attempts"][-1]
            attempt_summary["state"] = state["state"]
            if state["state"] in _TERMINAL:
                # Keep the last launcher identity as evidence even though
                # terminal status must not expose it as a signal target.
                for field in ("pid", "pgid"):
                    if attempt_summary.get(field) is not None:
                        attempt_summary[f"last_launcher_{field}"] = attempt_summary[
                            field
                        ]
            attempt_summary["pid"] = state.get("pid")
            attempt_summary["pgid"] = state.get("pgid")
            for field in ("batch_id", "manifest_ref", "experience_deltas"):
                value = state.get(field)
                if value is not None:
                    attempt_summary[field] = copy.deepcopy(value)
        if states and all(state == "succeeded" for state in states):
            manifest["state"] = "succeeded"
        elif any(
            state
            in {
                "created",
                "running",
                "starting",
                "resuming",
                "validating",
                "publishing",
            }
            for state in states
        ):
            manifest["state"] = "running"
        elif any(state == "succeeded" for state in states):
            manifest["state"] = "partial"
        elif any(state == "stopped" for state in states):
            manifest["state"] = "stopped"
        elif states and all(state == "failed" for state in states):
            manifest["state"] = "failed"
        manifest["updated_at"] = _utc_now()
        _write_round_manifest(path, manifest)
        return manifest


def _public_target_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project private attempt state onto the handbook's target contract."""

    result = {
        "target_id": state.get("target_id"),
        "state": state.get("state"),
    }
    for field in ("batch_id", "manifest_ref"):
        value = state.get(field)
        if value is not None:
            result[field] = copy.deepcopy(value)

    current = state.get("current")
    if isinstance(current, Mapping):
        failure = current.get("failure")
        if isinstance(failure, Mapping) and isinstance(failure.get("reason"), str):
            result["failure_reason"] = failure["reason"]
        elif isinstance(current.get("stop_reason"), str):
            result["stop_reason"] = current["stop_reason"]
    return result


def status(round_id: str) -> dict[str, Any]:
    config = _config()
    layout = _layout(config)
    manifest = _refresh_round_state(layout, _round_id(round_id))
    targets = [
        _public_target_state(_attempt_state(layout, _round_id(round_id), item))
        for item in manifest["targets"]
    ]
    result = {
        "schema_version": _SCHEMA_VERSION,
        "round_id": round_id,
        "state": manifest["state"],
        "syn_data_source": str(config.data_source_dir),
        "targets": targets,
    }
    if manifest.get("disposition") is not None:
        result["disposition"] = copy.deepcopy(manifest["disposition"])
    return result


def _active_targets_in_other_rounds(layout: _SynLayout, round_id: str) -> set[str]:
    """Collect target_ids whose latest attempt is active in another round.

    A target_id may be launched again under a new round while an earlier
    round's worker is still running.  Two live workers for one target would
    publish overlapping batches for the same capability, so every round
    belongs to a single concurrency domain per target.  A missing or empty
    rounds directory yields an empty set.
    """

    active: set[str] = set()
    synthesis_dir = layout.synthesis_dir
    if not synthesis_dir.is_dir():
        return active
    for entry in sorted(synthesis_dir.iterdir()):
        if entry.name == round_id or not entry.is_dir():
            continue
        try:
            other_round_id = _round_id(entry.name)
        except _SynError:
            continue
        if not layout.round_manifest(other_round_id).is_file():
            continue
        # round.json is a projection and can lag the append-only attempt
        # event stream.  Reconcile before reading any state so a finished
        # worker is not mistaken for an active one.
        manifest = _refresh_round_state(layout, other_round_id)
        for item in manifest.get("targets", []):
            if not isinstance(item, Mapping):
                continue
            state = _attempt_state(layout, other_round_id, item)
            if state["state"] in _ACTIVE_ATTEMPT_STATES:
                active.add(str(state["target_id"]))
    return active


def _launch_created_targets(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    # One dispatcher per run prevents concurrent commands from exceeding the
    # worker limit or starting the same target in two rounds.
    with _exclusive_file_lock(layout.synthesis_dir / ".dispatch.lock"):
        manifest = _refresh_round_state(layout, round_id)
        _require_open_round(manifest)
        return _dispatch_created_targets(config, layout, round_id, manifest)


def _dispatch_created_targets(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Start created attempts up to the round's durable worker limit."""

    configured = manifest.get(
        "max_concurrency",
        getattr(config, "max_concurrency", _MAX_TARGETS_PER_LAUNCH),
    )
    try:
        max_concurrency = max(1, int(configured))
    except (TypeError, ValueError) as exc:
        raise _SynContractError("round max_concurrency is invalid") from exc
    active_count = sum(
        1
        for item in manifest.get("targets", [])
        if isinstance(item, Mapping)
        and _attempt_state(layout, round_id, item)["state"] in _ACTIVE_ATTEMPT_STATES
    )
    busy_targets = _active_targets_in_other_rounds(layout, round_id)
    launched: list[str] = []
    skipped: list[str] = []
    for item in manifest.get("targets", []):
        if active_count >= max_concurrency:
            break
        if not isinstance(item, Mapping):
            continue
        target_id = str(item["target_id"])
        current = _attempt_state(layout, round_id, item)
        if current["state"] != "created":
            continue
        if target_id in busy_targets:
            skipped.append(target_id)
            continue
        latest = _latest_attempt(item)
        pid = _start_attempt(
            config, layout, round_id, target_id, str(latest["attempt_id"])
        )
        if pid is not None:
            launched.append(target_id)
            active_count += 1
    return launched, skipped


def _require_root_synthesis() -> None:
    """Apply the same run and role restrictions to lifecycle decisions."""
    if not synthesis_enabled():
        raise _SynContractError("Synthesis is disabled for this run")

    roles = {
        os.environ.get(name, "")
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {"", "meta"}
    if roles or any(
        os.environ.get(name)
        for name in (
            "AUTO_SFT_TASK_DIR",
            "AUTO_SFT_SEARCH_TASK_DIR",
            "AUTO_SFT_SYN_WORKER",
        )
    ):
        raise _SynContractError("Only the meta agent may manage synthesis")


def launch(round_id: str, requests_dir: Path) -> dict[str, Any]:
    _require_root_synthesis()
    config = _config()
    config.validate_tokenizer_runtime()
    layout = _layout(config)
    round_id = _round_id(round_id)
    if layout.round_manifest(round_id).is_file():
        with _exclusive_file_lock(_round_lock(layout, round_id)):
            _require_open_round(_round_manifest(layout, round_id))
    requested = _discover_requests(
        Path(requests_dir),
        allowed_root=config.run_dir,
        minimum_targets=1,
    )
    if len(requested) > _MAX_TARGETS_PER_LAUNCH:
        raise _SynContractError(
            f"launch requests {len(requested)} targets; the per-round "
            f"capacity is {_MAX_TARGETS_PER_LAUNCH}"
        )
    manifest = _freeze_and_manifest(config, layout, round_id, Path(requests_dir))
    # Existing rounds may have a stale projection after a worker exited. Do
    # not relaunch an attempt until its append-only state has been reconciled.
    manifest = _refresh_round_state(layout, round_id)
    if "max_concurrency" not in manifest:
        with _exclusive_file_lock(_round_lock(layout, round_id)):
            manifest = _round_manifest(layout, round_id)
            _require_open_round(manifest)
            manifest.setdefault(
                "max_concurrency",
                max(
                    1,
                    int(getattr(config, "max_concurrency", _MAX_TARGETS_PER_LAUNCH)),
                ),
            )
            manifest["updated_at"] = _utc_now()
            _write_round_manifest(layout.round_manifest(round_id), manifest)
    launched_targets, skipped_active_targets = _launch_created_targets(
        config, layout, round_id, manifest
    )
    result = status(round_id)
    result["launched_targets"] = launched_targets
    result["skipped_active_targets"] = skipped_active_targets
    _write_launch_receipt(config, layout, round_id, result)
    return result


def dispatch(round_id: str) -> dict[str, Any]:
    """Fill available slots from frozen pending requests, without new retries."""
    _require_root_synthesis()
    round_id = _round_id(round_id)
    config = _config()
    config.validate_tokenizer_runtime()
    layout = _layout(config)
    manifest = _refresh_round_state(layout, round_id)
    _require_open_round(manifest)
    launched, skipped = _launch_created_targets(config, layout, round_id, manifest)
    result = status(round_id)
    result.update(launched_targets=launched, skipped_active_targets=skipped)
    result["pending_targets"] = [
        target["target_id"]
        for target in result["targets"]
        if target["state"] == "created"
    ]
    _write_launch_receipt(config, layout, round_id, result, operation="dispatch")
    return result


def cancel_pending(round_id: str, *, reason: str) -> dict[str, Any]:
    """Stop unlaunched requests, leaving active and terminal workers intact."""
    _require_root_synthesis()
    round_id = _round_id(round_id)
    if not isinstance(reason, str) or not reason.strip():
        raise _SynContractError("cancel-pending requires a nonempty reason")
    config = _config()
    layout = _layout(config)
    cancelled = []
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        _recover_round_intent(layout, round_id)
        manifest = _round_manifest(layout, round_id)
        _require_open_round(manifest)
        for job in manifest["targets"]:
            target_id = str(job["target_id"])
            attempt = _latest_attempt(job)
            attempt_id = str(attempt["attempt_id"])
            store = _attempt_store(layout, round_id, target_id)
            current = store.current(attempt_id)
            if current.get("pending_cancellation") is not None:
                cancelled.append(target_id)
                continue
            if current.get("state") != "created":
                continue
            _require_attempt_quiescent(
                layout, round_id, target_id, attempt, current=current
            )
            store.append_event(
                attempt_id,
                "pending_cancelled",
                {
                    "reason": reason.strip(),
                    "operation": "cancel_pending",
                },
                event_id=f"cancel-pending-{attempt_id}",
            )
            cancelled.append(target_id)
    result = status(round_id)
    result.update(
        cancelled_targets=cancelled, launched_targets=[], skipped_active_targets=[]
    )
    _write_launch_receipt(config, layout, round_id, result, operation="cancel_pending")
    return result


def _create_retry_attempts(
    layout_or_config: _SynLayout | Any,
    layout_or_round_id: _SynLayout | str,
    round_id: str | None = None,
) -> list[str]:
    """Append fresh attempts for failed or stopped targets in a round."""

    # Keep the upstream private-call shape ``(config, layout, round_id)``
    # usable for old launchers and tests.  The config is only used to backfill
    # a missing durable concurrency limit, so the current implementation also
    # accepts the compact ``(layout, round_id)`` form.
    config = None
    if round_id is None:
        layout = layout_or_config
        round_id = str(layout_or_round_id)
    else:
        config = layout_or_config
        layout = layout_or_round_id
        round_id = str(round_id)
    round_id = _round_id(round_id)
    created: list[str] = []
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        _recover_round_intent(layout, round_id)
        manifest = _round_manifest(layout, round_id)
        _require_open_round(manifest)
        planned_attempts: list[dict[str, Any]] = []
        for item in manifest.get("targets", []):
            if not isinstance(item, dict):
                continue
            target_id = str(item["target_id"])
            _reconcile_orphaned_attempt(layout, round_id, item)
            current = _attempt_state(layout, round_id, item)
            if current["state"] not in {"failed", "stopped"}:
                continue
            if current.get("current", {}).get("pending_cancellation") is not None:
                continue
            attempts = item.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise _SynContractError(f"target {target_id} has no attempts")
            for previous_attempt in attempts:
                _require_attempt_quiescent(
                    layout, round_id, target_id, previous_attempt
                )
            previous = attempts[-1]
            numbers = [
                int(entry.get("attempt_number", 0))
                for entry in attempts
                if isinstance(entry, Mapping)
                and str(entry.get("attempt_number", "")).isdigit()
            ]
            attempt_number = max(numbers or [len(attempts)]) + 1
            attempt_id = f"attempt-{attempt_number:03d}"
            planned_attempts.append(
                {
                    "attempt_id": attempt_id,
                    "target_id": target_id,
                    "metadata": {
                        "frozen_target": item["frozen_target"],
                        "experience_versions": item.get("experience_versions", {}),
                        "retry_of": previous.get("attempt_id"),
                    },
                }
            )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "state": "created",
                    "pid": None,
                    "retry_of": previous.get("attempt_id"),
                }
            )
            item["state"] = "created"
            created.append(target_id)
        if "max_concurrency" not in manifest:
            configured = getattr(config, "max_concurrency", _MAX_TARGETS_PER_LAUNCH)
            try:
                manifest["max_concurrency"] = max(1, int(configured))
            except (TypeError, ValueError) as exc:
                raise _SynContractError("round max_concurrency is invalid") from exc
        manifest["updated_at"] = _utc_now()
        _commit_round_intent(
            layout, round_id, manifest, planned_attempts, operation="retry"
        )
    return created


def retry(round_id: str) -> dict[str, Any]:
    """Retry failed or stopped Synthesis targets without changing frozen inputs."""

    _require_root_synthesis()
    config = _config()
    config.validate_tokenizer_runtime()
    layout = _layout(config)
    round_id = _round_id(round_id)
    if not layout.round_manifest(round_id).is_file():
        raise _SynContractError(f"synthesis round does not exist: {round_id}")
    created = _create_retry_attempts(config, layout, round_id)
    manifest = _refresh_round_state(layout, round_id)
    launched, skipped = _launch_created_targets(config, layout, round_id, manifest)
    result = status(round_id)
    result.update(
        {
            "created_retry_targets": created,
            "launched_targets": launched,
            "skipped_active_targets": skipped,
        }
    )
    _write_launch_receipt(config, layout, round_id, result, operation="retry")
    return result


def _abandon_process_may_be_live(
    pid: int,
    layout: _SynLayout,
    round_id: str,
    target_id: str,
    attempt_id: str,
) -> bool:
    """Exclude PID reuse, retaining uncertainty about an orphan process group."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError, ValueError):
        return True
    process = Path(f"/proc/{pid}")
    try:
        state = (process / "stat").read_text().rsplit(")", 1)[-1].split()[0]
        # A group may outlive its leader. Without a live leader's identity,
        # do not mistake surviving children for an unrelated reused PID.
        if state in {"Z", "X"}:
            return True
        environment = (process / "environ").read_bytes().split(b"\0")
        if _lifecycle_process_matches(
            pid,
            round_id=round_id,
            target_id=target_id,
            attempt_id=attempt_id,
        ):
            run_entries = [
                entry.partition(b"=")[2]
                for entry in environment
                if entry.startswith(b"AUTO_SFT_RUN_DIR=")
            ]
            if not run_entries:
                return True
            process_run = Path(os.fsdecode(run_entries[0])).expanduser()
            if not process_run.is_absolute():
                process_run = (process / "cwd").resolve() / process_run
            return process_run.resolve() == layout.run_dir
        worker_root = os.fsencode(
            "AUTO_SFT_SYN_WRITE_ROOT="
            + str(layout.attempt_output_dir(round_id, target_id, attempt_id))
        )
        return b"AUTO_SFT_SYN_WORKER=1" in environment and worker_root in environment
    except (OSError, IndexError, ValueError):
        # The group existed, but its leader disappeared or could not be read.
        return True


def _require_attempt_quiescent(
    layout: _SynLayout,
    round_id: str,
    target_id: str,
    attempt: Mapping[str, Any],
    *,
    current: Mapping[str, Any] | None = None,
) -> None:
    attempt_id = str(attempt["attempt_id"])
    if current is None:
        current = _attempt_store(layout, round_id, target_id).current(attempt_id)
    pids = [
        current.get(key) for key in ("launcher_pgid", "launcher_pid", "pgid", "pid")
    ]
    pids.extend(
        attempt.get(key)
        for key in ("pgid", "pid", "last_launcher_pgid", "last_launcher_pid")
    )
    for pid in pids:
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            if _abandon_process_may_be_live(
                pid, layout, round_id, target_id, attempt_id
            ):
                raise _SynConflictError(
                    f"synthesis attempt {target_id}/{attempt_id} still has a live process"
                )


def _abandon_evidence(
    layout: _SynLayout,
    round_id: str,
    manifest: Mapping[str, Any],
    *,
    check_live_processes: bool = True,
) -> dict[str, Any]:
    """Bind fallback to terminal event streams, never mutable projections."""
    if str(manifest.get("round_id")) != round_id:
        raise _SynContractError("synthesis round identity does not match abandonment")
    if (
        Path(str(manifest.get("data_source_dir", ""))).resolve()
        != layout.data_source_dir
    ):
        raise _SynContractError("synthesis Data Source changed since launch")
    jobs = manifest.get("targets")
    if not isinstance(jobs, list) or not jobs:
        raise _SynContractError("abandon requires a nonempty synthesis round")
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise _SynContractError("invalid synthesis target")
        target_id = _validate_safe_id(job.get("target_id"), "target_id")
        if target_id in seen:
            raise _SynContractError("duplicate synthesis target")
        seen.add(target_id)
        attempts = job.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise _SynContractError(f"target {target_id} has no attempt evidence")
        store = _attempt_store(layout, round_id, target_id)
        records = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise _SynContractError("invalid synthesis attempt")
            attempt_id = _validate_safe_id(attempt.get("attempt_id"), "attempt_id")
            current = store.current(attempt_id)
            if (
                current.get("round_id") != round_id
                or current.get("target_id") != target_id
            ):
                raise _SynContractError("synthesis attempt identity mismatch")
            if current.get("state") not in {"failed", "stopped"}:
                raise _SynContractError(
                    "abandon requires every attempt to be terminal failed/stopped, with no success"
                )
            if check_live_processes:
                # Terminal events may precede process exit. First abandonment
                # must exclude a surviving launcher or nested worker group.
                _require_attempt_quiescent(
                    layout, round_id, target_id, attempt, current=current
                )
            records.append(
                {
                    "attempt_id": attempt_id,
                    "state": current["state"],
                    "events_sha256": hashlib.sha256(
                        store.events_path(attempt_id).read_bytes()
                    ).hexdigest(),
                }
            )
        recorded_ids = [record["attempt_id"] for record in records]
        actual_ids = {path.parent.name for path in store.root.glob("*/events.jsonl")}
        if (
            len(set(recorded_ids)) != len(recorded_ids)
            or set(recorded_ids) != actual_ids
        ):
            raise _SynContractError("synthesis attempt inventory changed")
        targets.append({"target_id": target_id, "attempts": records})
    actual_targets = {
        path.parent.parent.parent.name
        for path in (layout.round_dir(round_id) / "targets").glob(
            "*/attempts/*/events.jsonl"
        )
    }
    if seen != actual_targets:
        raise _SynContractError("synthesis target inventory changed")
    source = _DataSource(layout.data_source_dir)
    if not source.manifest_path.is_file():
        raise _SynContractError("launched synthesis has no Data Source manifest")
    source_manifest = source.manifest()
    for collection in ("batches", "retired_batches"):
        for batch in source_manifest.get(collection, {}).values():
            if str(batch.get("round_id")) == round_id:
                raise _SynConflictError("synthesis round already has a published batch")
    return {
        "round_id": round_id,
        "data_source_dir": str(layout.data_source_dir),
        "targets": targets,
    }


def _evidence_sha256(evidence: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_abandon_receipt(
    layout: _SynLayout, round_id: str, manifest: Mapping[str, Any], receipt: Any
) -> dict[str, Any]:
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != _SCHEMA_VERSION
        or receipt.get("operation") != "abandon"
        or not isinstance(receipt.get("reason"), str)
        or not receipt["reason"].strip()
        or not isinstance(receipt.get("abandoned_at"), str)
        or not receipt["abandoned_at"]
    ):
        raise _SynContractError("invalid synthesis abandonment receipt")
    # A committed disposition prevents this round from starting new attempts.
    # Recheck durable terminal evidence, not historical PIDs that can be reused.
    evidence = _abandon_evidence(layout, round_id, manifest, check_live_processes=False)
    if any(receipt.get(key) != value for key, value in evidence.items()) or receipt.get(
        "evidence_sha256"
    ) != _evidence_sha256(evidence):
        raise _SynConflictError("synthesis abandonment evidence changed")
    return receipt


def verify_abandonment(
    run_dir: Path, round_id: str, decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Check the durable fallback disposition before admitting training data."""
    source = Path(
        os.environ.get("AUTO_SFT_SYN_DATA_SOURCE_DIR") or run_dir / "syn_data_source"
    )
    if not source.is_absolute():
        source = run_dir / source
    layout = _SynLayout(run_dir, source)
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        manifest = _round_manifest(layout, round_id)
        receipt = _validate_abandon_receipt(
            layout, round_id, manifest, manifest.get("disposition")
        )
        if decision.get("abandon_receipt") != receipt:
            raise _SynConflictError(
                "Synthesis decision has no matching abandonment receipt"
            )
        return receipt


def abandon(round_id: str, *, reason: str) -> dict[str, Any]:
    """Close an unsuccessful Synthesis round while retaining its training round."""
    _require_root_synthesis()
    if not isinstance(reason, str) or not reason.strip():
        raise _SynContractError("abandon requires a nonempty reason")
    config = _config()
    layout = _layout(config)
    round_id = _round_id(round_id)
    if not layout.round_manifest(round_id).is_file():
        raise _SynContractError(f"synthesis round does not exist: {round_id}")
    # Refresh outside the round lock: reconciliation also takes that lock.
    # Subsequent validation uses the event streams again while excluding retry.
    _refresh_round_state(layout, round_id)
    path = config.run_dir / "syn_decision" / f"{round_id}.json"
    with _exclusive_file_lock(_round_lock(layout, round_id)):
        with _exclusive_file_lock(path.with_name(path.name + ".lock")):
            decision = _load_json(path, "syn decision")
            if (
                str(decision.get("round_id")) != round_id
                or decision.get("launch") is not True
            ):
                raise _SynContractError(
                    "abandon requires the original launch=true Synthesis decision"
                )
            history = decision.get("launch_receipts")
            launch_receipt = (
                history[-1]
                if isinstance(history, list) and history
                else decision.get("launch_receipt")
            )
            if (
                not isinstance(launch_receipt, dict)
                or not launch_receipt.get("executed_at")
                or not isinstance(launch_receipt.get("launched_targets"), list)
                or not isinstance(launch_receipt.get("skipped_active_targets"), list)
            ):
                raise _SynContractError("abandon requires a complete launch_receipt")
            manifest = _round_manifest(layout, round_id)
            receipt = manifest.get("disposition")
            if receipt is not None:
                receipt = _validate_abandon_receipt(layout, round_id, manifest, receipt)
                if receipt["reason"] != reason.strip():
                    raise _SynConflictError(
                        "synthesis abandonment already records a different reason"
                    )
            else:
                evidence = _abandon_evidence(layout, round_id, manifest)
                receipt = {
                    "schema_version": _SCHEMA_VERSION,
                    "operation": "abandon",
                    "reason": reason.strip(),
                    "abandoned_at": _utc_now(),
                    **evidence,
                    "evidence_sha256": _evidence_sha256(evidence),
                }
                manifest["disposition"] = receipt
                _write_round_manifest(layout.round_manifest(round_id), manifest)
            # Round disposition is canonical and written first. Repeating the
            # command repairs an interrupted decision write without new facts.
            decision["abandon_receipt"] = receipt
            _atomic_write_json(path, decision)
    return {
        "round_id": round_id,
        "state": manifest["state"],
        "disposition": receipt,
        "decision_path": str(path),
        "continue_training_round": round_id,
    }


def _write_launch_receipt(
    config: _SynConfig,
    layout: _SynLayout,
    round_id: str,
    result: dict[str, Any],
    *,
    operation: str = "launch",
) -> None:
    """Stamp the execution receipt into the round's syn_decision file.

    The decision file is otherwise the meta agent's self-declaration; a
    ``launch: true`` decision without a ``launch_receipt`` therefore means
    "declared but never executed". Only the launch command writes this key.
    """
    path = config.run_dir / "syn_decision" / f"{round_id}.json"
    receipt = {
        "executed_at": _utc_now(),
        "operation": operation,
        "launched_targets": list(result.get("launched_targets", [])),
        "skipped_active_targets": list(result.get("skipped_active_targets", [])),
        "synthesis_round_dir": str(layout.round_dir(round_id)),
    }
    try:
        payload: dict[str, Any] = {}
        with _exclusive_file_lock(path.with_name(path.name + ".lock")):
            if path.is_file():
                existing = _load_json(path, "syn decision")
                if isinstance(existing, dict):
                    payload = existing
            history = payload.get("launch_receipts")
            if not isinstance(history, list):
                history = []
            history.append(receipt)
            payload["launch_receipts"] = history
            payload["launch_receipt"] = receipt
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(path, payload)
        result["launch_receipt_path"] = str(path)
    except Exception as exc:
        # The launch itself succeeded; surface the receipt failure to the
        # caller instead of destroying a completed launch over bookkeeping.
        result["launch_receipt_error"] = repr(exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = sys.argv[1:] if argv is None else list(argv)

    # The detached target process is an implementation detail, not part of the
    # handbook/CLI contract.  Parse it separately so argparse's public command
    # inventory cannot expose the private lifecycle topology in ``--help``.
    if arguments and arguments[0] == "_attempt":
        internal = argparse.ArgumentParser(add_help=False)
        internal.set_defaults(command="_attempt")
        internal.add_argument("_command")
        internal.add_argument("--round-id", required=True)
        internal.add_argument("--target-id", required=True)
        internal.add_argument("--attempt-id", required=True)
        return internal.parse_args(arguments)

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--round-id", required=True)
    launch_parser.add_argument("--requests-dir", required=True, type=Path)
    retry_parser = sub.add_parser("retry")
    retry_parser.add_argument("--round-id", required=True)
    abandon_parser = sub.add_parser("abandon")
    abandon_parser.add_argument("--round-id", required=True)
    abandon_parser.add_argument("--reason", required=True)
    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--round-id", required=True)
    cancel_parser = sub.add_parser("cancel-pending")
    cancel_parser.add_argument("--round-id", required=True)
    cancel_parser.add_argument("--reason", required=True)
    return parser.parse_args(arguments)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "_attempt":
        result = _attempt_worker(
            _config(),
            _layout(_config()),
            _round_id(args.round_id),
            _validate_safe_id(args.target_id, "target_id"),
            _validate_safe_id(args.attempt_id, "attempt_id"),
        )
    elif args.command == "retry":
        result = retry(args.round_id)
    elif args.command == "abandon":
        result = abandon(args.round_id, reason=args.reason)
    elif args.command == "dispatch":
        result = dispatch(args.round_id)
    elif args.command == "cancel-pending":
        result = cancel_pending(args.round_id, reason=args.reason)
    else:
        result = launch(args.round_id, args.requests_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
