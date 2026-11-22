"""Provenance and usage recording for data taken from the Synthesis pool.

The pool is browsable. A consumer reads a pinned view, the actual batch rows,
quality reports, and the usage history, then assembles the training file
that tests its current research hypothesis. The consumer decides which batches
and rows to use, how to compare them with the current base recipe, and how much
exposure to give each one. This module records the resulting artifact rather
than pretending that publication or row count proves usefulness.

What does need machinery is only the part the agent cannot self-certify:

* **What was actually used.** Derived here by fingerprinting the assembled
  training file and matching each row against the immutable per-batch token
  sidecars, so the record comes from the artifact rather than from the agent's
  claim. The verified party cannot also be the verifier.
* **A consistent view of the pool.** Resolved against a pinned snapshot: the
  pool is append-only and may be written concurrently by several targets in the
  same round, so reading the mutable root manifest can yield a half-written view.
* **Retry-stable usage ids.** Derived from content, so a recording that crashes
  and retries appends the *same* event instead of a duplicate. The event log is
  append-only; a duplicate usage event permanently corrupts the accounting of
  how much of a batch was really trained on, and nothing surfaces the error.

Selection remains a research decision rather than a universal recipe. The
caller should materialize a deliberate row-level sample, account for total and
loss tokens, inspect overlap with the base data, and record why the exposure is
appropriate for the current target. The API deliberately verifies what was
used, while leaving the research judgment with the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from autosft.synthesis._internal.filesystem import (
    message_fingerprint,
    read_json_object as _read_object,
    reject_symlink_components,
    sha256_file,
    validate_safe_id,
)
from autosft.synthesis._internal.storage import stable_usage_event_id
from autosft.synthesis.errors import SynContractError, SynValidationError

__all__ = ["pool_view", "record_usage"]


def _reject_symlink_components(path: Path, label: str) -> None:
    """Recording runs in the meta agent's context, where the read side is not
    otherwise constrained, so a symlink inside the pool would be a way out of
    the pinned snapshot."""

    reject_symlink_components(path, label, error=SynContractError)


def _regular_directory(path: Path, label: str) -> Path:
    resolved = Path(path)
    _reject_symlink_components(resolved, label)
    if not resolved.is_dir():
        raise SynContractError(f"{label} is missing or not a directory: {resolved}")
    return resolved


def _regular_file(path: Path, label: str, *, root: Path | None = None) -> Path:
    resolved = Path(path)
    _reject_symlink_components(resolved, label)
    if not resolved.is_file():
        raise SynContractError(f"{label} is missing or not a regular file: {resolved}")
    if root is not None:
        try:
            resolved.resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise SynContractError(f"{label} escapes {root}: {resolved}") from exc
    return resolved


#: The pool's row identity (single shared definition, see ``message_fingerprint``).
fingerprint = message_fingerprint


def _snapshot_root(data_source_dir: Path, snapshot_id: str) -> Path:
    root = _regular_directory(Path(data_source_dir), "Synthesis Data Source directory")
    safe_snapshot_id = validate_safe_id(snapshot_id, "snapshot_id")
    snapshots_root = _regular_directory(
        root / "snapshots", "Synthesis Data Source snapshots directory"
    )
    candidate = _regular_directory(
        snapshots_root / safe_snapshot_id, "pinned data source snapshot"
    )
    try:
        candidate.resolve().relative_to(snapshots_root.resolve())
    except ValueError as exc:
        raise SynContractError(
            f"pinned data source snapshot escapes snapshots directory: {candidate}"
        ) from exc
    return candidate


def _latest_snapshot_id(data_source_dir: Path) -> str:
    manifest = _read_object(
        _regular_file(Path(data_source_dir) / "manifest.json", "Synthesis manifest"),
        "Synthesis manifest",
    )
    snapshot_id = manifest.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise SynContractError("Synthesis manifest has no snapshot_id to pin")
    return snapshot_id


def pool_view(
    data_source_dir: str | Path, *, snapshot_id: str | None = None
) -> dict[str, Any]:
    """Return a pinned, read-only view of the pool for browsing.

    Pins a snapshot so the batch set cannot shift while the caller is deciding.
    Entries carry what selection judgment needs -- including ``target_snapshot_id``
    and ``round_id``, which say what evidence a batch was generated against and
    therefore why two batches for the same target are not interchangeable.
    """

    source = Path(data_source_dir)
    pinned = snapshot_id or _latest_snapshot_id(source)
    root = _snapshot_root(source, pinned)
    manifest = _read_object(
        _regular_file(root / "manifest.json", "snapshot manifest", root=root),
        "snapshot manifest",
    )
    batches = manifest.get("batches")
    if not isinstance(batches, Mapping):
        raise SynContractError("snapshot manifest has no batches object")
    available = {
        batch_id: dict(entry)
        for batch_id, entry in batches.items()
        if isinstance(entry, Mapping) and entry.get("status") == "available"
    }
    return {
        "snapshot_id": pinned,
        "snapshot_dir": str(root),
        "revision": manifest.get("revision"),
        "batches": available,
    }


def _sidecar_index(
    root: Path, entry: Mapping[str, Any], batch_id: str
) -> dict[str, dict[str, int]]:
    """Map every published row fingerprint of one batch to its token counts.

    The sidecar is verified against the digest the manifest recorded at publish
    time, so a batch whose bytes changed cannot silently back a usage claim.
    """

    sidecar_rel = entry.get("token_counts_path")
    digest = entry.get("token_counts_sha256")
    if not isinstance(sidecar_rel, str) or not isinstance(digest, str):
        raise SynContractError(f"batch {batch_id} has no verifiable token sidecar")
    path = _regular_file(
        root / sidecar_rel, f"batch {batch_id} token sidecar", root=root
    )
    actual = sha256_file(path)
    if actual != digest:
        raise SynValidationError(
            f"batch {batch_id} token sidecar digest mismatch: {actual} != {digest}"
        )
    payload = _read_object(path, f"batch {batch_id} token sidecar")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SynContractError(f"batch {batch_id} token sidecar has no rows")
    index: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = row.get("message_fingerprint")
        if isinstance(key, str) and key:
            index.setdefault(
                key,
                {
                    "total_tokens": int(row.get("total_tokens") or 0),
                    "loss_tokens": int(row.get("loss_tokens") or 0),
                },
            )
    return index


def _used_fingerprints(path: Path) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    total = 0
    target = _regular_file(path, "assembled training data")
    with target.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SynValidationError(
                    f"assembled training data line {lineno} is not JSON: {exc}"
                ) from exc
            messages = row.get("messages") if isinstance(row, Mapping) else None
            if messages is None:
                continue
            key = fingerprint(messages)
            counts[key] = counts.get(key, 0) + 1
    return counts, total


def record_usage(
    data_source_dir: str | Path,
    used_data_path: str | Path,
    *,
    training_run_id: str,
    selection_id: str,
    recipe_ref: str | None = None,
    snapshot_id: str | None = None,
    append: bool = True,
) -> dict[str, Any]:
    """Derive, from the assembled training file, what the pool actually contributed.

    ``used_data_path`` is the caller's own assembled training set. Every row is
    fingerprinted and matched against the published fingerprints of each batch in
    the pinned snapshot, so per-batch row and token counts are computed from the
    artifact -- not accepted from the caller. Rows matching no batch are reported
    as ``unattributed_rows`` rather than silently dropped.

    One ``usage`` event is appended per contributing batch, with an id derived
    from ``(training_run_id, batch_id, selection_id)``. Re-running after a crash
    reproduces the same ids, so the append is idempotent. Callers must never
    construct these ids themselves.

    ``recipe_ref`` is whatever the host calls the thing that defined this
    training set; it is recorded verbatim and defaults to the data path, since
    not every host has a separate recipe artifact.
    """

    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise SynContractError("training_run_id must be a non-empty string")
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise SynContractError("selection_id must be a non-empty string")

    view = pool_view(data_source_dir, snapshot_id=snapshot_id)
    root = Path(view["snapshot_dir"])
    remaining, total_rows = _used_fingerprints(Path(used_data_path))

    per_batch: list[dict[str, Any]] = []
    attributed = 0
    for batch_id, entry in sorted(view["batches"].items()):
        index = _sidecar_index(root, entry, batch_id)
        rows = tokens = loss = unique = 0
        for key, counts in index.items():
            hits = remaining.get(key, 0)
            if not hits:
                continue
            unique += 1
            rows += hits
            tokens += counts["total_tokens"] * hits
            loss += counts["loss_tokens"] * hits
            # A row published by more than one batch is attributed once, to the
            # first batch in sorted order, so totals never double-count.
            remaining[key] = 0
        if rows:
            attributed += rows
            per_batch.append(
                {
                    "batch_id": batch_id,
                    "target_id": entry.get("target_id"),
                    "round_id": entry.get("round_id"),
                    # materialized_rows counts occurrences in the training file;
                    # unique_rows counts distinct published rows matched, so an
                    # upsampled batch reads as e.g. 4000 materialized / 1000
                    # unique instead of being indistinguishable from inflation.
                    "materialized_rows": rows,
                    "unique_rows": unique,
                    "materialized_tokens": tokens,
                    "materialized_loss_tokens": loss,
                    "event_id": stable_usage_event_id(
                        training_run_id, batch_id, selection_id
                    ),
                }
            )

    record: dict[str, Any] = {
        "selection_id": selection_id,
        "training_run_id": training_run_id,
        "snapshot_id": view["snapshot_id"],
        "manifest_revision": view["revision"],
        "used_data_path": str(Path(used_data_path)),
        "used_data_sha256": sha256_file(Path(used_data_path)),
        "total_rows": total_rows,
        "attributed_rows": attributed,
        "unattributed_rows": total_rows - attributed,
        "per_batch": per_batch,
    }
    if append:
        appended, append_errors = _append_events(
            data_source_dir,
            training_run_id,
            selection_id,
            per_batch,
            dataset_ref=str(Path(used_data_path)),
            recipe_ref=recipe_ref or str(Path(used_data_path)),
        )
        record["appended"] = appended
        if append_errors:
            record["append_errors"] = append_errors
    return record


def _append_events(
    data_source_dir: str | Path,
    training_run_id: str,
    selection_id: str,
    per_batch: Iterable[Mapping[str, Any]],
    *,
    dataset_ref: str,
    recipe_ref: str,
) -> tuple[list[str], list[dict[str, str]]]:
    from autosft.synthesis._internal.storage import DataSource, append_usage

    source = DataSource(Path(data_source_dir))
    appended: list[str] = []
    # One batch failing validation must not void the whole round's books: the
    # remaining batches still get their usage events, and the failure is
    # reported per batch instead of erasing the derived record.
    errors: list[dict[str, str]] = []
    for item in per_batch:
        try:
            append_usage(
                source,
                {
                    "event_id": item["event_id"],
                    "event_type": "usage",
                    "training_run_id": training_run_id,
                    "selection_id": selection_id,
                    "batch_id": item["batch_id"],
                    "materialized_rows": item["materialized_rows"],
                    "unique_rows": item.get("unique_rows"),
                    "materialized_tokens": item["materialized_tokens"],
                    "materialized_loss_tokens": item["materialized_loss_tokens"],
                    "dataset_ref": dataset_ref,
                    "recipe_ref": recipe_ref,
                },
            )
        except (SynContractError, SynValidationError) as exc:
            errors.append({"batch_id": str(item["batch_id"]), "error": str(exc)})
            continue
        appended.append(str(item["event_id"]))
    return appended, errors
