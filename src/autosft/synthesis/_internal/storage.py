"""Durable storage boundary for Synthesis: Data Source, publication, feedback, tokens.

Four faces of the same append-only pool live together:

* ``DataSource`` -- atomic manifest, immutable snapshots, and append-only
  event storage shared by the Synthesis black box and training consumers.
* ``Publisher`` -- admission and immutable publication of validated worker
  deliveries into that Data Source.
* the feedback appenders -- the sanctioned append-only API for usage
  events.
* ``count_jsonl_tokens`` -- Synthesis batch token counting via the same Qwen3 path
  used by SFT conversion.

Publisher commits through the same in-module manifest transaction the Data
Source uses (``_load_current_unlocked``/``_commit_unlocked`` under the single
manifest lock); merging the two sides makes that a normal module-internal
collaboration instead of a cross-module reach into private methods.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import stat
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from autosft.synthesis.errors import (
    SynConflictError,
    SynContractError,
    SynIOError,
    SynValidationError,
)
from autosft.backends.megatron.tokenization_runtime import (
    missing_tokenizer_files,
    tokenizer_subprocess_env,
)
from autosft.synthesis._internal.filesystem import (
    atomic_write_text,
    canonical_json,
    make_read_only_tree,
    read_json_object,
    reject_symlink_components,
    sha256_file,
    sha256_text,
    utc_now,
    validate_safe_id,
)


SCHEMA_VERSION = "1.0"
EVENT_KINDS = ("batch_status", "usage")

README = """# Synthesis Data Source

This directory is the durable public boundary of Synthesis System. The operating contract for the consumer is `skills/meta/synthesis-handbook/SKILL.md`; this file only describes the layout.

Read `manifest.json` instead of scanning directories. Its `batches` table holds at most one entry per target -- the latest successfully delivered batch. A newer batch for the same target moves its predecessor out of `batches` (and out of `batches/`) into `retired_batches` / `_internal/retired/`; retired batches are not selectable and exist only for dedup and worker reference. `target_attempts` records each target's most recent worker attempt (`state`: running / succeeded / failed).

All paths declared by the manifest are relative to this directory. Batch files, target snapshots, delivery receipts, and event records are immutable; new facts are appended under `events/` and never rewrite prior events.

`events/usage.jsonl` is appended by the host training pipeline and records which batch contributed how many rows to which training run. Do not edit event files directly.
"""


# ---------------------------------------------------------------------------
# Data Source: atomic manifest, immutable snapshots, append-only events
#
# ---------------------------------------------------------------------------


def _atomic_text(path: Path, content: str) -> None:
    _reject_symlink_components(path, "atomic Data Source write")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, "atomic Data Source write")
    atomic_write_text(path, content)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _snapshot_id(manifest: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("snapshot_id", None)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise SynContractError("manifest.revision must be a non-negative integer")
    return f"r{revision:08d}-{digest[:20]}"


def _root_path(value: Any) -> Path:
    candidate = getattr(value, "data_source_dir", value)
    try:
        raw = Path(candidate).expanduser()
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        raw = Path(os.path.abspath(raw))
        _reject_symlink_components(raw, "Synthesis Data Source root")
        if raw.is_symlink():
            raise SynContractError(
                f"Synthesis Data Source root cannot be a symlink: {raw}"
            )
        return raw.resolve(strict=False)
    except (TypeError, ValueError, OSError, UnicodeError) as exc:
        raise SynContractError("data source must be a path or SynLayout") from exc


def _reject_symlink_components(path: Path, label: str) -> None:
    reject_symlink_components(path, label, error=SynContractError)


class DataSource:
    """Atomic manifest, immutable snapshots, and append-only event storage."""

    def __init__(self, root: Any):
        self.root = _root_path(root)
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".manifest.lock"
        self.events_dir = self.root / "events"
        self.snapshots_dir = self.root / "snapshots"

    @contextmanager
    def locked(self) -> Iterator[None]:
        self._ensure_storage_layout()
        if self.lock_path.is_symlink():
            raise SynContractError(
                f"Synthesis Data Source lock cannot be a symlink: {self.lock_path}"
            )
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def ensure(self) -> dict[str, Any]:
        self._ensure_storage_layout()
        if not (self.root / "README.md").exists():
            _atomic_text(self.root / "README.md", README)
        else:
            self._regular_file(self.root / "README.md", "Synthesis Data Source README")

        with self.locked():
            if self.manifest_path.exists():
                manifest = self._load_current_unlocked()
                self._validate_manifest(manifest)
                self._write_snapshot_unlocked(manifest)
                self._write_event_projections_unlocked(manifest)
                return copy.deepcopy(manifest)

            now = utc_now()
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "data_source_type": "syn_batches",
                "revision": 0,
                "created_at": now,
                "updated_at": now,
                "targets": {},
                "batches": {},
                "events": {kind: [] for kind in EVENT_KINDS},
                "event_cursors": {kind: 0 for kind in EVENT_KINDS},
            }
            manifest["snapshot_id"] = _snapshot_id(manifest)
            _atomic_json(self.manifest_path, manifest)
            self._write_snapshot_unlocked(manifest)
            return copy.deepcopy(manifest)

    def _ensure_storage_layout(self) -> None:
        """Create the fixed Data Source layout only through safe components."""

        _reject_symlink_components(self.root, "Synthesis Data Source root")
        if self.root.exists() or self.root.is_symlink():
            self._regular_directory(self.root, "Synthesis Data Source root")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self._regular_directory(self.root, "Synthesis Data Source root")

        directories = (
            (self.root / "targets", "Synthesis Data Source targets directory"),
            (self.root / "batches", "Synthesis Data Source batches directory"),
            (self.retired_dir, "Synthesis Data Source retired batches directory"),
            (self.events_dir, "Synthesis Data Source events directory"),
            (self.snapshots_dir, "Synthesis Data Source snapshots directory"),
        )
        for path, label in directories:
            _reject_symlink_components(path, label)
            if path.exists() or path.is_symlink():
                self._regular_directory(path, label)
            else:
                path.mkdir(parents=True, exist_ok=True)
                self._regular_directory(path, label)

        # Event kind directories are fixed write targets.  Validate/create them
        # before any append or projection operation can open a path beneath one.
        for kind in EVENT_KINDS:
            path = self.events_dir / kind
            _reject_symlink_components(path, f"Synthesis {kind} events directory")
            if path.exists() or path.is_symlink():
                self._regular_directory(path, f"Synthesis {kind} events directory")
            else:
                path.mkdir(parents=True, exist_ok=True)
                self._regular_directory(path, f"Synthesis {kind} events directory")

        for path, label in (
            (self.manifest_path, "Synthesis Data Source manifest"),
            (self.root / "README.md", "Synthesis Data Source README"),
        ):
            _reject_symlink_components(path, label)
            if path.exists() or path.is_symlink():
                self._regular_file(path, label)
        for kind in EVENT_KINDS:
            projection = self.events_dir / f"{kind}.jsonl"
            _reject_symlink_components(projection, f"Synthesis {kind} event projection")
            if projection.exists() or projection.is_symlink():
                self._regular_file(projection, f"Synthesis {kind} event projection")
        _reject_symlink_components(self.lock_path, "Synthesis Data Source lock")
        if self.lock_path.exists() or self.lock_path.is_symlink():
            self._regular_file(self.lock_path, "Synthesis Data Source lock")

        # Existing children must not redirect the canonical target/batch roots.
        for directory, label in (
            (self.root / "targets", "target entry"),
            (self.root / "batches", "batch entry"),
        ):
            for child in directory.iterdir():
                if child.is_symlink() or not child.is_dir():
                    raise SynContractError(
                        f"Synthesis {label} must be a regular directory: {child}"
                    )

    def manifest(self) -> dict[str, Any]:
        # ``ensure`` reads an existing root manifest while rebuilding its
        # projections.  Validate its inode type first so a FIFO, directory,
        # or symlink cannot be followed before the public check below.
        if self.manifest_path.exists() or self.manifest_path.is_symlink():
            self._regular_file(self.manifest_path, "Synthesis Data Source manifest")
        self.ensure()
        path = self._regular_file(self.manifest_path, "Synthesis Data Source manifest")
        manifest = read_json_object(path, "Synthesis Data Source manifest")
        self._validate_manifest(manifest)
        return manifest

    @property
    def retired_dir(self) -> Path:
        return self.root / "_internal" / "retired"

    def retire_batch(self, batch_id: str, *, superseded_by: str) -> dict[str, Any]:
        """Move a superseded batch out of the public surface.

        The manifest entry leaves ``batches`` for ``retired_batches`` and the
        directory leaves ``batches/`` for ``_internal/retired/``; the files
        themselves stay immutable. Dedup and worker snapshots still see the
        rows; the selecting consumer no longer does. Idempotent: a batch that
        is already retired is left alone.
        """
        self.ensure()
        with self.locked():
            manifest = self._load_current_unlocked()
            retired = manifest.setdefault("retired_batches", {})
            entry = manifest["batches"].get(batch_id)
            if entry is None:
                if batch_id in retired:
                    return {
                        "revision": manifest["revision"],
                        "snapshot_id": manifest["snapshot_id"],
                    }
                raise SynContractError(f"cannot retire unknown batch {batch_id!r}")
            old_rel = str(entry["path"])
            new_rel = _relative(self.retired_dir / batch_id, self.root)
            src_dir = self.root / old_rel
            dst_dir = self.retired_dir / batch_id
            self._regular_directory(src_dir, "retiring batch directory")
            if dst_dir.exists() or dst_dir.is_symlink():
                raise SynConflictError(f"retired batch already present: {dst_dir}")
            moved = copy.deepcopy(entry)
            for key, value in list(moved.items()):
                if key.endswith("_path") or key in ("path", "delivery_receipt"):
                    if isinstance(value, str) and value.startswith(old_rel):
                        moved[key] = new_rel + value[len(old_rel) :]
            moved["status"] = "unavailable"
            moved["superseded_by"] = superseded_by
            moved["retired_at"] = utc_now()
            next_manifest = copy.deepcopy(manifest)
            del next_manifest["batches"][batch_id]
            next_manifest.setdefault("retired_batches", {})[batch_id] = moved
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            # Moving a read-only directory across parents requires updating
            # its '..' entry on some filesystems. Keep payload files immutable
            # and restore the directory mode after the locked transaction.
            original_mode = src_dir.stat().st_mode & 0o777
            src_dir.chmod(original_mode | 0o200)
            try:
                os.replace(src_dir, dst_dir)
                try:
                    return self._commit_unlocked(next_manifest, current=manifest)
                except Exception:
                    os.replace(dst_dir, src_dir)
                    raise
            finally:
                destination = dst_dir if dst_dir.exists() else src_dir
                if destination.exists():
                    destination.chmod(original_mode)

    def record_target_attempt(
        self, target_id: str, attempt: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Project the latest worker attempt per target into the manifest.

        Without this, a target whose worker failed is indistinguishable from
        one still running: both show "no batches". The manifest is the single
        entry point consumers already read, so the projection puts worker
        liveness there; the attempt event stream stays the source of truth.
        """
        if not isinstance(attempt, Mapping):
            raise SynContractError("attempt projection must be an object")
        self.ensure()
        with self.locked():
            manifest = self._load_current_unlocked()
            attempts = manifest.get("target_attempts")
            if not isinstance(attempts, dict):
                attempts = {}
                manifest["target_attempts"] = attempts
            attempts[str(target_id)] = copy.deepcopy(dict(attempt))
            return self._commit_unlocked(manifest, current=manifest)

    def snapshot(self, destination: Path | None = None) -> dict[str, Any]:
        self.ensure()
        with self.locked():
            manifest = self._load_current_unlocked()
            snapshot_dir = self._write_snapshot_unlocked(manifest)
            if destination is not None:
                destination = self._safe_snapshot_destination(destination)
                if destination.exists() or destination.is_symlink():
                    if destination.is_symlink():
                        raise SynContractError(
                            f"snapshot destination cannot be a symlink: {destination}"
                        )
                    self._validate_snapshot_tree(destination, manifest)
                    snapshot_dir = destination
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self._safe_snapshot_destination(destination)
                    temporary = Path(
                        tempfile.mkdtemp(
                            prefix=f".{destination.name}.", dir=destination.parent
                        )
                    )
                    try:
                        shutil.copytree(
                            self.snapshots_dir / str(manifest["snapshot_id"]),
                            temporary,
                            dirs_exist_ok=True,
                            copy_function=self._copy_regular_file,
                        )
                        os.replace(temporary, destination)
                    except Exception:
                        shutil.rmtree(temporary, ignore_errors=True)
                        raise
                    snapshot_dir = destination
                    self._make_read_only(snapshot_dir)

                # A caller may pass a destination that was created by another
                # process between the existence check and the copy/replace.
                # Validate the installed tree before exposing it as a pinned
                # snapshot so a partial destination can never be accepted.
                self._validate_snapshot_tree(snapshot_dir, manifest)

        snapshot_manifest = snapshot_dir / "manifest.json"
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": manifest["snapshot_id"],
            "revision": manifest["revision"],
            "data_source_dir": str(snapshot_dir),
            "source_data_source_dir": str(self.root),
            "snapshot_dir": str(snapshot_dir),
            "manifest_path": str(snapshot_manifest),
            "manifest_sha256": sha256_file(snapshot_manifest),
        }

    def _safe_snapshot_destination(self, destination: Path) -> Path:
        """Return a non-symlink snapshot destination outside the Data Source."""

        try:
            candidate = Path(destination).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = Path(os.path.abspath(candidate))
        except (TypeError, ValueError, OSError, UnicodeError) as exc:
            raise SynContractError("snapshot destination is invalid") from exc
        _reject_symlink_components(candidate, "snapshot destination")
        if candidate == self.root or self.root in candidate.parents:
            raise SynContractError(
                "snapshot destination must not be inside the Synthesis Data Source"
            )
        parent = candidate.parent
        if parent.exists() or parent.is_symlink():
            self._regular_directory(parent, "snapshot destination parent")
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink():
                raise SynContractError(
                    f"snapshot destination cannot be a symlink: {candidate}"
                )
            if not candidate.is_dir():
                raise SynContractError(
                    f"snapshot destination must be a directory: {candidate}"
                )
        return candidate

    def append_event(
        self,
        kind: str,
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        if kind not in EVENT_KINDS:
            raise SynContractError(f"unsupported Synthesis event kind: {kind}")
        if not isinstance(event, Mapping):
            raise SynContractError("event must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise SynContractError("event.event_id must be a non-empty string")
        incoming = copy.deepcopy(dict(event))
        declared_kind = incoming.setdefault("event_type", kind)
        if declared_kind != kind:
            raise SynContractError(
                f"event.event_type must be {kind!r}, got {declared_kind!r}"
            )

        self.ensure()
        with self.locked():
            manifest = self._load_current_unlocked()
            existing = self._find_event_file_unlocked(event_id)
            if existing is not None:
                existing_kind, existing_event = existing
                if not self._same_event(existing_event, incoming):
                    raise SynConflictError(
                        f"event_id {event_id!r} already exists with different content"
                    )
                indexed = self._event_is_indexed(manifest, existing_kind, event_id)
                if indexed:
                    return {
                        "event": existing_event,
                        "idempotent": True,
                        "revision": manifest["revision"],
                        "snapshot_id": manifest["snapshot_id"],
                    }
                if existing_kind != kind:
                    raise SynConflictError(
                        f"event_id {event_id!r} belongs to {existing_kind}, not {kind}"
                    )
                incoming = existing_event

            if kind == "batch_status":
                self._validate_batch_status(incoming, manifest)
            if existing is None:
                self._install_event_unlocked(kind, incoming)
            next_manifest = copy.deepcopy(manifest)
            event_index = next_manifest.setdefault(
                "events", {item: [] for item in EVENT_KINDS}
            )
            event_index.setdefault(kind, []).append(event_id)
            cursors = next_manifest.setdefault("event_cursors", {})
            cursors[kind] = len(event_index[kind])
            if kind == "batch_status":
                batch_id = str(incoming["batch_id"])
                next_manifest["batches"][batch_id]["status"] = incoming["status"]
                next_manifest["batches"][batch_id]["status_event_id"] = event_id
            committed = self._commit_unlocked(next_manifest, current=manifest)
            self._write_event_projections_unlocked(committed)
            return {
                "event": incoming,
                "idempotent": False,
                "revision": committed["revision"],
                "snapshot_id": committed["snapshot_id"],
            }

    def _load_current_unlocked(self) -> dict[str, Any]:
        return read_json_object(self.manifest_path, "Synthesis Data Source manifest")

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise SynContractError(f"manifest.schema_version must be {SCHEMA_VERSION}")
        revision = manifest.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise SynContractError("manifest.revision must be a non-negative integer")
        if manifest.get("snapshot_id") != _snapshot_id(manifest):
            raise SynContractError("manifest.snapshot_id does not match its content")
        for field in ("targets", "batches", "events", "event_cursors"):
            if not isinstance(manifest.get(field), dict):
                raise SynContractError(f"manifest.{field} must be an object")
        events = manifest["events"]
        cursors = manifest["event_cursors"]
        for kind in EVENT_KINDS:
            ids = events.get(kind)
            if not isinstance(ids, list) or any(
                not isinstance(item, str) or not item for item in ids
            ):
                raise SynContractError(f"manifest.events.{kind} must be a string list")
            if len(ids) != cursors.get(kind):
                raise SynContractError(
                    f"manifest event cursor does not match events.{kind}"
                )

    def _commit_unlocked(
        self,
        next_manifest: Mapping[str, Any],
        *,
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = copy.deepcopy(dict(next_manifest))
        payload["schema_version"] = SCHEMA_VERSION
        payload["revision"] = int(current["revision"]) + 1
        payload["created_at"] = current["created_at"]
        payload["updated_at"] = utc_now()
        payload["snapshot_id"] = _snapshot_id(payload)
        self._validate_manifest(payload)
        _atomic_json(self.manifest_path, payload)
        self._write_snapshot_unlocked(payload)
        return copy.deepcopy(payload)

    def _write_snapshot_unlocked(self, manifest: Mapping[str, Any]) -> Path:
        snapshot_id = str(manifest["snapshot_id"])
        final = self.snapshots_dir / snapshot_id
        _reject_symlink_components(final, "Synthesis Data Source snapshot directory")
        if final.is_symlink():
            raise SynContractError(f"snapshot directory cannot be a symlink: {final}")
        if final.exists():
            self._validate_snapshot_tree(final, manifest)
            return final

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=self.snapshots_dir)
        )
        try:
            _atomic_text(temporary / "README.md", README)
            _atomic_json(temporary / "manifest.json", manifest)
            snapshot_events = temporary / "events"
            snapshot_events.mkdir()
            for kind in EVENT_KINDS:
                kind_dir = snapshot_events / kind
                kind_dir.mkdir()
                lines: list[str] = []
                for event_id in manifest["events"][kind]:
                    source = self._event_path(kind, event_id)
                    event = read_json_object(source, f"{kind} event {event_id}")
                    if event.get("event_id") != event_id:
                        raise SynContractError(
                            f"event file identity mismatch for {event_id!r}"
                        )
                    self._copy_regular_file(source, kind_dir / source.name)
                    lines.append(canonical_json(event) + "\n")
                _atomic_text(snapshot_events / f"{kind}.jsonl", "".join(lines))

            for target_id, target in manifest["targets"].items():
                snapshots = target.get("snapshots", {})
                if not isinstance(snapshots, dict):
                    raise SynContractError(
                        f"manifest target {target_id!r} snapshots must be an object"
                    )
                for snapshot in snapshots.values():
                    relative = snapshot.get("path")
                    self._link_manifest_tree(temporary, relative, "target snapshot")
            for batch in manifest["batches"].values():
                self._link_manifest_tree(
                    temporary, batch.get("path"), "batch directory"
                )
            for batch in manifest.get("retired_batches", {}).values():
                self._link_manifest_tree(
                    temporary, batch.get("path"), "retired batch directory"
                )
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._make_read_only(final)
        return final

    @staticmethod
    def _make_read_only(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink():
                raise SynContractError(f"cannot make symlink read-only: {path}")
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise SynContractError(f"cannot inspect snapshot path: {path}") from exc
            if stat.S_ISREG(mode):
                if stat.S_IMODE(mode) != 0o444:
                    path.chmod(0o444)
            elif stat.S_ISDIR(mode):
                if stat.S_IMODE(mode) != 0o555:
                    path.chmod(0o555)
            else:
                raise SynContractError(f"snapshot contains a special file: {path}")
        root.chmod(0o555)

    def _link_manifest_tree(
        self, snapshot_root: Path, relative: Any, label: str
    ) -> None:
        if not isinstance(relative, str) or not relative:
            raise SynContractError(f"manifest {label} has an invalid path")
        raw_source = self.root / relative
        if raw_source.is_symlink():
            raise SynContractError(
                f"manifest {label} cannot be a symlink: {raw_source}"
            )
        self._reject_symlink_components(raw_source, label)
        source = raw_source.resolve(strict=False)
        try:
            relative_path = source.relative_to(self.root)
        except ValueError as exc:
            raise SynContractError(f"manifest {label} escapes data source") from exc
        if not source.is_dir():
            raise SynContractError(f"manifest {label} is missing: {source}")
        for child in source.rglob("*"):
            if child.is_symlink():
                raise SynContractError(f"manifest {label} contains a symlink: {child}")
        destination = snapshot_root / relative_path
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            copy_function=self._copy_regular_file,
        )

    @staticmethod
    def _reject_symlink_components(path: Path, label: str) -> None:
        _reject_symlink_components(path, f"manifest {label}")

    @staticmethod
    def _copy_regular_file(source: str | Path, destination: str | Path) -> None:
        """Place a snapshot entry, rejecting links and special files.

        Everything a snapshot captures is already immutable at the moment it is
        captured: batch files and target snapshots are chmod 0444 by their
        publisher and event files by ``_install_event_unlocked``. A hard link is
        therefore a stronger identity than a copy -- it is the same inode, so it
        cannot drift from the source -- and it makes a commit cost O(entries)
        instead of O(pool bytes). Snapshots are written on *every* commit, so
        copying bytes made a run's disk usage grow quadratically in the number
        of published batches.

        Falls back to copying when the destination is on another filesystem.
        """

        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise SynContractError(
                f"snapshot source must be a regular non-symlink file: {source_path}"
            )
        target = Path(destination)
        try:
            os.link(source_path, target)
            return
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EMLINK, errno.EPERM, errno.EACCES):
                raise SynIOError(
                    f"could not link snapshot file {source_path} -> {target}: {exc}"
                ) from exc
        try:
            shutil.copy2(source_path, target)
        except OSError as exc:
            raise SynIOError(
                f"could not copy snapshot file {source_path} -> {target}: {exc}"
            ) from exc

    @classmethod
    def _regular_file(cls, path: Path, label: str) -> Path:
        """Return a regular, non-symlink file without following links."""

        candidate = Path(path)
        if candidate.is_symlink():
            raise SynContractError(f"{label} cannot be a symlink: {candidate}")
        cls._reject_symlink_components(candidate, label)
        resolved = candidate.resolve(strict=False)
        try:
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise SynContractError(f"{label} is missing or unsafe: {resolved}") from exc
        if not stat.S_ISREG(mode):
            raise SynContractError(f"{label} must be a regular file: {resolved}")
        return resolved

    @classmethod
    def _regular_directory(cls, path: Path, label: str) -> Path:
        """Return a regular, non-symlink directory without following links."""

        candidate = Path(path)
        if candidate.is_symlink():
            raise SynContractError(f"{label} cannot be a symlink: {candidate}")
        cls._reject_symlink_components(candidate, label)
        resolved = candidate.resolve(strict=False)
        try:
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise SynContractError(f"{label} is missing or unsafe: {resolved}") from exc
        if not stat.S_ISDIR(mode):
            raise SynContractError(f"{label} must be a directory: {resolved}")
        return resolved

    @staticmethod
    def _snapshot_file(root: Path, relative: Any, label: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise SynContractError(f"{label} has an invalid path")
        raw_candidate = root / relative
        DataSource._reject_symlink_components(raw_candidate, label)
        if raw_candidate.is_symlink():
            raise SynContractError(f"{label} cannot be a symlink: {raw_candidate}")
        candidate = raw_candidate.resolve(strict=False)
        try:
            if not stat.S_ISREG(candidate.stat().st_mode):
                raise SynContractError(
                    f"{label} must be a regular snapshot file: {candidate}"
                )
        except OSError as exc:
            raise SynContractError(
                f"{label} must be a regular snapshot file: {candidate}"
            ) from exc
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SynContractError(f"{label} escapes snapshot: {candidate}") from exc
        return candidate

    def _validate_snapshot_tree(
        self, snapshot_root: Path, manifest: Mapping[str, Any]
    ) -> None:
        """Fail closed when a pinned snapshot is missing or has changed bytes."""

        root = self._regular_directory(Path(snapshot_root), "snapshot directory")
        existing = read_json_object(
            self._regular_file(root / "manifest.json", "snapshot manifest"),
            "snapshot manifest",
        )
        if canonical_json(existing) != canonical_json(manifest):
            raise SynConflictError(
                f"snapshot_id collision with different content: {manifest.get('snapshot_id')}"
            )
        # Every file in a pinned tree must be a regular file.  This also
        # catches a partially copied tree that left a symlink behind.
        for child in root.rglob("*"):
            if child.is_symlink():
                raise SynContractError(f"snapshot contains a symlink: {child}")
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise SynContractError(
                    f"snapshot path cannot be inspected: {child}"
                ) from exc
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise SynContractError(f"snapshot contains a non-regular file: {child}")

        for kind in EVENT_KINDS:
            projection_path = self._snapshot_file(
                root,
                f"events/{kind}.jsonl",
                f"snapshot {kind} event projection",
            )
            projection_lines: list[str] = []
            for event_id in manifest["events"][kind]:
                event_path = self._snapshot_file(
                    root,
                    f"events/{kind}/{self._event_digest(event_id)}.json",
                    f"snapshot {kind} event {event_id}",
                )
                event = read_json_object(
                    event_path, f"snapshot {kind} event {event_id}"
                )
                if event.get("event_id") != event_id:
                    raise SynContractError(
                        f"snapshot event identity mismatch for {event_id!r}"
                    )
                projection_lines.append(canonical_json(event) + "\n")
            try:
                projection = projection_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise SynContractError(
                    f"snapshot {kind} event projection is invalid: {projection_path}"
                ) from exc
            if projection != "".join(projection_lines):
                raise SynContractError(
                    f"snapshot {kind} event projection does not match event files"
                )

        for target_id, target in manifest["targets"].items():
            snapshots = target.get("snapshots", {})
            if not isinstance(snapshots, Mapping):
                raise SynContractError(
                    f"manifest target {target_id!r} snapshots must be an object"
                )
            for target_snapshot in snapshots.values():
                relative = target_snapshot.get("path")
                target_root = self._snapshot_file_or_dir(
                    root, relative, "target snapshot"
                )
                expected_target_root = (
                    root
                    / "targets"
                    / str(target_id)
                    / str(target_snapshot.get("target_snapshot_id", ""))
                ).resolve(strict=False)
                if target_root != expected_target_root:
                    raise SynContractError(
                        f"target snapshot path is not canonical: {relative}"
                    )
                receipt_path = self._snapshot_file(
                    root, target_snapshot.get("receipt"), "target receipt"
                )
                if sha256_file(receipt_path) != target_snapshot.get("receipt_sha256"):
                    raise SynContractError(
                        f"target receipt hash mismatch: {receipt_path}"
                    )
                receipt = read_json_object(receipt_path, "target receipt")
                files = receipt.get("files")
                if not isinstance(files, Mapping):
                    raise SynContractError("target receipt.files must be an object")
                for file_relative, info in files.items():
                    if not isinstance(info, Mapping):
                        raise SynContractError(
                            "target receipt file entry must be an object"
                        )
                    # Receipt paths are relative to the target snapshot itself.
                    if not isinstance(file_relative, str) or not file_relative:
                        raise SynContractError("target receipt file path is invalid")
                    raw_path = target_root / file_relative
                    self._reject_symlink_components(raw_path, "target receipt file")
                    path = raw_path.resolve(strict=False)
                    try:
                        path.relative_to(target_root)
                    except ValueError as exc:
                        raise SynContractError(
                            f"target receipt file escapes target snapshot: {file_relative}"
                        ) from exc
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or not stat.S_ISREG(path.stat().st_mode)
                    ):
                        raise SynContractError(
                            f"target receipt file is missing: {path}"
                        )
                    if sha256_file(path) != info.get("sha256"):
                        raise SynContractError(
                            f"target receipt file hash mismatch: {path}"
                        )
                    if path.stat().st_size != info.get("bytes"):
                        raise SynContractError(
                            f"target receipt file size mismatch: {path}"
                        )

        for batch_id, batch in manifest["batches"].items():
            batch_root = self._snapshot_file_or_dir(
                root, batch.get("path"), "batch directory"
            )
            expected_batch_root = (root / "batches" / str(batch_id)).resolve(
                strict=False
            )
            if batch_root != expected_batch_root:
                raise SynContractError(
                    f"batch {batch_id} path is not canonical: {batch.get('path')}"
                )
            for field in (
                "data_path",
                "report_path",
                "experience_delta_path",
                "token_counts_path",
                "metadata_path",
                "delivery_receipt",
            ):
                path = self._snapshot_file(root, batch.get(field), f"batch {field}")
                try:
                    path.relative_to(batch_root)
                except ValueError as exc:
                    raise SynContractError(
                        f"batch {batch_id} {field} escapes its directory"
                    ) from exc
                digest_field = field.removesuffix("_path") + "_sha256"
                expected = batch.get(digest_field)
                if expected is not None and sha256_file(path) != expected:
                    raise SynContractError(
                        f"batch {batch_id} {field} hash mismatch: {path}"
                    )
            receipt_path = self._snapshot_file(
                root, batch.get("delivery_receipt"), "batch delivery receipt"
            )
            receipt = read_json_object(receipt_path, "batch delivery receipt")
            if (
                receipt.get("batch_id") != batch_id
                or receipt.get("status") != "complete"
            ):
                raise SynContractError(
                    f"batch {batch_id} delivery receipt identity is invalid"
                )

    @staticmethod
    def _snapshot_file_or_dir(root: Path, relative: Any, label: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise SynContractError(f"{label} has an invalid path")
        raw_candidate = root / relative
        DataSource._reject_symlink_components(raw_candidate, label)
        if raw_candidate.is_symlink():
            raise SynContractError(f"{label} cannot be a symlink: {raw_candidate}")
        candidate = raw_candidate.resolve(strict=False)
        try:
            if not stat.S_ISDIR(candidate.stat().st_mode):
                raise SynContractError(f"{label} is missing or unsafe: {candidate}")
        except OSError as exc:
            raise SynContractError(
                f"{label} is missing or unsafe: {candidate}"
            ) from exc
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SynContractError(f"{label} escapes snapshot: {candidate}") from exc
        return candidate

    @staticmethod
    def _event_digest(event_id: str) -> str:
        return hashlib.sha256(event_id.encode("utf-8")).hexdigest()

    def _event_path(self, kind: str, event_id: str) -> Path:
        return self.events_dir / kind / f"{self._event_digest(event_id)}.json"

    @staticmethod
    def _event_is_indexed(
        manifest: Mapping[str, Any], kind: str, event_id: str
    ) -> bool:
        events = manifest.get("events", {})
        return isinstance(events, Mapping) and event_id in events.get(kind, [])

    def _find_event_file_unlocked(
        self, event_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        name = f"{self._event_digest(event_id)}.json"
        for kind in EVENT_KINDS:
            path = self.events_dir / kind / name
            if path.exists() or path.is_symlink():
                self._regular_file(path, f"{kind} event")
                event = read_json_object(path, f"{kind} event")
                if event.get("event_id") != event_id:
                    raise SynConflictError(
                        f"event digest collision for event_id {event_id!r}"
                    )
                return kind, event
        return None

    @staticmethod
    def _same_event(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> bool:
        left = dict(existing)
        right = dict(incoming)
        left.pop("resolution", None)
        right.pop("resolution", None)
        return canonical_json(left) == canonical_json(right)

    def _install_event_unlocked(self, kind: str, event: Mapping[str, Any]) -> None:
        path = self._event_path(kind, str(event["event_id"]))
        self._reject_symlink_components(path, f"{kind} event")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise SynContractError(f"{kind} event cannot be a symlink: {path}")
        if path.exists():
            self._regular_file(path, f"{kind} event")
            raise SynConflictError(f"event file already exists: {path}")
        _atomic_json(path, event)
        path.chmod(0o444)

    def _write_event_projections_unlocked(self, manifest: Mapping[str, Any]) -> None:
        for kind in EVENT_KINDS:
            lines = []
            for event_id in manifest["events"][kind]:
                event = read_json_object(
                    self._event_path(kind, event_id), f"{kind} event"
                )
                lines.append(canonical_json(event) + "\n")
            _atomic_text(self.events_dir / f"{kind}.jsonl", "".join(lines))

    @staticmethod
    def _validate_batch_status(
        event: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        batch_id = event.get("batch_id")
        status = event.get("status")
        known = batch_id in manifest["batches"] or batch_id in manifest.get(
            "retired_batches", {}
        )
        if not isinstance(batch_id, str) or not known:
            raise SynContractError(
                f"batch status references unknown batch_id: {batch_id!r}"
            )
        if status not in {"available", "unavailable"}:
            raise SynContractError(
                "batch status event.status must be 'available' or 'unavailable'"
            )


# ---------------------------------------------------------------------------
# Publication of validated deliveries
# ---------------------------------------------------------------------------


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SynContractError(f"{label} must be a non-empty string")
    return value


def _required_path(value: Any, label: str) -> Path:
    try:
        raw_path = Path(value).expanduser()
    except TypeError as exc:
        raise SynContractError(f"{label} must be a path") from exc
    if raw_path.is_symlink():
        raise SynValidationError(f"{label} cannot be a symlink: {raw_path}")
    reject_symlink_components(raw_path, label, error=SynValidationError)
    path = raw_path.resolve(strict=False)
    if not path.is_file():
        raise SynValidationError(f"{label} is missing: {path}")
    return path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise SynContractError(f"published path escapes data source: {path}") from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SynContractError(f"{label} must be a positive integer")
    return value


def _loss_int(value: Any, total: int, label: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > total
    ):
        raise SynContractError(
            f"{label} must be null or an integer between zero and total_tokens"
        )
    return value


def _token_sidecar(
    row_count: int,
    fingerprints: Sequence[str],
    *,
    token_counts: Sequence[int],
    loss_counts: Sequence[int | None] | None,
    tokenizer: str | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len(token_counts) != row_count:
        raise SynContractError(
            "token_counts length does not match delivery rows: "
            f"{len(token_counts)} != {row_count}"
        )
    if loss_counts is not None and len(loss_counts) != row_count:
        raise SynContractError(
            "loss_counts length does not match delivery rows: "
            f"{len(loss_counts)} != {row_count}"
        )
    counts: list[tuple[int, int | None]] = []
    for index, raw_total in enumerate(token_counts):
        total = _positive_int(raw_total, f"token_counts[{index}]")
        raw_loss = loss_counts[index] if loss_counts is not None else None
        counts.append((total, _loss_int(raw_loss, total, f"loss_counts[{index}]")))

    if tokenizer is None:
        identity = {"counter": "provided_token_counts"}
    elif isinstance(tokenizer, str):
        identity = {"name": _required_string(tokenizer, "tokenizer")}
    elif isinstance(tokenizer, Mapping):
        identity = copy.deepcopy(dict(tokenizer))
    else:
        raise SynContractError("tokenizer must be a string or object")

    entries = [
        {
            "row_index": index,
            "total_tokens": total,
            "loss_tokens": loss,
            "message_fingerprint": fingerprints[index],
        }
        for index, (total, loss) in enumerate(counts)
    ]
    known_loss = all(item["loss_tokens"] is not None for item in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": identity,
        "row_count": len(entries),
        "total_tokens": sum(item["total_tokens"] for item in entries),
        "loss_tokens": (
            sum(int(item["loss_tokens"]) for item in entries) if known_loss else None
        ),
        "rows": entries,
    }


def _tree_entries(target_file: Path, materials_dir: Path) -> list[tuple[str, Path]]:
    if target_file.is_symlink() or not target_file.is_file():
        raise SynValidationError(
            f"target file must be a regular non-symlink file: {target_file}"
        )
    if materials_dir.is_symlink():
        raise SynValidationError(
            f"materials directory cannot be a symlink: {materials_dir}"
        )
    if not materials_dir.is_dir():
        raise SynValidationError(f"materials directory is missing: {materials_dir}")
    entries = [("target.md", target_file)]
    for path in sorted(materials_dir.rglob("*")):
        if path.is_symlink():
            raise SynValidationError(
                f"target materials cannot contain symlinks: {path}"
            )
        if path.is_file():
            entries.append(
                (f"materials/{path.relative_to(materials_dir).as_posix()}", path)
            )
    return entries


def _tree_fingerprint(entries: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for relative, path in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _freeze_target(
    source: DataSource,
    *,
    target_id: str,
    frozen_target: Any,
) -> dict[str, Any]:
    target_file = Path(_field(frozen_target, "target_file"))
    materials_dir = Path(_field(frozen_target, "materials_dir"))
    target_snapshot_id = _field(frozen_target, "snapshot_id")
    declared_target = _field(frozen_target, "target_id")
    if declared_target is not None and declared_target != target_id:
        raise SynContractError(
            "frozen target identity does not match delivery target_id"
        )
    target_file = _required_path(target_file, "target_file")
    raw_materials_dir = Path(materials_dir).expanduser()
    if raw_materials_dir.is_symlink():
        raise SynValidationError(
            f"materials directory cannot be a symlink: {raw_materials_dir}"
        )
    reject_symlink_components(
        raw_materials_dir, "materials directory", error=SynValidationError
    )
    materials_dir = raw_materials_dir.resolve(strict=False)
    entries = _tree_entries(target_file, materials_dir)
    fingerprint = _tree_fingerprint(entries)
    snapshot_id = validate_safe_id(
        target_snapshot_id or f"target-{fingerprint[:20]}", "target_snapshot_id"
    )
    final = source.root / "targets" / target_id / snapshot_id
    receipt_payload = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "target_snapshot_id": snapshot_id,
        "input_fingerprint": fingerprint,
        "files": {
            relative: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for relative, path in entries
        },
    }
    if final.exists():
        _verify_target_snapshot(final, receipt_payload)
    else:
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=final.parent))
        try:
            for relative, path in entries:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            _atomic_json(staging / "target_receipt.json", receipt_payload)
            os.replace(staging, final)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        make_read_only_tree(final)
    return {
        "target_id": target_id,
        "target_snapshot_id": snapshot_id,
        "path": _relative(final, source.root),
        "target_file": _relative(final / "target.md", source.root),
        "materials_dir": _relative(final / "materials", source.root),
        "input_fingerprint": fingerprint,
        "receipt": _relative(final / "target_receipt.json", source.root),
        "receipt_sha256": sha256_file(final / "target_receipt.json"),
    }


def _verify_target_snapshot(final: Path, expected_receipt: Mapping[str, Any]) -> None:
    """Validate every immutable target file before treating a retry as idempotent."""

    if final.is_symlink() or not final.is_dir():
        raise SynConflictError(f"target snapshot is not a regular directory: {final}")
    receipt_path = final / "target_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SynConflictError(f"target snapshot receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SynConflictError(
            f"target snapshot receipt is invalid: {receipt_path}"
        ) from exc
    if canonical_json(receipt) != canonical_json(expected_receipt):
        raise SynConflictError("target snapshot exists with different content")
    for relative, info in expected_receipt.get("files", {}).items():
        if not isinstance(relative, str) or not isinstance(info, Mapping):
            raise SynConflictError(
                "target snapshot receipt contains an invalid file entry"
            )
        path = final / relative
        if path.is_symlink() or not path.is_file():
            raise SynConflictError(f"target snapshot file is missing: {path}")
        if sha256_file(path) != info.get("sha256") or path.stat().st_size != info.get(
            "bytes"
        ):
            raise SynConflictError(f"target snapshot file changed: {path}")


def _publication_identity(receipt: Mapping[str, Any]) -> str:
    fields = {
        key: receipt.get(key)
        for key in (
            "batch_id",
            "target_id",
            "target_snapshot_id",
            "data_sha256",
            "token_counts_sha256",
            "metadata_sha256",
            "report_sha256",
            "experience_delta_sha256",
        )
    }
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()


# Registration writes a manifest entry once; the availability event afterwards
# updates "status" and backfills "status_event_id" in place.  Republish
# idempotency must compare batch identity without those event-owned fields,
# or a fully published batch would look like "different content" to a retry.
_EVENT_OWNED_ENTRY_FIELDS = frozenset({"status", "status_event_id"})


def _entry_identity(entry: Mapping[str, Any]) -> str:
    return canonical_json(
        {k: v for k, v in entry.items() if k not in _EVENT_OWNED_ENTRY_FIELDS}
    )


def _verify_existing_batch(
    final: Path, expected_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify an already-installed batch before returning an idempotent result."""

    if final.is_symlink() or not final.is_dir():
        raise SynConflictError(f"batch path is not a regular directory: {final}")
    receipt_path = final / "delivery_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SynConflictError(f"batch delivery receipt is missing: {receipt_path}")
    try:
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SynConflictError(
            f"batch delivery receipt is invalid: {receipt_path}"
        ) from exc
    if not isinstance(existing, Mapping) or existing.get("status") != "complete":
        raise SynConflictError(f"batch delivery receipt is incomplete: {receipt_path}")
    normalized_expected = dict(expected_receipt)
    # Publication time is diagnostic metadata, while all content identity is
    # covered by the receipt hashes and publication identity.
    normalized_expected["published_at"] = existing.get("published_at")
    if canonical_json(existing) != canonical_json(normalized_expected):
        raise SynConflictError("batch_id already exists with different content")

    expected_files = {
        "data.jsonl": "data_sha256",
        "token_counts.json": "token_counts_sha256",
        "metadata.json": "metadata_sha256",
        "report.md": "report_sha256",
        "experience_delta.md": "experience_delta_sha256",
        "delivery_receipt.json": None,
    }
    for filename, digest_field in expected_files.items():
        path = final / filename
        if path.is_symlink() or not path.is_file():
            raise SynConflictError(f"batch file is missing or unsafe: {path}")
        if digest_field is not None and sha256_file(path) != existing.get(digest_field):
            raise SynConflictError(f"batch file hash mismatch: {path}")
    if _publication_identity(existing) != existing.get("publication_identity"):
        raise SynConflictError("batch receipt publication identity is invalid")
    return dict(existing)


class Publisher:
    """Publish one complete target delivery without exposing worker internals."""

    def __init__(self, data_source: DataSource | Any):
        self.data_source = (
            data_source
            if isinstance(data_source, DataSource)
            else DataSource(data_source)
        )

    def publish(
        self,
        delivery: Any,
        *,
        batch_id: str | None = None,
        frozen_target: Any = None,
        token_counts: Sequence[int],
        loss_counts: Sequence[int | None] | None = None,
        tokenizer: str | Mapping[str, Any] | None = None,
        round_id: str | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        source = self.data_source
        source.ensure()
        if batch_id is None:
            batch_id = _field(delivery, "batch_id")
        batch_id = validate_safe_id(batch_id, "batch_id")
        target_id = validate_safe_id(
            _required_string(_field(delivery, "target_id"), "delivery.target_id"),
            "target_id",
        )
        data_path = _required_path(_field(delivery, "data_path"), "delivery.data_path")
        report_path = _required_path(
            _field(delivery, "report_path"), "delivery.report_path"
        )
        experience_delta_path = _required_path(
            _field(delivery, "experience_delta_path"),
            "delivery.experience_delta_path",
        )
        declared_hashes = {
            data_path: _field(delivery, "data_sha256"),
            report_path: _field(delivery, "report_sha256"),
            experience_delta_path: _field(delivery, "experience_delta_sha256"),
        }
        for path, expected in declared_hashes.items():
            actual = sha256_file(path)
            if actual != expected:
                raise SynValidationError(
                    f"delivery changed after validation: {path}: {actual} != {expected}"
                )

        # data_sha256 was just verified against what validation saw. Identical
        # bytes imply identical rows, so re-parsing the file and recomputing
        # every fingerprint could only confirm what the hash already proved --
        # at the cost of a full re-read and N sha256 passes per publish.
        row_count = _positive_int(_field(delivery, "rows"), "delivery.rows")
        fingerprints = tuple(_field(delivery, "message_fingerprints", ()))
        if len(fingerprints) != row_count:
            raise SynValidationError(
                "delivery message fingerprints do not cover every row: "
                f"{len(fingerprints)} != {row_count}"
            )

        token_sidecar = _token_sidecar(
            row_count,
            fingerprints,
            token_counts=token_counts,
            loss_counts=loss_counts,
            tokenizer=tokenizer,
        )
        target = _freeze_target(
            source,
            target_id=target_id,
            frozen_target=frozen_target,
        )

        final = source.root / "batches" / batch_id
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{batch_id}.", dir=final.parent))
        published_at = utc_now()
        try:
            shutil.copy2(data_path, staging / "data.jsonl")
            shutil.copy2(report_path, staging / "report.md")
            shutil.copy2(experience_delta_path, staging / "experience_delta.md")
            _atomic_json(staging / "token_counts.json", token_sidecar)
            batch_metadata = {
                "schema_version": SCHEMA_VERSION,
                "batch_id": batch_id,
                "target_id": target_id,
                "target_snapshot_id": target["target_snapshot_id"],
                "round_id": round_id,
                "attempt_id": attempt_id,
                "row_count": row_count,
                "total_tokens": token_sidecar["total_tokens"],
                "loss_tokens": token_sidecar["loss_tokens"],
                "tokenizer": token_sidecar["tokenizer"],
            }
            _atomic_json(staging / "metadata.json", batch_metadata)
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "batch_id": batch_id,
                "target_id": target_id,
                "target_snapshot_id": target["target_snapshot_id"],
                "published_at": published_at,
                "row_count": row_count,
                "total_tokens": token_sidecar["total_tokens"],
                "loss_tokens": token_sidecar["loss_tokens"],
                "data_sha256": sha256_file(staging / "data.jsonl"),
                "token_counts_sha256": sha256_file(staging / "token_counts.json"),
                "metadata_sha256": sha256_file(staging / "metadata.json"),
                "report_sha256": sha256_file(staging / "report.md"),
                "experience_delta_sha256": sha256_file(staging / "experience_delta.md"),
            }
            receipt["publication_identity"] = _publication_identity(receipt)
            _atomic_json(staging / "delivery_receipt.json", receipt)

            if final.exists():
                receipt = _verify_existing_batch(final, receipt)
                shutil.rmtree(staging)
                staging = final
            else:
                os.replace(staging, final)
                staging = final
                make_read_only_tree(final)
        finally:
            if staging != final and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        receipt_path = final / "delivery_receipt.json"
        entry = {
            "batch_id": batch_id,
            "target_id": target_id,
            "target_snapshot_id": target["target_snapshot_id"],
            "status": "available",
            "path": _relative(final, source.root),
            "data_path": _relative(final / "data.jsonl", source.root),
            "data_sha256": receipt["data_sha256"],
            "report_path": _relative(final / "report.md", source.root),
            "report_sha256": receipt["report_sha256"],
            "experience_delta_path": _relative(
                final / "experience_delta.md", source.root
            ),
            "experience_delta_sha256": receipt["experience_delta_sha256"],
            "token_counts_path": _relative(final / "token_counts.json", source.root),
            "token_counts_sha256": receipt["token_counts_sha256"],
            "metadata_path": _relative(final / "metadata.json", source.root),
            "metadata_sha256": receipt["metadata_sha256"],
            "delivery_receipt": _relative(receipt_path, source.root),
            "delivery_receipt_sha256": sha256_file(receipt_path),
            "row_count": receipt["row_count"],
            "total_tokens": receipt["total_tokens"],
            "loss_tokens": receipt["loss_tokens"],
            "round_id": round_id,
            "attempt_id": attempt_id,
            "published_at": receipt["published_at"],
        }

        with source.locked():
            current = source._load_current_unlocked()
            existing = current["batches"].get(batch_id)
            if existing is not None:
                if _entry_identity(existing) != _entry_identity(entry):
                    raise SynConflictError(
                        f"manifest already contains another batch {batch_id}"
                    )
                entry = existing
                was_existing = True
            else:
                was_existing = False

            if not was_existing:
                next_manifest = copy.deepcopy(current)
                targets = next_manifest["targets"]
                target_entry = targets.setdefault(
                    target_id,
                    {
                        "target_id": target_id,
                        "snapshots": {},
                        "batches": [],
                    },
                )
                snapshots = target_entry.setdefault("snapshots", {})
                prior_target = snapshots.get(target["target_snapshot_id"])
                if prior_target is not None and canonical_json(
                    prior_target
                ) != canonical_json(target):
                    raise SynConflictError(
                        "manifest target snapshot conflicts with immutable target files"
                    )
                snapshots[target["target_snapshot_id"]] = target
                target_entry["latest_snapshot_id"] = target["target_snapshot_id"]
                if batch_id not in target_entry["batches"]:
                    target_entry["batches"].append(batch_id)
                next_manifest["batches"][batch_id] = entry
                source._commit_unlocked(next_manifest, current=current)

        # Availability is itself an append-only fact.  The batch is first
        # registered so the event can reference a concrete immutable batch;
        # a retry either appends the missing event or observes it idempotently.
        status_result = {
            "revision": source.manifest()["revision"],
            "snapshot_id": source.manifest()["snapshot_id"],
        }
        if not (was_existing and entry.get("status") == "unavailable"):
            status_event_id = "batch-status-" + receipt["publication_identity"][:32]
            status_result = source.append_event(
                "batch_status",
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_type": "batch_status",
                    "event_id": status_event_id,
                    "batch_id": batch_id,
                    "status": "available",
                    "reason": "complete immutable delivery admitted",
                    "publication_identity": receipt["publication_identity"],
                },
            )
        # A target exposes exactly one selectable batch: the latest. Each
        # launch is made from the newest eval evidence, so a fresh batch
        # supersedes its predecessors rather than sitting beside them. The
        # retire events carry ids derived from (old, new) so a crashed publish
        # re-run appends nothing twice. Older batch files stay immutable on
        # disk; only their selectability changes.
        superseded: list[str] = []
        if not was_existing:
            batches_now = source.manifest()["batches"]
            for other_id, other in sorted(batches_now.items()):
                if other_id == batch_id or other.get("target_id") != target_id:
                    continue
                retire_event_id = (
                    "batch-status-superseded-"
                    + sha256_text(f"{other_id}->{batch_id}")[:32]
                )
                source.append_event(
                    "batch_status",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "event_type": "batch_status",
                        "event_id": retire_event_id,
                        "batch_id": other_id,
                        "status": "unavailable",
                        "reason": f"superseded by newer batch {batch_id} for the same target",
                        "superseded_by": batch_id,
                    },
                )
                status_result = source.retire_batch(other_id, superseded_by=batch_id)
                superseded.append(other_id)

        current_after_status = source.manifest()["batches"][batch_id]
        return {
            "batch": copy.deepcopy(current_after_status),
            "idempotent": was_existing,
            "superseded": superseded,
            "revision": status_result["revision"],
            "snapshot_id": status_result["snapshot_id"],
            "manifest_path": str(source.manifest_path),
        }


# ---------------------------------------------------------------------------
# Sanctioned append-only feedback API
# ---------------------------------------------------------------------------


def stable_usage_event_id(
    training_run_id: str, batch_id: str, selection_id: str
) -> str:
    """Derive the retry-stable id used after final materialization succeeds."""

    for value, label in (
        (training_run_id, "training_run_id"),
        (batch_id, "batch_id"),
        (selection_id, "selection_id"),
    ):
        _required_string(value, label)
    digest = hashlib.sha256(
        canonical_json([training_run_id, batch_id, selection_id]).encode("utf-8")
    ).hexdigest()
    return f"usage-{digest[:32]}"


def _non_negative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SynContractError(f"{label} must be a non-negative integer")
    return value


def _event(value: Mapping[str, Any], event_type: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SynContractError("feedback event must be an object")
    event = dict(value)
    _required_string(event.get("event_id"), "event_id")
    declared = event.setdefault("event_type", event_type)
    if declared != event_type:
        raise SynContractError(f"event_type must be {event_type!r}, got {declared!r}")
    return event


def append_usage(
    data_source: DataSource | Any,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    source = (
        data_source if isinstance(data_source, DataSource) else DataSource(data_source)
    )
    payload = _event(event, "usage")
    training_run_id = _required_string(
        payload.get("training_run_id"), "usage.training_run_id"
    )
    batch_id = _required_string(payload.get("batch_id"), "usage.batch_id")
    rows = _non_negative(payload.get("materialized_rows"), "usage.materialized_rows")
    tokens = _non_negative(
        payload.get("materialized_tokens"), "usage.materialized_tokens"
    )
    manifest = source.manifest()
    batch = manifest["batches"].get(batch_id)
    if batch is None:
        # Training can finish after a newer publication retires its selected
        # batch. Retirement changes selectability, not the immutable identity
        # or capacity against which historical exposure is validated.
        batch = manifest.get("retired_batches", {}).get(batch_id)
    if batch is None:
        raise SynContractError(f"usage references unknown batch_id: {batch_id}")
    # Materialized counts may exceed the published batch size: usage is derived
    # from the assembled training file, and repeating a row (upsampling) is a
    # legitimate recipe decision, not inflation.  What cannot exceed publication
    # is the number of *distinct* published rows matched, so that is the bound
    # enforced when the deriver supplies it.
    if payload.get("unique_rows") is not None:
        unique_rows = _non_negative(payload["unique_rows"], "usage.unique_rows")
        if unique_rows > rows:
            raise SynContractError(
                "usage.unique_rows cannot exceed usage.materialized_rows"
            )
        if unique_rows > batch["row_count"]:
            raise SynContractError(
                f"usage.unique_rows exceeds published rows of batch {batch_id}"
            )
    # Loss-token accounting is optional because a published batch may not have
    # a loss mask.  An explicit null therefore means "unknown"; a concrete
    # value is bounded by this materialization (upsampling scales the loss
    # tokens with the rows, so the publish-time capacity does not apply).
    if "materialized_loss_tokens" in payload:
        loss_tokens = payload["materialized_loss_tokens"]
        if loss_tokens is not None:
            loss_tokens = _non_negative(loss_tokens, "usage.materialized_loss_tokens")
            if loss_tokens > tokens:
                raise SynContractError(
                    "usage.materialized_loss_tokens cannot exceed "
                    "usage.materialized_tokens"
                )
    selection_id = _required_string(payload.get("selection_id"), "usage.selection_id")
    for field in ("dataset_ref", "recipe_ref"):
        _required_string(payload.get(field), f"usage.{field}")
    expected_event_id = stable_usage_event_id(training_run_id, batch_id, selection_id)
    if payload.get("event_id") != expected_event_id:
        raise SynContractError(
            "usage.event_id must equal the stable id derived from "
            "training_run_id, batch_id, and selection_id"
        )
    return source.append_event("usage", payload)


# ---------------------------------------------------------------------------
# Qwen3 token counting for Synthesis batches
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenCounts:
    total_tokens: tuple[int, ...]
    loss_tokens: tuple[int, ...]


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SynValidationError(
                    f"invalid Synthesis JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SynValidationError(
                    f"Synthesis JSONL row must be an object at {path}:{line_number}"
                )
            yield row


def _parse_count_tsv(path: Path, expected_rows: int) -> TokenCounts:
    totals: list[int | None] = [None] * expected_rows
    losses: list[int | None] = [None] * expected_rows
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SynValidationError(f"token counter did not produce {path}") from exc
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise SynValidationError(f"invalid token count row at {path}:{line_number}")
        try:
            row_id, total, loss = (int(item) for item in fields)
        except ValueError as exc:
            raise SynValidationError(
                f"non-integer token count at {path}:{line_number}"
            ) from exc
        if not 0 <= row_id < expected_rows:
            raise SynValidationError(
                f"token count row id out of range at {path}:{line_number}"
            )
        if totals[row_id] is not None:
            raise SynValidationError(f"duplicate token count for row {row_id}")
        if total <= 0 or loss < 0 or loss > total:
            raise SynValidationError(
                f"invalid token counts for row {row_id}: total={total}, loss={loss}"
            )
        totals[row_id] = total
        losses[row_id] = loss
    if any(value is None for value in totals):
        missing = [index for index, value in enumerate(totals) if value is None]
        raise SynValidationError(
            "token counter omitted Synthesis rows: " + ", ".join(map(str, missing[:20]))
        )
    return TokenCounts(
        total_tokens=tuple(int(value) for value in totals),
        loss_tokens=tuple(int(value) for value in losses),
    )


def count_jsonl_tokens(
    data_path: Path,
    *,
    tokenizer_code_dir: Path | None = None,
    python_executable: str | None = None,
    repo_root: Path | None = None,
) -> TokenCounts:
    """Return per-row total/loss tokens using ``prepare_tokens.encode_row``.

    The converter's count mode keys its TSV by ``pool_row_id``. Synthesis batches do
    not belong to the fixed pool, so this adapter creates a private annotated
    copy with temporary dense row ids. The published training rows remain the
    worker's original ``messages`` objects.
    """

    source = Path(data_path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise SynValidationError(
            f"Synthesis data must be a regular JSONL file: {source}"
        )
    # ``repo_root`` remains accepted for callers from older run artifacts;
    # new callers pass the launcher-selected local tokenizer runtime directly.
    if tokenizer_code_dir is None:
        root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[4]
        )
        converter_dir = root / "src" / "autosft" / "data"
    else:
        converter_dir = Path(tokenizer_code_dir).expanduser().resolve(strict=False)
    missing = missing_tokenizer_files(converter_dir)
    if missing:
        raise SynValidationError(
            "local SSTable runtime is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    converter = converter_dir / "prepare_tokens.py"

    rows = list(_jsonl_rows(source))
    if not rows:
        raise SynValidationError("Synthesis data contains no rows to tokenize")
    with tempfile.TemporaryDirectory(prefix="autosft-syn-token-count-") as temp:
        temp_dir = Path(temp)
        annotated = temp_dir / "annotated.jsonl"
        counts_path = temp_dir / "counts.tsv"
        with annotated.open("w", encoding="utf-8") as stream:
            for row_id, row in enumerate(rows):
                payload = dict(row)
                payload["pool_row_id"] = row_id
                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        command = [
            (
                python_executable
                or os.environ.get("AUTO_SFT_TOKENIZER_PYTHON")
                or sys.executable
            ),
            str(converter),
            "--json-file",
            str(annotated),
            "--tokenizer",
            "qwen3",
            "--workers",
            "1",
            "--count-only",
            "--count-output",
            str(counts_path),
        ]
        env = tokenizer_subprocess_env()
        env["TOKENIZERS_PARALLELISM"] = "false"
        result = subprocess.run(
            command,
            cwd=converter_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            raise SynValidationError(
                f"Qwen3 token counter failed with exit {result.returncode}: {detail}"
            )
        return _parse_count_tsv(counts_path, len(rows))


__all__ = [
    "DataSource",
    "Publisher",
    "append_usage",
    "count_jsonl_tokens",
    "stable_usage_event_id",
]
