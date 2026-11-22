"""Private, rebuildable lifecycle for the Synthesis experience library.

Experience is derived from immutable batches and feedback events.  A worker
may propose a complete next summary, but publication is performed here with
a single compare-and-swap transaction over the global experience library.
The framework only owns storage, versioning, and evidence mechanics; how the
library is organized internally belongs to the maintainer worker.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from autosft.synthesis.errors import SynConflictError, SynContractError
from autosft.synthesis._internal.filesystem import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    exclusive_file_lock,
    sha256_file,
    validate_safe_id,
)
from autosft.synthesis._internal.runtime import (
    EXPERIENCE_PLACEHOLDERS,
    CodexRuntime,
    WorkerRunRequest,
    render_prompt,
)
from autosft.synthesis._internal.storage import DataSource


SCHEMA_VERSION = "1.0"
INITIAL_EXPERIENCE = "# Experience\n\nNo prior evidence.\n"
_SHA256 = frozenset("0123456789abcdef")
#: CAS retry budget for one maintenance call; a loser recomputes against the
#: winning version each iteration before giving up with ``status: conflict``.
_MAX_CAS_RETRIES = 3


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SynContractError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SynContractError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SynContractError(f"{label} must be a JSON object: {path}")
    return value


def _safe_read_text(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise SynContractError(f"{label} is missing or unsafe: {path}")
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SynContractError(f"{label} is invalid: {path}: {exc}") from exc
    if not value.strip():
        raise SynContractError(f"{label} must not be empty: {path}")
    return value


def _empty_cursor() -> dict[str, Any]:
    return {
        "revision": 0,
        "snapshot_id": None,
        "batch_ids": [],
        "event_ids": [],
    }


def _validate_cursor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SynContractError(f"{label} must be an object")
    revision = value.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise SynContractError(f"{label}.revision must be non-negative")
    snapshot_id = value.get("snapshot_id")
    if snapshot_id is not None and (
        not isinstance(snapshot_id, str) or not snapshot_id
    ):
        raise SynContractError(f"{label}.snapshot_id must be null or non-empty")
    cursor: dict[str, Any] = {
        "revision": revision,
        "snapshot_id": snapshot_id,
    }
    for field in ("batch_ids", "event_ids"):
        items = value.get(field)
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item for item in items
        ):
            raise SynContractError(f"{label}.{field} must be a string list")
        if items != sorted(set(items)):
            raise SynContractError(f"{label}.{field} must be sorted and unique")
        cursor[field] = list(items)
    return cursor


def _cursor_covers(current: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    return set(requested["batch_ids"]).issubset(current["batch_ids"]) and set(
        requested["event_ids"]
    ).issubset(current["event_ids"])


def _unprocessed_event_ids(
    evidence_event_ids: Iterable[str], base_cursor: Mapping[str, Any]
) -> set[str]:
    """Event ids inside the frozen scope that the base version has not consumed.

    The maintainer prompt hands this set to the worker verbatim and
    ``_validate_candidate`` compares the receipt against it.  Both callers must
    derive it here: when the two sides computed it independently the worker had
    no way to guess the ``batch:`` prefix rule, and every receipt was rejected.
    """

    consumed = set(base_cursor["event_ids"]) | {
        f"batch:{batch_id}" for batch_id in base_cursor["batch_ids"]
    }
    return set(evidence_event_ids) - consumed


@dataclass(frozen=True, slots=True)
class ExperienceVersion:
    version: int
    version_id: str
    cursor: Mapping[str, Any]
    experience_path: Path
    experience_sha256: str
    update_event_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "version_id": self.version_id,
            "cursor": copy.deepcopy(dict(self.cursor)),
            "experience_path": str(self.experience_path),
            "experience_sha256": self.experience_sha256,
            "update_event_id": self.update_event_id,
        }


@dataclass(frozen=True, slots=True)
class FrozenEvidence:
    target_id: str
    batch_id: str | None
    update_id: str
    snapshot_dir: Path
    snapshot_id: str
    revision: int
    evidence_cursor: Mapping[str, Any]
    cursor: Mapping[str, Any]
    event_ids: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExperienceCandidateContext:
    update_id: str
    target_id: str
    batch_id: str | None
    generation: int
    evidence: FrozenEvidence
    base: ExperienceVersion
    events_file: Path
    new_evidence_dir: Path
    output_dir: Path
    log_path: Path


@dataclass(frozen=True, slots=True)
class _ExperienceWorkerPaths:
    data_source_snapshot_dir: Path
    events_file: Path
    experience_file: Path
    new_evidence_dir: Path
    output_dir: Path
    log_path: Path

    def allowed_read_roots(self) -> tuple[Path, ...]:
        return (
            self.data_source_snapshot_dir,
            self.events_file,
            self.experience_file,
            self.new_evidence_dir,
            self.output_dir,
        )


class CodexExperienceCandidateGenerator:
    """Run the private maintainer prompt without exposing it via public APIs."""

    def __init__(
        self,
        *,
        repo_root: Path,
        prompt_path: Path,
        max_resumes: int,
        resume_backoff_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.prompt_path = Path(prompt_path).resolve()
        self.max_resumes = max_resumes
        self.resume_backoff_seconds = resume_backoff_seconds
        self.timeout_seconds = timeout_seconds

    def __call__(self, context: ExperienceCandidateContext) -> Any:
        paths = _ExperienceWorkerPaths(
            data_source_snapshot_dir=context.evidence.snapshot_dir,
            events_file=context.events_file,
            experience_file=context.base.experience_path,
            new_evidence_dir=context.new_evidence_dir,
            output_dir=context.output_dir,
            log_path=context.log_path,
        )
        values = {
            "SYN_EXPERIENCE_UPDATE_ID": context.update_id,
            "SYN_TARGET_ID": context.target_id,
            "SYN_BATCH_ID": context.batch_id or "",
            "SYN_DATA_SOURCE_DIR": str(context.evidence.snapshot_dir),
            "SYN_EVIDENCE_CURSOR": canonical_json(context.evidence.evidence_cursor),
            "SYN_EXPERIENCE_EVENTS_FILE": str(context.events_file),
            "SYN_EXPERIENCE_FILE": str(context.base.experience_path),
            "SYN_EXPERIENCE_BASE_VERSION": context.base.version_id,
            "SYN_EXPERIENCE_BASE_CURSOR": canonical_json(context.base.cursor),
            "SYN_EXPERIENCE_NEXT_CURSOR": canonical_json(context.evidence.cursor),
            "SYN_EXPERIENCE_PROCESSED_EVENT_IDS": canonical_json(
                sorted(
                    _unprocessed_event_ids(
                        context.evidence.event_ids,
                        context.base.cursor,
                    )
                )
            ),
            "SYN_NEW_EVIDENCE_DIR": str(context.new_evidence_dir),
            "SYN_EXPERIENCE_OUTPUT_DIR": str(context.output_dir),
        }
        rendered = render_prompt(
            self.prompt_path,
            values,
            allowed_placeholders=EXPERIENCE_PLACEHOLDERS,
        )
        request = WorkerRunRequest(
            attempt_id=f"{context.update_id}-g{context.generation:03d}",
            target_id=context.target_id,
            rendered_prompt=rendered,
            paths=paths,  # type: ignore[arg-type]
            rows_required=1,
            max_resumes=self.max_resumes,
            resume_backoff_seconds=self.resume_backoff_seconds,
            timeout_seconds=self.timeout_seconds,
            resume_prompt=(
                "Continue the same frozen Synthesis experience update. Finish both "
                "candidate output files without reading newer evidence."
            ),
        )
        runtime = CodexRuntime(
            repo_root=self.repo_root,
            python_executable=sys.executable,
        )
        return runtime.run(request)


class ExperienceStore:
    """Immutable experience versions plus a single CAS projection."""

    def __init__(self, data_source: DataSource | Any):
        self.source = (
            data_source
            if isinstance(data_source, DataSource)
            else DataSource(data_source)
        )
        self.root = self.source.root / "_internal" / "experience"
        self.versions_dir = self.root / "versions"
        self.locks_dir = self.root / "locks"
        self.events_dir = self.root / "events"
        self.current_path = self.root / "current.json"
        self.lock_path = self.locks_dir / "experience.lock"

    def ensure(self) -> ExperienceVersion:
        self.source.ensure()
        with exclusive_file_lock(self.lock_path):
            if self.current_path.exists():
                return self.current()
            content = INITIAL_EXPERIENCE
            version_id = "experience-v00000000"
            final_dir = self.versions_dir / version_id
            event_id = "bootstrap-experience"
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_type": "experience_bootstrap",
                "event_id": event_id,
                "next_version": version_id,
                "next_cursor": _empty_cursor(),
                "candidate_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
            self._install_version(final_dir, content, event)
            self._install_event(self.events_dir / f"{event_id}.json", event)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "version": 0,
                "version_id": version_id,
                "cursor": _empty_cursor(),
                "experience_path": str(final_dir / "experience.md"),
                "experience_sha256": event["candidate_sha256"],
                "update_event_id": event_id,
            }
            atomic_write_json(self.current_path, payload)
            return self._version_from_payload(payload)

    def current(self) -> ExperienceVersion:
        payload = _safe_read_json(self.current_path, "experience current")
        return self._version_from_payload(payload)

    def publish(
        self,
        *,
        expected_version_id: str,
        next_cursor: Mapping[str, Any],
        candidate_path: Path,
        update_id: str,
        receipt: Mapping[str, Any],
    ) -> ExperienceVersion:
        cursor = _validate_cursor(next_cursor, "experience.next_cursor")
        with exclusive_file_lock(self.lock_path):
            current = self.current()
            if current.version_id != expected_version_id:
                raise SynConflictError(
                    "experience version conflict: expected "
                    f"{expected_version_id}, got {current.version_id}"
                )
            if not _cursor_covers(cursor, current.cursor):
                raise SynContractError(
                    "experience next cursor would discard already processed evidence"
                )
            content = _safe_read_text(candidate_path, "experience candidate")
            candidate_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            version = current.version + 1
            version_id = f"experience-v{version:08d}-{candidate_sha[:12]}"
            event_id = (
                "experience-"
                + _digest([update_id, expected_version_id, cursor, candidate_sha])[:40]
            )
            final_dir = self.versions_dir / version_id
            event_path = self.events_dir / f"{event_id}.json"
            payload = {
                "schema_version": SCHEMA_VERSION,
                "version": version,
                "version_id": version_id,
                "cursor": cursor,
                "experience_path": str(final_dir / "experience.md"),
                "experience_sha256": candidate_sha,
                "update_event_id": event_id,
            }
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_type": "experience_update",
                "event_id": event_id,
                "update_id": update_id,
                "base_version": expected_version_id,
                "next_version": version_id,
                "base_cursor": copy.deepcopy(dict(current.cursor)),
                "next_cursor": cursor,
                "candidate_sha256": candidate_sha,
                "receipt": copy.deepcopy(dict(receipt)),
            }
            self._install_version(final_dir, content, event)
            self._install_event(event_path, event)
            atomic_write_json(self.current_path, payload)
            return self._version_from_payload(payload)

    def _version_from_payload(self, payload: Mapping[str, Any]) -> ExperienceVersion:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SynContractError("experience schema is unsupported")
        version = payload.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise SynContractError("experience version is invalid")
        version_id = payload.get("version_id")
        if not isinstance(version_id, str) or not version_id:
            raise SynContractError("experience version_id is invalid")
        path = Path(str(payload.get("experience_path", ""))).resolve(strict=False)
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise SynContractError("experience path escapes store") from exc
        digest = payload.get("experience_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in _SHA256 for char in digest)
        ):
            raise SynContractError("experience sha256 is invalid")
        if sha256_file(path) != digest:
            raise SynContractError("experience content hash mismatch")
        event_id = payload.get("update_event_id")
        if not isinstance(event_id, str) or not event_id:
            raise SynContractError("experience event id is invalid")
        return ExperienceVersion(
            version=version,
            version_id=version_id,
            cursor=_validate_cursor(payload.get("cursor"), "experience.cursor"),
            experience_path=path,
            experience_sha256=digest,
            update_event_id=event_id,
        )

    def _install_version(
        self, final_dir: Path, content: str, event: Mapping[str, Any]
    ) -> None:
        expected_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if final_dir.exists():
            path = final_dir / "experience.md"
            receipt = _safe_read_json(
                final_dir / "version_receipt.json", "experience version receipt"
            )
            if sha256_file(path) != expected_sha or canonical_json(
                receipt
            ) != canonical_json(event):
                raise SynConflictError(
                    f"experience version exists with different content: {final_dir}"
                )
            return
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=final_dir.parent)
        )
        try:
            atomic_write_text(staging / "experience.md", content, mode=0o444)
            atomic_write_json(staging / "version_receipt.json", event, mode=0o444)
            os.replace(staging, final_dir)
            for path in final_dir.iterdir():
                path.chmod(0o444)
            final_dir.chmod(0o555)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _install_event(self, path: Path, event: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = _safe_read_json(path, "experience event")
            if canonical_json(existing) != canonical_json(event):
                raise SynConflictError(
                    f"experience event exists with different content: {path}"
                )
            return
        atomic_write_json(path, event, mode=0o444)


class ExperienceMaintainer:
    """Freeze facts, obtain a candidate, and publish with a single CAS."""

    def __init__(
        self,
        data_source: DataSource | Any,
        *,
        run_dir: Path,
        candidate_generator: "CodexExperienceCandidateGenerator",
    ) -> None:
        self.source = (
            data_source
            if isinstance(data_source, DataSource)
            else DataSource(data_source)
        )
        self.store = ExperienceStore(self.source)
        self.run_dir = Path(run_dir).expanduser().resolve(strict=False)
        self.candidate_generator = candidate_generator

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        data_source: DataSource | None = None,
    ) -> "ExperienceMaintainer":
        generator = CodexExperienceCandidateGenerator(
            repo_root=config.repo_root,
            prompt_path=config.experience_prompt_path,
            max_resumes=config.max_resumes,
            resume_backoff_seconds=config.resume_backoff_seconds,
            timeout_seconds=config.attempt_timeout_seconds,
        )
        return cls(
            data_source or DataSource(config.data_source_dir),
            run_dir=config.run_dir,
            candidate_generator=generator,
        )

    def maintain(
        self, target_id: str, *, batch_id: str | None = None
    ) -> dict[str, Any]:
        target_id = validate_safe_id(target_id, "target_id")
        if batch_id is not None:
            batch_id = validate_safe_id(batch_id, "batch_id")
        base = self.store.ensure()
        evidence = self._freeze_evidence(target_id, batch_id=batch_id)
        if _cursor_covers(base.cursor, evidence.cursor):
            return self._result(evidence, base, conflicts=0, status="up_to_date")

        update_root = (
            self.run_dir
            / "synthesis"
            / "_internal"
            / "experience_updates"
            / evidence.update_id
        )
        events_file, evidence_dir = self._write_frozen_inputs(update_root, evidence)
        conflicts = 0
        generation = self._next_generation(update_root)
        for _ in range(_MAX_CAS_RETRIES + 1):
            base = self.store.current()
            if _cursor_covers(base.cursor, evidence.cursor):
                return self._result(
                    evidence, base, conflicts=conflicts, status="up_to_date"
                )
            output_dir = update_root / f"generation-{generation:03d}" / "output"
            log_path = update_root / f"generation-{generation:03d}" / "worker.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            context = ExperienceCandidateContext(
                update_id=evidence.update_id,
                target_id=target_id,
                batch_id=batch_id,
                generation=generation,
                evidence=evidence,
                base=base,
                events_file=events_file,
                new_evidence_dir=evidence_dir,
                output_dir=output_dir,
                log_path=log_path,
            )
            self.candidate_generator(context)
            candidate = self._validate_candidate(context)
            try:
                version = self.store.publish(
                    expected_version_id=base.version_id,
                    next_cursor=evidence.cursor,
                    candidate_path=candidate["candidate_path"],
                    update_id=evidence.update_id,
                    receipt=candidate["receipt"],
                )
            except SynConflictError:
                # The CAS loser recomputes against the winning version on the
                # next iteration instead of overwriting it.
                conflicts += 1
                generation += 1
                continue
            return self._result(
                evidence, version, conflicts=conflicts, status="published"
            )
        return self._result(
            evidence,
            self.store.current(),
            conflicts=conflicts,
            status="conflict",
        )

    @staticmethod
    def _result(
        evidence: FrozenEvidence,
        version: ExperienceVersion,
        *,
        conflicts: int,
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "update_id": evidence.update_id,
            "target_id": evidence.target_id,
            "batch_id": evidence.batch_id,
            "evidence_cursor": copy.deepcopy(dict(evidence.evidence_cursor)),
            "status": status,
            "cas_conflicts": conflicts,
            "experience": version.to_dict(),
        }

    def _freeze_evidence(
        self, target_id: str, *, batch_id: str | None
    ) -> FrozenEvidence:
        snapshot = self.source.snapshot()
        snapshot_dir = Path(snapshot["snapshot_dir"]).resolve()
        manifest = _safe_read_json(snapshot_dir / "manifest.json", "evidence manifest")
        batches = manifest.get("batches")
        events = manifest.get("events")
        cursors = manifest.get("event_cursors")
        if not isinstance(batches, Mapping) or not isinstance(events, Mapping):
            raise SynContractError("evidence manifest has invalid batches/events")
        if not isinstance(cursors, Mapping):
            raise SynContractError("evidence manifest has invalid event cursors")

        event_records: list[dict[str, Any]] = []
        events_by_id: dict[str, dict[str, Any]] = {}
        for kind in ("batch_status", "usage"):
            identifiers = events.get(kind)
            if not isinstance(identifiers, list):
                raise SynContractError(f"evidence events.{kind} must be a list")
            projection = snapshot_dir / "events" / f"{kind}.jsonl"
            try:
                lines = projection.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise SynContractError(
                    f"could not read frozen {kind} events: {projection}"
                ) from exc
            if len(lines) != len(identifiers):
                raise SynContractError(f"frozen {kind} projection length mismatch")
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SynContractError(
                        f"frozen {kind} event is invalid at index {index}"
                    ) from exc
                if (
                    not isinstance(record, dict)
                    or record.get("event_id") != identifiers[index]
                ):
                    raise SynContractError(
                        f"frozen {kind} event identity mismatch at index {index}"
                    )
                copied = copy.deepcopy(record)
                event_records.append(copied)
                events_by_id[str(record["event_id"])] = copied

        batch_records: list[dict[str, Any]] = []
        target_batch_ids: set[str] = set()
        for identifier in sorted(batches):
            entry = batches[identifier]
            if not isinstance(entry, Mapping):
                raise SynContractError(f"batch {identifier} manifest entry is invalid")
            record = {
                "schema_version": SCHEMA_VERSION,
                "event_type": "batch_published",
                "event_id": f"batch:{identifier}",
                "batch_id": identifier,
                "target_id": entry.get("target_id"),
                "status": entry.get("status"),
                "immutable_batch": copy.deepcopy(dict(entry)),
            }
            batch_records.append(record)
            if entry.get("target_id") == target_id:
                target_batch_ids.add(identifier)
        if batch_id is not None and batch_id not in target_batch_ids:
            raise SynContractError(
                f"published batch {batch_id!r} is absent or belongs to another target"
            )

        # Cursor identity lists are canonicalized below, but candidates
        # receive the frozen records in append order for interpretation.
        ordered_event_ids = [str(record["event_id"]) for record in event_records]
        cursor = {
            "revision": int(manifest["revision"]),
            "snapshot_id": str(manifest["snapshot_id"]),
            "batch_ids": sorted(str(item) for item in batches),
            "event_ids": sorted(events_by_id),
        }
        evidence_cursor = {
            "revision": int(manifest["revision"]),
            "snapshot_id": str(manifest["snapshot_id"]),
            "event_cursors": {
                str(key): int(value) for key, value in sorted(cursors.items())
            },
        }
        update_id = "experience-" + _digest([target_id, batch_id, evidence_cursor])[:40]
        records = batch_records + [
            events_by_id[event_id] for event_id in ordered_event_ids
        ]
        return FrozenEvidence(
            target_id=target_id,
            batch_id=batch_id,
            update_id=update_id,
            snapshot_dir=snapshot_dir,
            snapshot_id=str(manifest["snapshot_id"]),
            revision=int(manifest["revision"]),
            evidence_cursor=evidence_cursor,
            cursor=_validate_cursor(cursor, "experience evidence cursor"),
            event_ids=tuple(str(record["event_id"]) for record in records),
            records=tuple(records),
        )

    def _write_frozen_inputs(
        self, update_root: Path, evidence: FrozenEvidence
    ) -> tuple[Path, Path]:
        input_dir = update_root / "input"
        evidence_dir = input_dir / "new_evidence"
        events_file = input_dir / "experience_events.jsonl"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        content = "".join(canonical_json(record) + "\n" for record in evidence.records)
        self._write_once(events_file, content)
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "update_id": evidence.update_id,
            "target_id": evidence.target_id,
            "batch_id": evidence.batch_id,
            "snapshot_id": evidence.snapshot_id,
            "revision": evidence.revision,
            "snapshot_dir": str(evidence.snapshot_dir),
            "event_ids": list(evidence.event_ids),
            "cursor": copy.deepcopy(dict(evidence.cursor)),
            "events_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        manifest_path = evidence_dir / "evidence_manifest.json"
        encoded = json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n"
        self._write_once(manifest_path, encoded)
        return events_file, evidence_dir

    @staticmethod
    def _write_once(path: Path, content: str) -> None:
        if path.exists():
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise SynConflictError(
                    f"frozen experience input changed for update: {path}"
                )
            return
        atomic_write_text(path, content, mode=0o444)

    @staticmethod
    def _next_generation(update_root: Path) -> int:
        generations: list[int] = []
        if update_root.exists():
            for path in update_root.glob("generation-*"):
                try:
                    generations.append(int(path.name.removeprefix("generation-")))
                except ValueError:
                    continue
        return max(generations, default=0) + 1

    def _validate_candidate(
        self, context: ExperienceCandidateContext
    ) -> dict[str, Any]:
        candidate_path = context.output_dir / "experience.next.md"
        receipt_path = context.output_dir / "experience_update_receipt.json"
        _safe_read_text(candidate_path, "experience candidate")
        receipt = _safe_read_json(receipt_path, "experience update receipt")
        if receipt.get("update_id") != context.update_id:
            raise SynContractError("experience receipt update_id mismatch")
        if receipt.get("target_id") != context.target_id:
            raise SynContractError("experience receipt target_id mismatch")
        if receipt.get("batch_id") != context.batch_id:
            raise SynContractError("experience receipt batch_id mismatch")
        if receipt.get("base_version") != context.base.version_id:
            raise SynContractError("experience receipt base_version mismatch")
        if canonical_json(receipt.get("base_cursor")) != canonical_json(
            context.base.cursor
        ):
            raise SynContractError("experience receipt base_cursor mismatch")
        if canonical_json(receipt.get("next_cursor")) != canonical_json(
            context.evidence.cursor
        ):
            raise SynContractError("experience receipt next_cursor mismatch")
        allowed_ids = _unprocessed_event_ids(
            context.evidence.event_ids, context.base.cursor
        )
        processed = receipt.get("processed_event_ids")
        if not isinstance(processed, list) or any(
            not isinstance(item, str) or not item for item in processed
        ):
            raise SynContractError(
                "experience receipt.processed_event_ids must be a string list"
            )
        if len(processed) != len(set(processed)) or set(processed) != allowed_ids:
            raise SynContractError(
                "experience receipt.processed_event_ids must exactly "
                "cover the frozen evidence scope"
            )
        status = receipt.get("candidate_status")
        if status not in {"changed", "no_change"}:
            raise SynContractError("experience receipt.candidate_status is invalid")
        for field in ("added", "revised", "retired", "unresolved_conflicts"):
            if not isinstance(receipt.get(field), list):
                raise SynContractError(f"experience receipt.{field} must be a list")
        return {
            "candidate_path": candidate_path,
            "receipt_path": receipt_path,
            "receipt": receipt,
        }


__all__ = [
    "ExperienceMaintainer",
    "ExperienceStore",
]
