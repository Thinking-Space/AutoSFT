"""Value objects, request freezing, and delivery validation for Synthesis.

Three faces of the same external contract live together: the stable data
models exchanged across module boundaries, the discovery/fingerprint/freeze
path that turns a mutable request into an immutable input, and the fail-closed
validation of one worker delivery. Validation does not inspect message content
or structure; that contract is owned by the worker-side self-check tooling and
the host training gate. It covers delivery completeness (the three required
files), per-row JSON well-formedness, duplicate detection within the batch and
against target history, the row count, and the publication ledger material
(hashes and message fingerprints).
"""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from autosft.synthesis.catalog import CODE_BENCHMARKS, validate_request_target
from autosft.synthesis.errors import (
    SynConflictError,
    SynContractError,
    SynRequestError,
    SynValidationError,
)
from autosft.synthesis._internal.filesystem import (
    atomic_copy,
    atomic_install_directory,
    atomic_write_json,
    make_read_only_tree,
    message_fingerprint,
    reject_symlink,
    remove_staging_directory,
    require_within,
    sha256_file,
    validate_safe_id,
)


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Stable data models
# ---------------------------------------------------------------------------


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SynContractError(f"{label} must be an object")
    return dict(value)


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise SynContractError(f"{label} must be {qualifier}")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise SynContractError(f"{label} must be a {relation} integer")
    return value


def _path(value: Any, label: str) -> Path:
    return Path(_string(value, label)).expanduser().resolve(strict=False)


def _schema(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise SynContractError(f"{label}.schema_version must be {SCHEMA_VERSION}")


@dataclass(frozen=True, slots=True)
class TargetRequest:
    """One validated target discovered in an external requests directory."""

    target_id: str
    request_dir: Path
    target_file: Path
    materials_dir: Path
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class FrozenTarget:
    """Immutable request snapshot consumed by a single or retried attempt."""

    target_id: str
    snapshot_id: str
    frozen_dir: Path
    target_file: Path
    materials_dir: Path
    input_fingerprint: str
    rows_required: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "target_id": self.target_id,
            "snapshot_id": self.snapshot_id,
            "frozen_dir": str(self.frozen_dir),
            "target_file": str(self.target_file),
            "materials_dir": str(self.materials_dir),
            "input_fingerprint": self.input_fingerprint,
            "rows_required": self.rows_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenTarget":
        payload = _object(value, "frozen target")
        _schema(payload, "frozen target")
        return cls(
            target_id=_string(payload.get("target_id"), "frozen target.target_id"),
            snapshot_id=_string(
                payload.get("snapshot_id"), "frozen target.snapshot_id"
            ),
            frozen_dir=_path(payload.get("frozen_dir"), "frozen target.frozen_dir"),
            target_file=_path(payload.get("target_file"), "frozen target.target_file"),
            materials_dir=_path(
                payload.get("materials_dir"), "frozen target.materials_dir"
            ),
            input_fingerprint=_string(
                payload.get("input_fingerprint"),
                "frozen target.input_fingerprint",
            ),
            rows_required=_positive_int(
                payload.get("rows_required"), "frozen target.rows_required"
            ),
        )


@dataclass(frozen=True, slots=True)
class DeliveryValidation:
    """Machine-verifiable result of validating one worker output directory."""

    target_id: str
    rows: int
    unique_messages: int
    data_path: Path
    report_path: Path
    experience_delta_path: Path
    data_sha256: str
    report_sha256: str
    experience_delta_sha256: str
    message_fingerprints: tuple[str, ...]

    def to_dict(self, *, include_message_fingerprints: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "target_id": self.target_id,
            "rows": self.rows,
            "unique_messages": self.unique_messages,
            "data_path": str(self.data_path),
            "report_path": str(self.report_path),
            "experience_delta_path": str(self.experience_delta_path),
            "data_sha256": self.data_sha256,
            "report_sha256": self.report_sha256,
            "experience_delta_sha256": self.experience_delta_sha256,
        }
        if include_message_fingerprints:
            payload["message_fingerprints"] = list(self.message_fingerprints)
        return payload


# ---------------------------------------------------------------------------
# Request discovery, fingerprinting, and atomic freezing
# ---------------------------------------------------------------------------


def _regular_file(path: Path, label: str, *, non_empty: bool = False) -> Path:
    candidate = reject_symlink(path, label)
    if not candidate.is_file():
        raise SynRequestError(f"{label} must be a regular file: {candidate}")
    if non_empty:
        try:
            if not candidate.read_text(encoding="utf-8").strip():
                raise SynRequestError(f"{label} cannot be empty: {candidate}")
        except UnicodeDecodeError as exc:
            raise SynRequestError(f"{label} must be UTF-8 text: {candidate}") from exc
    return candidate


def _tree_entries(root: Path, label: str) -> tuple[list[tuple[str, str, Path]], int]:
    directory = reject_symlink(root, label)
    if not directory.is_dir():
        raise SynRequestError(f"{label} must be a directory: {directory}")
    entries: list[tuple[str, str, Path]] = []
    file_count = 0
    try:
        candidates = sorted(
            directory.rglob("*"),
            key=lambda item: item.relative_to(directory).as_posix(),
        )
    except OSError as exc:
        raise SynRequestError(f"could not walk {label} {directory}: {exc}") from exc
    for path in candidates:
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise SynRequestError(f"{label} cannot contain symlinks: {path}")
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise SynRequestError(f"could not inspect {path}: {exc}") from exc
        if stat.S_ISDIR(mode):
            entries.append((relative, "directory", path))
        elif stat.S_ISREG(mode):
            entries.append((relative, "file", path))
            file_count += 1
        else:
            raise SynRequestError(f"{label} contains a non-regular entry: {path}")
    return entries, file_count


def _update_file(digest: "hashlib._Hash", name: str, path: Path) -> None:
    digest.update(b"file\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SynRequestError(f"could not fingerprint {path}: {exc}") from exc
    digest.update(b"\0")


def tree_fingerprint(
    target_file: Path,
    materials_dir: Path,
) -> str:
    """Hash names, entry types, and bytes for the complete request tree."""

    target = _regular_file(Path(target_file), "target.md", non_empty=True)
    entries, file_count = _tree_entries(Path(materials_dir), "materials")

    digest = hashlib.sha256()
    _update_file(digest, "target.md", target)
    digest.update(b"directory\0materials\0")
    for relative, kind, entry in entries:
        name = f"materials/{relative}"
        if kind == "directory":
            digest.update(b"directory\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
        else:
            _update_file(digest, name, entry)
    return digest.hexdigest()


def discover_requests(
    requests_dir: Path,
    *,
    allowed_root: Path | None = None,
    minimum_targets: int = 1,
) -> list[TargetRequest]:
    if isinstance(minimum_targets, bool) or minimum_targets < 1:
        raise SynRequestError("minimum_targets must be a positive integer")
    directory = Path(requests_dir).expanduser()
    if directory.is_symlink():
        raise SynRequestError(f"requests directory cannot be a symlink: {directory}")
    directory = directory.resolve(strict=False)
    if allowed_root is not None:
        try:
            directory = require_within(
                directory,
                Path(allowed_root),
                "requests directory",
                allow_root=True,
            )
        except Exception as exc:
            raise SynRequestError(str(exc)) from exc
    if not directory.is_dir():
        raise SynRequestError(f"requests directory does not exist: {directory}")

    requests: list[TargetRequest] = []
    try:
        children = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise SynRequestError(
            f"could not list requests directory {directory}: {exc}"
        ) from exc
    for child in children:
        if child.is_symlink():
            raise SynRequestError(f"request entries cannot be symlinks: {child}")
        if not child.is_dir():
            continue
        try:
            target_id = validate_safe_id(child.name, "target_id")
        except Exception as exc:
            raise SynRequestError(str(exc)) from exc
        target_file = child / "target.md"
        materials_dir = child / "materials"
        fingerprint = tree_fingerprint(target_file, materials_dir)
        # Code targets opt in to the EvalPlus evidence contract; retain the
        # existing instruction-target discovery behavior and empty materials.
        if target_id.split(".", 1)[0] in CODE_BENCHMARKS:
            validate_request_target(target_id, materials_dir)
        requests.append(
            TargetRequest(
                target_id=target_id,
                request_dir=child.resolve(),
                target_file=target_file.resolve(),
                materials_dir=materials_dir.resolve(),
                input_fingerprint=fingerprint,
            )
        )
    if len(requests) < minimum_targets:
        raise SynRequestError(
            f"expected at least {minimum_targets} synthesis targets; found {len(requests)}"
        )
    return requests


def _copy_materials(source: Path, destination: Path) -> None:
    destination.mkdir()
    entries, file_count = _tree_entries(source, "materials")
    for relative, kind, path in entries:
        target = destination / Path(relative)
        if kind == "directory":
            target.mkdir()
        else:
            atomic_copy(path, target, mode=0o444)


def _load_existing(path: Path) -> FrozenTarget:
    manifest = path / "frozen_target.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SynConflictError(
            f"existing frozen target has no valid manifest: {manifest}"
        ) from exc
    try:
        return FrozenTarget.from_dict(payload)
    except Exception as exc:
        raise SynConflictError(
            f"existing frozen target is invalid: {manifest}"
        ) from exc


def freeze_request(
    request: TargetRequest,
    destination_dir: Path,
    *,
    rows_required: int,
) -> FrozenTarget:
    """Freeze one request into an immutable directory installed atomically."""

    if (
        isinstance(rows_required, bool)
        or not isinstance(rows_required, int)
        or rows_required < 1
    ):
        raise SynRequestError("rows_required must be a positive integer")
    destination = Path(destination_dir).expanduser().resolve(strict=False)
    if destination.is_symlink():
        raise SynConflictError(
            f"frozen target destination cannot be a symlink: {destination}"
        )

    current_fingerprint = tree_fingerprint(
        request.target_file,
        request.materials_dir,
    )
    if current_fingerprint != request.input_fingerprint:
        raise SynConflictError(f"request {request.target_id} changed after discovery")
    snapshot_id = f"snapshot-{current_fingerprint[:24]}"

    if destination.exists():
        existing = _load_existing(destination)
        if (
            existing.target_id == request.target_id
            and existing.input_fingerprint == current_fingerprint
            and existing.rows_required == rows_required
        ):
            return existing
        raise SynConflictError(
            f"existing frozen target does not match request: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{request.target_id}.",
            dir=destination.parent,
        )
    )
    try:
        target_file = staging / "target.md"
        materials_dir = staging / "materials"
        atomic_copy(request.target_file, target_file, mode=0o444)
        _copy_materials(request.materials_dir, materials_dir)
        frozen = FrozenTarget(
            target_id=request.target_id,
            snapshot_id=snapshot_id,
            frozen_dir=destination,
            target_file=destination / "target.md",
            materials_dir=destination / "materials",
            input_fingerprint=current_fingerprint,
            rows_required=rows_required,
        )
        atomic_write_json(staging / "frozen_target.json", frozen.to_dict(), mode=0o444)
        make_read_only_tree(staging)
        atomic_install_directory(staging, destination)
        return frozen
    except Exception:
        try:
            staging.chmod(0o755)
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
        except OSError:
            pass
        remove_staging_directory(staging)
        raise


# ---------------------------------------------------------------------------
# Fail-closed delivery validation
# ---------------------------------------------------------------------------


DELIVERY_DATA_NAME = "data.jsonl"
DELIVERY_REPORT_NAME = "report.md"
EXPERIENCE_DELTA_NAME = "experience_delta.md"


def _required_text(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SynValidationError(f"{label} is missing or not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SynValidationError(f"could not read {label} {path}: {exc}") from exc
    if not text.strip():
        raise SynValidationError(f"{label} is empty: {path}")
    return path.resolve()


def iter_validated_rows(path: Path) -> Iterator[tuple[dict[str, Any], str]]:
    data_path = Path(path)
    if data_path.is_symlink() or not data_path.is_file():
        raise SynValidationError(
            f"data.jsonl is missing or not a regular file: {data_path}"
        )
    try:
        stream = data_path.open(encoding="utf-8")
    except OSError as exc:
        raise SynValidationError(
            f"could not open data.jsonl {data_path}: {exc}"
        ) from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SynValidationError(
                    f"invalid JSON at {data_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SynValidationError(f"line {line_number} must be a JSON object")
            yield row, message_fingerprint(row.get("messages"))


def validate_delivery(
    output_dir: Path,
    *,
    target_id: str,
    rows_required: int,
) -> DeliveryValidation:
    """Validate a complete delivery: row count, in-batch uniqueness, required files."""

    try:
        target = validate_safe_id(target_id, "target_id")
    except Exception as exc:
        raise SynValidationError(str(exc)) from exc
    if (
        isinstance(rows_required, bool)
        or not isinstance(rows_required, int)
        or rows_required < 1
    ):
        raise SynValidationError("rows_required must be a positive integer")
    root = Path(output_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise SynValidationError(f"output directory is missing or invalid: {root}")
    root = root.resolve()

    data_path = root / DELIVERY_DATA_NAME
    report_path = _required_text(root / DELIVERY_REPORT_NAME, "report.md")
    experience_delta_path = _required_text(
        root / EXPERIENCE_DELTA_NAME,
        EXPERIENCE_DELTA_NAME,
    )

    seen: set[str] = set()
    ordered_fingerprints: list[str] = []
    for _, fingerprint in iter_validated_rows(data_path):
        row_number = len(ordered_fingerprints) + 1
        if fingerprint in seen:
            raise SynValidationError(
                f"duplicate messages within delivery at row {row_number}"
            )
        seen.add(fingerprint)
        ordered_fingerprints.append(fingerprint)
    if len(ordered_fingerprints) != rows_required:
        raise SynValidationError(
            f"data.jsonl must contain exactly {rows_required} non-empty rows; "
            f"got {len(ordered_fingerprints)}"
        )

    return DeliveryValidation(
        target_id=target,
        rows=len(ordered_fingerprints),
        unique_messages=len(seen),
        data_path=data_path.resolve(),
        report_path=report_path,
        experience_delta_path=experience_delta_path,
        data_sha256=sha256_file(data_path),
        report_sha256=sha256_file(report_path),
        experience_delta_sha256=sha256_file(experience_delta_path),
        message_fingerprints=tuple(ordered_fingerprints),
    )
