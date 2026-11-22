"""Crash-safe filesystem primitives and shared path policy for Synthesis modules.

Path policy (safe ids, boundary checks, symlink rejection) and the atomic
write/copy/lock primitives live together: every higher-level operation uses
both, and keeping one definition of each prevents the checks from drifting.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from autosft.synthesis.errors import SynContractError, SynIOError, SynPathError


# ---------------------------------------------------------------------------
# Identifier and path-boundary policy
# ---------------------------------------------------------------------------

SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def validate_safe_id(value: object, label: str = "id") -> str:
    if not isinstance(value, str) or not SAFE_ID_PATTERN.fullmatch(value):
        raise SynPathError(
            f"{label} must match {SAFE_ID_PATTERN.pattern} and contain no path separators"
        )
    return value


def is_within(path: Path, root: Path) -> bool:
    resolved = Path(path).expanduser().resolve(strict=False)
    boundary = Path(root).expanduser().resolve(strict=False)
    return resolved == boundary or boundary in resolved.parents


def require_within(
    path: Path,
    root: Path,
    label: str,
    *,
    allow_root: bool = True,
) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    boundary = Path(root).expanduser().resolve(strict=False)
    if not is_within(resolved, boundary) or (not allow_root and resolved == boundary):
        raise SynPathError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def reject_symlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise SynPathError(f"{label} cannot be a symlink: {candidate}")
    return candidate


def reject_symlink_components(
    path: Path,
    label: str,
    *,
    error: type[Exception] = SynPathError,
) -> None:
    """Fail if any component of ``path`` is a symlink.

    ``reject_symlink`` only inspects the leaf. A symlinked parent directory can
    still smuggle in a file from outside the pool, so every component between
    the anchor and the leaf is checked.
    """

    try:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = Path(os.path.abspath(candidate))
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            current /= component
            if current.is_symlink():
                raise error(f"{label} cannot traverse a symlink component: {current}")
    except error:
        raise
    except (TypeError, ValueError, OSError, UnicodeError) as exc:
        raise error(f"{label} is invalid or unsafe") from exc


# ---------------------------------------------------------------------------
# Time, hashing, and canonical JSON identity
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(content: str) -> str:
    return sha256_bytes(content.encode("utf-8"))


def sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SynIOError(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SynIOError(f"could not hash file {source}: {exc}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def message_fingerprint(messages: Any) -> str:
    """The row identity used throughout the pool.

    Publication writes this into each batch's token sidecar and selection
    recomputes it from the assembled training file; reverse attribution in
    ``selection.record_usage`` only works while the two agree, so there is
    exactly one definition.
    """

    return sha256_text(canonical_json(messages))


# ---------------------------------------------------------------------------
# Shared readers and tree helpers (single definitions for what used to be
# per-module private ``_read_object``/``_make_read_only`` copies)
# ---------------------------------------------------------------------------


def read_json_object(
    path: Path,
    label: str = "JSON",
    *,
    error: type[Exception] = SynContractError,
) -> dict[str, Any]:
    """Read one UTF-8 JSON object, failing with a labelled contract error."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise error(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise error(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error(f"{label} must be an object: {path}")
    return value


def make_read_only_tree(root: Path) -> None:
    """Chmod a staged tree immutable: files 0o444, directories 0o555."""

    base = Path(root)
    for path in sorted(base.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    base.chmod(0o555)


# ---------------------------------------------------------------------------
# Atomic write, copy, lock, and staging installation
# ---------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise SynIOError(
            f"could not open directory for fsync {directory}: {exc}"
        ) from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(mode)
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise SynIOError(f"could not atomically write {destination}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    mode: int = 0o644,
) -> None:
    if not isinstance(payload, Mapping):
        raise SynIOError("atomic JSON payload must be an object")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, content, mode=mode)


def atomic_copy(
    source: Path,
    destination: Path,
    *,
    mode: int | None = None,
) -> str:
    """Copy one regular file and atomically replace the destination.

    The returned digest identifies the exact bytes placed at the destination.
    Metadata is deliberately not part of the identity.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.is_symlink() or not source_path.is_file():
        raise SynIOError(
            f"copy source must be a regular non-symlink file: {source_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with (
            source_path.open("rb") as source_stream,
            os.fdopen(descriptor, "wb") as destination_stream,
        ):
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                destination_stream.write(chunk)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        destination_mode = mode
        if destination_mode is None:
            destination_mode = source_path.stat().st_mode & 0o777
        temporary.chmod(destination_mode)
        os.replace(temporary, destination_path)
        temporary = None
        _fsync_directory(destination_path.parent)
    except OSError as exc:
        raise SynIOError(
            f"could not atomically copy {source_path} to {destination_path}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return digest.hexdigest()


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize a local read-check-write transaction using ``flock``."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise SynIOError(f"lock path cannot be a symlink: {lock_path}")
    try:
        with lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise SynIOError(f"could not lock {lock_path}: {exc}") from exc


def atomic_install_directory(staging_dir: Path, destination: Path) -> None:
    """Install a complete sibling staging directory without overwriting state."""

    staging = Path(staging_dir)
    target = Path(destination)
    if staging.is_symlink() or not staging.is_dir():
        raise SynIOError(f"staging directory must be regular: {staging}")
    if staging.parent.resolve() != target.parent.resolve():
        raise SynIOError("staging and destination directories must be siblings")
    if target.exists() or target.is_symlink():
        raise SynIOError(f"destination already exists: {target}")
    try:
        os.replace(staging, target)
        _fsync_directory(target.parent)
    except OSError as exc:
        raise SynIOError(f"could not install directory {target}: {exc}") from exc


def remove_staging_directory(path: Path) -> None:
    """Best-effort cleanup restricted to an explicitly supplied staging path."""

    staging = Path(path)
    if staging.is_symlink():
        try:
            staging.unlink()
        except FileNotFoundError:
            return
        return
    shutil.rmtree(staging, ignore_errors=True)
