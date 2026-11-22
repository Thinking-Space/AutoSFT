"""Attempt evidence and supervision: append-only events plus the state machine.

The event stream and the state machine that drives it live together. The store
holds immutable per-attempt events with rebuildable current-state projections;
the supervisor runs exactly one immutable attempt over those events, joining a
worker runtime to atomic publication.
"""

from __future__ import annotations

import json
import os
import re
import signal
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

import fcntl

from autosft.synthesis._internal.filesystem import (
    SAFE_ID_PATTERN as _SAFE_ID,
    atomic_write_json as _atomic_json,
    utc_now as _utc_now,
)
from autosft.synthesis._internal.runtime import WorkerRunRequest, WorkerRunResult

if TYPE_CHECKING:
    from autosft.synthesis._internal.runtime import CodexRuntime


# ---------------------------------------------------------------------------
# Append-only attempt evidence
# ---------------------------------------------------------------------------

#: The attempt state machine's terminal states. Imported rather than
#: re-declared by supervisor and the lifecycle CLI.
TERMINAL_STATES = frozenset({"succeeded", "failed", "stopped"})
#: ``desired_state`` answers one question -- "has a stop been requested?" -- and
#: is NOT a mirror of ``state``. A terminal attempt keeps whatever intent it
#: last had (usually "running"), because ``set_desired_state`` refuses to change
#: it once terminal; that preserves the difference between "succeeded and nobody
#: ever asked it to stop" and "succeeded, but a stop request arrived too late".
#: Read ``state`` for the outcome. A reconciler that treats
#: desired=running + state=succeeded as "should be running but isn't" is
#: misreading this field, not finding a bug.
_DESIRED_STATES = frozenset({"running", "stopped"})
_STATE_EVENTS = {
    "attempt_created": "created",
    # The lifecycle launcher records its claim before spawning the detached
    # process.  Keeping the claim in the append-only stream makes a concurrent
    # launch observe ``starting`` instead of racing on a disposable round
    # projection.
    "launch_claimed": "starting",
    # The detached process records its own identity as soon as it starts.  It
    # is distinct from the nested Codex worker PID recorded by
    # ``worker_started`` so orphan reconciliation can safely inspect the
    # launcher process group.
    "launcher_started": "starting",
    "supervisor_started": "starting",
    "worker_started": "running",
    "worker_resuming": "resuming",
    "validation_started": "validating",
    "publishing_started": "publishing",
    "attempt_stopped": "stopped",
    "pending_cancelled": "stopped",
    "attempt_failed": "failed",
    "delivery_published": "succeeded",
}


class AttemptStoreError(RuntimeError):
    """Raised when durable attempt evidence is malformed or contradictory."""


def _value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


class AttemptStore:
    """Store immutable events and disposable projections for each attempt.

    events.jsonl is the source of truth. current.json and desired_state.json
    are rebuilt after every append and may be deleted without losing evidence.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def attempt_dir(self, attempt_id: str) -> Path:
        if not _SAFE_ID.fullmatch(attempt_id):
            raise AttemptStoreError(f"unsafe attempt_id: {attempt_id!r}")
        return self.root / attempt_id

    def events_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "events.jsonl"

    def current_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "current.json"

    def desired_state_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "desired_state.json"

    @contextmanager
    def _locked(self, attempt_id: str) -> Iterator[None]:
        directory = self.attempt_dir(attempt_id)
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".lock"
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def create(
        self,
        attempt_id: str,
        *,
        target_id: str,
        round_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target_id": target_id,
            "round_id": round_id,
            "desired_state": "running",
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return self.append_event(attempt_id, "attempt_created", payload)

    def ensure_created(
        self,
        attempt_id: str,
        *,
        target_id: str,
        round_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recover a round transaction without replacing existing evidence.

        The caller serializes round mutations. An existing attempt is reusable
        only when its immutable creation identity and metadata match exactly.
        """
        if self.events_path(attempt_id).is_file():
            current = self.current(attempt_id)
            if (
                current.get("target_id") != target_id
                or current.get("round_id") != round_id
                or current.get("metadata", {}) != dict(metadata or {})
            ):
                raise AttemptStoreError(
                    f"attempt creation conflicts with durable intent: {attempt_id}"
                )
            return current
        self.create(
            attempt_id, target_id=target_id, round_id=round_id, metadata=metadata
        )
        return self.current(attempt_id)

    def append_event(
        self,
        attempt_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        if not event_type or not re.fullmatch(r"[a-z][a-z0-9_]*", event_type):
            raise AttemptStoreError(f"invalid event type: {event_type!r}")
        with self._locked(attempt_id):
            events = self._read_events_unlocked(attempt_id)
            if not events and event_type != "attempt_created":
                raise AttemptStoreError(
                    f"attempt {attempt_id!r} must begin with attempt_created"
                )
            if events and event_type == "attempt_created":
                raise AttemptStoreError(f"attempt already exists: {attempt_id}")
            identifier = event_id or str(uuid.uuid4())
            if any(item["event_id"] == identifier for item in events):
                raise AttemptStoreError(f"duplicate attempt event_id: {identifier}")
            event = {
                "schema_version": 1,
                "event_id": identifier,
                "attempt_id": attempt_id,
                "sequence": len(events) + 1,
                "event_type": event_type,
                "occurred_at": _utc_now(),
                "payload": dict(payload or {}),
            }
            projected = self._project([*events, event])
            path = self.events_path(attempt_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            self._write_projections(attempt_id, projected)
            return event

    def set_desired_state(
        self,
        attempt_id: str,
        desired_state: object,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        desired = _value(desired_state)
        if desired not in _DESIRED_STATES:
            raise AttemptStoreError(f"unsupported desired state: {desired!r}")
        current = self.current(attempt_id)
        existing = str(current["desired_state"])
        if existing == desired:
            return current
        if str(current["state"]) in TERMINAL_STATES:
            raise AttemptStoreError("terminal attempt cannot change its desired state")
        if existing == "stopped" and desired != "stopped":
            raise AttemptStoreError(
                "stop is sticky; retry must create a new attempt instead"
            )
        payload: dict[str, Any] = {"desired_state": desired}
        if reason:
            payload["reason"] = reason
        self.append_event(attempt_id, "desired_state_changed", payload)
        return self.current(attempt_id)

    def current(self, attempt_id: str) -> dict[str, Any]:
        with self._locked(attempt_id):
            events = self._read_events_unlocked(attempt_id)
            if not events:
                raise AttemptStoreError(f"attempt does not exist: {attempt_id}")
            projected = self._project(events)
            self._write_projections(attempt_id, projected)
            return projected

    def _read_events_unlocked(self, attempt_id: str) -> list[dict[str, Any]]:
        path = self.events_path(attempt_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AttemptStoreError(f"cannot read attempt events: {path}") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AttemptStoreError(
                    f"invalid attempt event JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise AttemptStoreError(
                    f"attempt event must be an object at {path}:{line_number}"
                )
            expected_sequence = len(events) + 1
            if event.get("attempt_id") != attempt_id:
                raise AttemptStoreError(
                    f"attempt identity mismatch at {path}:{line_number}"
                )
            if event.get("sequence") != expected_sequence:
                raise AttemptStoreError(
                    f"non-contiguous attempt sequence at {path}:{line_number}"
                )
            identifier = event.get("event_id")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen_ids
            ):
                raise AttemptStoreError(
                    f"invalid or duplicate event_id at {path}:{line_number}"
                )
            if not isinstance(event.get("payload"), dict):
                raise AttemptStoreError(
                    f"attempt event payload must be an object at {path}:{line_number}"
                )
            seen_ids.add(identifier)
            events.append(event)
        return events

    def _project(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        if not events or events[0].get("event_type") != "attempt_created":
            raise AttemptStoreError(
                "attempt event stream must begin with attempt_created"
            )
        first = events[0]
        first_payload = first["payload"]
        projection: dict[str, Any] = {
            "schema_version": 1,
            "attempt_id": first["attempt_id"],
            "round_id": first_payload.get("round_id"),
            "target_id": first_payload.get("target_id"),
            "state": "created",
            "desired_state": _value(first_payload.get("desired_state", "running")),
            "sequence": 0,
            "resume_count": 0,
            "created_at": first["occurred_at"],
            "updated_at": first["occurred_at"],
        }
        if projection["desired_state"] not in _DESIRED_STATES:
            raise AttemptStoreError("attempt_created contains an invalid desired_state")
        if isinstance(first_payload.get("metadata"), dict):
            projection["metadata"] = dict(first_payload["metadata"])

        for event in events:
            event_type = str(event["event_type"])
            payload = event["payload"]
            projection["sequence"] = event["sequence"]
            projection["updated_at"] = event["occurred_at"]
            if event_type == "desired_state_changed":
                desired = _value(payload.get("desired_state"))
                if desired not in _DESIRED_STATES:
                    raise AttemptStoreError(
                        f"event {event['event_id']} has invalid desired_state"
                    )
                if projection["desired_state"] == "stopped" and desired != "stopped":
                    raise AttemptStoreError(
                        "event stream tries to clear a sticky stop request"
                    )
                projection["desired_state"] = desired
                if payload.get("reason"):
                    projection["stop_reason"] = payload["reason"]

            new_state = _STATE_EVENTS.get(event_type)
            old_state = str(projection["state"])
            if new_state is not None:
                if old_state in TERMINAL_STATES and new_state != old_state:
                    raise AttemptStoreError(
                        f"terminal attempt state {old_state!r} cannot become {new_state!r}"
                    )
                projection["state"] = new_state

            if event_type in {"worker_started", "worker_resuming"}:
                for name in ("pid", "pgid"):
                    if payload.get(name) is not None:
                        projection[name] = payload[name]
                if event_type == "worker_resuming":
                    projection["resume_count"] = int(
                        payload.get("resume_count", projection["resume_count"] + 1)
                    )
            elif event_type == "launch_claimed":
                claim_id = payload.get("claim_id")
                if claim_id is not None:
                    projection["launch_claim_id"] = str(claim_id)
                projection["launch_claimed_at"] = event["occurred_at"]
            elif event_type == "launcher_started":
                for name in ("pid", "pgid"):
                    if payload.get(name) is not None:
                        projection[f"launcher_{name}"] = payload[name]
                claim_id = payload.get("claim_id")
                if claim_id is not None:
                    projection["launch_claim_id"] = str(claim_id)
            elif event_type == "thread_observed" and payload.get("thread_id"):
                projection["thread_id"] = str(payload["thread_id"])
            elif event_type == "worker_finished":
                projection["returncode"] = payload.get("returncode")
                projection["last_event_type"] = payload.get("last_event_type")
                projection["resume_count"] = int(
                    payload.get("resume_count", projection["resume_count"])
                )
            elif event_type == "attempt_failed":
                projection["failure"] = dict(payload)
            elif event_type == "pending_cancelled":
                projection["pending_cancellation"] = dict(payload)
                projection["stop_reason"] = payload.get("reason")
            elif event_type == "delivery_published":
                projection["delivery"] = dict(payload)
                for name in ("batch_id", "manifest_ref"):
                    if payload.get(name) is not None:
                        projection[name] = payload[name]
            elif event_type in {
                "experience_maintenance_succeeded",
                "experience_maintenance_failed",
            }:
                projection["experience_maintenance"] = {
                    "state": (
                        "succeeded"
                        if event_type == "experience_maintenance_succeeded"
                        else "failed"
                    ),
                    **dict(payload),
                }

        return projection

    def _write_projections(
        self, attempt_id: str, projection: Mapping[str, Any]
    ) -> None:
        _atomic_json(self.current_path(attempt_id), projection)
        _atomic_json(
            self.desired_state_path(attempt_id),
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "desired_state": projection["desired_state"],
                "sequence": projection["sequence"],
                "updated_at": projection["updated_at"],
            },
        )


# ---------------------------------------------------------------------------
# Attempt supervision
# ---------------------------------------------------------------------------


ValidationCallback = Callable[[WorkerRunRequest], Any]
PublisherCallback = Callable[[Any, WorkerRunRequest], Any]


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raw = getattr(value, "value", value)
    if raw is not value:
        return _plain(raw)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class AttemptSupervisor:
    """Run exactly one immutable attempt.

    A failed or stopped attempt is terminal. This class deliberately has no
    retry operation; the service creates a new attempt_id instead.
    """

    def __init__(
        self,
        *,
        store: AttemptStore,
        runtime: "CodexRuntime",
        publisher: PublisherCallback,
        validator: ValidationCallback,
        delivery_complete: Callable[[WorkerRunRequest], bool] | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.publisher = publisher
        self.validator = validator
        # Cheap file-presence predicate handed to the runtime so a clean
        # worker exit without a delivery resumes the session instead of
        # falling through to a terminal validation failure.
        self.delivery_complete = delivery_complete
        self._active_attempt_id: str | None = None

    def request_stop(
        self,
        attempt_id: str,
        *,
        signum: int = signal.SIGTERM,
        reason: str = "stop_requested",
    ) -> dict[str, Any]:
        current = self.store.current(attempt_id)
        if str(current["state"]) in TERMINAL_STATES:
            return current
        self.store.set_desired_state(attempt_id, "stopped", reason=reason)
        self.runtime.stop(signum)
        return self.store.current(attempt_id)

    def run(
        self,
        attempt_id: str,
        request: WorkerRunRequest,
        *,
        install_signal_handlers: bool = False,
    ) -> dict[str, Any]:
        # Keep the handler installed through validation and publication too.
        # A SIGTERM in either phase must be recorded as a terminal stop rather
        # than killing the detached launcher between append-only events.
        with self._signal_handlers(attempt_id, install_signal_handlers):
            return self._run_impl(attempt_id, request)

    def _run_impl(
        self,
        attempt_id: str,
        request: WorkerRunRequest,
    ) -> dict[str, Any]:
        if request.attempt_id != attempt_id:
            raise ValueError("runtime request does not match supervised attempt_id")
        current = self.store.current(attempt_id)
        if str(current["state"]) in TERMINAL_STATES:
            return current
        if str(current["desired_state"]) == "stopped":
            self.store.append_event(
                attempt_id,
                "attempt_stopped",
                {"reason": "stop_requested_before_start"},
            )
            return self.store.current(attempt_id)

        self._active_attempt_id = attempt_id
        self.store.append_event(attempt_id, "supervisor_started", {})
        try:
            result = self.runtime.run(
                request,
                should_stop=lambda: self._stop_requested(attempt_id),
                on_event=lambda event_type, payload: self._runtime_event(
                    attempt_id, event_type, payload
                ),
                work_complete=(
                    (lambda: self.delivery_complete(request))
                    if self.delivery_complete is not None
                    else None
                ),
            )
        except Exception as exc:
            if self._stop_requested(attempt_id):
                self.store.append_event(
                    attempt_id,
                    "attempt_stopped",
                    {"reason": "stop_requested", "runtime_error": repr(exc)},
                )
                return self.store.current(attempt_id)
            self.store.append_event(
                attempt_id,
                "attempt_failed",
                {"reason": "runtime_exception", "error": repr(exc)},
            )
            return self.store.current(attempt_id)
        finally:
            self._active_attempt_id = None

        self.store.append_event(attempt_id, "worker_finished", result.to_dict())
        if self._stop_requested(attempt_id) or result.stopped:
            self.store.append_event(
                attempt_id,
                "attempt_stopped",
                {
                    "reason": result.reason,
                    "returncode": result.returncode,
                    "resume_count": result.resume_count,
                },
            )
            return self.store.current(attempt_id)

        self.store.append_event(
            attempt_id,
            "validation_started",
            {"worker_returncode": result.returncode},
        )
        try:
            delivery = self.validator(request)
        except Exception as exc:
            if self._stop_requested(attempt_id):
                self.store.append_event(
                    attempt_id,
                    "attempt_stopped",
                    {"reason": "stop_requested_during_validation", "error": str(exc)},
                )
                return self.store.current(attempt_id)
            reason = self._validation_failure_reason(request, result)
            self.store.append_event(
                attempt_id,
                "attempt_failed",
                {
                    "reason": reason,
                    "worker": result.to_dict(),
                    "validation_error": str(exc),
                },
            )
            return self.store.current(attempt_id)

        if self._stop_requested(attempt_id):
            self.store.append_event(
                attempt_id,
                "attempt_stopped",
                {"reason": "stop_requested_before_publication"},
            )
            return self.store.current(attempt_id)

        self.store.append_event(
            attempt_id,
            "publishing_started",
            {"validation": _plain(delivery)},
        )
        try:
            publication = self.publisher(delivery, request)
            published_payload = self._publication_payload(publication)
        except Exception as exc:
            if self._stop_requested(attempt_id):
                self.store.append_event(
                    attempt_id,
                    "attempt_stopped",
                    {"reason": "stop_requested_during_publication", "error": repr(exc)},
                )
                return self.store.current(attempt_id)
            self.store.append_event(
                attempt_id,
                "attempt_failed",
                {
                    "reason": "publication_failed",
                    "error": repr(exc),
                    "validation": _plain(delivery),
                },
            )
            return self.store.current(attempt_id)

        # Once publisher() returns, the immutable batch may already be
        # registered in the Data Source.  Always commit delivery_published so
        # a stop racing publication cannot orphan a usable batch.
        self.store.append_event(
            attempt_id,
            "delivery_published",
            published_payload,
        )
        return self.store.current(attempt_id)

    def _stop_requested(self, attempt_id: str) -> bool:
        return str(self.store.current(attempt_id)["desired_state"]) == "stopped"

    @staticmethod
    def _validation_failure_reason(
        request: WorkerRunRequest, result: WorkerRunResult
    ) -> str:
        """Classify a post-worker validation failure by what actually happened.

        This used to be derived from ``result.returncode`` alone, so every
        rejection by the validator was labelled ``delivery_missing_after_worker
        _exit``.  All three real failures in this run were content defects
        (duplicate rows, a row not ending in an assistant message) with an
        intact delivery, and the misleading label sent the operator down a
        transient-loss retry path instead of a data-quality one.  Only claim the
        delivery is missing when it really is; otherwise say validation rejected
        it and let ``validation_error`` carry the specifics.
        """

        if result.returncode != 0:
            return "worker_failed_without_valid_delivery"
        try:
            delivered = (Path(request.paths.output_dir) / "data.jsonl").is_file()
        except (OSError, TypeError, ValueError):
            delivered = False
        return (
            "delivery_rejected_by_validation"
            if delivered
            else "delivery_missing_after_worker_exit"
        )

    def _runtime_event(
        self, attempt_id: str, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        if event_type not in {
            "worker_started",
            "worker_resuming",
            "thread_observed",
        }:
            raise ValueError(f"unsupported runtime event: {event_type}")
        self.store.append_event(attempt_id, event_type, payload)

    @staticmethod
    def _publication_payload(publication: Any) -> dict[str, Any]:
        plain = _plain(publication)
        if not isinstance(plain, dict):
            raise ValueError("publisher must return a structured receipt")
        batch = plain.get("batch")
        batch_object = batch if isinstance(batch, dict) else plain
        batch_id = batch_object.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("publisher receipt is missing batch_id")
        manifest_path = plain.get("manifest_path")
        snapshot_id = plain.get("snapshot_id")
        if not manifest_path and not snapshot_id:
            raise ValueError("publisher receipt is missing a manifest reference")
        return {
            "batch_id": batch_id,
            "manifest_ref": {
                "manifest_path": manifest_path,
                "snapshot_id": snapshot_id,
                "revision": plain.get("revision"),
            },
            "publication": plain,
        }

    @contextmanager
    def _signal_handlers(self, attempt_id: str, enabled: bool) -> Iterator[None]:
        if not enabled:
            yield
            return
        previous: dict[int, Any] = {}

        def handle(signum: int, _frame: Any) -> None:
            self.request_stop(
                attempt_id,
                signum=signum,
                reason=f"signal_{signum}",
            )

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handle)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
