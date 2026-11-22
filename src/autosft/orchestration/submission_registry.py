"""Canonical lookup and locking for SFT/evaluation submissions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


_TASK_RE = re.compile(r"^tasks/[^/\s]+$")
FAILED_EVAL_STATUSES = frozenset(
    {"Failed", "Stopped", "Killed", "Cancelled", "Canceled"}
)


def submission_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield canonical submission records and reject a damaged manifest."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"submission manifest contains invalid JSON at line "
                    f"{line_number}: {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"submission manifest line {line_number} must be a JSON object: "
                    f"{path}"
                )
            if record.get("event"):
                continue
            task = _task(record)
            if task is None:
                raise ValueError(
                    f"submission manifest line {line_number} is missing a canonical "
                    f"task (expected tasks/<name>): {path}"
                )
            yield record


def _canonical_path(value: object) -> str:
    if not isinstance(value, (str, Path)) or not str(value):
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def _task(record: dict[str, Any]) -> str | None:
    """Return a canonical ``tasks/<name>`` value from a submission record."""
    value = record.get("task")
    if not isinstance(value, str):
        return None
    task = value.strip()
    return task if _TASK_RE.fullmatch(task) else None


@contextmanager
def submission_lock(run_dir: Path) -> Iterator[None]:
    """Serialize local check-and-create operations for one durable run."""
    lock_path = Path(run_dir).resolve() / ".local-submission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def find_sft_submission(
    manifest: Path,
    *,
    run_name: str,
) -> dict[str, Any] | None:
    """Return the first durable task for a run name."""
    matches = [
        record
        for record in submission_records(Path(manifest))
        if record.get("run_name") == run_name and _task(record)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda record: str(record.get("submitted_at") or ""),
    )


def sft_submission_conflicts(
    record: dict[str, Any],
    *,
    output_dir: Path,
    dataset_path: Path,
    train_iters: int,
    seq_length: int,
) -> list[str]:
    """Describe identity-changing differences hidden behind one run name."""
    expected = {
        "output_dir": _canonical_path(output_dir),
        "dataset_path": _canonical_path(dataset_path),
        "train_iters": str(train_iters),
        "seq_length": str(seq_length),
    }
    actual = {
        "output_dir": _canonical_path(record.get("output_dir")),
        "dataset_path": _canonical_path(record.get("dataset_path")),
        "train_iters": str(record.get("train_iters") or ""),
        "seq_length": str(record.get("seq_length") or ""),
    }
    return [
        f"{field}: existing={actual[field]!r}, requested={value!r}"
        for field, value in expected.items()
        if actual[field] != value
    ]


def record_repeats(record: dict[str, Any]) -> int:
    """How many repeats one eval submission record covers.

    Current records carry ``repeats`` (one task evaluates its benchmark N
    times). Records written by the earlier one-task-per-repeat scheme carry
    ``repeat`` (an index) instead and each covers exactly one repeat.
    """
    value = record.get("repeats")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return 1


def record_repeat_indices(record: dict[str, Any]) -> list[int]:
    """Which repeat indices one eval submission record's artifacts cover."""
    if "repeats" not in record:
        legacy = record.get("repeat")
        if isinstance(legacy, int) and not isinstance(legacy, bool) and legacy >= 1:
            return [legacy]
    return list(range(1, record_repeats(record) + 1))


def eval_identity(
    *,
    model_path: object,
    model_identity_value: object = None,
    benchmark: object,
    eval_backend: object,
    limit: object,
    repeats: object = None,
) -> tuple[str, str, str, str, str]:
    """Identity of one benchmark evaluation independent of agent run naming.

    ``repeats`` is the number of times the task evaluates the benchmark, so a
    task that sampled once never satisfies a request for four samples. A
    missing/empty value normalizes to ``"1"`` so records written before
    repeats existed keep matching a single-repeat request.
    """
    return (
        (
            str(model_identity_value)
            if model_identity_value
            else model_identity(model_path)
        ),
        str(benchmark or "").strip().lower(),
        str(eval_backend or "").strip().lower(),
        str(limit if limit is not None else ""),
        str(repeats or "1"),
    )


def model_identity(model_path: object) -> str:
    """Pin a converted model to checkpoint provenance, not a reusable path."""
    canonical_model = _canonical_path(model_path)
    if not canonical_model:
        raise ValueError("model_path is required")
    manifest_path = Path(canonical_model) / "conversion_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"converted model is missing a valid manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"conversion manifest must be an object: {manifest_path}")
    checkpoint = _canonical_path(manifest.get("checkpoint"))
    step = manifest.get("ckpt_step")
    if (
        not checkpoint
        or not isinstance(step, str)
        or not step.isdigit()
        or step.startswith("0")
    ):
        raise ValueError(
            f"conversion manifest needs canonical checkpoint and ckpt_step: {manifest_path}"
        )
    digest = hashlib.sha256(f"{checkpoint}\0{step}".encode("utf-8")).hexdigest()
    return f"checkpoint:{digest}"


def _eval_submission_manifest_path(search_root: Path, manifest: Path) -> Path:
    search_root = Path(search_root).resolve()
    eval_runs = search_root / "eval_runs"
    resolved = Path(manifest).resolve()
    if resolved.name != "eval_submissions.jsonl" or resolved.parent.parent != eval_runs:
        raise ValueError(
            "eval manifest must be "
            f"{eval_runs}/<name>/eval_submissions.jsonl; got {resolved}"
        )
    return resolved


def eval_manifest_paths(
    search_root: Path,
    target_manifest: Path | None = None,
) -> list[Path]:
    """List only current per-eval-run submission manifests."""
    search_root = Path(search_root).resolve()
    eval_runs = search_root / "eval_runs"
    candidates: set[Path] = set()
    if target_manifest is not None:
        candidates.add(_eval_submission_manifest_path(search_root, target_manifest))
    if eval_runs.is_dir():
        for candidate in eval_runs.glob("*/eval_submissions.jsonl"):
            try:
                candidates.add(_eval_submission_manifest_path(search_root, candidate))
            except ValueError:
                continue
    return sorted(candidates, key=str)


def eval_backends(search_root: Path, target_manifest: Path) -> set[str]:
    """Return evaluation backends already committed to one durable run."""
    backends: set[str] = set()
    for manifest in eval_manifest_paths(search_root, target_manifest):
        for record in submission_records(manifest):
            if not _task(record):
                continue
            backend = record.get("eval_backend")
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError(f"eval submission is missing eval_backend: {manifest}")
            backends.add(backend.strip().lower())
    return backends


def _eval_record_identity(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return eval_identity(
        model_path=record.get("model_path"),
        model_identity_value=record.get("model_identity"),
        benchmark=record.get("benchmark"),
        eval_backend=record.get("eval_backend"),
        limit=record.get("limit"),
        repeats=record_repeats(record),
    )


def current_eval_submissions(
    search_root: Path,
    target_manifest: Path | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    """Keep current attempts while retaining superseded tasks in their manifests."""
    records = [
        (record, manifest)
        for manifest in eval_manifest_paths(search_root, target_manifest)
        for record in submission_records(manifest)
    ]
    by_task = {submission_task(record): record for record, _ in records}
    superseded: set[str] = set()
    parents: dict[str, str] = {}
    replacements: dict[str, str] = {}
    for record, manifest in records:
        request_path = manifest.parent / "eval_request.json"
        if not request_path.is_file():
            continue
        request = json.loads(request_path.read_text(encoding="utf-8"))
        retry_of = request.get("retry_of_task") if isinstance(request, dict) else None
        if retry_of is None:
            continue
        if not isinstance(retry_of, str):
            raise ValueError(f"eval retry needs a canonical retry_of_task: {manifest}")
        source = by_task.get(retry_of)
        if (
            source is None
            or retry_of == submission_task(record)
            or _eval_record_identity(source) != _eval_record_identity(record)
        ):
            raise ValueError(
                f"eval retry does not match an existing task identity: {manifest}"
            )
        task = submission_task(record)
        if retry_of in replacements and replacements[retry_of] != task:
            raise ValueError(f"eval task has multiple replacement tasks: {retry_of}")
        parents[task] = retry_of
        replacements[retry_of] = task
        superseded.add(retry_of)
    for task in parents:
        visited: set[str] = set()
        while task in parents:
            if task in visited:
                raise ValueError("eval retry lineage contains a cycle")
            visited.add(task)
            task = parents[task]
    return [
        (record, manifest)
        for record, manifest in records
        if submission_task(record) not in superseded
    ]


def eval_task_status(record: dict[str, Any], manifest: Path) -> str | None:
    """Read the latest lifecycle observation; submission status alone is insufficient."""
    task = submission_task(record)
    observations: list[tuple[float | None, str | None]] = []

    def observed_time(value: Any) -> float | None:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return None

    final_path = Path(manifest).parent / "eval_final_status.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if isinstance(final, dict):
            for item in final.get("tasks") or []:
                if isinstance(item, dict) and item.get("task") == task:
                    observations.append(
                        (
                            observed_time(
                                item.get("status_observed_at")
                                or final.get("generated_at")
                            ),
                            item.get("local_status")
                            if not item.get("local_task_error")
                            else None,
                        )
                    )
    events = Path(manifest).parent / "local_events.jsonl"
    last_event = None
    if events.is_file():
        for line in events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if isinstance(event, dict) and event.get("task") == task:
                    last_event = event
    if last_event is not None:
        status = last_event.get("status")
        if last_event.get("local_task_exit_code", 0) != 0:
            status = None
        observations.append((observed_time(last_event.get("polled_at")), status))
    if not observations:
        return None
    if len(observations) > 1 and any(time is None for time, _ in observations):
        statuses = {status for _, status in observations}
        return next(iter(statuses)) if len(statuses) == 1 else None
    latest_time = max(time or 0 for time, _ in observations)
    latest_statuses = {
        status for time, status in observations if (time or 0) == latest_time
    }
    return next(iter(latest_statuses)) if len(latest_statuses) == 1 else None


def find_eval_submission(
    search_root: Path,
    target_manifest: Path,
    *,
    model_path: Path,
    benchmark: str,
    eval_backend: str,
    limit: int,
    repeats: int = 1,
) -> tuple[dict[str, Any], Path] | None:
    """Find an existing task across every eval manifest in the durable run.

    Only records that cover the whole repeat set are candidates: a legacy
    per-repeat record (``repeat: k``) never stands in for a task that must
    produce every ``rep_<k>/`` itself.
    """
    requested = eval_identity(
        model_path=model_path,
        benchmark=benchmark,
        eval_backend=eval_backend,
        limit=limit,
        repeats=repeats,
    )
    matches: list[tuple[str, str, dict[str, Any], Path]] = []
    for record, manifest in current_eval_submissions(search_root, target_manifest):
        if not _task(record):
            continue
        if "repeats" not in record and record.get("repeat") not in (None, 1):
            continue
        existing = _eval_record_identity(record)
        if existing == requested:
            matches.append(
                (
                    str(record.get("submitted_at") or ""),
                    str(manifest),
                    record,
                    manifest,
                )
            )
    if not matches:
        return None
    _, _, record, manifest = min(matches, key=lambda item: (item[0], item[1]))
    return record, manifest


def submission_task(record: dict[str, Any]) -> str | None:
    """Return the canonical task name stored in a submission record."""
    return _task(record)
