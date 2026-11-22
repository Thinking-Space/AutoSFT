#!/usr/bin/env python3
"""Bulk chat-completion runner over an agent-authored JSONL of prompts.

The caller writes one JSON object per line carrying the `messages` it wants
sent; this script sends them concurrently and records what happened. It holds
no opinion about what a prompt should say, what a good answer looks like, or
what the caller does with the results.

Every job persists its record under ``${AUTO_SFT_RUN_DIR}/llm_calls/``:

    round_<id>/<label>.manifest.json   job-level record, refreshed at each
                                       run's start; keeps the first started_at
    round_<id>/<label>.prompts.jsonl   exactly what is sent, written once,
                                       before the first request goes out
    round_<id>/<label>.responses.jsonl what came back, appended as it arrives
    round_<id>/<label>.events.jsonl    retries, rate changes, progress
    round_<id>/<label>.aligned.jsonl   input-order answers, written only once
                                       every request has succeeded

The prompts file is the point. Rows produced by a model are only
interpretable against the instruction that produced them, and a rate or a
rewrite whose prompt lives in a shell history is not reproducible by anyone
reading the run later. Sending is what writes the record, so the record cannot
drift from the request.

The aligned file exists because joining answers back to source rows is where
callers introduce bugs. It is written only when every row's latest attempt
succeeded, so its existence is the completeness check: no per-row error
branching, no re-derived coverage, line k answers the k-th non-blank input
line.

Resume is by `request_index` over the input's non-blank lines: a request whose
latest response record succeeded is skipped, one that failed is re-sent and a
new record appended. The responses file is append-only, so a re-run leaves the
earlier attempt readable rather than overwriting it. A label is bound to one
input digest and one temperature; a finished label re-run prints ``over`` and
mutates nothing; and one label runs in one process at a time, under an OS file
lock the kernel releases with the process, so a crashed run never wedges its
label.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, IO

import httpx

# Fixed deployment contract. Neither the model nor the sampling temperature
# is a command-line option: two rounds whose data was shaped by different
# models — or by the same model sampled differently — are not comparable, and
# the comparison is the reason this run exists. What to ask is the caller's;
# what answers, and how it is sampled, is the run's.
#
MODEL = os.environ.get("AUTO_SFT_LLM_MODEL", "").strip()
TEMPERATURE = 0.0
DEFAULT_BASE_URL = os.environ.get("AUTO_SFT_LLM_BASE_URL", "").strip().rstrip("/")
ENABLE_THINKING_ENV = "AUTO_SFT_LLM_ENABLE_THINKING"
API_KEY_ENV = "AUTO_SFT_LLM_API_KEY"

MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 300.0
INPUT_KEYS_REQUIRED = frozenset({"messages"})
INPUT_KEYS_ALLOWED = frozenset({"messages", "ref"})
# How many offending input lines to name before rejecting the batch. The
# whole batch is always rejected; the cap only bounds the error message.
MAX_REPORTED_INPUT_ERRORS = 10

# A response body past this size is recorded as a failure rather than
# retried: pulling the same oversize body again spends bandwidth to learn
# nothing, and an answer this large is not going to become a data row.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# In-flight ceiling, not a target. Select --concurrency for provider capacity;
# adaptive pacing governs the realized request rate.
DEFAULT_CONCURRENCY = 192

# Adaptive launch rate. The runner starts unpaced — or, warm-started from the
# rate hint, just under the last observed wall — and only meters itself once
# the gateway has pushed back. On a 429 the launch rate is cut to
# RPM_THROTTLE_FACTOR of the rate it was asking for, and the wall is raised to
# that asking rate if the pushback came higher than any before; between
# pushbacks the rate creeps back up toward the wall. The rate the pacer ASKS
# for is what the gateway rejected, so that is what the cut multiplies — using
# the achieved rate instead makes each cut shrink the next one, which is how a
# job walks from 1,145 rpm to the floor and cannot climb back.
# `--concurrency` is the in-flight CEILING; the realized in-flight count is
# governed by the Throttle below, and the realized request rate by the Pacer.
RATE_WINDOW_SECONDS = 60.0
RPM_THROTTLE_FACTOR = 0.90
# Recovery has to outrun collapse, or a gateway that pushes back for a while
# and then recovers leaves the job crawling. At 1.05 per 30s a rate cut to a
# tenth of the wall needed 24 minutes to climb back, while sustained pushback
# cuts every 10s -- collapse was 6.5x faster than recovery. Growing 1.30 every
# 5s inverts that: the same tenth is recovered in under a minute, and the wall
# below still stops the climb before it reaches the rate that drew pushback.
RPM_GROWTH_FACTOR = 1.30
RPM_GROWTH_SECONDS = 5.0
RATE_CUT_COOLDOWN_SECONDS = 10.0
# How long a learned wall stays authoritative. The wall only rises, which is
# what keeps a cut from capping its own recovery, but it also means a ceiling
# learned while the gateway was degraded outlives the degradation: once the
# quota is restored the job stays pinned under a limit that no longer exists,
# with no way to discover the new capacity. After this long without pushback
# the wall is retired and growth probes upward again; a real ceiling is
# re-learned from the next 429 at the cost of one pushback.
WALL_TTL_SECONDS = 300.0
MIN_RPM = 6.0
# Sanity bound for growth before any pushback is seen. Observed walls on this
# deployment run 1,100-2,600 rpm, so this is above any real ceiling: the first
# 429 replaces it with the measured wall, and exceeding it just means the job
# finds the wall rather than failing.
MAX_RPM = 6000.0
RATE_HINT_NAME = ".rate_hint.json"

# Bound outstanding requests independently from the launch rate. Cut the
# in-flight limit on a 429 and grow it toward --concurrency during recovery;
# otherwise a low limit can continue to suppress throughput after rate recovery.
INFLIGHT_CUT_FACTOR = 0.5
INFLIGHT_GROWTH_FACTOR = 1.5
INFLIGHT_GROWTH_SECONDS = 10.0
MIN_INFLIGHT = 1


class InputError(RuntimeError):
    """The caller's input file or environment is not usable."""


class ResponseTooLarge(RuntimeError):
    """A response body exceeded MAX_RESPONSE_BYTES."""


def normalize_chat_completions_url(value: str) -> str:
    """Accept an OpenAI-compatible base URL or a full chat endpoint."""
    endpoint = value.strip().rstrip("/")
    if not endpoint:
        return endpoint
    suffix = "/chat/completions"
    return endpoint if endpoint.endswith(suffix) else endpoint + suffix


def read_enable_thinking() -> bool | None:
    """Read the optional provider extension without inventing a default.

    Public OpenAI-compatible APIs generally do not define ``enable_thinking``.
    Leaving the setting unset therefore omits it from the request entirely.
    When supplied, accept only the explicit boolean spellings below so a
    typo cannot silently change the provider's reasoning mode.
    """
    raw = os.environ.get(ENABLE_THINKING_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise InputError(
        f"{ENABLE_THINKING_ENV} must be one of: 1, 0, true, false; got {raw!r}"
    )


# --------------------------------------------------------------------------
# strict JSON
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _strict_loads(text: str) -> Any:
    """Parse JSON, rejecting duplicate keys and NaN/Infinity.

    A silently-deduplicated key means the row that was sent is not the row the
    caller wrote, which defeats the point of persisting prompts at all.
    """
    return json.loads(
        text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Write-then-rename so a killed run leaves the old file or the new one,
    never a torn half. The tmp name carries the pid so two processes racing on
    a shared file (the rate hint) cannot cross-write each other's tmp."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    index: int
    line_number: int
    messages: list[dict[str, Any]]
    ref: str | None


@dataclass(frozen=True)
class Manifest:
    requests: list[Request]
    sha256: str


@dataclass(frozen=True)
class Scope:
    """The declared set of row ids a job is allowed to touch.

    Declared up front with --rows rather than inferred from the input: the
    input says what is being sent, the scope says what the caller decided to
    reach, and enforcement is the runner checking the first against the
    second. A ref outside the scope rejects the whole batch before anything
    is sent.
    """

    canonical: str
    size: int
    exact: frozenset[str]
    ranges: dict[str, tuple[tuple[int, int], ...]]


def _parse_span(text: str) -> tuple[int, int] | None:
    """`"12"` -> (12, 12); `"1-20"` -> (1, 20); non-numeric -> None."""
    if text.isdigit():
        value = int(text)
        return (value, value)
    lo_text, sep, hi_text = text.partition("-")
    if sep and lo_text.isdigit() and hi_text.isdigit():
        lo, hi = int(lo_text), int(hi_text)
        if lo > hi:
            raise InputError(f"--rows span {text!r} runs backwards ({lo} > {hi})")
        return (lo, hi)
    return None


def _merge_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def parse_rows_scope(specs: list[str]) -> Scope:
    """Parse repeated/comma-separated --rows items into one Scope.

    An item is `<prefix>:<n>` for a single row, `<prefix>:<a>-<b>` for an
    inclusive range, a bare `<n>` or `<a>-<b>` when refs carry no file
    prefix, or any other string taken as an exact ref match (for rows whose
    ids are not numeric). The prefix is everything before the item's last
    colon, matching how refs name rows.

    A comma list continues the file it started on: `<file>:2,4` is rows 2 and
    4 of that file, the same as `<file>:2,<file>:4`. Repeating the prefix
    stays valid — the canonical scope written to the manifest uses that form,
    and it is the only way to span two files — but writing it once is what
    the skills document. Carrying does not cross a `--rows` flag or reach
    past a leading bare number, so `--rows '1-20,45'` still declares two
    prefix-less spans for refs that carry no file.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    exact: set[str] = set()
    item_count = 0
    for spec in specs:
        carried = ""
        for part in spec.split(","):
            item = part.strip()
            if not item:
                continue
            item_count += 1
            prefix, sep, tail = item.rpartition(":")
            span = _parse_span(tail if sep else item)
            if span is None:
                exact.add(item)
                continue
            if sep:
                carried = prefix
            ranges.setdefault(prefix if sep else carried, []).append(span)
    if not item_count:
        raise InputError("--rows declared nothing")
    merged = {prefix: _merge_spans(spans) for prefix, spans in ranges.items()}
    parts: list[str] = []
    size = 0
    for prefix in sorted(merged):
        for lo, hi in merged[prefix]:
            span_text = str(lo) if lo == hi else f"{lo}-{hi}"
            parts.append(f"{prefix}:{span_text}" if prefix else span_text)
            size += hi - lo + 1
    parts.extend(sorted(exact))
    size += len(exact)
    return Scope(
        canonical=",".join(parts), size=size, exact=frozenset(exact), ranges=merged
    )


def _ref_in_scope(ref: str, scope: Scope) -> bool:
    if ref in scope.exact:
        return True
    prefix, sep, tail = ref.rpartition(":")
    text = tail if sep else ref
    if text.isdigit():
        value = int(text)
        for lo, hi in scope.ranges.get(prefix if sep else "", ()):
            if lo <= value <= hi:
                return True
    return False


def _scope_mismatch_hint(manifest: Manifest, scope: Scope) -> str:
    """Name the cause when out-of-scope refs differ from the scope by prefix.

    A scope rejection reads as "you asked for rows you did not declare", but
    the common cause is a scope that does not say what its author meant: the
    row numbers are right and the file prefix is not. Diagnosing that from
    the ref alone costs a job, so say it here when the evidence supports it,
    and stay silent when it does not — a guess offered as a diagnosis is
    worse than no hint.
    """
    declared = {prefix for prefix in scope.ranges if prefix}
    hints: list[str] = []

    covered_elsewhere: list[tuple[str, str]] = []
    unknown_prefix: set[str] = set()
    for request in manifest.requests:
        if request.ref is None or _ref_in_scope(request.ref, scope):
            continue
        prefix, sep, tail = request.ref.rpartition(":")
        if not sep or not tail.isdigit():
            continue
        value = int(tail)
        for other, spans in scope.ranges.items():
            if other == prefix:
                continue
            if any(lo <= value <= hi for lo, hi in spans):
                covered_elsewhere.append((request.ref, other or "<no prefix>"))
                break
        else:
            if prefix not in declared:
                unknown_prefix.add(prefix)

    if covered_elsewhere:
        ref, other = covered_elsewhere[0]
        hints.append(
            f"row {ref.rpartition(':')[2]} is declared under "
            f"'{other}' but ref {ref!r} names a different file. A comma list "
            f"continues the file it started on, and a leading bare number "
            f"declares a prefix-less span, so '<file>:2,4' means rows 2 and 4 "
            f"of <file> while '2,4' matches only refs with no file prefix. "
            f"Give the scope the same prefix the refs use."
        )
    elif unknown_prefix and declared:
        hints.append(
            f"the input's refs name {sorted(unknown_prefix)[0]!r}, which the "
            f"scope never declares (it declares {sorted(declared)[0]!r}). A "
            f"ref's prefix is its file as a run-dir-relative path; the scope "
            f"must name that same path."
        )
    if not hints:
        return ""
    return "\n  hint: " + "\n  hint: ".join(hints)


def enforce_scope(manifest: Manifest, scope: Scope) -> None:
    """Every request must name a row inside the declared scope.

    Batch-wide like input validation: every violation is collected and up to
    MAX_REPORTED_INPUT_ERRORS named in one rejection. A row without a ref
    cannot be shown to be in scope, so under a declared scope refs stop
    being optional.
    """
    errors: list[str] = []
    for request in manifest.requests:
        if request.ref is None:
            errors.append(
                f"line {request.line_number}: ref is required when --rows "
                f"declares a scope"
            )
        elif not _ref_in_scope(request.ref, scope):
            errors.append(
                f"line {request.line_number}: ref {request.ref!r} is outside "
                f"the declared --rows scope"
            )
    if errors:
        shown = errors[:MAX_REPORTED_INPUT_ERRORS]
        more = len(errors) - len(shown)
        tail = f"\n  ... and {more} more" if more > 0 else ""
        raise InputError(
            "input rejected: requests fall outside the declared row scope\n"
            f"  declared scope: {scope.canonical}\n  "
            + "\n  ".join(shown)
            + tail
            + _scope_mismatch_hint(manifest, scope)
        )


def _validate_messages(messages: Any, line_number: int) -> list[dict[str, Any]]:
    """Check the shape the gateway requires, before spending a request on it.

    The upstream runner this is modelled on validates the envelope and lets a
    malformed `messages` fail at the API, one request at a time. Over a pool
    the size these jobs run against, that is a slow and expensive way to
    discover the input was broken at line 3.
    """
    if not isinstance(messages, list) or not messages:
        raise InputError(f"line {line_number}: messages must be a non-empty list")
    for position, turn in enumerate(messages):
        if not isinstance(turn, dict):
            raise InputError(
                f"line {line_number}: messages[{position}] must be an object"
            )
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not role:
            raise InputError(
                f"line {line_number}: messages[{position}].role must be a non-empty string"
            )
        if not isinstance(content, str):
            raise InputError(
                f"line {line_number}: messages[{position}].content must be a string"
            )
    return messages


def read_input(path: Path) -> Manifest:
    """Parse and validate the whole input before anything is sent.

    Every offending line is collected (and up to MAX_REPORTED_INPUT_ERRORS
    named) rather than stopping at the first: the caller fixes the file once
    instead of re-discovering it one broken row per attempt.
    """
    if not path.is_file():
        raise InputError(f"input is not a readable file: {path}")
    digest = hashlib.sha256()
    requests: list[Request] = []
    errors: list[str] = []
    index = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                payload = _strict_loads(text)
            except ValueError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: each row must be a JSON object")
                continue
            keys = set(payload)
            if not INPUT_KEYS_REQUIRED <= keys or not keys <= INPUT_KEYS_ALLOWED:
                errors.append(
                    f"line {line_number}: row keys must be 'messages' and optionally "
                    f"'ref'; got {sorted(keys)}"
                )
                continue
            ref = payload.get("ref")
            if ref is not None and not isinstance(ref, str):
                errors.append(f"line {line_number}: ref must be a string when present")
                continue
            try:
                messages = _validate_messages(payload["messages"], line_number)
            except InputError as exc:
                errors.append(str(exc))
                continue
            requests.append(
                Request(
                    index=index, line_number=line_number, messages=messages, ref=ref
                )
            )
            index += 1
    if errors:
        shown = errors[:MAX_REPORTED_INPUT_ERRORS]
        more = len(errors) - len(shown)
        tail = f"\n  ... and {more} more" if more > 0 else ""
        raise InputError(
            "input rejected as a whole batch:\n  " + "\n  ".join(shown) + tail
        )
    if not requests:
        raise InputError(f"input holds no requests: {path}")
    return Manifest(requests=requests, sha256=digest.hexdigest())


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def job_paths(run_dir: Path, round_id: str, label: str) -> dict[str, Path]:
    base = Path(run_dir) / "llm_calls" / f"round_{round_id}"
    return {
        "dir": base,
        "manifest": base / f"{label}.manifest.json",
        "prompts": base / f"{label}.prompts.jsonl",
        "responses": base / f"{label}.responses.jsonl",
        "events": base / f"{label}.events.jsonl",
        "aligned": base / f"{label}.aligned.jsonl",
        "lock": base / f"{label}.lock",
        "detach_log": base / f"{label}.detach.log",
    }


def _body_for(request: Request, temperature: float) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "stream": False,
        "temperature": temperature,
        "messages": request.messages,
    }
    enable_thinking = read_enable_thinking()
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


def write_prompt_record(
    paths: dict[str, Path], manifest: Manifest, temperature: float
) -> None:
    """Persist the exact request body for every row, before any of them is sent.

    Written up front rather than alongside each response: a job that dies
    halfway still leaves a complete record of what it was going to ask, and
    the file is a snapshot of one decision rather than an interleaving of
    whatever finished first. Written atomically because a resume skips
    rewriting it — its existence must mean the record is complete.
    """
    lines: list[str] = []
    for request in manifest.requests:
        lines.append(
            _json_line(
                {
                    "request_index": request.index,
                    "input_line_number": request.line_number,
                    "ref": request.ref,
                    "request_body": _body_for(request, temperature),
                }
            )
        )
    atomic_write_text(paths["prompts"], "".join(lines))


def load_latest(
    paths: dict[str, Path], manifest: Manifest
) -> dict[int, dict[str, Any]]:
    """Latest response record per request index.

    Last record wins: the file is append-only, so a retried request has an
    earlier failure and a later success, and only the later one decides
    whether to send it again. Every record's input digest is checked, so a
    label can never silently serve answers computed from a different input.
    """
    path = paths["responses"]
    if not path.is_file():
        return {}
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = _strict_loads(line)
            except ValueError:
                # A torn final line from a killed run; everything before it is
                # still authoritative.
                continue
            if not isinstance(record, dict):
                continue
            index = record.get("request_index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if record.get("input_sha256") != manifest.sha256:
                raise InputError(
                    f"{path} was produced from a different input; move it aside or "
                    f"choose another --label"
                )
            latest[index] = record
    return latest


def completed_of(latest: dict[int, dict[str, Any]]) -> set[int]:
    return {index for index, record in latest.items() if record.get("error") is None}


def materialize_aligned(
    paths: dict[str, Path], manifest: Manifest, latest: dict[int, dict[str, Any]]
) -> bool:
    """Input-order answers file, written only for a fully-succeeded job.

    Joining answers back to source rows is where callers introduce bugs, so
    the runner does the join once, deterministically: one line per input row,
    in input order, each carrying its `ref`. Existence is the contract — the
    file is only written when every request's latest attempt succeeded, so a
    reader never branches on per-row errors or re-derives coverage. Written
    atomically so a killed run cannot leave a torn aligned file.
    """
    rows: list[str] = []
    for request in manifest.requests:
        record = latest.get(request.index)
        if record is None or record.get("error") is not None:
            return False
        rows.append(
            _json_line(
                {
                    "request_index": request.index,
                    "input_line_number": request.line_number,
                    "ref": request.ref,
                    "answer": record.get("answer", ""),
                    "model": record.get("model", MODEL),
                    "temperature": record.get("temperature"),
                    "finish_reason": record.get("finish_reason"),
                    "input_sha256": manifest.sha256,
                }
            )
        )
    atomic_write_text(paths["aligned"], "".join(rows))
    return True


def read_job_manifest(paths: dict[str, Path]) -> dict[str, Any] | None:
    path = paths["manifest"]
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(
            f"job manifest is unreadable: {path} ({exc}); persisted jobs must not "
            f"be edited — use a new --label"
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(f"job manifest is not an object: {path}")
    return payload


def check_label_binding(
    existing: dict[str, Any] | None,
    manifest: Manifest,
    label: str,
    temperature: float | None,
    base_url: str | None = None,
) -> None:
    """A label belongs to one input, one model, and one temperature.

    Rebinding any of them would silently mix two populations under one name —
    the exact failure the durable record exists to prevent. The temperature is
    fixed now, so the check can only trip on a label whose job ran under an
    earlier runner version that still took a --temperature flag; resuming
    such a job at the fixed value would mix two sampling regimes in one
    responses file. `temperature` is None when the caller is only reading
    (status), where sampling settings are not in play.

    Endpoint and model identity are launcher settings. A resume must not mix
    answers from different providers or models in a single response record.
    """
    if existing is None:
        return
    bound_sha = existing.get("input_sha256")
    if isinstance(bound_sha, str) and bound_sha != manifest.sha256:
        raise InputError(
            f"--label {label!r} is bound to a different input "
            f"(manifest records input_sha256 {bound_sha[:12]}..., this file is "
            f"{manifest.sha256[:12]}...); a new question needs a new label"
        )
    bound_model = existing.get("model")
    if (
        temperature is not None
        and isinstance(bound_model, str)
        and bound_model != MODEL
    ):
        raise InputError(
            f"--label {label!r} already ran on model {bound_model!r}, and this "
            f"runner now answers with {MODEL!r}; use a new label — one label is "
            f"one question, asked of one model. Resuming would put two models' "
            f"answers in one responses file, and a verdict is only comparable "
            f"with one made by the same model."
        )
    if temperature is None:
        return
    if base_url is not None and existing.get("base_url") != base_url:
        raise InputError(
            f"--label {label!r} is bound to a different endpoint; use a new label"
        )
    bound_temp = existing.get("temperature")
    if isinstance(bound_temp, (int, float)) and not isinstance(bound_temp, bool):
        if float(bound_temp) != float(temperature):
            raise InputError(
                f"--label {label!r} already ran at temperature {bound_temp}, and "
                f"this runner samples only at {TEMPERATURE}; use a new label — "
                f"one label is one question, asked one way"
            )


def acquire_job_lock(paths: dict[str, Path], label: str) -> IO[str]:
    """One process per label, enforced by the kernel rather than a pidfile.

    flock dies with its holder — including a holder that has become a zombie
    and would still pass a kill-0 liveness probe — so a crashed or killed run
    never wedges its label. The pid written into the file is informational
    only; liveness is the lock itself. The returned handle must stay open for
    the life of the process.
    """
    handle = paths["lock"].open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise InputError(
            f"label {label!r} is already being run by another process "
            f"(lock: {paths['lock']}); wait for it, or watch it with --status"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def spawn_detached(
    *,
    input_path: Path,
    run_dir: Path,
    round_id: str,
    label: str,
    scope: Scope | None,
    concurrency: int,
    base_url: str,
    paths: dict[str, Path],
) -> int:
    """Re-run this job in its own session, then return without waiting.

    The caller is an agent whose turn ends: it runs inside a codex session
    whose whole process group is signalled when the turn closes, so a job
    started in the foreground -- or with a plain shell ``&``, which stays in
    the same group -- is killed partway through. Two full-pool jobs died that
    way, at 822 and 1,945 of 550,000 requests, each one ending on the same
    line as its session's ``turn.completed``. ``start_new_session`` is the
    setsid equivalent: the child gets its own process group, so the group
    signal that ends the turn no longer reaches it.

    The child is the same script under the same arguments with ``--detach``
    removed, which keeps one code path -- resume, locking, scope enforcement
    and the aligned contract are whatever the foreground run would have done.
    This process validates the input and the scope first, so a malformed
    batch still fails in the caller's turn rather than in a log nobody reads
    until later.

    Returns the child's pid. The caller prints it and exits; the child writes
    ``<label>.detach.log`` and its own events file, and ``--status`` reports
    progress from the responses file either way.
    """
    paths["detach_log"].parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(input_path.resolve()),
        "--run-dir",
        str(run_dir),
        "--round-id",
        str(round_id),
        "--label",
        label,
        "--concurrency",
        str(concurrency),
        "--base-url",
        base_url,
    ]
    if scope is not None:
        argv += ["--rows", scope.canonical]
    with paths["detach_log"].open("ab") as log:
        log.write(
            _json_line(
                {
                    "phase": "detach",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "argv": argv,
                }
            ).encode("utf-8")
        )
        log.flush()
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[4]),
            start_new_session=True,
        )
    return child.pid


class ResponseWriter:
    """Append-only response sink with a repaired tail."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = path.open("a", encoding="utf-8")
        if path.stat().st_size:
            with path.open("rb") as probe:
                probe.seek(-1, os.SEEK_END)
                if probe.read(1) != b"\n":
                    self._handle.write("\n")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(_json_line(record))
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()


class EventLog:
    """Structured progress log. Never raises: logging must not fail a job."""

    def __init__(self, path: Path | None) -> None:
        self._handle = path.open("a", encoding="utf-8") if path else None

    def emit(self, phase: str, **fields: Any) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(_json_line({"phase": phase, **fields}))
            self._handle.flush()
        except (OSError, ValueError, TypeError):
            pass

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


# --------------------------------------------------------------------------
# rate hint
# --------------------------------------------------------------------------


def rate_hint_path(run_dir: Path) -> Path:
    return Path(run_dir) / "llm_calls" / RATE_HINT_NAME


def read_rate_hint(run_dir: Path, base_url: str) -> float | None:
    """Warm-start rate from the last observed 429 wall of this deployment.

    The hint is an optimization, never authority: an unreadable, foreign, or
    nonsensical hint is ignored rather than failing the job, because the
    closed loop re-finds the wall on its own either way.
    """
    path = rate_hint_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("model") != MODEL or payload.get("base_url") != base_url:
        return None
    wall = payload.get("wall_rpm")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        return None
    if not (MIN_RPM <= float(wall) < 1e6):
        return None
    return float(wall)


def write_rate_hint(run_dir: Path, base_url: str, wall_rpm: float) -> None:
    try:
        atomic_write_text(
            rate_hint_path(run_dir),
            json.dumps(
                {
                    "model": MODEL,
                    "base_url": base_url,
                    "wall_rpm": round(wall_rpm, 3),
                    "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    except OSError:
        # A lost hint costs one cold start, never the job.
        pass


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


@dataclass
class Brake:
    """Shared 429 backoff.

    One worker being rate-limited is information for every other worker, so
    the pause is held here rather than per-request. The Brake is the transient
    reaction; the Pacer below holds the steady state.
    """

    until: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait(self) -> None:
        while True:
            delay = self.until - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(min(delay, 5.0))

    async def engage(self, seconds: float) -> None:
        async with self._lock:
            self.until = max(self.until, time.monotonic() + seconds)


class Throttle:
    """Adaptive in-flight ceiling: a semaphore whose size follows the gateway.

    A plain `asyncio.Semaphore(n)` is fixed at construction. This one starts
    at the caller's ceiling, halves on a 429, and grows back x1.5 every
    INFLIGHT_GROWTH_SECONDS while no pushback arrives, never exceeding the
    ceiling. A worker acquires a slot the same way it would from a semaphore;
    when the cap shrinks below the number of slots outstanding, the excess
    workers simply finish and their slots are not re-issued until the count
    is back under the cap.

    Cut and grow are deliberately the Pacer's shape so the two governors
    move together: the same 429 cuts both, the same quiet period grows both.
    """

    def __init__(self, ceiling: int) -> None:
        self._ceiling = max(MIN_INFLIGHT, int(ceiling))
        self._cap = self._ceiling
        self._active = 0
        self._last_cut = 0.0
        self._last_growth = time.monotonic()
        self._cond = asyncio.Condition()

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def ceiling(self) -> int:
        return self._ceiling

    @property
    def active(self) -> int:
        return self._active

    def _maybe_grow(self, now: float) -> bool:
        if self._cap >= self._ceiling:
            return False
        if now - self._last_cut < INFLIGHT_GROWTH_SECONDS:
            return False
        if now - self._last_growth < INFLIGHT_GROWTH_SECONDS:
            return False
        grown = max(self._cap + 1, int(self._cap * INFLIGHT_GROWTH_FACTOR))
        self._cap = min(grown, self._ceiling)
        self._last_growth = now
        return True

    async def acquire(self) -> None:
        async with self._cond:
            while True:
                self._maybe_grow(time.monotonic())
                if self._active < self._cap:
                    self._active += 1
                    return
                # Growth is wall-clock gated; a waiter must wake to re-check
                # even when no slot is released, or a cut-to-1 cap with one
                # long-lived request in flight would strand every other
                # worker until that one request finished.
                try:
                    await asyncio.wait_for(
                        self._cond.wait(), timeout=INFLIGHT_GROWTH_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass

    async def release(self) -> None:
        async with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    async def on_429(self) -> int | None:
        """Halve the in-flight cap. Returns the new cap, or None inside the
        cooldown of a previous cut (one wall-hit is one piece of information)."""
        async with self._cond:
            now = time.monotonic()
            if now - self._last_cut < RATE_CUT_COOLDOWN_SECONDS:
                return None
            self._cap = max(MIN_INFLIGHT, int(self._cap * INFLIGHT_CUT_FACTOR))
            self._last_cut = now
            self._last_growth = now
            return self._cap


class Pacer:
    """Adaptive request launch rate governed by provider pushback.

    A job starts unpaced or below a remembered ceiling. A 429 cuts the requested rate; quiet intervals allow multiplicative recovery. The observed ceiling is not lowered by a cut, so throttling cannot cap its own recovery path. Outstanding request capacity is governed separately by Throttle.
    """

    def __init__(self, initial_rpm: float | None, wall_rpm: float | None) -> None:
        self._rpm = initial_rpm
        self._wall = wall_rpm
        self._next_slot = 0.0
        self._last_cut = 0.0
        self._last_growth = time.monotonic()
        self._launches: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> float | None:
        return self._rpm

    @property
    def wall_rpm(self) -> float | None:
        return self._wall

    def _trim(self, now: float) -> None:
        while self._launches and now - self._launches[0] > RATE_WINDOW_SECONDS:
            self._launches.popleft()

    def _measured_rpm(self, now: float) -> float | None:
        self._trim(now)
        if len(self._launches) < 2:
            return None
        window = max(now - self._launches[0], 1.0)
        return len(self._launches) * 60.0 / window

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._launches.append(now)
            self._trim(now)
            if self._rpm is None:
                return
            if (
                now - self._last_cut > RPM_GROWTH_SECONDS
                and now - self._last_growth > RPM_GROWTH_SECONDS
            ):
                # Grow toward the wall, or toward MAX_RPM while no pushback has
                # been seen. The ceiling is MAX_RPM rather than the warm-start
                # hint because a hint is a cache of one past observation: a
                # stale or poisoned one (written while an earlier run sat at
                # the floor) would otherwise cap a healthy job at the floor
                # with no way up. The gateway is the authority on capacity, so
                # before it has said anything the only honest ceiling is the
                # sanity bound; the first pushback replaces it with a real one.
                # A wall that has gone quiet for WALL_TTL_SECONDS is retired
                # for the same reason: capacity that was restored upstream is
                # invisible from here until growth is allowed to probe past
                # the old limit.
                if self._wall is not None and now - self._last_cut > WALL_TTL_SECONDS:
                    self._wall = None
                grown = self._rpm * RPM_GROWTH_FACTOR
                ceiling = self._wall if self._wall is not None else MAX_RPM
                self._rpm = min(grown, ceiling)
                self._last_growth = now
            interval = 60.0 / self._rpm
            slot = max(now, self._next_slot)
            self._next_slot = slot + interval
            delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def on_429(self) -> float | None:
        """Cut the requested launch rate without lowering the recovery ceiling.

        Return the new rpm, or None when this pushback belongs to a recent burst. Use the requested rate rather than achieved throughput: the latter includes the previous cut's slowdown and can cause repeated cuts to collapse the recovery ceiling.
        """
        async with self._lock:
            now = time.monotonic()
            if now - self._last_cut < RATE_CUT_COOLDOWN_SECONDS:
                return None
            base = self._rpm
            if base is None:
                # Unpaced, so there is no asked-for rate: the trailing window
                # is the only estimate available before the first cut.
                base = self._measured_rpm(now)
            if base is None:
                return None
            self._rpm = max(MIN_RPM, base * RPM_THROTTLE_FACTOR)
            if self._wall is None or base > self._wall:
                self._wall = base
            self._last_cut = now
            self._last_growth = now
            return self._rpm


def _retry_after(response: httpx.Response) -> float | None:
    raw = (response.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value != value:
        return None
    return min(value, 300.0)


def _backoff(attempt: int) -> float:
    ceiling = min(30.0, 2.0**attempt)
    return random.uniform(0.5 * ceiling, ceiling)


def _answer_of(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response carried no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choices[0] is not an object")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("choices[0].message.content is not a string")
    return content


async def _read_bounded(response: httpx.Response) -> bytes:
    """Stream the body with a hard size ceiling.

    A pathological answer must not be able to exhaust memory or disk: past
    the cap the request is failed, not downloaded to completion.
    """
    raw = bytearray()
    async for chunk in response.aiter_bytes():
        raw.extend(chunk)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge(f"body exceeded {MAX_RESPONSE_BYTES} bytes")
    return bytes(raw)


async def send_one(
    client: httpx.AsyncClient,
    request: Request,
    *,
    endpoint: str,
    temperature: float,
    brake: Brake,
    pacer: Pacer,
    throttle: Throttle | None = None,
    save_wall: Callable[[float], None],
    log: EventLog,
) -> dict[str, Any]:
    body = _body_for(request, temperature)
    started = time.monotonic()
    last_error = "request did not complete"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        await brake.wait()
        await pacer.acquire()
        try:
            async with client.stream("POST", endpoint, json=body) as response:
                status = response.status_code
                if status == 200:
                    raw = await _read_bounded(response)
                    # NaN/Infinity in a response would survive parsing only to
                    # crash the writer (records are written with allow_nan
                    # off); reject them here so the row fails as
                    # invalid_response instead of killing the job.
                    payload = json.loads(
                        raw.decode("utf-8"), parse_constant=_reject_constant
                    )
                    if not isinstance(payload, dict):
                        raise ValueError("response body is not a JSON object")
                    answer = _answer_of(payload)
                    return {
                        "request_index": request.index,
                        "input_line_number": request.line_number,
                        "ref": request.ref,
                        "input_sha256": "",  # filled by the caller
                        "model": MODEL,
                        "temperature": temperature,
                        "attempts": attempt,
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "answer": answer,
                        "usage": payload.get("usage") or {},
                        "finish_reason": (payload.get("choices") or [{}])[0].get(
                            "finish_reason"
                        ),
                        "error": None,
                    }
                last_error = f"http_{status}"
                retryable = status in RETRYABLE_STATUS
                delay = _retry_after(response) or _backoff(attempt)
                if status == 429:
                    await brake.engage(max(delay, 5.0))
                    new_rpm = await pacer.on_429()
                    if new_rpm is not None and pacer.wall_rpm is not None:
                        log.emit(
                            "rate_cut",
                            wall_rpm=round(pacer.wall_rpm, 1),
                            new_rpm=round(new_rpm, 1),
                        )
                        save_wall(pacer.wall_rpm)
                    if throttle is not None:
                        new_cap = await throttle.on_429()
                        if new_cap is not None:
                            log.emit(
                                "inflight_cut",
                                ceiling=throttle.ceiling,
                                new_cap=new_cap,
                            )
            if not retryable or attempt >= MAX_ATTEMPTS:
                break
            log.emit(
                "retry",
                request_index=request.index,
                attempt=attempt,
                error=last_error,
                delay_seconds=round(delay, 3),
            )
            await asyncio.sleep(delay)
            continue
        except ResponseTooLarge as exc:
            last_error = f"response_too_large: {exc}"
            break
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except ValueError as exc:
            # Malformed or truncated body. Worth one more try: for a rewrite
            # job a silently short answer is corrupted data, not a slow one.
            last_error = f"invalid_response: {exc}"
        if attempt < MAX_ATTEMPTS:
            delay = _backoff(attempt)
            log.emit(
                "retry",
                request_index=request.index,
                attempt=attempt,
                error=last_error,
                delay_seconds=round(delay, 3),
            )
            await asyncio.sleep(delay)

    return {
        "request_index": request.index,
        "input_line_number": request.line_number,
        "ref": request.ref,
        "input_sha256": "",
        "model": MODEL,
        "temperature": temperature,
        "attempts": MAX_ATTEMPTS,
        "latency_seconds": round(time.monotonic() - started, 3),
        "answer": "",
        "usage": {},
        "finish_reason": None,
        "error": last_error,
    }


async def run_job(
    manifest: Manifest,
    pending: list[Request],
    *,
    base_url: str,
    credential: str,
    temperature: float,
    concurrency: int,
    pacer: Pacer,
    save_wall: Callable[[float], None],
    writer: ResponseWriter,
    log: EventLog,
) -> tuple[int, int]:
    brake = Brake()
    throttle = Throttle(concurrency)
    succeeded = 0
    failed = 0
    done = 0
    last_report = time.monotonic()
    last_cap_logged = throttle.cap

    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    # Pool sized to the ceiling: the throttle decides how many are in flight,
    # the pool only has to be able to hold that many when it is at the top.
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credential}",
    }

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, headers=headers
    ) as client:

        async def worker(request: Request) -> dict[str, Any]:
            await throttle.acquire()
            try:
                return await send_one(
                    client,
                    request,
                    endpoint=normalize_chat_completions_url(base_url),
                    temperature=temperature,
                    brake=brake,
                    pacer=pacer,
                    throttle=throttle,
                    save_wall=save_wall,
                    log=log,
                )
            finally:
                await throttle.release()

        tasks = [asyncio.create_task(worker(request)) for request in pending]
        try:
            for coro in asyncio.as_completed(tasks):
                record = await coro
                record["input_sha256"] = manifest.sha256
                writer.write(record)
                done += 1
                if record["error"] is None:
                    succeeded += 1
                else:
                    failed += 1
                now = time.monotonic()
                # Growth happens inside acquire(); surface it here so the
                # events log shows the cap climbing back, not just falling.
                if throttle.cap > last_cap_logged:
                    log.emit(
                        "inflight_grow",
                        ceiling=throttle.ceiling,
                        new_cap=throttle.cap,
                    )
                    last_cap_logged = throttle.cap
                elif throttle.cap < last_cap_logged:
                    last_cap_logged = throttle.cap
                if now - last_report >= 30.0:
                    log.emit(
                        "progress",
                        completed=done,
                        total=len(pending),
                        succeeded=succeeded,
                        failed=failed,
                        inflight_cap=throttle.cap,
                    )
                    last_report = now
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    return succeeded, failed


# --------------------------------------------------------------------------
# credential
# --------------------------------------------------------------------------


def read_credential() -> str:
    for name in (API_KEY_ENV,):
        raw = os.environ.get(name, "")
        if raw.strip():
            secret = raw.strip()
            if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in secret):
                raise InputError(f"{name} contains control characters")
            return secret
    raise InputError(
        f"{API_KEY_ENV} is required. "
        f"It is supplied by the launcher; do not set it yourself."
    )


def fingerprint(secret: str) -> str:
    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send recorded chat batches to the configured compatible provider.",
    )
    parser.add_argument(
        "input", type=Path, help="JSONL with one {'messages': [...]} per line"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="defaults to $AUTO_SFT_RUN_DIR"
    )
    parser.add_argument("--round-id", required=True, help="round this job belongs to")
    parser.add_argument(
        "--label", required=True, help="short name for this job, naming what it was for"
    )
    parser.add_argument(
        "--rows",
        action="append",
        required=False,
        help=(
            "declared row scope this job may touch; repeatable and "
            "comma-separable. Items: '<file>:<n>' for one row, "
            "'<file>:<a>-<b>' for an inclusive range, bare '<n>'/'<a>-<b>' "
            "for prefix-less refs, or an exact ref string. A comma list "
            "continues the file it started on, so '<file>:2-4,7' is rows 2-4 "
            "and 7 of that file; name a second file to switch. Every "
            "request's ref must fall inside it. Required for a new job."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only", action="store_true", help="check the input, send nothing"
    )
    mode.add_argument(
        "--status", action="store_true", help="report progress, send nothing"
    )
    mode.add_argument(
        "--detach",
        action="store_true",
        help=(
            "run the job in its own session and return immediately, printing "
            "the child's pid; for a job that must outlive the caller's turn"
        ),
    )
    return parser


def resolve_run_dir(value: Path | None) -> Path:
    raw = value or os.environ.get("AUTO_SFT_RUN_DIR")
    if not raw:
        raise InputError("--run-dir is required when AUTO_SFT_RUN_DIR is unset")
    run_dir = Path(raw).expanduser()
    if not run_dir.is_dir():
        raise InputError(f"run dir does not exist: {run_dir}")
    return run_dir


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency < 1:
        raise InputError("--concurrency must be at least 1")
    base_url = (args.base_url or "").strip().rstrip("/")
    if not base_url and not (args.validate_only or args.status):
        # An empty override must fail here, loudly: silently falling back to
        # the default would send this job to a different deployment than the
        # caller believes it configured.
        raise InputError("AUTO_SFT_LLM_BASE_URL is required")
    if not (args.validate_only or args.status):
        from urllib.parse import urlsplit

        endpoint = urlsplit(base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise InputError("provider URL must be an absolute HTTP(S) URL")
        if (
            endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise InputError(
                "provider URL must not contain credentials, query, or fragment"
            )
        if DEFAULT_BASE_URL and base_url != DEFAULT_BASE_URL:
            raise InputError("--base-url must match the launcher-configured endpoint")
        if not MODEL:
            raise InputError("AUTO_SFT_LLM_MODEL is required")

    run_dir = resolve_run_dir(args.run_dir)
    label = args.label.strip()
    if not label or "/" in label or label.startswith("."):
        raise InputError("--label must be a bare name without '/' or a leading '.'")
    paths = job_paths(run_dir, str(args.round_id).strip(), label)
    manifest = read_input(args.input)
    scope = parse_rows_scope(args.rows) if args.rows else None

    if args.validate_only:
        if scope is not None:
            enforce_scope(manifest, scope)
        print(f"ok {len(manifest.requests)} requests")
        return 0

    existing = read_job_manifest(paths)

    if args.status:
        check_label_binding(existing, manifest, label, temperature=None)
        latest = load_latest(paths, manifest)
        ok = len(completed_of(latest))
        permanent_failures = len(latest) - ok
        total = len(manifest.requests)
        # A job whose input failed on every attempt stops here rather than
        # retrying forever: the count is what tells the caller to re-run the
        # same command, which re-sends only the failures.
        if ok == total:
            print("over")
        elif permanent_failures:
            print(f"{ok}/{total} ({permanent_failures} failed)")
        else:
            print(f"{ok}/{total}")
        return 0

    check_label_binding(
        existing, manifest, label, temperature=TEMPERATURE, base_url=base_url
    )

    # The declared row scope is part of what a label means. A new job must
    # declare one; a resume may repeat it verbatim or omit it to inherit the
    # bound scope, but never change it — a scope that could widen on resume
    # would make the manifest's declaration worthless.
    bound_scope_text = existing.get("row_scope") if existing else None
    if scope is None:
        if isinstance(bound_scope_text, str) and bound_scope_text:
            scope = parse_rows_scope([bound_scope_text])
        elif existing is None:
            raise InputError(
                "--rows is required for a new job: declare the row ids this "
                "job may touch, e.g. --rows '<file>:1-20,<file>:45'"
            )
    elif isinstance(bound_scope_text, str) and bound_scope_text:
        if scope.canonical != parse_rows_scope([bound_scope_text]).canonical:
            raise InputError(
                f"--label {label!r} is bound to row scope "
                f"[{bound_scope_text}]; a different scope is a different "
                f"job — use a new label"
            )
    if scope is not None:
        enforce_scope(manifest, scope)

    paths["dir"].mkdir(parents=True, exist_ok=True)

    if args.detach:
        # Validate and declare before spawning, so a bad batch or a bad scope
        # fails here, in the caller's turn, rather than in a log the caller
        # reads an hour later. The child then takes the lock itself.
        pid = spawn_detached(
            input_path=args.input,
            run_dir=run_dir,
            round_id=str(args.round_id).strip(),
            label=label,
            scope=scope,
            concurrency=args.concurrency,
            base_url=base_url,
            paths=paths,
        )
        print(
            json.dumps(
                {
                    "state": "detached",
                    "pid": pid,
                    "label": label,
                    "requests": len(manifest.requests),
                    "log": str(paths["detach_log"]),
                    "status_file": str(paths["responses"]),
                    "lock_file": str(paths["lock"]),
                    "manifest_file": str(paths["manifest"]),
                },
                ensure_ascii=False,
            )
        )
        return 0

    # Held for the life of the process; the kernel releases it on any exit.
    job_lock = acquire_job_lock(paths, label)

    latest = load_latest(paths, manifest)
    completed = completed_of(latest)
    total = len(manifest.requests)
    if len(completed) == total:
        # Idempotent short-circuit: a finished job answers "over" and mutates
        # nothing, so re-running the same command is always a safe move. The
        # aligned file is materialized here if a run under an older version
        # (or a crash after the last response) left it missing.
        if not paths["aligned"].is_file():
            if materialize_aligned(paths, manifest, latest):
                print(f"aligned -> {paths['aligned']}")
        print("over")
        return 0

    credential = read_credential()
    pending = [r for r in manifest.requests if r.index not in completed]

    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev_runs = existing.get("run_count") if existing else None
    if not isinstance(prev_runs, int) or isinstance(prev_runs, bool):
        prev_runs = 1 if existing else 0
    started_at = existing.get("started_at") if existing else None
    if not isinstance(started_at, str) or not started_at:
        started_at = now_utc
    manifest_payload: dict[str, Any] = {
        "label": label,
        "round_id": str(args.round_id).strip(),
        "model": MODEL,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "concurrency": args.concurrency,
        "input_path": str(args.input.resolve()),
        "input_sha256": manifest.sha256,
        "row_scope": scope.canonical if scope is not None else None,
        "row_scope_size": scope.size if scope is not None else None,
        "request_count": len(manifest.requests),
        "resumed_completed": len(completed),
        "credential": fingerprint(credential),
        "started_at": started_at,
        "run_count": prev_runs + 1,
    }
    if existing:
        manifest_payload["last_resumed_at"] = now_utc
    atomic_write_text(
        paths["manifest"],
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
    )
    # The label-to-input binding guarantees an existing prompts file already
    # describes exactly these requests at exactly this temperature, so it is
    # written once and never churned by resumes.
    if not paths["prompts"].is_file():
        write_prompt_record(paths, manifest, TEMPERATURE)

    log = EventLog(paths["events"])
    log.emit(
        "start",
        model=MODEL,
        request_count=len(manifest.requests),
        pending=len(pending),
        resumed=len(completed),
        concurrency=args.concurrency,
        temperature=TEMPERATURE,
    )

    wall_hint = read_rate_hint(run_dir, base_url)
    if wall_hint is not None:
        # The hint seeds the STARTING RATE only, never the ceiling. A hint is
        # a cache of one past observation; treating it as the wall caps growth
        # at whatever value it holds, so a stale or poisoned hint (one written
        # while a previous run was collapsed at the floor) would pin a fresh
        # job at the floor with no way up. The wall is for pushbacks this run
        # actually receives, so it starts unknown.
        pacer = Pacer(
            initial_rpm=max(MIN_RPM, wall_hint * RPM_THROTTLE_FACTOR),
            wall_rpm=None,
        )
        log.emit(
            "rate_warm_start",
            wall_rpm=round(wall_hint, 1),
            initial_rpm=round(pacer.rpm or 0.0, 1),
        )
    else:
        pacer = Pacer(initial_rpm=None, wall_rpm=None)

    writer = ResponseWriter(paths["responses"])
    try:
        succeeded, failed = asyncio.run(
            run_job(
                manifest,
                pending,
                base_url=normalize_chat_completions_url(base_url),
                credential=credential,
                temperature=TEMPERATURE,
                concurrency=args.concurrency,
                pacer=pacer,
                save_wall=lambda wall: write_rate_hint(run_dir, base_url, wall),
                writer=writer,
                log=log,
            )
        )
    finally:
        writer.close()

    try:
        recheck = read_input(args.input)
        input_changed = recheck.sha256 != manifest.sha256
    except InputError:
        # The input became unreadable mid-run; the responses can no longer be
        # attributed to the persisted prompts with confidence.
        input_changed = True

    aligned_written = False
    if not input_changed:
        final_latest = load_latest(paths, manifest)
        if len(completed_of(final_latest)) == total:
            aligned_written = materialize_aligned(paths, manifest, final_latest)

    log.emit(
        "complete",
        succeeded=succeeded,
        failed=failed,
        input_changed=input_changed,
        responses=str(paths["responses"]),
        aligned=str(paths["aligned"]) if aligned_written else None,
    )
    log.close()

    print(f"{succeeded} ok, {failed} failed -> {paths['responses']}")
    if aligned_written:
        print(f"aligned -> {paths['aligned']}")
    if input_changed:
        print(
            "ERROR: the input file changed while the job ran; the responses are "
            "not attributable to the persisted prompts",
            file=sys.stderr,
        )
        return 1
    del job_lock  # released by process exit; named here so its life is visible
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)
