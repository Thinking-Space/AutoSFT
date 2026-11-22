"""Report which intervention candidates actually appear in assembled data.

Candidates are JSONL objects with ``row_id``, ``status`` (accepted, rejected,
or unknown), ``reasons`` (strings), and ChatML ``messages``. Selected rows
contain ``messages`` and may carry ``source_row_id``. A source ID only matches
when the complete messages and any top-level ``tools`` also match exactly.
Without a source ID, content matches are attributed only when a single
candidate has that content. Mixed ChatML and ``input``/``output`` records are
rejected because the converter prioritizes that other serialization format.

The report is descriptive: statuses are supplied by the caller, and neither
rejected nor unknown rows cause a policy rejection. No input is modified.
``candidates.provided`` counts the supplied valid ChatML candidates, not total
gateway requests. Requests that failed or produced no usable ChatML require
separate accounting from gateway records and are outside this denominator.

    python -m autosft.data.intervention_audit \
      --candidates candidates.jsonl --selected final_data.jsonl \
      --output intervention_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from autosft.core.common import utc_now, write_json
from autosft.data.preparation import _invalid_row_reason


STATUSES = ("accepted", "rejected", "unknown")
MAX_PROVENANCE_EXAMPLES = 20


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            reason = _invalid_row_reason(row)
            if reason:
                raise ValueError(f"{path}:{line_number}: {reason}")
            if "input" in row and "output" in row:
                raise ValueError(
                    f"{path}:{line_number}: ambiguous mixed ChatML and input/output "
                    "record; the converter prioritizes input/output, so provide "
                    "unambiguous ChatML records for this audit"
                )
            yield line_number, row


def _messages_fingerprint(row: dict[str, Any]) -> str:
    # Chat templates can consume reasoning, tool calls, and other message fields.
    # Retain them all and distinguish absent tools from an explicit value.
    content = {"messages": row["messages"]}
    if "tools" in row:
        content["tools"] = row["tools"]
    canonical = json.dumps(
        content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit_interventions(candidates_path: Path, selected_path: Path) -> dict[str, Any]:
    """Compare candidate text to final text and return a read-only audit.

    ``selection_fraction`` uses unique attributed candidates as numerator;
    duplicated selected rows contribute only to occurrence counts. Ambiguous
    content without a source ID is unresolved, not evidence of selection or
    non-selection for any individual candidate. ``not_selected`` therefore
    means no exact, unambiguous attribution was found.
    """
    candidates_path, selected_path = Path(candidates_path), Path(selected_path)
    candidates: dict[str, dict[str, Any]] = {}
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for line_number, row in _rows(candidates_path):
        location = f"{candidates_path}:{line_number}"
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{location}: row_id must be a non-empty string")
        if row_id in candidates:
            raise ValueError(f"{location}: duplicate candidate row_id {row_id!r}")
        status = row.get("status")
        if status not in STATUSES:
            raise ValueError(
                f"{location}: status must be accepted, rejected, or unknown"
            )
        reasons = row.get("reasons")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons
        ):
            raise ValueError(f"{location}: reasons must be a list of non-empty strings")
        fingerprint = _messages_fingerprint(row)
        candidates[row_id] = {
            "row_id": row_id,
            "status": status,
            "reasons": list(dict.fromkeys(reasons)),
            "messages_sha256": fingerprint,
            "selected_occurrences": 0,
            "selected_lines": [],
        }
        by_fingerprint[fingerprint].append(row_id)

    provenance: dict[str, Any] = {
        "with_source_row_id_rows": 0,
        "missing_source_row_id_rows": 0,
        "matched_by_source_and_messages_rows": 0,
        "matched_by_messages_only_rows": 0,
        "unresolved_rows": 0,
        "unresolved": {
            issue: {"rows": 0, "examples": []}
            for issue in (
                "unknown_source_row_id",
                "source_content_mismatch",
                "ambiguous_messages",
                "no_candidate_match",
            )
        },
        "example_limit_per_issue": MAX_PROVENANCE_EXAMPLES,
    }
    selected_rows = 0
    for line_number, row in _rows(selected_path):
        selected_rows += 1
        fingerprint = _messages_fingerprint(row)
        source_id = row.get("source_row_id")
        matched_id = None
        issue = None
        example: dict[str, Any] = {"line": line_number}
        if "source_row_id" in row:
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    f"{selected_path}:{line_number}: source_row_id must be a non-empty string"
                )
            provenance["with_source_row_id_rows"] += 1
            example["source_row_id"] = source_id
            candidate = candidates.get(source_id)
            if candidate is None:
                issue = "unknown_source_row_id"
            elif candidate["messages_sha256"] != fingerprint:
                issue = "source_content_mismatch"
            else:
                matched_id = source_id
                provenance["matched_by_source_and_messages_rows"] += 1
        else:
            provenance["missing_source_row_id_rows"] += 1
            ids = by_fingerprint.get(fingerprint, [])
            if len(ids) == 1:
                matched_id = ids[0]
                provenance["matched_by_messages_only_rows"] += 1
            elif ids:
                issue = "ambiguous_messages"
                example["candidate_ids"] = ids
            else:
                issue = "no_candidate_match"
        if matched_id is not None:
            candidate = candidates[matched_id]
            candidate["selected_occurrences"] += 1
            candidate["selected_lines"].append(line_number)
        else:
            provenance["unresolved_rows"] += 1
            bucket = provenance["unresolved"][issue]
            bucket["rows"] += 1
            if len(bucket["examples"]) < MAX_PROVENANCE_EXAMPLES:
                bucket["examples"].append(example)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        unique = sum(row["selected_occurrences"] > 0 for row in rows)
        return {
            "candidates": len(rows),
            "selected_unique_candidates": unique,
            "selected_row_occurrences": sum(
                row["selected_occurrences"] for row in rows
            ),
            "not_selected": len(rows) - unique,
            "selection_fraction": _fraction(unique, len(rows)),
        }

    outcomes = list(candidates.values())
    counts = Counter(row["status"] for row in outcomes)
    selected_occurrences = sum(row["selected_occurrences"] for row in outcomes)
    selected_unique = sum(row["selected_occurrences"] > 0 for row in outcomes)
    status_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reason_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        status_rows[row["status"]].append(row)
        for reason in row["reasons"]:
            reason_rows[reason].append(row)
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "inputs": {
            "candidates": str(candidates_path.resolve()),
            "selected": str(selected_path.resolve()),
        },
        "candidates": {
            "provided": len(outcomes),
            **{status: counts[status] for status in STATUSES},
        },
        "selection": {
            "total_rows": selected_rows,
            "exact_match_rows": selected_occurrences,
            "unique_candidates": selected_unique,
            "duplicate_match_rows": selected_occurrences - selected_unique,
            "selection_fraction": _fraction(selected_unique, len(outcomes)),
            "by_status": {
                status: summarize(status_rows[status]) for status in STATUSES
            },
        },
        "reasons": {
            reason: summarize(reason_rows[reason]) for reason in sorted(reason_rows)
        },
        "provenance": provenance,
        "candidate_outcomes": outcomes,
    }


def write_intervention_audit(
    candidates_path: Path, selected_path: Path, output_path: Path
) -> dict[str, Any]:
    """Write an audit atomically, rejecting an output alias of either input."""
    candidates_path, selected_path, output_path = (
        Path(candidates_path),
        Path(selected_path),
        Path(output_path),
    )
    for source in (candidates_path, selected_path):
        if output_path.resolve() == source.resolve() or (
            output_path.exists() and source.exists() and output_path.samefile(source)
        ):
            raise ValueError("audit output must not overwrite either input")
    report = audit_interventions(candidates_path, selected_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".intervention-audit-", dir=output_path.parent
    ) as temp:
        temporary = Path(temp) / "report.json"
        write_json(temporary, report)
        os.replace(temporary, output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        write_intervention_audit(args.candidates, args.selected, args.output)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Intervention audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
