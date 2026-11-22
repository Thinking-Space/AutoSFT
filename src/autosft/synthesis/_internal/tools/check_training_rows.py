#!/usr/bin/env python3
"""Training row contract checker: shows and validates what SFT training actually consumes.

SFT training reads only the ``content`` strings of the user and assistant turns in each
row's ``messages``; everything else (extra top-level fields, extra message fields) never
enters training. Row contract: ``messages`` is a non-empty list; only the two roles
user/assistant are allowed (system is forbidden); the first entry must be user; user and
assistant strictly alternate; the last entry must be assistant; every content is a
non-empty string. Both single-turn (one user, one assistant) and multi-turn alternation
are valid. This tool validates a JSONL file row by row against that contract:

  ERROR   — rows that violate the delivery contract (cannot be trained on as intended)
  IGNORED — content silently ignored during training (usually means what you meant to teach was written in the wrong place)

Usage::

    python check_training_rows.py <data.jsonl>            # full validation
    python check_training_rows.py <data.jsonl> --show 3   # additionally print the training view of the first 3 rows

Exit code 0 means every row passes the contract; non-zero means at least one ERROR exists.
Standard library only; runs standalone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_row(obj: object) -> tuple[list[str], list[str]]:
    """Return (errors, ignored_notes) for one parsed JSONL row."""
    errors: list[str] = []
    ignored: list[str] = []
    if not isinstance(obj, dict):
        return ["row is not a JSON object"], ignored
    for key in obj:
        if key != "messages":
            ignored.append(f"top-level field {key!r} does not enter training")
    messages = obj.get("messages")
    if not isinstance(messages, list):
        errors.append("missing messages list")
        return errors, ignored
    if not messages:
        errors.append("messages list is empty")
        return errors, ignored
    roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    if any(role not in ("user", "assistant") for role in roles):
        errors.append(
            f"messages role sequence is {roles}; the contract allows only the two roles user/assistant, "
            "forbidding system and any other role"
        )
        if "system" in roles:
            errors.append(
                "system turn present: the contract forbids system — what you want to teach must be written into"
                " the content of user/assistant"
            )
    else:
        if roles[0] != "user":
            errors.append(
                f"messages role sequence is {roles}; the contract requires the first entry to be user"
            )
        if any(a == b for a, b in zip(roles, roles[1:])):
            errors.append(
                f"messages role sequence is {roles}; the contract requires user and assistant to strictly alternate"
            )
        if roles[-1] != "assistant":
            errors.append(
                f"messages role sequence is {roles}; the contract requires the last entry to be assistant"
            )
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message[{index}] is not an object")
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"message[{index}] content is empty or not a string")
        for key in message:
            if key not in ("role", "content"):
                ignored.append(
                    f"message[{index}] field {key!r} does not enter training"
                )
    return errors, ignored


def training_view(obj: dict) -> str:
    lines = [
        "=== Training view (everything the model will learn; there is nothing else) ==="
    ]
    for message in obj.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        tag = "[assistant → counted in loss]" if role == "assistant" else f"[{role}]"
        lines.append(f"{tag} {message.get('content')!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSONL file to check")
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="print the training view of the first N rows",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        metavar="M",
        help="print at most M ERROR/IGNORED detail lines (default 20)",
    )
    args = parser.parse_args(argv)

    total = error_rows = 0
    ignored_rows = 0
    detail_budget = args.max_report
    shown = 0
    try:
        handle = args.data.open("r", encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.data}: {exc}", file=sys.stderr)
        return 2
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                error_rows += 1
                if detail_budget > 0:
                    print(f"line {line_number}: ERROR JSON parse failed: {exc}")
                    detail_budget -= 1
                continue
            errors, ignored = check_row(obj)
            if errors:
                error_rows += 1
            if ignored:
                ignored_rows += 1
            for note in errors:
                if detail_budget > 0:
                    print(f"line {line_number}: ERROR {note}")
                    detail_budget -= 1
            for note in ignored:
                if detail_budget > 0:
                    print(f"line {line_number}: IGNORED {note}")
                    detail_budget -= 1
            if shown < args.show and isinstance(obj, dict):
                print(training_view(obj))
                shown += 1

    print(
        f"Total {total} rows: {total - error_rows} rows conform to the contract, "
        f"{error_rows} rows ERROR, {ignored_rows} rows contain ignored content."
    )
    print(
        "Reminder: SFT training consumes only the content of the user/assistant turns in messages; "
        "no other field or accompanying file is ever seen by the model; system turns are forbidden by the contract."
    )
    return 1 if error_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
