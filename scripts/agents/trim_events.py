#!/usr/bin/env python3
"""Shrink a role session's event stream before it lands on disk.

`codex --json` inlines each command's entire stdout into the event that
reports it, so one `rg` over the run directory becomes a single megabyte-long
line. The root polls these logs for liveness, and reading a few such lines
costs it more context than every task report in the run combined.

Only the bulky output fields are shortened, and every line stays valid JSON:
`run_role.sh` parses this file to recover the `thread.started` id that lets an
interrupted role resume, so a byte-level truncation would silently destroy
resumability. Lines that are not JSON, and JSON without an oversized field,
pass through byte-identical.

Reads stdin, writes stdout, and never fails the session: on any internal
error the original line is emitted unchanged.
"""

from __future__ import annotations

import json
import os
import sys

# Long enough to keep a failure diagnosable, short enough that the pathological
# lines stop dominating the file. Both ends are kept: a command's output
# usually opens with what it found and closes with the error that matters.
LIMIT = int(os.environ.get("AUTO_SFT_EVENT_FIELD_LIMIT", "2000"))
HEAD = LIMIT // 2
TAIL = LIMIT - HEAD - 1

# The fields codex uses to carry raw command output. Anything else in the
# event (type, ids, exit codes, timestamps) is structural and stays intact.
BULKY_FIELDS = ("aggregated_output", "output", "stdout", "stderr", "text")


def _trim(value: str) -> str:
    dropped = len(value) - HEAD - TAIL
    return f"{value[:HEAD]}\n...[trimmed {dropped} bytes]...\n{value[-TAIL:]}"


def _walk(node: object) -> bool:
    """Shorten oversized bulky fields in place. True if anything changed."""
    changed = False
    if isinstance(node, dict):
        for key, value in node.items():
            if key in BULKY_FIELDS and isinstance(value, str) and len(value) > LIMIT:
                node[key] = _trim(value)
                changed = True
            elif isinstance(value, (dict, list)):
                changed |= _walk(value)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                changed |= _walk(item)
    return changed


def main() -> int:
    out = sys.stdout
    for line in sys.stdin:
        # Cheap guard: the vast majority of lines are already small, and a
        # session's output should not pay a JSON round-trip per event.
        if len(line) <= LIMIT:
            out.write(line)
            out.flush()
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            out.write(line)
            out.flush()
            continue
        try:
            if _walk(event):
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
            else:
                out.write(line)
        except (ValueError, RecursionError, TypeError):
            out.write(line)
        out.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
