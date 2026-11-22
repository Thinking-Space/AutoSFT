#!/usr/bin/env python3
"""Estimate Codex agent context composition from the run JSONL log.

This is a control-plane audit, not an exact model-context reconstruction. Codex
does not expose the final packed prompt, so we parse the durable JSON stream in
``logs/agent.log`` and bucket the text that can enter or summarize the working
context: reasoning, assistant messages, command strings, command output, todo
state, runner lines, and metadata/errors. Token counts use ``tiktoken`` when its verified vocabulary is cached
locally and otherwise fall back to a simple character heuristic, so the
audit never adds a hard dependency to eval collection.
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import os
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from autosft.core.common import utc_now, write_json


TEXT_FIELDS = ("text", "command", "aggregated_output", "output", "message", "error")
CATEGORY_LABELS = {
    "runner_log": "runner / shell log lines",
    "reasoning": "agent reasoning",
    "assistant_message": "assistant messages",
    "command_input": "command strings",
    "command_output": "command output",
    "todo": "todo state",
    "tool_or_event_metadata": "tool/event metadata",
    "error": "errors",
}

# Context accounting is diagnostic. Missing optional assets must never cause
# a network request or delay controller startup; the trainer has its own exact
# tokenizer-based budget accounting.
_TOKEN_ENCODER = None
TOKEN_COUNTER = "chars_per_4"
try:
    cache_dir = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get(
            "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")
        ),
    )
    source = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cached = (
        Path(cache_dir) / hashlib.sha1(source.encode()).hexdigest()
        if cache_dir
        else None
    )
    expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    if (
        cached is not None
        and cached.is_file()
        and hashlib.sha256(cached.read_bytes()).hexdigest() == expected
    ):
        import tiktoken

        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        TOKEN_COUNTER = "tiktoken:cl100k_base"
except (ImportError, OSError, ValueError):
    pass


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return max(1, len(_TOKEN_ENCODER.encode(text, disallowed_special=())))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _add_bucket(
    buckets: dict[str, dict[str, Any]],
    category: str,
    text: str,
    *,
    event_type: str | None = None,
    item_type: str | None = None,
    line_number: int | None = None,
) -> None:
    if not text:
        return
    bucket = buckets.setdefault(
        category,
        {
            "label": CATEGORY_LABELS.get(category, category),
            "chars": 0,
            "estimated_tokens": 0,
            "records": 0,
            "examples": [],
        },
    )
    bucket["chars"] += len(text)
    bucket["estimated_tokens"] += _estimate_tokens(text)
    bucket["records"] += 1
    examples = bucket["examples"]
    if len(examples) < 5:
        examples.append(
            {
                "line": line_number,
                "event_type": event_type,
                "item_type": item_type,
                "chars": len(text),
                "snippet": text.replace("\n", " ")[:240],
            }
        )


def _extract_item_text(
    buckets: dict[str, dict[str, Any]],
    item: dict[str, Any],
    *,
    event_type: str,
    line_number: int,
) -> None:
    item_type = str(item.get("type") or "unknown")
    if item_type == "reasoning":
        _add_bucket(
            buckets,
            "reasoning",
            _stringify(item.get("text")),
            event_type=event_type,
            item_type=item_type,
            line_number=line_number,
        )
        return
    if item_type == "agent_message":
        _add_bucket(
            buckets,
            "assistant_message",
            _stringify(item.get("text")),
            event_type=event_type,
            item_type=item_type,
            line_number=line_number,
        )
        return
    if item_type == "command_execution":
        _add_bucket(
            buckets,
            "command_input",
            _stringify(item.get("command")),
            event_type=event_type,
            item_type=item_type,
            line_number=line_number,
        )
        for key in ("aggregated_output", "output"):
            _add_bucket(
                buckets,
                "command_output",
                _stringify(item.get(key)),
                event_type=event_type,
                item_type=item_type,
                line_number=line_number,
            )
        if item.get("exit_code") not in (None, 0):
            _add_bucket(
                buckets,
                "error",
                _stringify(
                    {"exit_code": item.get("exit_code"), "status": item.get("status")}
                ),
                event_type=event_type,
                item_type=item_type,
                line_number=line_number,
            )
        return
    if item_type == "todo_list":
        _add_bucket(
            buckets,
            "todo",
            _stringify(item.get("items")),
            event_type=event_type,
            item_type=item_type,
            line_number=line_number,
        )
        return

    parts = []
    for key in TEXT_FIELDS:
        value = item.get(key)
        if value:
            parts.append(_stringify(value))
    if not parts:
        parts.append(_stringify(item))
    _add_bucket(
        buckets,
        "tool_or_event_metadata",
        "\n".join(parts),
        event_type=event_type,
        item_type=item_type,
        line_number=line_number,
    )


def analyze_log(log_path: Path) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    total_lines = 0
    json_lines = 0

    if not log_path.exists():
        return {
            "log_path": str(log_path),
            "exists": False,
            "token_counter": TOKEN_COUNTER,
            "total_lines": 0,
            "json_lines": 0,
            "item_counts": {},
            "event_counts": {},
            "buckets": {},
            "totals": {"chars": 0, "estimated_tokens": 0},
        }

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            total_lines += 1
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                _add_bucket(buckets, "runner_log", stripped, line_number=line_number)
                continue
            if not isinstance(event, dict):
                _add_bucket(
                    buckets,
                    "tool_or_event_metadata",
                    _stringify(event),
                    line_number=line_number,
                )
                continue
            json_lines += 1
            event_type = str(event.get("type") or "unknown")
            event_counts[event_type] += 1
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type") or "unknown")
                item_counts[item_type] += 1
                _extract_item_text(
                    buckets, item, event_type=event_type, line_number=line_number
                )
                continue
            if event_type.endswith("error") or event.get("error"):
                _add_bucket(
                    buckets,
                    "error",
                    _stringify(event),
                    event_type=event_type,
                    line_number=line_number,
                )
            else:
                _add_bucket(
                    buckets,
                    "tool_or_event_metadata",
                    _stringify(event),
                    event_type=event_type,
                    line_number=line_number,
                )

    totals = {
        "chars": sum(bucket["chars"] for bucket in buckets.values()),
        "estimated_tokens": sum(
            bucket["estimated_tokens"] for bucket in buckets.values()
        ),
    }
    for bucket in buckets.values():
        denom = totals["estimated_tokens"] or 1
        bucket["token_ratio"] = bucket["estimated_tokens"] / denom
    return {
        "log_path": str(log_path),
        "exists": True,
        "token_counter": TOKEN_COUNTER,
        "total_lines": total_lines,
        "json_lines": json_lines,
        "item_counts": dict(sorted(item_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "buckets": dict(sorted(buckets.items())),
        "totals": totals,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    analysis = payload["analysis"]
    lines = [
        f"# Context Audit - {Path(payload['run_dir']).name}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- trigger: `{payload.get('trigger') or 'manual'}`",
        f"- log: `{analysis['log_path']}`",
        f"- token_counter: `{analysis.get('token_counter')}`",
        f"- estimated total tokens: `{analysis['totals']['estimated_tokens']}`",
        "",
        "| category | records | chars | est tokens | ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, bucket in sorted(
        analysis["buckets"].items(),
        key=lambda item: item[1]["estimated_tokens"],
        reverse=True,
    ):
        lines.append(
            f"| {category} | {bucket['records']} | {bucket['chars']} | "
            f"{bucket['estimated_tokens']} | {bucket['token_ratio']:.2%} |"
        )
    lines.extend(["", "## Examples"])
    for category, bucket in sorted(analysis["buckets"].items()):
        lines.append(f"### {category}")
        for example in bucket.get("examples", []):
            lines.append(
                f"- line {example.get('line')}: `{example.get('snippet', '')}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_context_audit(
    run_dir: Path,
    *,
    trigger: str = "manual",
    related_manifest: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    generated = utc_now()
    stamp = generated.replace(":", "").replace("-", "").replace("+", "Z").split(".")[0]
    out_dir = run_dir / "context_audits"
    json_path = out_dir / f"context_audit_{stamp}.json"
    md_path = out_dir / f"context_audit_{stamp}.md"
    analysis = analyze_log(run_dir / "logs" / "agent.log")
    payload = {
        "generated_at": generated,
        "trigger": trigger,
        "related_manifest": related_manifest,
        "run_dir": str(run_dir),
        "analysis": analysis,
    }
    write_json(json_path, payload)
    _write_markdown(md_path, payload)
    latest_json = out_dir / "context_audit_latest.json"
    latest_md = out_dir / "context_audit_latest.md"
    shutil.copy2(json_path, latest_json)
    shutil.copy2(md_path, latest_md)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    payload["latest_json_path"] = str(latest_json)
    payload["latest_markdown_path"] = str(latest_md)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Codex agent log content proportions for an AutoSFT run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--related-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_context_audit(
        args.run_dir,
        trigger=args.trigger,
        related_manifest=args.related_manifest,
    )
    print(
        json.dumps(
            {
                "json_path": payload["json_path"],
                "markdown_path": payload["markdown_path"],
                "estimated_tokens": payload["analysis"]["totals"]["estimated_tokens"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
