#!/usr/bin/env python3
"""Parse Megatron TensorBoard events into agent-friendly loss artifacts.

Produces two files for the coding agent to consume after each SFT run:
- loss_summary.json: per-tag aggregates (first/last/min/max/mean/slope/nan flags).
- loss_curve.csv:    one row per training step, one column per scalar tag.

Implementation note: we read raw tfrecord events with the lightweight
``tensorboard.compat.proto`` package and do not touch ``EventAccumulator``.
That keeps the parser working on devspace envs where the heavier tensorboard
pipeline is broken (e.g. NumPy 2 + older tensorboard). The script is
read-only with respect to TensorBoard data; outputs are overwritten in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard events into loss_summary.json + loss_curve.csv."
    )
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--curve-out", type=Path, required=True)
    parser.add_argument(
        "--loss-pattern",
        default="loss",
        help="Case-insensitive substring used to flag loss-style tags in the summary.",
    )
    parser.add_argument(
        "--slope-pct",
        type=float,
        default=0.2,
        help="Fraction of the tail used to estimate the late-stage slope (default 0.2).",
    )
    return parser.parse_args()


def _import_event_proto():
    try:
        from tensorboard.compat.proto.event_pb2 import Event  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to parse events. Install it in the control-plane env: "
            "`pip install tensorboard`."
        ) from exc
    return Event


def _iter_tfrecords(path: Path) -> Iterable[bytes]:
    """Yield raw record payloads from a tfrecord file.

    The format is: <uint64 length><uint32 length_crc><bytes data><uint32 data_crc>.
    We do not verify CRCs; partial/truncated records are skipped silently so
    we can also parse logs from runs that are still in progress.
    """
    with path.open("rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                return
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            if len(data) < length:
                return
            if len(f.read(4)) < 4:
                return
            yield data


def _scalar_from_value(value) -> float | None:
    """Extract a Python float from a Summary.Value (simple or tensor scalar)."""
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        tensor = value.tensor
        if tensor.tensor_shape.dim:
            return None
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.int_val:
            return float(tensor.int_val[0])
        if tensor.tensor_content:
            dtype = tensor.dtype
            try:
                if dtype == 1 and len(tensor.tensor_content) >= 4:
                    return float(struct.unpack("<f", tensor.tensor_content[:4])[0])
                if dtype == 2 and len(tensor.tensor_content) >= 8:
                    return float(struct.unpack("<d", tensor.tensor_content[:8])[0])
            except struct.error:
                return None
    return None


def _collect_series(tb_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    Event = _import_event_proto()
    event_files = sorted(p for p in tb_dir.rglob("events.out.tfevents*") if p.is_file())
    series: dict[str, list[tuple[int, float, float]]] = {}
    for event_file in event_files:
        for raw in _iter_tfrecords(event_file):
            event = Event()
            try:
                event.ParseFromString(raw)
            except Exception:
                continue
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                scalar = _scalar_from_value(value)
                if scalar is None:
                    continue
                series.setdefault(value.tag, []).append(
                    (int(event.step), float(event.wall_time), float(scalar))
                )

    for tag, points in series.items():
        points.sort(key=lambda p: (p[0], p[1]))
        series[tag] = points
    return series


def _summarize_series(values: list[float]) -> dict[str, Any]:
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    finite = [v for v in values if math.isfinite(v)]
    summary: dict[str, Any] = {
        "num_points": len(values),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    if not finite:
        summary.update(
            {"first": None, "last": None, "min": None, "max": None, "mean": None}
        )
        return summary
    summary["first"] = finite[0]
    summary["last"] = finite[-1]
    summary["min"] = min(finite)
    summary["max"] = max(finite)
    summary["mean"] = sum(finite) / len(finite)
    return summary


def _tail_slope(steps: list[int], values: list[float], pct: float) -> float | None:
    """Ordinary least squares slope over the last ``pct`` fraction of points."""
    n = len(steps)
    if n < 2:
        return None
    k = max(2, int(round(n * pct)))
    xs = steps[-k:]
    ys = values[-k:]
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    if den == 0:
        return None
    return num / den


def _write_summary(
    out_path: Path,
    tb_dir: Path,
    series: dict[str, list[tuple[int, float, float]]],
    loss_pattern: str,
    slope_pct: float,
) -> dict[str, Any]:
    scalar_tags = sorted(series)
    summary: dict[str, Any] = {
        "tensorboard_dir": str(tb_dir),
        "num_scalar_tags": len(scalar_tags),
        "scalar_tags": scalar_tags,
        "loss_tags": [],
        "tags": {},
    }
    pattern = loss_pattern.lower()
    for tag in scalar_tags:
        pts = series[tag]
        steps = [p[0] for p in pts]
        values = [p[2] for p in pts]
        tag_summary = _summarize_series(values)
        if steps:
            tag_summary["first_step"] = steps[0]
            tag_summary["last_step"] = steps[-1]
        if pattern in tag.lower():
            summary["loss_tags"].append(tag)
            tag_summary["last_pct_slope"] = _tail_slope(steps, values, slope_pct)
            tag_summary["slope_pct"] = slope_pct
        summary["tags"][tag] = tag_summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_curve(
    out_path: Path,
    series: dict[str, list[tuple[int, float, float]]],
) -> int:
    scalar_tags = sorted(series)
    per_tag_index: dict[str, dict[int, float]] = {
        tag: {p[0]: p[2] for p in pts} for tag, pts in series.items()
    }
    all_steps = sorted({p[0] for pts in series.values() for p in pts})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *scalar_tags])
        for step in all_steps:
            row: list[Any] = [step]
            for tag in scalar_tags:
                value = per_tag_index[tag].get(step)
                row.append("" if value is None else format(value, ".6g"))
            writer.writerow(row)
    return len(all_steps)


def main() -> None:
    args = parse_args()
    tb_dir: Path = args.tensorboard_dir
    if not tb_dir.exists():
        raise SystemExit(f"tensorboard dir does not exist: {tb_dir}")

    series = _collect_series(tb_dir)
    if not series:
        raise SystemExit(f"no scalar events found under {tb_dir}")

    summary = _write_summary(
        args.summary_out, tb_dir, series, args.loss_pattern, args.slope_pct
    )
    num_rows = _write_curve(args.curve_out, series)

    print(f"summary: {args.summary_out}")
    print(f"curve:   {args.curve_out} ({num_rows} rows)")
    for tag in summary["loss_tags"]:
        stats = summary["tags"][tag]
        print(
            f"  {tag}: first={stats['first']} last={stats['last']} "
            f"min={stats['min']} slope={stats.get('last_pct_slope')} "
            f"nan={stats['has_nan']} inf={stats['has_inf']} "
            f"points={stats['num_points']}"
        )


if __name__ == "__main__":
    main()
