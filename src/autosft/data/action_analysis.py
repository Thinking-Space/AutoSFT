"""Retrospective data-action labels; not imported by the research policy."""

from __future__ import annotations
import argparse
from enum import Enum
from pathlib import Path
from autosft.core.common import append_jsonl, utc_now


class DataAction(str, Enum):
    MIXTURE = "mixture"
    PRUNING = "pruning"
    REPAIR = "repair"
    RESUPERVISION = "resupervision"
    SYNTHESIS = "synthesis"
    ORDERING = "ordering"


def annotate(
    run_dir: Path, round_id: int, labels: list[str], evidence: list[Path], note: str
) -> dict:
    if (
        type(round_id) is not int
        or round_id < 1
        or not labels
        or not evidence
        or not note.strip()
    ):
        raise ValueError(
            "annotation requires a positive round, labels, evidence and an explanation"
        )
    values = sorted({DataAction(label).value for label in labels})
    from hashlib import sha256

    record = {
        "round_id": round_id,
        "labels": values,
        "note": note,
        "evidence": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in evidence
        ],
        "annotated_at": utc_now(),
    }
    append_jsonl(run_dir / "reports/action_annotations.jsonl", record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=[item.value for item in DataAction],
        required=True,
    )
    parser.add_argument("--evidence", nargs="+", type=Path, required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    annotate(args.run_dir, args.round_id, args.labels, args.evidence, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
