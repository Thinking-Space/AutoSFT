"""Pinned public EvalPlus datasets and offline manifest validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from autosft.benchmarks.evalplus import EVALPLUS_PROBLEM_IDS


EVALPLUS_VERSION = "0.3.1"
GENERATION_INPUT_REVISION = "opencompass-humaneval-v2-20210705-mbpp-plus-v0.2.0"
DATASET_SPECS = {
    "humaneval_plus": {
        "file": "humaneval_plus.jsonl",
        "version": "v0.1.10",
        "count": 164,
        "url": "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    },
    "mbpp_plus": {
        "file": "mbpp_plus.jsonl",
        "version": "v0.2.0",
        "count": 378,
        "url": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
    },
}


def generation_task_prompt(row: dict[str, Any]) -> str:
    """Use the original generation input while preserving release test data.

    The OpenCompass HumanEval input and EvalPlus v0.1.10 differ only on
    HumanEval/115: the former imports math inside max_fill. Keep the pinned
    EvalPlus release unchanged for reference execution and data checksums.
    """
    prompt = str(row["prompt"])
    if row.get("task_id") == "HumanEval/115":
        prefix = "import math\n\ndef max_fill(grid, capacity):\n"
        if prompt.startswith(prefix):
            return (
                "\ndef max_fill(grid, capacity):\n    import math\n"
                + prompt[len(prefix) :]
            )
    return prompt


def dataset_root() -> Path:
    configured = os.environ.get("AUTO_SFT_EVALPLUS_DATA_DIR")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3] / "assets/benchmarks/evalplus"
    )


def validate_dataset_bytes(data: bytes, benchmark: str) -> list[dict[str, Any]]:
    """Validate the complete raw release before deserializing MBPP input types."""
    spec = DATASET_SPECS[benchmark]
    rows = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    if len(rows) != spec["count"] or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{benchmark} requires {spec['count']} problem objects")
    ids = [row.get("task_id") for row in rows]
    if (
        not all(isinstance(value, str) for value in ids)
        or set(ids) != EVALPLUS_PROBLEM_IDS[benchmark]
    ):
        raise ValueError(
            f"{benchmark} problem identities do not match its public release"
        )
    for row in rows:
        for key in ("prompt", "canonical_solution", "entry_point", "contract"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{row['task_id']} requires string {key}")
        base_input = row.get("base_input")
        if not isinstance(base_input, list) or not base_input:
            raise ValueError(f"{row['task_id']} requires nonempty base_input")
        # The public MBPP+ v0.2.0 release uses ``{}`` for Mbpp/793 to denote
        # that no additional plus cases were generated. EvalPlus treats that
        # empty mapping as an empty test list; preserve the release verbatim.
        plus_input = row.get("plus_input")
        if benchmark == "mbpp_plus":
            if not (
                (isinstance(plus_input, list) and bool(plus_input)) or plus_input == {}
            ):
                raise ValueError(
                    f"{row['task_id']} requires nonempty plus_input or {{}}"
                )
        elif not isinstance(plus_input, list) or not plus_input:
            raise ValueError(f"{row['task_id']} requires nonempty plus_input")
        atol = row.get("atol")
        if (
            isinstance(atol, bool)
            or not isinstance(atol, (int, float))
            or not math.isfinite(atol)
            or atol < 0
        ):
            raise ValueError(f"{row['task_id']} requires finite nonnegative atol")
    return rows


def load_staged_rows(data_dir: str | Path, benchmark: str) -> list[dict[str, Any]]:
    """Check the manifest and content without importing the evaluation runtime."""
    spec = DATASET_SPECS[benchmark]
    root = Path(data_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported EvalPlus data manifest schema")
    if manifest.get("evaluator") != f"evalplus=={EVALPLUS_VERSION}":
        raise ValueError("EvalPlus data manifest uses a different evaluator version")
    entries = manifest.get("datasets")
    entry = entries.get(benchmark) if isinstance(entries, dict) else None
    if not isinstance(entry, dict) or any(
        entry.get(key) != spec[key] for key in ("file", "version", "count")
    ):
        raise ValueError(f"manifest does not describe the pinned {benchmark} release")
    data = (root / spec["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise ValueError(f"{benchmark} dataset SHA-256 does not match its manifest")
    return validate_dataset_bytes(data, benchmark)
