"""Independent sampled quality review and decontamination before publication."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from autosft.core.common import REPO_ROOT, write_json
from autosft.data.decontamination import benchmark_paths, check_rows


def sample_indices(rows: int, size: int = 32, seed: int = 0) -> list[int]:
    if rows < 1 or size < 1:
        raise ValueError("quality inspection needs positive row and sample counts")
    return sorted(random.Random(seed).sample(range(rows), min(rows, size)))


def validate_review(review: dict, *, data_sha256: str, indices: list[int]) -> None:
    if (
        review.get("data_sha256") != data_sha256
        or review.get("sample_indices") != indices
    ):
        raise ValueError(
            "quality inspection belongs to different data or a different sample"
        )
    verdicts = review.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(indices):
        raise ValueError("quality inspection must cover every sampled row exactly once")
    if [v.get("row_index") for v in verdicts] != indices:
        raise ValueError("quality verdicts are not aligned to the sample")
    if any(
        v.get("passed") is not True
        or not isinstance(v.get("reason"), str)
        or not v["reason"].strip()
        for v in verdicts
    ):
        raise ValueError("quality inspection contains rejected or unknown rows")


def inspect_file(
    data: Path, run_dir: Path, round_id: str, *, size: int = 32, n: int = 13
) -> dict:
    content = data.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    rows = [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    references = benchmark_paths(
        [
            item
            for item in os.environ.get("AUTO_SFT_BENCHMARKS", "ifeval,ifbench").split(
                ","
            )
            if item
        ]
    )
    decontamination = check_rows(rows, references, n=n)
    report_dir = run_dir / "synthesis_inspections" / f"round_{round_id}" / digest
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        report_dir / "decontamination.json", {"data_sha256": digest, **decontamination}
    )
    if not decontamination["passed"]:
        raise ValueError(
            "synthetic data overlap benchmark texts; see decontamination.json"
        )
    indices = sample_indices(len(rows), size)
    requests = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Independently inspect a synthetic SFT example for semantic correctness, coherence, useful supervision and instruction/response agreement. Return only JSON with passed (boolean) and reason (non-empty string). Reject incorrect, incoherent or unsupported answers.",
                },
                {
                    "role": "user",
                    "content": json.dumps(rows[i]["messages"], ensure_ascii=False),
                },
            ],
            "ref": str(k + 1),
        }
        for k, i in enumerate(indices)
    ]
    requests_path = report_dir / "requests.jsonl"
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests)
    if requests_path.exists() and requests_path.read_text() != rendered:
        raise ValueError("quality request inputs changed")
    requests_path.write_text(rendered)
    label = "quality_" + digest[:20]
    runner = REPO_ROOT / "src/autosft/backends/providers/batch.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(requests_path),
            "--run-dir",
            str(run_dir),
            "--round-id",
            str(round_id),
            "--label",
            label,
            "--rows",
            f"1-{len(indices)}",
        ],
        check=True,
    )
    aligned = run_dir / "llm_calls" / f"round_{round_id}" / f"{label}.aligned.jsonl"
    responses = [
        json.loads(line) for line in aligned.read_text().splitlines() if line.strip()
    ]
    if len(responses) != len(indices):
        raise ValueError("incomplete quality inspection results")
    verdicts = []
    for k, (index, response) in enumerate(zip(indices, responses)):
        if response.get("ref") != str(k + 1):
            raise ValueError("quality results refer to different input rows")
        verdict = json.loads(response["answer"])
        verdicts.append(
            {
                "row_index": index,
                "passed": verdict.get("passed"),
                "reason": verdict.get("reason"),
            }
        )
    review = {
        "data_sha256": digest,
        "sample_indices": indices,
        "sample_seed": 0,
        "verdicts": verdicts,
        "review_responses": str(aligned),
        "review_responses_sha256": hashlib.sha256(aligned.read_bytes()).hexdigest(),
    }
    write_json(report_dir / "quality.json", review)
    validate_review(review, data_sha256=digest, indices=indices)
    if hashlib.sha256(data.read_bytes()).hexdigest() != digest:
        raise ValueError("delivery changed during inspection")
    return {
        "inspection_dir": str(report_dir),
        "data_sha256": digest,
        "decontamination": decontamination,
        "quality": review,
    }


def inspect_delivery(delivery, run_dir: Path, round_id: str):
    inspect_file(
        delivery.data_path,
        run_dir,
        round_id,
        size=int(os.environ.get("AUTO_SFT_QUALITY_SAMPLE_SIZE", "32")),
        n=int(os.environ.get("AUTO_SFT_DECONTAMINATION_N", "13")),
    )
    return delivery


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--n", type=int, default=13)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            inspect_file(
                args.data, args.run_dir, args.round_id, size=args.sample_size, n=args.n
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
