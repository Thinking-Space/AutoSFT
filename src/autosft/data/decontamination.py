"""Conservative word n-gram decontamination against local benchmark texts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from autosft.core.common import REPO_ROOT, write_json


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def row_texts(row: dict) -> list[str]:
    if "messages" in row:
        return [
            turn["content"]
            for turn in row["messages"]
            if isinstance(turn.get("content"), str)
        ]
    return [
        row[key]
        for key in ("prompt", "text", "canonical_solution", "test", "test_list", "code")
        if isinstance(row.get(key), str)
    ]


def benchmark_paths(names: list[str]) -> list[Path]:
    files = {
        "ifeval": "ifeval/input_data.jsonl",
        "ifbench": "ifbench/input_data.jsonl",
        "humaneval_plus": "evalplus/humaneval_plus.jsonl",
        "mbpp_plus": "evalplus/mbpp_plus.jsonl",
    }
    result = []
    for name in names:
        if name not in files:
            raise ValueError(f"no decontamination source for benchmark {name!r}")
        path = REPO_ROOT / "assets/benchmarks" / files[name]
        if not path.is_file():
            # EvalPlus files are selected from their content manifest.
            from autosft.benchmarks.evalplus_data import load_staged_rows

            rows = load_staged_rows(REPO_ROOT / "assets/benchmarks/evalplus", name)
            candidates = list(
                (REPO_ROOT / "assets/benchmarks/evalplus").glob("*.jsonl")
            )
            expected = "HumanEval" if name == "humaneval_plus" else "Mbpp"
            matched = [
                p for p in candidates if expected.casefold() in p.name.casefold()
            ]
            if len(matched) != 1 or not rows:
                raise ValueError(f"missing benchmark reference: {name}")
            path = matched[0]
        result.append(path)
    return result


def build_index(paths: list[Path], n: int) -> tuple[set[tuple[str, ...]], list[dict]]:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    index: set[tuple[str, ...]] = set()
    sources = []
    for path in paths:
        content = path.read_bytes()
        rows = 0
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            texts = row_texts(row)
            if not texts:
                raise ValueError(
                    f"benchmark row has no recognized text: {path}:{rows + 1}"
                )
            for text in texts:
                index.update(ngrams(text, n))
            rows += 1
        if not rows:
            raise ValueError(f"benchmark reference is empty: {path}")
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": rows,
            }
        )
    if not paths:
        raise ValueError("at least one benchmark reference is required")
    return index, sources


def check_rows(rows: list[dict], references: list[Path], *, n: int = 13) -> dict:
    index, sources = build_index(references, n)
    rejected = [
        i
        for i, row in enumerate(rows)
        if any(ngrams(text, n) & index for text in row_texts(row))
    ]
    return {
        "method": "nfkc_casefold_word_ngram",
        "n": n,
        "references": sources,
        "rows": len(rows),
        "rejected_row_indices": rejected,
        "passed": not rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--benchmarks", nargs="+", default=[])
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="Write rows without detected overlap"
    )
    args = parser.parse_args(argv)
    if args.output and args.output.resolve() == args.data.resolve():
        parser.error("filtered output must not replace the input")
    rows = [
        json.loads(line) for line in args.data.read_text().splitlines() if line.strip()
    ]
    report = check_rows(
        rows, [*args.reference, *benchmark_paths(args.benchmarks)], n=args.n
    )
    report["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    write_json(args.report, report)
    if args.output:
        excluded = set(report["rejected_row_indices"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for i, row in enumerate(rows)
                if i not in excluded
            )
        )
    return 0 if report["passed"] or args.output else 1


if __name__ == "__main__":
    raise SystemExit(main())
