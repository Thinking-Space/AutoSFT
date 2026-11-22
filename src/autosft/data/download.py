"""Download the paper's four public data sources and prepare inspectable pools."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from autosft.core.common import write_json

DATASETS = {
    "ifeval_like": "argilla/ifeval-like-data",
    "opc": "OpenCoder-LLM/opc-sft-stage2",
    "kodcode": "KodCode/KodCode-V1-SFT-R1",
    "magicoder": "ise-uiuc/Magicoder-OSS-Instruct-75K",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(row: dict) -> dict:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    else:
        prompt = next(
            (
                row[key]
                for key in ("prompt", "instruction", "question", "problem")
                if isinstance(row.get(key), str)
            ),
            None,
        )
        answer = next(
            (
                row[key]
                for key in ("response", "output", "answer", "solution", "completion")
                if isinstance(row.get(key), str)
            ),
            None,
        )
        if prompt is None or not answer:
            raise ValueError("unrecognized instruction/response fields")
        if isinstance(row.get("input"), str) and row["input"].strip():
            prompt += "\n\n" + row["input"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    from autosft.data.preparation import _invalid_row_reason

    result = {"messages": messages}
    error = _invalid_row_reason(result)
    if error:
        raise ValueError(error)
    return result


def is_python(row: dict, converted: dict) -> bool:
    tag = row.get("language", row.get("lang"))
    if isinstance(tag, str):
        return tag.casefold() in {"python", "python3", "py"}
    answer = "\n".join(
        turn["content"] for turn in converted["messages"] if turn["role"] == "assistant"
    )
    blocks = re.findall(
        r"```(?:python|py)\s*\n(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        if "```" in answer:
            return False
        blocks = [answer]
    try:
        return all(bool(ast.parse(code).body) for code in blocks)
    except SyntaxError:
        return False


def prepare(
    source: str,
    output: Path,
    *,
    revision: str = "main",
    python_only: bool = False,
    deduplicate: bool = True,
) -> dict:
    destination = output / f"{source}.jsonl"
    manifest_path = output / f"{source}.manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise ValueError(f"existing pool has no manifest: {destination}")
        report = json.loads(manifest_path.read_text())
        if not isinstance(report, dict) or report.get("sha256") != _file_sha256(
            destination
        ):
            raise ValueError(
                f"existing pool content or manifest changed: {destination}"
            )
        if (
            report.get("python_only") != python_only
            or report.get("deduplicate") != deduplicate
        ):
            raise ValueError("existing pool uses different filters")
        if revision != "main" and revision != report.get("revision"):
            raise ValueError(
                "existing pool uses a different revision; choose a new output directory"
            )
        return report
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi

    repo = DATASETS[source]
    info = HfApi().dataset_info(repo, revision=revision)
    resolved = info.sha
    configurations = get_dataset_config_names(repo, revision=resolved)
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "downloaded": 0,
        "invalid": 0,
        "non_python": 0,
        "duplicates": 0,
        "accepted": 0,
    }
    seen = set()
    splits = []
    temporary = destination.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for config in configurations:
                dataset = load_dataset(repo, config, revision=resolved)
                if "train" not in dataset:
                    raise ValueError(f"{repo}/{config} has no train split")
                splits.append(
                    {"config": config, "split": "train", "rows": len(dataset["train"])}
                )
                for index, row in enumerate(dataset["train"]):
                    counts["downloaded"] += 1
                    try:
                        converted = normalize(row)
                    except ValueError:
                        counts["invalid"] += 1
                        continue
                    if python_only and not is_python(row, converted):
                        counts["non_python"] += 1
                        continue
                    fingerprint = hashlib.sha256(
                        json.dumps(converted["messages"], sort_keys=True).encode()
                    ).hexdigest()
                    if deduplicate and fingerprint in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(fingerprint)
                    converted["pool_row_id"] = (
                        list(DATASETS).index(source) * 10**12 + counts["accepted"]
                    )
                    converted["source"] = {
                        "dataset": repo,
                        "revision": resolved,
                        "config": config,
                        "split": "train",
                        "row": index,
                    }
                    supervision = {
                        "messages",
                        "prompt",
                        "instruction",
                        "question",
                        "problem",
                        "response",
                        "output",
                        "answer",
                        "solution",
                        "completion",
                        "input",
                    }
                    converted["metadata"] = {
                        key: value
                        for key, value in row.items()
                        if key not in supervision
                    }
                    stream.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    counts["accepted"] += 1
        if not counts["accepted"]:
            raise ValueError("no valid training rows were found")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    card = info.card_data.to_dict() if info.card_data else {}
    report = {
        "dataset": repo,
        "revision": resolved,
        "license": card.get("license"),
        "splits": splits,
        "python_only": python_only,
        "deduplicate": deduplicate,
        "counts": counts,
        "sha256": _file_sha256(destination),
    }
    write_json(manifest_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", choices=["instruction_following", "code"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        help="SOURCE=REVISION; repeat to fix individual source revisions",
    )
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args(argv)
    sources = (
        ["ifeval_like"]
        if args.pool == "instruction_following"
        else ["opc", "kodcode", "magicoder"]
    )
    revisions = {}
    for binding in args.revision:
        source, separator, revision = binding.partition("=")
        if (
            not separator
            or source not in sources
            or not revision
            or source in revisions
        ):
            parser.error(
                "revision bindings must be unique SOURCE=REVISION entries for this pool"
            )
        revisions[source] = revision
    for source in sources:
        print(
            json.dumps(
                prepare(
                    source,
                    args.output,
                    revision=revisions.get(source, "main"),
                    python_only=args.pool == "code",
                    deduplicate=not args.no_dedup,
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
