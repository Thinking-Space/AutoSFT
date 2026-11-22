#!/usr/bin/env python3
"""Fetch and verify an unmodified public Megatron-LM checkout."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PUBLIC_COMMIT = "23e00ed0963c35382dfe8a5a94fb3cda4d21e133"  # core_v0.14.0

def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", default=PUBLIC_COMMIT, help="Full public upstream commit hash"
    )
    parser.add_argument(
        "--repository", default="https://github.com/NVIDIA/Megatron-LM.git"
    )
    parser.add_argument("--output", type=Path, default=root / "third_party/megatron")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        parser.error("a full 40-character upstream commit hash is required")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        parser.error(
            "output exists; use a fresh directory to avoid replacing local source"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.fetch-", dir=output.parent)
    )
    try:
        subprocess.run(["git", "init", str(staging)], check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", args.repository],
            cwd=staging,
            check=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", args.commit],
            cwd=staging,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", "FETCH_HEAD"], cwd=staging, check=True
        )
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=staging, text=True
        ).strip()
        if actual != args.commit or not (staging / "pretrain_gpt.py").is_file():
            raise SystemExit(
                "public Megatron checkout does not satisfy the requested identity"
            )
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    (root / "third_party/megatron.lock.json").write_text(
        json.dumps(
            {
                "repository": args.repository,
                "commit": actual,
            },
            indent=2,
        )
        + "\n"
    )
    print(actual)


if __name__ == "__main__":
    main()
