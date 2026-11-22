#!/usr/bin/env python3
"""Pin the implementation a run uses and detect checkout changes.

The launcher records the source tree before the first agent session. Every continuation compares the live checkout with that baseline, so changed prompts, skills or runtime implementations cannot silently become part of the same experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_FILENAME = "code_identity.json"

# What a round's behavior actually depends on. A path may be a file or a
# directory; missing entries are recorded as absent rather than skipped, so
# deleting one is itself drift.
TRACKED_PATHS = (
    "AGENTS.md",
    "src/autosft/backends/megatron/convert.py",
    "src/autosft/benchmarks/submit.py",
    "src/autosft/backends/megatron/train.py",
    "skills",
    "src",
    "scripts/agents",
    "scripts/local",
    "src/autosft/backends/megatron/checkpoint",
    "assets/benchmarks",
    "third_party/megatron",
    "third_party/opencompass_local",
    "third_party/ifbench",
    "third_party/instruction_following_eval",
)

_EXCLUDED_DIRS = {"__pycache__", ".git", ".pytest_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(base.rglob("*")):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix in _EXCLUDED_SUFFIXES:
            continue
        if path.is_file():
            found.append(path)
    return found


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Commit and dirty-file list, when this checkout is a git work tree."""

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty_paths": []}
    status = run("status", "--porcelain") or ""
    dirty = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    return {"commit": commit, "dirty_paths": dirty}


def compute_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Digest the tracked tree plus the checkout's git state."""
    root = Path(repo_root or REPO_ROOT).resolve()
    digest = hashlib.sha256()
    file_count = 0
    per_path: dict[str, str] = {}
    for entry in TRACKED_PATHS:
        base = root / entry
        entry_digest = hashlib.sha256()
        for path in _iter_files(base):
            relative = path.relative_to(root).as_posix()
            entry_digest.update(relative.encode("utf-8"))
            entry_digest.update(b"\0")
            try:
                entry_digest.update(path.read_bytes())
            except OSError:
                # Unreadable is a state worth pinning too: it must not hash
                # the same as the file's contents.
                entry_digest.update(b"<unreadable>")
            entry_digest.update(b"\0")
            file_count += 1
        per_path[entry] = entry_digest.hexdigest()
        digest.update(entry.encode("utf-8"))
        digest.update(b"\0")
        digest.update(per_path[entry].encode("utf-8"))
        digest.update(b"\0")
    return {
        "schema_version": 1,
        "repo_root": str(root),
        "tree_sha256": digest.hexdigest(),
        "tracked_paths": {name: per_path[name] for name in sorted(per_path)},
        "file_count": file_count,
        "git": _git_state(root),
    }


def identity_path(run_dir: Path) -> Path:
    return Path(run_dir) / IDENTITY_FILENAME


def record_identity(run_dir: Path, repo_root: Path | None = None) -> Path:
    """Pin the current code version as this run's baseline.

    Idempotent: an existing baseline is never overwritten, because the
    baseline is what a later attempt is compared against. Re-recording it
    after an edit would erase the very drift this detects.
    """
    path = identity_path(run_dir)
    if path.is_file():
        return path
    payload = compute_identity(repo_root)
    payload["recorded_at"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_identity(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(identity_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def check_drift(run_dir: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Compare the live checkout with this run's pinned baseline.

    Returns a verdict with ``drifted`` plus the tracked entries that differ.
    A run without a recorded baseline reports
    ``drifted`` false with ``baseline`` null: an unknown baseline cannot be
    evidence of a change.
    """
    baseline = read_identity(run_dir)
    current = compute_identity(repo_root)
    if baseline is None:
        return {
            "drifted": False,
            "baseline": None,
            "current_tree_sha256": current["tree_sha256"],
            "changed_paths": [],
        }
    changed = sorted(
        name
        for name, digest in current["tracked_paths"].items()
        if baseline.get("tracked_paths", {}).get(name) != digest
    )
    # A path present in the baseline but gone from the current scan is a
    # change too; TRACKED_PATHS is fixed, so this only differs when the
    # constant itself changed between the two runs of this code.
    changed += sorted(
        name
        for name in baseline.get("tracked_paths", {})
        if name not in current["tracked_paths"]
    )
    return {
        "drifted": bool(changed),
        "baseline": baseline.get("tree_sha256"),
        "baseline_recorded_at": baseline.get("recorded_at"),
        "baseline_commit": (baseline.get("git") or {}).get("commit"),
        "current_tree_sha256": current["tree_sha256"],
        "current_commit": (current.get("git") or {}).get("commit"),
        "changed_paths": sorted(set(changed)),
    }


def drift_is_fatal() -> bool:
    """Whether drift should stop the run.

    Results produced before and after a mid-run code change are not
    comparable, so stopping is the default. An operator who knows an edit
    was irrelevant to the experiment can set
    ``AUTO_SFT_ALLOW_CODE_DRIFT=1`` to continue with a warning; the drift
    stays recorded either way.
    """
    return os.environ.get("AUTO_SFT_ALLOW_CODE_DRIFT", "0") != "1"


def describe_drift(verdict: dict[str, Any]) -> str:
    changed = ", ".join(verdict.get("changed_paths") or []) or "(none)"
    return (
        f"code drift: this run was pinned to tree "
        f"{str(verdict.get('baseline'))[:12]} "
        f"(commit {str(verdict.get('baseline_commit'))[:12]}, recorded "
        f"{verdict.get('baseline_recorded_at')}), but the checkout now "
        f"hashes {str(verdict.get('current_tree_sha256'))[:12]} "
        f"(commit {str(verdict.get('current_commit'))[:12]}). "
        f"Changed: {changed}. Rounds before and after this edit are not the "
        f"same experiment. Set AUTO_SFT_ALLOW_CODE_DRIFT=1 to continue "
        f"anyway; the drift stays recorded in "
        f"{IDENTITY_FILENAME}/agent_attempts.jsonl either way."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="Print the current code identity.")
    show.add_argument("--repo-root", type=Path, default=None)
    record = commands.add_parser(
        "record", help="Pin the current version as a run's baseline."
    )
    record.add_argument("--run-dir", type=Path, required=True)
    record.add_argument("--repo-root", type=Path, default=None)
    check = commands.add_parser(
        "check", help="Report drift against a run's baseline; exit 3 on drift."
    )
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "show":
        print(json.dumps(compute_identity(args.repo_root), indent=2))
        return 0
    if args.command == "record":
        print(record_identity(args.run_dir, args.repo_root))
        return 0
    verdict = check_drift(args.run_dir, args.repo_root)
    print(json.dumps(verdict, indent=2))
    return 3 if verdict["drifted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
