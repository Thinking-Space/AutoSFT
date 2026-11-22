#!/usr/bin/env python3
from __future__ import annotations
from autosft.config.components import executor_enabled

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from autosft.core.common import REPO_ROOT, run_command, utc_now, write_json
from autosft.agents.executor_handoff import refresh_executor_handoff
from autosft.orchestration.policy_guard import (
    check_policy_or_exit,
    policy_enabled,
    policy_summary,
    record_policy_violation_and_exit,
)
from autosft.core.round_ids import RoundIdError, storage_round_id
from autosft.orchestration.submission_registry import (
    FAILED_EVAL_STATUSES,
    eval_backends,
    eval_identity,
    eval_manifest_paths,
    eval_task_status,
    find_eval_submission,
    model_identity,
    submission_lock,
    submission_records,
    submission_task,
)


DEFAULT_BENCHMARKS = "ifeval,ifbench"
DEFAULT_EVAL_BACKEND = "opencompass"
EVAL_BACKENDS = ("opencompass",)


def eval_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EVAL_BACKENDS:
        raise argparse.ArgumentTypeError(
            f"unsupported eval backend {value!r}; choose one of {', '.join(EVAL_BACKENDS)}"
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed benchmark evaluation entrypoint for Local."
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        default=os.environ.get("AUTO_SFT_EVAL_BENCHMARK", DEFAULT_BENCHMARKS),
        help=(
            "One or more benchmarks to evaluate, separated by commas or whitespace. "
            "Each benchmark is submitted as its own Local eval job. "
            f"Defaults to '{DEFAULT_BENCHMARKS}'."
        ),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Canonical storage round name: round_<positive integer>.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=int(os.environ.get("AUTO_SFT_EVAL_LIMIT", "-1"))
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=int(os.environ.get("AUTO_SFT_EVAL_REPEATS", "1")),
        help=(
            "Number of times each benchmark is evaluated. One Local task "
            "per benchmark loads the model once and samples every prompt N "
            "times; repeat k is written to "
            "eval/<benchmark>/rep_<k>/ (single-repeat eval/<benchmark>/ when 1). "
            "Defaults to AUTO_SFT_EVAL_REPEATS or 1."
        ),
    )
    parser.add_argument(
        "--eval-backend",
        type=eval_backend,
        choices=EVAL_BACKENDS,
        default=os.environ.get("AUTO_SFT_EVAL_BACKEND", DEFAULT_EVAL_BACKEND),
        help=(
            "The 'opencompass' runner uses local vLLM generation with "
            "OpenCompass-compatible IFEval/IFBench scoring or EvalPlus for "
            "HumanEval+/MBPP+. Defaults to AUTO_SFT_EVAL_BACKEND or "
            f"'{DEFAULT_EVAL_BACKEND}'."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-failed-task",
        help="Explicitly replace one confirmed terminal failed eval task, preserving its identity and artifacts.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    configured_backend = os.environ.get("AUTO_SFT_EVAL_BACKEND")
    if configured_backend is not None:
        fixed_backend = eval_backend(configured_backend)
        if args.eval_backend != fixed_backend:
            parser.error(
                "--eval-backend cannot override the run-fixed "
                f"AUTO_SFT_EVAL_BACKEND={fixed_backend}"
            )
    return args


def parse_benchmarks(raw: str) -> list[str]:
    """Split a benchmark spec on commas/whitespace and de-duplicate, preserving order."""
    tokens = [token for chunk in raw.split(",") for token in chunk.split()]
    benchmarks: list[str] = []
    for token in tokens:
        if token and token not in benchmarks:
            benchmarks.append(token)
    return benchmarks


def resolve_eval_run_dir(
    requested: Path | None,
    *,
    run_name: str,
    guarded: bool,
) -> tuple[Path, Path]:
    """Resolve the one canonical directory for an evaluation request."""
    policy_value = os.environ.get("AUTO_SFT_RUN_DIR")
    if not policy_value:
        raise SystemExit("AUTO_SFT_RUN_DIR is required for evaluation")
    policy_root = Path(policy_value)
    # An eval run is named for the round it evaluates, exactly as the SFT side
    # is (src/autosft/backends/megatron/train.py). Without this the directory name was whatever string
    # the caller passed, so the same run could end up with both `round_6` and
    # `<run_id>_round_6_final_step_54` depending on which entrypoint submitted
    # it, and nothing on disk tied either name to a round.
    try:
        run_name = storage_round_id(run_name, label="--run-name")
    except RoundIdError as exc:
        message = f"eval run name must be the round it evaluates: {exc}"
        if guarded:
            record_policy_violation_and_exit(
                stage="eval",
                run_dir=policy_root,
                rule_id="eval_run_name_not_canonical",
                message=message,
                path=str(run_name),
            )
        raise SystemExit(message) from exc
    canonical = policy_root / "eval_runs" / run_name
    run_dir = Path(requested) if requested is not None else canonical
    if run_dir.resolve() != canonical.resolve():
        message = f"Eval output must equal {canonical}; got {run_dir}"
        if guarded:
            record_policy_violation_and_exit(
                stage="eval",
                run_dir=policy_root,
                rule_id="eval_run_dir_mismatch",
                message=message,
                path=str(run_dir),
            )
        raise SystemExit(message)
    return run_dir, policy_root


def main() -> None:
    args = parse_args()
    benchmarks = parse_benchmarks(args.benchmark)
    if not benchmarks:
        raise SystemExit(
            "No benchmark specified; pass --benchmark or set AUTO_SFT_EVAL_BENCHMARK "
            f"(default '{DEFAULT_BENCHMARKS}')."
        )
    if args.retry_failed_task and len(benchmarks) != 1:
        raise SystemExit("--retry-failed-task requires exactly one benchmark")
    guarded = policy_enabled()
    run_dir, policy_run_dir = resolve_eval_run_dir(
        args.run_dir,
        run_name=args.run_name,
        guarded=guarded,
    )
    # resolve_eval_run_dir returned, so --run-name is the canonical round_<n>.
    run_name = args.run_name
    if args.retry_failed_task:
        retry_key = hashlib.sha256(args.retry_failed_task.encode("utf-8")).hexdigest()[
            :16
        ]
        run_dir = run_dir.with_name(f"{run_name}_retry_{retry_key}")
    run_dir.mkdir(parents=True, exist_ok=True)
    check_policy_or_exit(stage="eval", run_dir=policy_run_dir)

    eval_manifest = str(run_dir / "eval_submissions.jsonl")
    existing_backends = eval_backends(policy_run_dir, Path(eval_manifest))
    if existing_backends and existing_backends != {args.eval_backend}:
        message = (
            "Evaluation backend is fixed for the entire run: existing submissions "
            f"use {sorted(existing_backends)}, requested {args.eval_backend!r}."
        )
        if guarded:
            record_policy_violation_and_exit(
                stage="eval",
                run_dir=policy_run_dir,
                rule_id="eval_backend_mismatch",
                message=message,
                path=str(run_dir),
            )
        raise SystemExit(message)
    requested_model_identity = model_identity(args.model_path)
    submissions: list[dict] = []
    failures: list[tuple[str, int]] = []
    actual_manifests: set[str] = set()
    repeats = args.repeats
    for benchmark in benchmarks:
        bench_run_name = f"{run_name}_{benchmark}" if len(benchmarks) > 1 else run_name
        # One task owns the whole benchmark directory; the task itself writes
        # rep_<k>/ below it for every repeat (submit_local_eval.sh).
        bench_output_dir = run_dir / "eval" / benchmark
        env = {
            "AUTO_SFT_EVAL_MANIFEST": eval_manifest,
            "AUTO_SFT_EVAL_OUTPUT_DIR": str(bench_output_dir),
            "AUTO_SFT_EVAL_LIMIT": str(args.limit),
            "AUTO_SFT_EVAL_BACKEND": args.eval_backend,
            "AUTO_SFT_EVAL_MODEL_IDENTITY": requested_model_identity,
            "AUTO_SFT_EVAL_REPEATS": str(repeats),
            "AUTO_SFT_LOCAL_DRY_RUN": "1"
            if args.dry_run
            else os.environ.get("AUTO_SFT_LOCAL_DRY_RUN", "0"),
            "AUTO_SFT_POLICY_FEEDBACK": str(policy_run_dir / "policy_feedback.json"),
        }
        existing_submission = None
        existing_manifest = None
        completed = None
        try:
            with submission_lock(policy_run_dir):
                found = find_eval_submission(
                    policy_run_dir,
                    Path(eval_manifest),
                    model_path=args.model_path,
                    benchmark=benchmark,
                    eval_backend=args.eval_backend,
                    limit=args.limit,
                    repeats=repeats,
                )
                if args.retry_failed_task:
                    found = validate_failed_retry(
                        policy_run_dir,
                        args.retry_failed_task,
                        found,
                        run_name=run_name,
                        requested_identity=eval_identity(
                            model_path=args.model_path,
                            benchmark=benchmark,
                            eval_backend=args.eval_backend,
                            limit=args.limit,
                            repeats=repeats,
                        ),
                    )
                if found is not None:
                    existing_submission, existing_manifest = found
                else:
                    if args.retry_failed_task:
                        # Persist lineage before create so a replay after submission
                        # can recover its replacement from the canonical manifest.
                        write_json(
                            run_dir / "eval_request.json",
                            {
                                "run_name": run_name,
                                "retry_of_task": args.retry_failed_task,
                                "model_identity": requested_model_identity,
                                "benchmarks": benchmarks,
                                "repeats": repeats,
                            },
                        )
                    completed = run_command(
                        [
                            sys.executable,
                            "-m",
                            "autosft.backends.local.submission",
                            "eval",
                            bench_run_name,
                            str(args.model_path),
                            benchmark,
                        ],
                        env=env,
                        cwd=REPO_ROOT,
                    )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                sys.stdout.write(exc.stdout)
                if not exc.stdout.endswith("\n"):
                    sys.stdout.write("\n")
            if exc.stderr:
                sys.stderr.write(exc.stderr)
                if not exc.stderr.endswith("\n"):
                    sys.stderr.write("\n")
            sys.stderr.write(
                f"ERROR: local evaluation submission exited with code {exc.returncode} for benchmark "
                f"'{benchmark}'. Inspect the manifest and local_tasks receipts before retrying.\n"
            )
            failures.append((benchmark, exc.returncode))
            submissions.append(
                {
                    "benchmark": benchmark,
                    "repeats": repeats,
                    "run_name": bench_run_name,
                    "eval_output_dir": str(bench_output_dir),
                    "eval_backend": args.eval_backend,
                    "status": "submit_failed",
                    "returncode": exc.returncode,
                }
            )
            continue
        if existing_submission is not None and existing_manifest is not None:
            actual_manifests.add(str(existing_manifest))
            existing_task = submission_task(existing_submission)
            print(
                json.dumps(
                    {
                        "status": "existing_task_found",
                        "benchmark": benchmark,
                        "repeats": repeats,
                        "task": existing_task,
                        "submission_manifest": str(existing_manifest),
                    },
                    ensure_ascii=False,
                )
            )
            submissions.append(
                {
                    "benchmark": benchmark,
                    "repeats": repeats,
                    "run_name": existing_submission.get("run_name"),
                    "requested_run_name": bench_run_name,
                    "task": existing_task,
                    "model_identity": requested_model_identity,
                    "eval_output_dir": existing_submission.get("output_dir"),
                    "eval_backend": args.eval_backend,
                    "submission_manifest": str(existing_manifest),
                    "status": "existing_task_found",
                }
            )
            continue

        if completed is not None:
            print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="")
        created = find_eval_submission(
            policy_run_dir,
            Path(eval_manifest),
            model_path=args.model_path,
            benchmark=benchmark,
            eval_backend=args.eval_backend,
            limit=args.limit,
            repeats=repeats,
        )
        created_record, created_manifest = (
            created if created is not None else ({}, Path(eval_manifest))
        )
        actual_manifests.add(str(created_manifest))
        submissions.append(
            {
                "benchmark": benchmark,
                "repeats": repeats,
                "run_name": bench_run_name,
                "task": submission_task(created_record),
                "model_identity": requested_model_identity,
                "eval_output_dir": str(bench_output_dir),
                "eval_backend": args.eval_backend,
                "submission_manifest": str(created_manifest),
                "status": "submitted",
            }
        )

    manifest = {
        "run_name": run_name,
        "created_at": utc_now(),
        "model_path": str(args.model_path),
        "model_identity": requested_model_identity,
        "benchmarks": benchmarks,
        "benchmark": args.benchmark,
        "eval_backend": args.eval_backend,
        "repeats": repeats,
        "run_dir": str(run_dir),
        "eval_manifest": (
            next(iter(actual_manifests))
            if len(actual_manifests) == 1
            else eval_manifest
        ),
        "eval_manifests": sorted(actual_manifests) or [eval_manifest],
        "submissions": submissions,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "policy": policy_summary(policy_run_dir),
    }
    if args.retry_failed_task:
        manifest["retry_of_task"] = args.retry_failed_task
    write_json(run_dir / "eval_request.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if executor_enabled():
        try:
            refresh_executor_handoff(
                policy_run_dir,
                benchmarks=benchmarks,
            )
        except Exception as exc:
            sys.stderr.write(f"WARN: failed to refresh executor handoff: {exc}\n")

    if failures:
        sys.stderr.write(
            "ERROR: "
            + ", ".join(f"{name} (code {code})" for name, code in failures)
            + f" failed to submit out of {len(benchmarks)} benchmark(s); "
            + "the submissions that succeeded are recorded in eval_submissions.jsonl.\n"
        )
        raise SystemExit(failures[0][1])


def validate_failed_retry(
    run_root: Path,
    failed_task: str,
    current: tuple[dict, Path] | None,
    *,
    run_name: str,
    requested_identity: tuple[str, str, str, str, str],
) -> tuple[dict, Path] | None:
    """Return an existing replacement, or authorize one terminal-failure retry."""
    known = {
        submission_task(record): (record, manifest)
        for manifest in eval_manifest_paths(run_root)
        for record in submission_records(manifest)
    }
    source = known.get(failed_task)
    if source is None:
        raise SystemExit(
            f"retry task is not in the durable eval manifests: {failed_task}"
        )
    record, source_manifest = source
    identity = eval_identity(
        model_path=record.get("model_path"),
        model_identity_value=record.get("model_identity"),
        benchmark=record.get("benchmark"),
        eval_backend=record.get("eval_backend"),
        limit=record.get("limit"),
        repeats=record.get("repeats", 1),
    )
    if identity != requested_identity:
        raise SystemExit(
            "retry must preserve the failed task's model, benchmark, backend, limit, and repeats"
        )
    request_path = source_manifest.parent / "eval_request.json"
    request = (
        json.loads(request_path.read_text(encoding="utf-8"))
        if request_path.is_file()
        else {}
    )
    source_round = request.get("run_name", source_manifest.parent.name)
    if source_round != run_name:
        raise SystemExit(f"retry must preserve the original round {source_round!r}")
    if current is None:
        raise SystemExit("retry task is not a current evaluation attempt")
    cursor = current
    seen: set[str] = set()
    while submission_task(cursor[0]) != failed_task:
        cursor_task = submission_task(cursor[0])
        if cursor_task in seen:
            raise SystemExit("eval retry lineage contains a cycle")
        seen.add(cursor_task)
        cursor_request = cursor[1].parent / "eval_request.json"
        payload = (
            json.loads(cursor_request.read_text(encoding="utf-8"))
            if cursor_request.is_file()
            else {}
        )
        cursor = known.get(payload.get("retry_of_task"))
        if cursor is None:
            raise SystemExit(
                "retry task is not the current attempt or its recorded ancestor"
            )
    if submission_task(current[0]) != failed_task:
        return current
    status = eval_task_status(record, source_manifest)
    if status not in FAILED_EVAL_STATUSES:
        raise SystemExit(
            f"retry requires a confirmed terminal failed task; {failed_task} has status {status or 'unknown'}"
        )
    return None


if __name__ == "__main__":
    main()
