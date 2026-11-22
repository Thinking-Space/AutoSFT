"""Collect local task receipts without confusing artifacts with task success."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from autosft.config.components import executor_enabled
from autosft.core.common import REPO_ROOT
from autosft.orchestration.submission_registry import (
    current_eval_submissions,
    eval_task_status,
)


def fetch_local_task_status(run: Path, task: str):
    from autosft.backends.local.tasks import status

    try:
        return status(run, task), None
    except (OSError, ValueError, RuntimeError) as exc:
        return None, str(exc)


def compact_local_task(payload):
    if not isinstance(payload, dict):
        return None
    return {
        "name": payload.get("name"),
        "status": payload.get("status"),
        "create_time": payload.get("create_time")
        or payload.get("createTime")
        or payload.get("createdAt"),
        "update_time": payload.get("update_time")
        or payload.get("updateTime")
        or payload.get("updatedAt"),
        "error": payload.get("error") or payload.get("lastError"),
    }


def refresh_handoff(run: Path) -> None:
    if executor_enabled():
        from autosft.agents.executor_handoff import refresh_executor_handoff

        try:
            refresh_executor_handoff(run)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"WARN: could not refresh executor handoff: {exc}", file=sys.stderr)


def collect_sft(run: Path, manifest: Path) -> None:
    manifest = Path(manifest)
    final_status_file = manifest.parent / "sft_final_status.json"
    policy_feedback_file = final_status_file.parent / "policy_feedback.json"

    if not manifest.exists():
        print(f"No SFT manifest found at {manifest}")
        return

    parser_script = REPO_ROOT / "src/autosft/backends/megatron/parse_loss.py"
    terminal_statuses = {
        "Succeeded",
        "Failed",
        "Stopped",
        "Killed",
        "Cancelled",
        "Canceled",
    }
    final_task_fields = (
        "task",
        "run_name",
        "output_dir",
        "submitted_at",
        "checkpoint_complete",
        "latest_step",
        "local_status",
        "local_task_summary",
        "local_task_error",
        "stopped_at",
        "operator_stop_recorded",
    )

    previous_tasks = {}
    if final_status_file.is_file():
        try:
            previous_payload = json.loads(final_status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_payload = {}
        for item in previous_payload.get("tasks") or []:
            if isinstance(item, dict) and item.get("task"):
                previous_tasks[str(item["task"])] = item

    stop_events = {}
    submission_records = []
    seen_submissions = set()

    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            event = record.get("event")
            if event == "stopped":
                task = record.get("task")
                if isinstance(task, str) and task:
                    stop_events[task] = record.get("stopped_at")
                continue
            if event:
                continue
            task = record.get("task")
            if not isinstance(task, str) or not task:
                continue
            if task in seen_submissions:
                continue
            seen_submissions.add(task)
            submission_records.append(record)

    final_tasks = []

    for record in submission_records:
        task = record.get("task", "<unknown>")
        previous = previous_tasks.get(task) or {}
        cached_status = previous.get("local_status")
        if cached_status in terminal_statuses and (
            cached_status != "Succeeded" or previous.get("checkpoint_complete") is True
        ):
            print(f"{task}\t{cached_status}\treused_terminal_result")
            reused = {
                key: previous[key] for key in final_task_fields if key in previous
            }
            reused.setdefault("local_task_summary", None)
            final_tasks.append(reused)
            continue
        output_dir = Path(record.get("output_dir", ""))
        checkpoints = output_dir / "checkpoints"
        latest = checkpoints / "latest_checkpointed_iteration.txt"
        tensorboard = output_dir / "tensorboard"
        checkpoint_complete = latest.exists()
        status_label = "complete" if checkpoint_complete else "incomplete"
        print(f"{task}\t{status_label}\t{output_dir}")

        latest_step = None
        if checkpoint_complete:
            latest_step = latest.read_text().strip()
            print(f"  latest_checkpointed_iteration: {latest_step}")
        else:
            print(f"  missing: {latest}")

        if tensorboard.exists():
            events = sorted(tensorboard.glob("events.out.tfevents*"))
            print(f"  tensorboard_events: {len(events)}")
            if events and parser_script.exists():
                summary_path = output_dir / "loss_summary.json"
                curve_path = output_dir / "loss_curve.csv"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(parser_script),
                        "--tensorboard-dir",
                        str(tensorboard),
                        "--summary-out",
                        str(summary_path),
                        "--curve-out",
                        str(curve_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"  loss_summary: {summary_path}")
                    print(f"  loss_curve:   {curve_path}")
                else:
                    err = (result.stderr or result.stdout).strip().splitlines()
                    tail = err[-1] if err else "unknown error"
                    print(f"  loss parse failed: {tail}")
        else:
            print(f"  missing: {tensorboard}")

        if cached_status in terminal_statuses:
            local_task_parsed = previous.get("local_task_summary")
            local_task_err = previous.get("local_task_error")
            local_status = cached_status
            print("  reused_terminal_status: true")
        else:
            local_task_parsed, local_task_err = fetch_local_task_status(run, task)
            local_status = (
                local_task_parsed.get("status") if local_task_parsed else None
            )
        print(f"  local_status: {local_status or 'unknown'}")
        if local_task_err:
            print(f"  local_task_error: {local_task_err}")

        stopped_at = stop_events.get(task)
        if stopped_at:
            print(f"  stop_event_recorded_at: {stopped_at}")

        final_tasks.append(
            {
                "task": task,
                "run_name": record.get("run_name"),
                "output_dir": str(output_dir),
                "submitted_at": record.get("submitted_at"),
                "checkpoint_complete": checkpoint_complete,
                "latest_step": latest_step,
                "local_status": local_status,
                "local_task_summary": compact_local_task(local_task_parsed),
                "local_task_error": local_task_err,
                "stopped_at": stopped_at,
                "operator_stop_recorded": stopped_at is not None,
            }
        )

    final_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "tasks": final_tasks,
    }
    if policy_feedback_file.exists():
        try:
            policy_feedback = json.loads(
                policy_feedback_file.read_text(encoding="utf-8")
            )
        except Exception as exc:
            print(f"policy_feedback_error: {exc}")
        else:
            final_payload["policy_status"] = policy_feedback.get("policy_status")
            final_payload["policy_feedback_path"] = str(policy_feedback_file)
            final_payload["policy_violations"] = [
                {
                    "rule_id": item.get("rule_id"),
                    "severity": item.get("severity"),
                    "path": item.get("path"),
                    "message": item.get("message"),
                }
                for item in policy_feedback.get("violations", [])[:10]
            ]
            print(f"policy_status: {final_payload['policy_status']}")
            print(f"policy_feedback: {policy_feedback_file}")
            for item in final_payload["policy_violations"][:3]:
                path = f" [{item.get('path')}]" if item.get("path") else ""
                print(f"  - {item.get('rule_id')}{path}: {item.get('message')}")
    final_status_file.parent.mkdir(parents=True, exist_ok=True)
    final_status_file.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"sft_final_status: {final_status_file}")
    refresh_handoff(run)


def collect_eval(run: Path, manifest: Path) -> None:
    manifest = Path(manifest)
    final_status_file = manifest.parent / "eval_final_status.json"
    valid_manifest_file = manifest.parent / "eval_successful_submissions.jsonl"

    if not manifest.exists():
        print(f"No eval manifest found at {manifest}")
        return

    terminal_statuses = {
        "Succeeded",
        "Failed",
        "Stopped",
        "Killed",
        "Cancelled",
        "Canceled",
    }
    final_task_fields = (
        "task",
        "benchmark",
        "repeats",
        "eval_backend",
        "run_name",
        "model_path",
        "output_dir",
        "submitted_at",
        "local_status",
        "status_observed_at",
        "local_task_summary",
        "local_task_error",
        "artifacts_complete",
        "artifact_paths",
        "artifact_exists",
    )
    previous_tasks = {}
    if final_status_file.is_file():
        try:
            previous_payload = json.loads(final_status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_payload = {}
        for item in previous_payload.get("tasks") or []:
            if isinstance(item, dict) and item.get("task"):
                previous_tasks[str(item["task"])] = item

    def repeat_dirs(output_dir: Path, record) -> list:
        """Every directory the task's repeats were written to.

        A task evaluating its benchmark N times writes rep_<k>/ below output_dir
        (mirrors autosft.benchmarks.aggregate.repeat_dirs); legacy one-task-
        per-repeat records point straight at their own directory.
        """
        repeats = record.get("repeats")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 1:
            return [output_dir]
        return [output_dir / f"rep_{k}" for k in range(1, repeats + 1)]

    submission_records = []
    seen_submissions = set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event"):
                continue
            task = record.get("task")
            if not isinstance(task, str) or not task:
                raise ValueError("eval submission record is missing task")
            for field in (
                "benchmark",
                "eval_backend",
                "run_name",
                "model_path",
                "output_dir",
            ):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(
                        f"eval submission record for {task} is missing {field}"
                    )
            if task in seen_submissions:
                continue
            seen_submissions.add(task)
            submission_records.append(record)

    final_tasks = []
    successful_records = []
    current_tasks = {record["task"] for record, _ in current_eval_submissions(run)}
    for record in submission_records:
        task = record["task"]
        previous = previous_tasks.get(str(task)) or {}
        cached_status = previous.get("local_status")
        observed_at = previous.get("status_observed_at") or (
            previous_payload.get("generated_at") if previous else None
        )
        if task not in current_tasks:
            # Keep historical outcomes, but never query a superseded attempt again.
            final_tasks.append(
                {
                    **previous,
                    "task": task,
                    "local_status": cached_status or eval_task_status(record, manifest),
                    "status_observed_at": observed_at,
                    "artifacts_complete": previous.get("artifacts_complete", False),
                }
            )
            continue
        if cached_status in terminal_statuses and cached_status != "Succeeded":
            print(f"{task}\t{cached_status}\treused_terminal_result")
            reused = {
                key: previous[key] for key in final_task_fields if key in previous
            }
            reused.setdefault("local_task_summary", None)
            reused["status_observed_at"] = observed_at
            final_tasks.append(reused)
            if (
                cached_status == "Succeeded"
                and previous.get("artifacts_complete") is True
            ):
                successful_records.append(record)
            continue
        benchmark = record["benchmark"]
        task_dir = Path(record["output_dir"])
        # The task owns every repeat; the listed artifact paths describe the first
        # repeat and completeness requires the core pair in every repeat dir.
        rep_dirs = repeat_dirs(task_dir, record)
        output_dir = rep_dirs[0]
        artifact_paths = {
            "metrics": output_dir / "metrics.json",
            "scores": output_dir / "scores.json",
            "failures": output_dir / "failures.jsonl",
            "failure_breakdown": output_dir / "failure_breakdown.json",
            "inspect_logs": output_dir / "inspect_logs",
        }
        artifact_exists = {name: path.exists() for name, path in artifact_paths.items()}
        artifacts_complete = all(
            artifact_exists[name] for name in ("metrics", "scores")
        ) and all(
            (rep_dir / name).is_file()
            for rep_dir in rep_dirs
            for name in (
                "metrics.json",
                "scores.json",
                "failures.jsonl",
                "failure_breakdown.json",
            )
        )

        if cached_status in terminal_statuses:
            local_task_parsed = previous.get("local_task_summary")
            local_task_error = previous.get("local_task_error")
            local_status = cached_status
        else:
            local_task_parsed, local_task_error = fetch_local_task_status(
                run, str(task)
            )
            local_status = (
                local_task_parsed.get("status") if local_task_parsed else None
            )
            observed_at = datetime.now(timezone.utc).isoformat()

        artifact_label = "complete" if artifacts_complete else "incomplete"
        print(
            f"{task}\t{local_status or 'unknown'}\t"
            f"artifacts_{artifact_label}\t{task_dir}"
        )
        print(f"  benchmark: {benchmark}")
        print(f"  repeats: {len(rep_dirs)}")
        print(f"  eval_backend: {record['eval_backend']}")
        print(f"  model_path: {record.get('model_path')}")
        if cached_status in terminal_statuses:
            print("  reused_terminal_status: true")
        for name, path in artifact_paths.items():
            state = "exists" if artifact_exists[name] else "missing"
            print(f"  {name}: {path} ({state})")
        if local_task_error:
            print(f"  local_task_error: {local_task_error}")

        final_record = {
            "task": task,
            "benchmark": benchmark,
            "repeats": len(rep_dirs),
            "eval_backend": record["eval_backend"],
            "run_name": record.get("run_name"),
            "model_path": record.get("model_path"),
            "output_dir": str(task_dir),
            "submitted_at": record.get("submitted_at"),
            "local_status": local_status,
            "status_observed_at": observed_at,
            "local_task_summary": compact_local_task(local_task_parsed),
            "local_task_error": local_task_error,
            "artifacts_complete": artifacts_complete,
            "artifact_paths": {
                name: str(path) for name, path in artifact_paths.items()
            },
            "artifact_exists": artifact_exists,
        }
        final_tasks.append(final_record)
        if local_status == "Succeeded" and artifacts_complete:
            successful_records.append(record)

    active_final_tasks = [item for item in final_tasks if item["task"] in current_tasks]
    successful_records = [
        record for record in successful_records if record["task"] in current_tasks
    ]
    if active_final_tasks and all(
        item["local_status"] == "Succeeded" and item["artifacts_complete"]
        for item in active_final_tasks
    ):
        overall_status = "succeeded"
    elif any(
        item["local_status"] in {"Failed", "Stopped", "Killed", "Cancelled", "Canceled"}
        for item in active_final_tasks
    ):
        overall_status = "failed"
    else:
        overall_status = "incomplete"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "ledger_manifest": str(valid_manifest_file),
        "status": overall_status,
        "tasks": final_tasks,
    }
    valid_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with valid_manifest_file.open("w", encoding="utf-8") as stream:
        for record in successful_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    final_status_file.parent.mkdir(parents=True, exist_ok=True)
    final_status_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"eval_successful_submissions: {valid_manifest_file}")
    print(f"eval_final_status: {final_status_file}")
    if overall_status != "succeeded":
        print(
            f"eval run not fully succeeded (status: {overall_status}); skipping aggregation and ledger recording until every repeat of every benchmark is terminal Succeeded"
        )
    else:
        try:
            request = json.loads((manifest.parent / "eval_request.json").read_text())
        except (OSError, ValueError):
            request = {}
        repeats = request.get("repeats")
        repeats = repeats if type(repeats) is int and repeats >= 1 else 1
        subprocess.run(
            [
                sys.executable,
                "-m",
                "autosft.benchmarks.aggregate",
                "--eval-run-dir",
                str(manifest.parent),
                "--repeats",
                str(repeats),
                "--manifest",
                str(valid_manifest_file),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "autosft.orchestration.ledger",
                "record-eval",
                "--run-dir",
                str(run),
                "--eval-manifest",
                str(valid_manifest_file),
            ],
            check=True,
        )

    refresh_handoff(run)
