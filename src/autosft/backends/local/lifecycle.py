"""Status, collection and stopping for the current run's local workloads."""

from autosft.config.components import executor_enabled
import argparse
import json
import os
from pathlib import Path
from autosft.backends.local.tasks import recover_unregistered_sft, status, stop
from autosft.backends.local.status import manifest_tasks
from autosft.core.common import append_jsonl, utc_now


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["sft", "eval"])
    p.add_argument("action", choices=["status", "collect", "stop"])
    p.add_argument("tasks", nargs="*")
    args = p.parse_args(argv)
    if not os.environ.get("AUTO_SFT_RUN_DIR"):
        p.error("AUTO_SFT_RUN_DIR is required")
    roles = {
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {None, "", "meta", "executor"}
    if (
        roles
        or os.environ.get("AUTO_SFT_TASK_DIR")
        or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")
    ):
        p.error("only the meta agent and executor manage workload lifecycle")
    run = Path(os.environ["AUTO_SFT_RUN_DIR"]).resolve()
    if args.kind == "eval" and not os.environ.get("AUTO_SFT_EVAL_MANIFEST"):
        p.error("AUTO_SFT_EVAL_MANIFEST must identify the evaluation request")
    manifest = Path(
        os.environ.get("AUTO_SFT_SFT_MANIFEST", str(run / "sft_submissions.jsonl"))
        if args.kind == "sft"
        else os.environ["AUTO_SFT_EVAL_MANIFEST"]
    ).resolve()
    if args.kind == "sft" and manifest != run / "sft_submissions.jsonl":
        p.error("SFT manifest must belong to the current run")
    if args.kind == "eval" and (
        manifest.parent.parent != run / "eval_runs"
        or manifest.name != "eval_submissions.jsonl"
    ):
        p.error("eval manifest must belong to the current run")
    if args.action == "collect":
        if args.tasks:
            p.error(
                "collection covers the exact manifest; task ids are for status and stop"
            )
        from autosft.backends.local.collection import collect_eval, collect_sft

        (collect_sft if args.kind == "sft" else collect_eval)(run, manifest)
        return
    if args.kind == "sft" and args.action == "status":
        for data in recover_unregistered_sft(run):
            event = {
                "task": data["task"],
                "status": data["status"],
                "polled_at": utc_now(),
                "source": "lifecycle:sft:status",
                "submission_registered": False,
                "next_action": "Resubmit the same training round through src/autosft/backends/megatron/train.py.",
            }
            append_jsonl(manifest.parent / "local_events.jsonl", event)
            print(json.dumps(event))
    allowed = manifest_tasks(manifest)
    tasks = args.tasks or allowed
    for task in tasks:
        if task not in allowed:
            raise SystemExit(f"task not in manifest: {task}")
        data = (stop if args.action == "stop" else status)(run, task)
        event = {
            "task": task,
            "status": data["status"],
            "polled_at": utc_now(),
            "source": f"lifecycle:{args.kind}:{args.action}",
            "create_time": data.get("create_time"),
        }
        append_jsonl(manifest.parent / "local_events.jsonl", event)
        print(json.dumps(event))
    if executor_enabled():
        from autosft.agents.executor_handoff import refresh_executor_handoff

        refresh_executor_handoff(run)


if __name__ == "__main__":
    main()
