---
name: local-job-lifecycle
description: Observe, stop, and collect durable same-host training and evaluation tasks.
---

Use this skill after src/autosft/backends/megatron/train.py or src/autosft/benchmarks/submit.py records a local task. Tasks are supervised local processes, not a remote service. Run commands from AUTO_SFT_REPO_ROOT.

- SFT: `autosft tools jobs sft status`
- SFT collection: `autosft tools jobs sft collect`
- Eval: set AUTO_SFT_EVAL_MANIFEST to the request's eval_submissions.jsonl, then `autosft tools jobs eval status`
- Eval collection: `autosft tools jobs eval collect`
- Stop: `autosft tools jobs sft stop` or `autosft tools jobs eval stop`, with task ids as arguments.

Status reads local_tasks/<id>.json and validates the supervisor's host, boot ID, PID and process start time. A GPU lock serializes SFT and benchmark tasks on the host. Stop signals the supervisor, which kills its process group and records a terminal receipt. Do not kill arbitrary PIDs or create tasks directly.

Before retrying, recover the recorded identity. Collection parses loss curves, checks final checkpoint/eval artifacts, aggregates every repeat, and updates the ledger. A checkpoint marker alone does not mean that training has exited. Do not evaluate a running or queued SFT task. Repeated status/collection calls are safe; repeated submission must go through the canonical public entrypoint.
