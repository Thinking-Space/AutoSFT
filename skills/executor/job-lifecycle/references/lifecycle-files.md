# Local lifecycle files

- `local_tasks/local-<uuid>.json`: supervisor PID/start token, child group, local command, host, timestamps, status and exit code.
- `sft_submissions.jsonl`, `eval_runs/round_<n>/eval_submissions.jsonl`: durable task identities, unchanged data/model/repeat identity contract.
- `local_events.jsonl`: compact status observations.
- `sft_final_status.json`, `eval_final_status.json`: collection receipts; `local_status` is the supervisor's observed state, separate from artifact completeness. Successful repeats are collected into aggregates and ledger.
- `<task-output>/local_job.log`: stdout/stderr including launch failures.

Interrupted launch is recorded as failure; status never invents success from files on disk. Historical remote tasks are not migrated into local identities. Start a new local run after moving this checkout to the GPU machine.
