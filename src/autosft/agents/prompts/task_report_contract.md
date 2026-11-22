# Durable Task Delivery

This contract applies to every proposal and review task. The task context supplies `task_id`, `round_id`, `role`, `input_digest` and `task_dir`.

## Persist the result

Write `<task_dir>/scratch/report_input.json` with this envelope:

```json
{
  "task_id": "<task-id>",
  "round_id": "<id>",
  "role": "<role>",
  "input_digest": "<exact digest from task context>",
  "summary": "Answer to the requested question, supporting evidence and limitations.",
  "result_artifacts": []
}
```

Artifact references are objects with `path` and optional `sha256` for existing run-local files. An executable proposal can live in the report itself with `result_artifacts: []`; no candidate dataset is required. Add plans, evidence, measurements, reusable task products, uncertainties and suggested interventions as useful. Negative or inconclusive findings are valid completed work, not a reason to fabricate approval.

```bash
python -m autosft.agents.search_tasks persist \
  --run-dir "${AUTO_SFT_RUN_DIR}" --round-id <id> --task-id <task-id> \
  --input "<task_dir>/scratch/report_input.json"
```

Persistence checks the task identity, pinned inputs and artifact digests, then writes canonical `report.json` and `receipt.json`. Fix validation errors before ending. Never overwrite canonical reports or another task's artifacts. Return only the receipt printed by the command. A report is a contribution, not approval to train. Root reads it through `search_tasks status`.

## Wait for a detached gateway job

Use the batch skill's fixed runner and a task-owned label of the form `<task-id>__<purpose>`. Do not submit the same job again while it is live. For a long job submitted with `--detach`, register its actual lock before ending the turn:

```bash
python -m autosft.agents.search_tasks wait \
  --run-dir "${AUTO_SFT_RUN_DIR}" --round-id <id> --task-id <task-id> \
  --job-label <task-id>__<purpose> \
  --job-lock "<lock_file from detached receipt>" \
  --job-pid <pid from detached receipt>
```

Use the runner's `lock_file` for `--job-lock`; `--job-progress` may use its `status_file`. Pass `--job-pid` from the detached receipt to cover startup before the child has created its manifest and acquired the lock. The wait receipt means paused, not completed; the harness resumes the task when the job stops running. Read the job result on resume. A released lock alone does not prove success: inspect status and resume failed requests under the same job label when appropriate. Persist the task report only after its claimed measurements exist.
