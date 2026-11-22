# Role dispatch

Each analysis request is identified by run, positive round id and a unique task id. Supply an objective and immutable artifact references; requests with changed inputs need a new task id. The harness launches, monitors and resumes requested tasks. Results remain advisory inputs to the meta agent's decision.

| Role | Evidence source | Deliverable |
| --- | --- | --- |
| `training_data` | Source pool and selected data | One executable proposal on quality, difficulty, coverage or redundancy |
| `failure_case` | Passing and failing evaluation outputs | One executable proposal grounded in observed behavior |
| `learning_dynamics` | Training signals and exposure | One executable proposal relating learning behavior to supervision |
| `history` | Earlier hypotheses, interventions, verdicts and lessons | One executable proposal addressing untested or insufficiently tested ideas |
| `executor` | Formal recipe and execution handoff | Artifact paths and a short structured lifecycle report |

Prepare a JSON request under the run:

```json
{
  "objective": "Examine the realized training data and propose the next data experiment.",
  "input_refs": [
    {"path": "data_rounds/round_1/final_data.jsonl", "snapshot": true},
    {"path": "data_manifest.json", "snapshot": true}
  ]
}
```

```bash
python -m autosft.agents.search_tasks request --run-dir "$AUTO_SFT_RUN_DIR" --round-id 2 --task-id data_proposal --role training_data --input "$AUTO_SFT_RUN_DIR/requests/data_proposal.json"
```

Use the same proposal contract for all four analysis roles. Reports identify their task, round, role and input digest and describe observations, supporting evidence, an executable data change, expected exposure, a predicted observation and uncertainty. Task-local measurements and isolated source-grounded trials may support the proposal. Analysis sessions cannot edit the shared pool, formal data or training settings.

The executor is delegated with `role_io/round_<n>/executor.task_context.md` and `executor_handoff.json`, rather than an analysis request. It performs submission, monitoring, collection, conversion and evaluation while the meta agent retains interpretation and data decisions.
