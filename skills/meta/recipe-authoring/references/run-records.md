# Run Records and Rationale

This ledger interface is shared by both architectures. Root records the chosen data changes and their rationale. Training and evaluation entrypoints record job state, data accounting, checkpoints, and results.

## Record the data decision

```bash
python -m autosft.orchestration.ledger note \
  --run-dir "${AUTO_SFT_RUN_DIR}" --round-id <id> \
  --rationale "What changes, why, and the relevant evidence or alternatives."
```

`--round-id` binds the note to the recipe's round, even when directory numbers and training counts differ. Before submission it is staged in `recipe_history/round_<id>.pending_rationale.json`; only that round's SFT submission consumes it. A recorded round receives a later note. `--iteration` selects an actual ledger index, not a round ID; do not supply both selectors. Use `--force-restage` to deliberately revise an unconsumed note; previous versions remain in `note_history`.

Notes retain their timestamp and phase based on submission/results already in the ledger. The recipe's `execution_summary` describes the actual data decision; do not add rationale-note fields to the strict recipe schema. These explanations record why a decision was made, not the proven cause of a later result.

The meta agent registers a hypothesis before training as part of its research procedure. Optional ledger fields `--expectation` and `--evidence-plan` record the prediction and intended comparison; later notes preserve earlier versions. Use `ledger review --round-id <id> --review "..."` to append an interpretation bound to the current evidence digest. `ledger hypothesis status` reads the recorded history without creating a claim. The execution harness does not gate submission on these learning records, so their presence and content remain evidence of the agent's research behavior.

The harness refreshes `experiment_memory.json` from this run's records for session recovery. The snapshot distinguishes recorded results from meta-authored interpretations and flags incomplete or changed evidence. It does not supply new conclusions or establish that a data change caused a score difference.

## Read completed and incomplete work

`iterations.jsonl` and `iterations.md` locate the per-round recipe, checkpoint, SFT state, and evaluation results. Compare completed evaluations using all configured benchmarks and repeats. Missing or incomplete results are unknown, and per-repeat spread matters when comparing scores.

Use `recipe_history/iter_<NNN>.yaml` and `recipe_history/iter_<NNN>.data_manifest.json` for the submitted recipe and data accounting. Inspect the actual training data and available assembly exposure reports for selected rows and transformations. SFT loss artifacts and evaluation outputs supply the measured results; their fields are documented in [run artifact fields](run-artifact-fields.md).

With analysis enabled, recover contributor tasks with `search_tasks status` and read their canonical reports and pinned inputs under `role_io/round_<id>/tasks/<task-id>/`. This works before any training round exists. A report's interpretation is not a measured training outcome.

## Resume

Recover existing task IDs, manifests, collected results, and pending rationale notes before submitting work. Read `recipe_history/round_<id>.pending_rationale.json` for a round-bound note or `pending_rationale.json` for an iteration-bound note. A pending note is not a submitted, evaluated, or failed round. Use `--force-restage` only when deliberately revising it.

A rejected preparation can require a new recipe and a revised rationale. Keep the note consistent with the actual data, recover existing jobs, and avoid duplicate submissions.
