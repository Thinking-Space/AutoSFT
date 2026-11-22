# Run Artifact Fields

What the run's loss and evaluation artifacts contain, so a reader knows which file answers which question and what a field name means. This is a field reference, not a reading of them: what counts as healthy, concerning, or decision-relevant is yours to judge from the round in front of you.

Every path in this reference is relative to `${AUTO_SFT_RUN_DIR}`. None of these files exist under the repository checkout or under `${CODEX_HOME}`.

## SFT loss artifacts

```text
sft/round_<id>/checkpoints/                      Megatron dist checkpoints
sft/round_<id>/tensorboard/events.out.tfevents*  Raw scalar events
sft/round_<id>/loss_summary.json                 Per-tag stats (below)
sft/round_<id>/loss_curve.csv                    Step and sample-indexed rows, one column per tag
sft_final_status.json                            Per-task local supervisor terminal state
```

`loss_summary.json` has `scalar_tags` (every tag logged), `loss_tags` (the loss-bearing subset), and `tags`, a map from tag to its statistics: `num_points`, `first`, `last`, `min`, `max`, `mean`, `first_step`, `last_step`, `has_nan`, `has_inf`, and — for loss tags — `last_pct_slope`, the least-squares slope over the last `slope_pct` fraction of points. `lm loss` is Megatron's overall training loss; `lm_loss/final_data` is the per-dataset loss over the round's one dataset, a separate series with its own values. `loss-scale`, `grad-norm`, `learning-rate`, `batch-size`, and `num-zeros` are optimizer diagnostics. Every tag also has a `<tag> vs samples` twin indexed by samples consumed rather than steps, with its own `first_step`/`last_step` and slope. There is one aggregate series per loss tag — no per-slice loss and no held-out validation series.

`loss_curve.csv` has a `step` column plus one column per tag, with the step-indexed and sample-indexed rows interleaved (so it holds more rows than there were steps; select columns rather than counting lines). Read it when a summary statistic is too coarse, such as locating the step where a change began.

`checkpoint_complete` and job success are different claims: a task killed after a save looks complete on disk while `local_status` in `sft_final_status.json` reads `Failed` or `Stopped`. The status file is the authority on whether the job finished. Without `tensorboard` in the control-plane environment, collect reports checkpoint status but writes no loss artifacts.

## Evaluation artifacts

Follow the actual output and manifest paths in `executor_handoff.json` and the ledger's evaluation records. The layout below is the normal first attempt; a confirmed failed task's replacement uses an isolated `eval_runs/round_<id>_retry_<key>/` directory and records its original round in `eval_request.json`. Old failed artifacts remain for history, not as the current round's result. Do not infer the active attempt from a directory name.

Per benchmark under `eval_runs/round_<id>/eval/<benchmark>/`, one local task loads the model and generates all repeats. With two or more repeats, each repeat has a `rep_<k>/` directory. With one repeat, these files sit directly in the benchmark directory. The local task writes a shared `local_job.log`; collection writes the benchmark's `aggregate.json`.

```text
rep_<k>/metrics.json                  Normalized metrics
rep_<k>/scores.json                   Run identity, primary score, failure summaries, paths
rep_<k>/failures.jsonl                One record per failed prompt/problem
rep_<k>/failure_breakdown.json        Counts and rates by instruction or failure type
rep_<k>/generation_config.json        Sampling settings and prompt protocol
rep_<k>/opencompass/results/<benchmark>.json  Full local scorer report
local_job.log                         Local generation and scorer stdout/stderr
aggregate.json                        Repeat scores, mean, spread, failure persistence
```

The `opencompass/results` directory name is retained for artifact compatibility. The local runner calls the bundled instruction scorers or EvalPlus in process; it does not deploy a model server. Follow paths in `scores.json` and the task manifest when locating logs and results.

### Instruction-following benchmarks

IFEval and IFBench expose `overall_accuracy` (mean of the four component accuracies; `primary_score` is the ledger's alias for it), `strict_prompt_accuracy` (fraction of prompts whose instructions all pass strict checking), `loose_prompt_accuracy` (prompt-level accuracy after the allowed response normalizations), `strict_instruction_accuracy` and `loose_instruction_accuracy` (the same two rules per instruction), and `num_failures`. The current scorer emits no `*_stderr` fields; use the repeat scores and `primary_score_sd` in `aggregate.json` to report spread.

`scores.json` adds run and benchmark identity, `top_failure_categories` (coarse buckets with strict/loose counts, totals, and rates), `top_failure_subcategories` (full `instruction_id` buckets with their parent category and the same fields), and paths to the artifacts. `failure_breakdown.json` holds strict/loose fail counts, totals, and rates for each full `instruction_id`.

A `failures.jsonl` record includes the original prompt and completion, strict and loose instruction verdicts, failed instruction ids, benchmark kwargs, the sample id, and an `opencompass_result_ref` pointing at that repeat's full scorer report. It contains strict-prompt failures only.

Read `opencompass/results/<benchmark>.json` within the same repeat for passing and failing samples. Its `details` mapping holds `prompt`, `pred`, `refer`, `is_strict_correct`, `is_loose_correct`, and `grade`. `refer[0].instruction_id_list` and `refer[0].kwargs` describe each sample's constraints. Normalize `refer[0].key` and failure `sample_id` values to strings before joining them. Use the full report for denominators; absence from the failure file alone does not establish a pass. The local runner does not write a separate prediction-shard tree or native generation-token counts.

### Code benchmarks

HumanEval+ and MBPP+ use `plus_pass@1` as their primary score: a problem passes only when its completion passes both the base and extra tests. `metrics.json` stores normalized `base_pass@1` and `plus_pass@1` values, problem counts, and `is_full_benchmark`. The full local report records raw and sanitized predictions, `task_id`, `base_result`, `plus_result`, and the individual suite statuses for every problem, including passes. It also records the pinned dataset version and EvalPlus scoring protocol. The report's top-level percentages use 0–100; normalized metric artifacts and target scores use 0–1.

`failure_breakdown.json` groups failed problems by test outcome. `generation_config.json` records seeds, sampling settings, template, dataset hash, and code-sanitization protocol. Compare outcomes by `task_id`; preserve the raw completion and its sanitized executable text when inspecting failures.

### Repeat aggregation

`aggregate.json` combines the repeats: `n_repeats`, `primary_scores` (the raw per-repeat list), `primary_score_mean`, `primary_score_sd` (sample standard deviation; `null` with one repeat), `repeats` (per-repeat metadata), and `failure_persistence`. Failure persistence identifies the repeats in which each sample failed; code benchmarks also preserve base/plus failure history. A round's benchmark score is the mean of its repeats. The ledger copies the mean and spread into the round's `eval.<benchmark>` block.

## Cross-round records

`iterations.jsonl` and `iterations.md` pair each round's recipe snapshot and checkpoint with SFT status, per-benchmark scores, and the rationale recorded for that round. Benchmark values are comparable only across valid completed evaluations; say so when an expected predecessor is missing rather than comparing across a gap.

`note_history` retains timestamped rationale changes. Unconsumed round-bound notes live in `recipe_history/round_<id>.pending_rationale.json`; `pending_rationale.json` stages an iteration-bound note. A pending note is not a submitted or evaluated round. See [run records](run-records.md) for rationale commands and recovery.

With analysis enabled, `python -m autosft.agents.search_tasks status --run-dir "${AUTO_SFT_RUN_DIR}"` lists contributor outcomes, including tasks completed before any training. Each task has its own request, state, receipt and canonical `role_io/round_<id>/tasks/<task-id>/report.json`. Read the pinned inputs and cited evidence before reusing a report.

`data_manifest.json` at the run root is the submit-time manifest of the most recently prepared round: the recipe identity (`recipe_id`, `round_id`, `execution_summary`, `rewrote_pool`), the resolved dataset entries with per-file `rows`, `sha256`, and `total_tokens`, and the row and token accounting (`row_accounting`, `token_accounting` with `budget_utilization`, `realized_iters_per_epoch`, `overlong_rows`). It is rewritten on every submission, so it answers questions about the latest prepared round only.

`recipe_history/` is the immutable per-iteration archive behind the ledger rows: `iter_<NNN>.yaml` is the exact recipe submitted for iteration `<NNN>`, and `iter_<NNN>.data_manifest.json` is that round's manifest with the same fields as the run-root `data_manifest.json`, where `<NNN>` is the zero-padded iteration index. Rounds that rewrote pool rows also keep their `rewritten_row_ids_file` here. Read the archive when the question is what an earlier round actually trained on; never edit it.
