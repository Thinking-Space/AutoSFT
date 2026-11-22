---
name: recipe-authoring
description: Assemble the next SFT training dataset and register an evidence-backed data decision under the run protocol.
---

# Recipe authoring

Read the staged pool, realized earlier training data and complete evaluation artifacts. You are the meta agent and own the final data decision. When enabled, consult the four analysis roles and weigh their supporting evidence. Use only this run's history, and distinguish observed score variation from evidence for the mechanism you plan to test.

Write `data_rounds/round_<n>/final_data.jsonl` and `data-recipe.yaml`. The dataset must be valid ChatML with at least one non-empty assistant response per row. The recipe records a unique `recipe_id`, positive `round_id`, one dataset path, `execution_summary`, `rewrote_pool`, and, when needed, `rewritten_row_ids_file`. Do not put training hyperparameters or the token budget in the recipe. Record changed source identities for provenance when supervision is modified.

Register the round hypothesis before training, including its comparison round, predicted direction and the finer observation that would support it. Use [run records](references/run-records.md) for the ledger commands and [artifact fields](references/run-artifact-fields.md) to interpret raw outputs. Preparation checks row validity, provenance, earlier-round disposition and the fixed token cap.

With the executor component enabled, delegate the lifecycle through its round context and durable handoff. Otherwise execute it yourself using [training submission](../../executor/training-submission/SKILL.md), [job lifecycle](../../executor/job-lifecycle/SKILL.md), [checkpoint conversion](../../executor/checkpoint-conversion/SKILL.md) and [evaluation submission](../../executor/evaluation-submission/SKILL.md). Read the applicable skill before each stage. Never resubmit a task without recovering its existing receipt. Complete all benchmark repeats before interpreting a round or preparing the next one. Finalize only when one checkpoint meets every configured target.

Read [role dispatch](references/role-dispatch.md) for analysis requests. The meta agent keeps decision authority; subagents return proposals rather than approvals or vetoes.

## Measure the exact training-token exposure

Use the shared converter for token measurement; a separate tokenizer estimate does not replace it. The converter includes the training protocol's markers and EOS handling. Keep measurements and intermediate artifacts under `role_io/round_<id>/scratch/`, outside formal round data.

```bash
mkdir -p "$AUTO_SFT_RUN_DIR/role_io/round_1/scratch"
"$AUTO_SFT_TOKENIZER_PYTHON" "$AUTO_SFT_TOKENIZER_CODE_DIR/prepare_tokens.py" -i "$AUTO_SFT_RUN_DIR/data_rounds/round_1/final_data.jsonl" -v qwen3 --count-only --count-output "$AUTO_SFT_RUN_DIR/role_io/round_1/scratch/token_lengths.tsv" --workers 1
```

The converter uses the fixed `AUTO_SFT_SFT_TOKENIZER_MODEL`. Read total and supervised token counts from the TSV and check the complete dataset against the run cap before handing it to the executor.
