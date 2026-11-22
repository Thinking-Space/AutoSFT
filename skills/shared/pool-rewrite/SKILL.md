---
name: pool-rewrite-operator
description: 'Rewrites specified data-pool rows with the launcher-selected model: builds per-row prompts, runs them through the batch runner, joins aligned answers into source-grounded candidate replacements, and records the rewrite so a later reader can re-derive it. Root may promote replacements into the pool. Use when a round''s data work changes what a row holds rather than which rows survive — the row''s content, a label, a category tag, the constraint ids and their parameters, the kwargs, or any other field on the row.'
compatibility: Calls the shared batch runner under ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/; requires the launcher-supplied provider settings in the environment and a writable ${AUTO_SFT_RUN_DIR}.
metadata:
  autosft-role: shared
  autosft-stage: pool-rewrite
---

# Rewriting Rows With the Configured Model

## Scope

This skill rewrites specified rows in the data pool using the configured model and records the rewrite so it stays re-derivable. Which rows to rewrite and what the rewrite instruction says are your decisions. Root may write the shared pool. All analysis subagents may use this operator to support executable full-scale plans and produce reusable replacements inside their own task directories. Root produces the formal training set; analysis subagents never edit it or the shared pool.

"Rewrite a row" means changing any of its fields, not only its content: the instruction/response text, a label or category tag, the constraint ids and their parameters, the kwargs, the model attribution, or any other field. A rewrite may expand or simplify any of them, as the round's failures and its supply show it needs.

Keep the substantive source task and provenance while changing instruction, response, and metadata together when useful. A relevant row with an incorrect answer is a repair candidate; it need not first pass a response-quality gate. Removing conflicting or distracting constraints is a valid source-grounded transformation. Validate the new pair against its current requirements, with unsupported checks or adapter errors recorded as unknown. Stale verdicts on the old response do not apply to the replacement.

Each rewrite request must refer to an existing row and preserve its substantive content and provenance. Pool and final-dataset row counts have no ceiling; the training-token cap still applies. Isolated candidate variants are alternative versions of identified source material, not new external examples. The model is fixed; what varies between rounds is what you asked, and which rows you asked it about.

## The shape of a rewrite job

The transport is the shared batch runner; read `${AUTO_SFT_REPO_ROOT}/skills/shared/batch-inference/references/recorded-calls.md` in full for its input contract, invocation, status polling, and the persisted record.

If root/meta owns a detached rewrite directly, register its receipt with
`python -m autosft.agents.root_wait wait` as described in recorded calls so the
harness resumes the same root session after the job stops. An analysis task
must use its task-scoped `search_tasks wait` contract instead; do not register
the task's job in the root wait marker.

1. **Build the input from source rows.** Each line's `messages` carries your rewrite instruction plus whatever of the source row the model needs — the row's instruction, its current response, the field being rewritten and its current value, and the constraints the result must respect. State explicitly which field the answer replaces. Set `ref` on every line to the row's durable identity: the pool file as a run-dir-relative path plus the row's position (`data_pool/<file>.jsonl:<line>`). Write the file under `${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/`.

2. **Run it under a declared row scope.**

   ```bash
   python ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/batch.py \
     "${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/<label>.input.jsonl" \
     --round-id <id> --label <label> \
     --rows '<file>:1-20,<file>:45'
   ```

`--rows` declares which rows this rewrite may touch, in the same terms the refs use. The runner refuses any request whose ref falls outside it, so the declared scope is a hard edge around the rewrite before the first answer exists, and the manifest carries it as the durable statement of how far this rewrite was allowed to reach — the same extent the written-back rows and `rewritten_row_ids_file` must stay within.

3. **Read answers from the aligned file** and join on `ref`. If `<label>.aligned.jsonl` is absent, the job is not done.

4. **Stage the rewritten rows outside the pool.** Materialize the joined result as a staging file under `${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/` — complete rows in the source file's own format, each the rewritten replacement for the row its `ref` names. This file is the rewrite as a reviewable object and survives unchanged after the pool moves on.

5. **Return reusable artifacts or produce formal data.** An analysis subagent joins the staged answers into candidate ChatML and provenance files in its task directory, checks current requirements, and returns those artifacts to the meta agent. It never changes the shared pool or final training set. The optional `autosft.data.candidate_data freeze` command can preserve a version under the task's `candidates/` directory using `--output-dir`.

Root may assemble candidate replacements directly into a scratch training set, or write the staged rows into the pool in place. For a pool edit, replace each named row's content in its pool file with the staged row. The recipe's `rewrote_pool` is `true`, and the rewritten rows' ids go to a JSON file under `${AUTO_SFT_RUN_DIR}/recipe_history/` named by `rewritten_row_ids_file`, in the same full-path form the refs use. Later rounds read the rewritten pool; the pre-rewrite text survives in the source snapshot and job record. Selected rewritten candidate versions also set `rewrote_pool` true even when the shared pool stays intact. The rewritten-row record names the original source identities of changed rows actually used in training. Unselected rewrite artifacts remain evidence, not a realized round change.

Cite the job's prompt, aligned, and staging paths where the round explains itself — the prompt file holds what was asked, but only you can say why that was the right question, on those rows. Do not edit or delete a persisted job or its staging file once the pool has taken the rows.

Check which replacements actually reach final_data.jsonl. Preserve candidate source ids and current ChatML content so the optional `autosft.data.intervention_audit` helper can measure exact inclusion. Distinguish produced, rejected, unknown, and selected rows and selected token volume. Low inclusion can reflect selection competition or invalid checks; it does not by itself show that rewriting is ineffective.

## What this skill does not do

It does not choose the rows, author the instruction, or write a single file into the pool — the runner stops at the aligned answers, and every step from there into the pool is your data work, recorded like the rest of it. The provider can produce incorrect text, so check its output against the source.
