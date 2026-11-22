---
name: row-analysis-operator
description: 'Puts a per-row question to the launcher-selected model and turns the answers into numbers a reader can re-derive: one prompt per row, run through the batch runner, answers joined back by ref, reported with the measurement''s definition and denominator. Use when a judgement is needed about more rows than can be read by hand — analysing, auditing, filtering, or classifying them.'
compatibility: Uses the batch transport under ${AUTO_SFT_REPO_ROOT}/skills/shared/batch-inference/references/; requires the launcher-supplied provider settings in the environment and a writable ${AUTO_SFT_RUN_DIR}.
metadata:
  autosft-role: shared
  autosft-stage: row-analysis
---

# Per-Row Analysis and Audit With the Configured Model

## Scope

This skill asks the configured model a question about each specified row and carries the answers into a measurement — a rate, a partition, a set of flagged rows — that whoever reads it can re-derive. Which rows to examine and what to ask are your decisions. The answers inform a judgement; they do not become row content. Actions based on a judgement follow your role contract: all analysis subagents may use measurements to support full-scale plans and isolated rewrite artifacts. Only the meta agent produces the formal training set or edits the shared pool.

## The shape of an analysis job

The transport is the shared batch runner; read `${AUTO_SFT_REPO_ROOT}/skills/shared/batch-inference/references/recorded-calls.md` in full for its input contract, invocation, status polling, and the persisted record.

If root/meta owns a detached analysis directly, register its receipt with
`python -m autosft.agents.root_wait wait` as described in recorded calls so the
harness resumes the same root session after the job stops. An analysis task
must use its task-scoped `search_tasks wait` contract instead; do not register
the task's job in the root wait marker.

1. **Build one prompt per row.** Each line's `messages` carries your question plus whatever of the row the model needs to answer it. Set `ref` on every line to the row's durable identity — its source file as a run-dir-relative path plus its position (`data_pool/<file>.jsonl:<line>`, `eval_runs/<...>/failures.jsonl:<line>`), or the id the row itself carries. Phrase the question so the answer is mechanically parseable. Write the file under `${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/`.

2. **Run it under a declared row scope.** Give the label a name that says what the question was, and pass `--rows` with the row ids the analysis covers, in the same terms the refs use. The runner refuses any request whose ref falls outside the scope, so the declaration is also the check that the input reaches exactly the rows you decided to examine, and the scope persists in the manifest as the population the job's numbers are about. Read answers from `<label>.aligned.jsonl`, which appears only when every request has succeeded, and join on `ref`, never by position; if the aligned file is absent the job is not done, and a number computed from a partial join has no defined denominator.

3. **The prompt is the measurement's definition.** `<label>.prompts.jsonl` holds the exact request behind every answer; a rate cited without that path is a value readers can only believe or ignore. Two rates are comparable only when the same question produced them — a changed question is a new label.

4. **The answers are evidence.** Model judgment and deterministic checks can both be wrong or inapplicable. Inspect consequential disagreements against the current row content; do not change a correct rule merely to agree with the model. State the denominator and scope behind every rate. Unparseable or undecidable answers are unknown, not negative or positive. Keep label accuracy distinct from whether the response satisfies the task.

5. **Record what was measured where your role records things.** Cite the job's prompts and aligned paths in whatever the measurement feeds — a task report's `summary` and `result_artifacts`, or the round's `execution_summary` and ledger rationale.

## What this skill does not do

It does not choose the rows, author the question, or accept the answers on your behalf, and it does not change data: an analysis job reads rows and produces judgement, and whatever happens to the rows afterwards is a different operation, governed by your role contract.
