---
name: synthesis-batch-generation
description: >-
  Synthesis batch inference: call batch.py under the shared recorded-calls
  contract to generate, review, rewrite, and repair candidates; read aligned
  answers by ref.
compatibility: 'Uses ${AUTO_SFT_REPO_ROOT}/skills/shared/batch-inference/references/recorded-calls.md as the transport contract; the runner is src/autosft/backends/providers/batch.py; requires AUTO_SFT_LLM_BASE_URL, AUTO_SFT_LLM_MODEL and optional AUTO_SFT_LLM_API_KEY in the inherited environment; the only writable root is ${SYN_OUTPUT_DIR}.'
metadata:
  autosft-role: synthesis
  autosft-stage: generation
---

# Synthesis batch inference operator

Generation, review, rewriting, repair and resume all use the same recorded batch runner. The transport contract is not repeated here: read the full `SYN_BATCH_REFERENCE_FILE` given in the worker prompt (that is, `skills/shared/batch-inference/references/recorded-calls.md`) and follow its input contract, invocation, status polling and persisted record.

This file only adds the synthesis-side conventions.

## Input

UTF-8 JSONL, one request per non-empty line:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"ref":"llm_calls/candidates/gen_a.jsonl:12"}
```

- `messages`: an OpenAI message array. Write the system prompt freely; few-shot turns may be included as preceding user/assistant turns; multi-turn conversations follow their real order. The last turn must be `user`.
- `ref`: the candidate identity, written as `<run-dir-relative path>:<candidate index>`, for example `llm_calls/candidates/gen_a.jsonl:12`. `--rows` validates the scope against it.
- Only `messages` and `ref` are accepted; extra fields or invalid turns reject the whole batch before sending.

## Invocation

Use the literal absolute paths given in the worker prompt for `<SYN_BATCH_RUNNER>` and `<OUTPUT_DIR>`; do not use shell variable expansion. See `SYN_BATCH_REFERENCE_FILE` for the full arguments, `--validate-only`, `--status`, `--detach` and the persisted record.

```bash
python "<SYN_BATCH_RUNNER>" \
  "<OUTPUT_DIR>/llm_calls/candidates/<label>.jsonl" \
  --run-dir "<OUTPUT_DIR>" \
  --round-id "syn" --label "<label>" \
  --rows 'llm_calls/candidates/<label>.jsonl:1-<N>'
```

- `<label>` names the purpose of this call: `pilot`, `generate_<template>`, `review_<dimension>`, `repair_<failure>`, `fill`. A label is bound to one input; a new question needs a new label.
- `--rows` declares the candidate indexes this job may touch; any `ref` outside it rejects the whole batch before sending.
- For long jobs add `--detach`, then register its receipt with the synthesis wait command below. All synthesis commands, including `--status` and retries, require the same explicit `--run-dir "<OUTPUT_DIR>"`; the worker does not inherit `AUTO_SFT_RUN_DIR`.

### Wait for detached synthesis batches

Synthesis uses its own wait registration because its output directory and `syn` batch round are separate from the controller's numbered training rounds. After `--detach`, use the exact `pid`, `lock_file` and label from the receipt:

```bash
python -m autosft.synthesis._internal.batch_wait \
  --output-dir "<OUTPUT_DIR>" --job-label "<label>" \
  --job-lock "<lock_file>" --job-pid <pid>
```

Register every active batch before ending the turn. The synthesis runtime waits outside the model session and resumes the same thread when any registered batch stops; this wait does not consume the error-recovery resume budget. On wake, inspect `--status`, the aligned output and detach log. Retry failed requests and register any still-running batches again before ending the turn. A stopped process is not proof of success. Do not use `root_wait` or `search_tasks wait` for synthesis, and do not sleep or poll toward completion in a model turn.

## Outputs

Under `<OUTPUT_DIR>/llm_calls/round_syn/`, the reference produces `prompts` / `responses` / `aligned` / `manifest` and related artifacts. Read answers only from `<label>.aligned.jsonl`: it is aligned row by row with the input and carries `ref`, `answer`, and `input_sha256`, and appears only after every request succeeds. Join on `ref`, never on `responses.jsonl` by position. If `aligned` is absent, rerun the same command to fill the failed rows.

## Rules

- One label is bound to one input; the input file must not change while the job runs. A new question needs a new label.
- Rerun the same command for failed rows; only the failed part is filled, with no loss or duplication.
- **Do not bypass the runner with direct API calls or another model endpoint.**
- Write only under `<OUTPUT_DIR>`; read only the allowlisted materials; do not train or evaluate models.
