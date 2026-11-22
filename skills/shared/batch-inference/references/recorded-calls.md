
# Batch Model Transport

## Scope

This skill runs one thing: a list of chat prompts you wrote, sent to a fixed model, with the answers written back to disk beside the prompts that produced them. It holds no opinion about what to ask, which rows to send, or what to do with the results — those belong to the operation you are performing.

The launcher selects `AUTO_SFT_LLM_BASE_URL`, `AUTO_SFT_LLM_MODEL`, and `AUTO_SFT_LLM_API_KEY` before the run. The endpoint can be a local server or a public API that implements OpenAI chat completions. Read `${AUTO_SFT_REPO_ROOT}/docs/model-provider.md` for operator setup; agents must use the configured values. There is no built-in endpoint or service-specific client. The model and sampling temperature stay fixed (temperature 0). The runner is the only sanctioned interface for these calls; direct calls, other clients, and other models are policy violations.

## Inputs

Write a JSONL file where each non-blank line is a JSON object with:

- `messages`: an OpenAI-format array; every turn a string `role` and a string `content`.
- `ref` (required under a `--rows` scope): the durable identity of the row this request came from, shaped `<source-path>:<row-id>` with the source as a run-dir-relative path — `data_pool/<file>.jsonl:<line>`, not a bare `<file>.jsonl:<line>`. The ref is copied to the response and the aligned output untouched; without it, matching answers back means trusting line order.

No other key is accepted. Validation covers the whole file before anything is sent, naming up to ten offending lines. Write the file under `${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/` with an absolute path, never under `data_pool`. Record source identities and file hashes; row counts are accounting fields and do not impose a ceiling.

Validate the input without starting a job:

```bash
python ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/batch.py \
  "${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/<label>.input.jsonl" \
  --round-id <id> --label <label> --validate-only
```

## Run a job

```bash
python ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/batch.py \
  "${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/<label>.input.jsonl" \
  --round-id <id> \
  --label <label> \
  --rows '<file>:1-20,<file>:45'
```

`--round-id` and `--label` are required and name the job's artifacts; give the label a name that says what the job was for. `--run-dir` defaults to `${AUTO_SFT_RUN_DIR}`. Temperature is fixed at `0.0` and recorded in the manifest and every response record.

`--rows` declares the job's row scope — the set of row ids this job may touch, stated before anything is sent. Items are comma-separated and the flag is repeatable: `<file>:<n>` names one row, `<file>:<a>-<b>` an inclusive range, a bare `<n>` or `<a>-<b>` covers refs with no file prefix, and any other string must match a ref exactly. A comma list continues the file it started on (`'<file>:2-4,7'` is rows 2–4 and 7 of that file); name a second file to switch (`'<a>:1,<b>:2'`). Every request's `ref` must fall inside the scope — a row outside it, or a request with no `ref`, rejects the whole batch before anything is sent. The scope is required for a new job, recorded in the manifest, and fixed thereafter: a resume may repeat it or omit the flag to inherit it, but a different scope under the same label is refused — that is a different job and needs a new label. `--validate-only` checks the scope too.

Exit status is `0` when every request succeeded, `1` when any failed or the input changed mid-job, and `2` when the input or environment was unusable. A job that ends non-zero has still written every answer it received.

Re-running the exact same command is safe: a finished job answers `over` without changes; an interrupted one resumes unfinished requests and preserves earlier failures. Only one process may run a label at a time; use `--status` when it is already live. The runner handles rate limits automatically, in both directions: `--concurrency` is only an in-flight ceiling, and the runner cuts below it on pushback and climbs back to it on its own once the provider goes quiet. Leave it at the default; lowering it on a resume does not help and only caps the recovery.

Poll a long job without re-reading its output:

```bash
python ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/batch.py \
  "${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/<label>.input.jsonl" \
  --round-id <id> --label <label> --status
```

This reads only local files and prints `<completed>/<total>`, appending `(<n> failed)` when some requests have exhausted their retries, or `over` when every request succeeded.

### A job that outlives the session

For a job that must survive the caller's turn, use `--detach`; a shell `&` does not isolate it from turn cleanup. The command starts a separate session, prints its pid and artifact paths, and returns:

```bash
python ${AUTO_SFT_REPO_ROOT}/src/autosft/backends/providers/batch.py \
  "${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/<label>.input.jsonl" \
  --round-id <id> --label <label> --rows '<scope>' --detach
```

```json
{"state": "detached", "pid": 480234, "label": "pool_audit_r1",
 "requests": 550000, "log": ".../pool_audit_r1.detach.log",
 "status_file": ".../pool_audit_r1.responses.jsonl",
 "lock_file": ".../pool_audit_r1.lock",
 "manifest_file": ".../pool_audit_r1.manifest.json"}
```

Input and scope validation happen before detaching. The returned `pid` covers startup before the manifest/lock are ready; logs go to `<label>.detach.log`. A durable analysis task uses `pid` and `lock_file` with `search_tasks wait`; root/meta uses the same durable wait semantics through `autosft.agents.root_wait`.

Synthesis workers use the wait command in their synthesis batch skill instead: their `--run-dir` is the worker output directory and their batch round is `syn`. The root/meta registration below does not apply to synthesis.

For a detached job owned directly by root/meta, register the exact receipt lock
and pid before ending the turn:

```bash
python -m autosft.agents.root_wait wait \
  --run-dir "${AUTO_SFT_RUN_DIR}" --round-id <id> \
  --job-label <label> --job-lock <lock_file> --job-pid <pid>
```

The harness monitors registered jobs without starting another model turn and
resumes the same root session when a job stops. A wake only means that the job
stopped; inspect `--status`, results, and logs before deciding success or
retrying. Re-register any still-running jobs after a wake.

Submit once, then ask `--status` once each time you are resumed and end the turn. Never sleep toward completion and never poll in a loop. Re-running the same command — with or without `--detach` — resumes the job, skipping the rows already answered.

**A job with failures must be re-run.** The aligned file is written only when every request has succeeded, so a job with one exhausted row sits at `N-1/N` forever and never produces its deliverable. `--status` reports those rows as `(<n> failed)`; re-run the same command to re-send just them.

## Persisted results

Every job writes to `${AUTO_SFT_RUN_DIR}/llm_calls/round_<id>/`:

- `<label>.prompts.jsonl` — the exact request body sent for every row, written before the first request goes out.
- `<label>.responses.jsonl` — attempt records in arrival order, including retries and failures; not an input-aligned answer file.
- `<label>.aligned.jsonl` — written only once every request has succeeded: one line per input row, in input order, carrying `ref`, `answer`, and the input digest. Read answers from this file, joined by `ref`, never from `responses.jsonl` by position. Its existence is the completeness check: if it is absent, the job is not done, and a partial join has no defined denominator.
- `<label>.manifest.json` — job identity, input digest, declared row scope, fixed request settings, and resume metadata.
- `<label>.events.jsonl` — retries, rate changes, progress, completion.
- `<label>.lock` — the process-held job lock; its path is returned as `lock_file` for a task's durable waiting interface. File existence alone does not indicate that the job is running or successful.

A `--label` belongs to one input; use a new label for a new question. Do not edit or delete persisted jobs, which may already support training data or reported measurements.

## Resources

- The [batch runner](../../../../src/autosft/backends/providers/batch.py) is the only supported way to call the configured provider; it writes the durable record.
- The credential is supplied by the launcher through the environment. Do not set, export, echo, or copy it, and do not read it out of `task.env`.
