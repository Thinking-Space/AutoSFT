# AutoSFT

AutoSFT is an agentic research service for supervised fine-tuning data and the first implementation of RSI-Service. Its meta agent plans data experiments; optional executor, analysis and synthesis components share a fixed objective, resource budget and experiment history.

This file applies throughout the repository. The development sections govern explicitly requested repository maintenance. During an AutoSFT experiment, follow the experiment-session section and the active role prompt; development guidance does not authorize editing the implementation during a run. More specific AGENTS.md files may provide directory-level instructions.

## Setup and commands

Activate a compatible Megatron environment, install AutoSFT with `pip install -e .`, and copy `.env.example` to the ignored `.env` for local model paths and API settings. Use the public Megatron commit selected by `scripts/setup/fetch_megatron.py`. See [runtime setup](docs/local-runtime.md) for GPU dependencies and base-checkpoint import.

```bash
# Check the checkout and resolve an experiment without starting workloads.
autosft check --agents --synthesis
bash scripts/run.sh --config configs/paper/analysis.json --dry-run

# Run the bundled small experiment after configuring local assets.
bash scripts/run.sh --config configs/examples/smoke-agent.json
```

The main commands are `autosft run`, `datasets`, `check` and `round`; individual pipeline and analysis operations are under `autosft tools`. Start an agent experiment with `scripts/run.sh` so local settings are loaded consistently.

## Repository layout

`src/autosft/config` resolves protocols and component selection; `core` contains shared paths and artifact identities; `agents` owns prompts, sessions and handoffs; `orchestration` owns history, round gates and completion evidence; `data` owns pools, recipes and token preparation; `backends` owns task execution, public Megatron and model-provider calls; `benchmarks` owns evaluation; and `synthesis` owns generated supervision and publication.

`scripts` contains launch and setup entrypoints. `skills` contains role procedures and references, with shared data operators and batch-transport materials under `skills/shared`, and role procedures under `skills/meta`, `skills/executor` and `skills/analysis`. The synthesis worker's model calls use the recorded batch runner documented under `skills/synthesis`. Runtime implementations live in the package. `configs/paper` binds the paper's experiment protocols, `examples` supplies runnable sample data, and `docs` explains the architecture and complete agent run. Model weights, downloaded pools, credentials and run state stay in ignored local directories.

## Development conventions

Keep changes focused on the requested behavior. Use the paper's role names and the existing independent `components` selection. Put implementation logic in its owning package module and keep launchers small. When moving a command or file, update its callers, prompts, skills, checks and documentation together; remove obsolete entrypoints.

Preserve training-data encoding, packing, loss-mask semantics, normal-round initialization and benchmark scoring unless the requested change explicitly covers them. Avoid speculative extension frameworks; new optimization domains should share objectives, task lifecycle and history while owning their own preparation and execution rules.

Write docstrings and comments to explain non-obvious decisions and artifact contracts. Use complete Markdown paragraphs and describe the current release. Keep API keys and machine-specific paths out of tracked files, and retain third-party attribution and licenses.

## Validation

Check Python and shell syntax and exercise the changed commands with `--help` or `--dry-run`. Run `autosft check --agents --synthesis` after changing paths or bundled assets. Validate affected experiment presets, including independent component selections. For training, conversion, evaluation or agent-lifecycle changes, use the bounded smoke configuration with the required local assets and inspect the resulting task receipts and complete round artifacts. See [agent-run validation](docs/agent-run.md#a-bounded-agent-validation-run) for the expected outputs.

## Code Review Rules

Flag changes that drop role identity or pinned task inputs, duplicate submissions during recovery, accept partial evaluations, or finalize scores from different checkpoints. Check that enabled components, configured models and fixed training/evaluation settings reach all their consumers. Review prompt and skill paths as executable dependencies. Treat score changes as observations; require additional evidence for causal conclusions.

## Experiment sessions

The current domain changes data while keeping the base model, training algorithm, training parameters, token cap and evaluation protocol fixed. The following rules apply to experiment sessions, including the meta agent and its subagents.

### Paths and data

`AUTO_SFT_REPO_ROOT` identifies the checkout, and `AUTO_SFT_RUN_DIR` identifies the current run. Source datasets are staged into the writable run pool with file hashes and row counts. Write scratch work and round artifacts inside the run; never change the source dataset directory outside it or repository code during an experiment. Use run-local evidence for data decisions.

Each round writes `data_rounds/round_<n>/final_data.jsonl` and `data-recipe.yaml`. Training data contain a non-empty `messages` list with string roles and content, including at least one non-empty assistant response. The recipe supplies `recipe_id`, `round_id`, one `datasets` entry with its path, `execution_summary`, and provenance bookkeeping through `rewrote_pool` and, when applicable, `rewritten_row_ids_file`. Explain what entered training and why. Register a hypothesis through the ledger before submission.

Preparation validates every row, confines paths to the run, records token exposure and enforces the fixed token cap. A rejected attempt may be corrected under a fresh recipe id. Do not override the budget or training protocol. Normal rounds initialize from the configured base model and train for three epochs.

### Roles

The meta agent owns the realized dataset, recipe, final decision and result interpretation. Enabled components are recorded in `experiment.json` and `AUTO_SFT_COMPONENTS`. The executor runs training, conversion and evaluation in a separate session and returns artifact paths with a short execution report. The four analysis subagents examine training data, failure cases, learning dynamics and history, each returning one evidence-backed executable proposal. The synthesis subagent produces supervision for undersupplied capabilities; verified, decontaminated and quality-inspected deliveries become available for selection.

Read the role's skill under `skills/meta`, `skills/executor` or `skills/analysis`; shared row-analysis, pool-rewrite and batch-transport materials live under `skills/shared`, while the synthesis worker reads its procedure under `skills/synthesis` and calls the recorded batch runner. Dispatch analysis tasks through `autosft.agents.search_tasks` using task-specific ids and immutable input references. Delegate the executor with its round context and handoff. The harness owns session polling and recovery. Disabled components must not be launched.

### Evidence and execution

Keep task inputs stable and reuse evidence only when its content hashes match. Recover durable task ids and submission records before retrying. Never evaluate a checkpoint while its training task is running. Read complete per-repeat evaluations; missing or partial results are unknown. Resolve a failed earlier round or record its digest-bound disposition before preparing another round.

Record hypotheses before training and retain the resulting observations, verdicts and lessons. A higher score does not by itself prove the proposed mechanism worked; compare evaluation variation and the finer observation predicted by the hypothesis. Finalize only when one checkpoint meets all configured targets with complete evaluations. Maxima from different checkpoints do not combine.

Use the documented submission and lifecycle commands. Do not change model assets, hardware settings, benchmark answers or backend configuration during a run. Model calls for data operations go through the recorded provider runners, using the configured model and endpoint. Keep credentials in local environment variables and owner-only run state. Do not use external dataset discovery or other runs as experiment inputs.
