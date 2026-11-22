# Architecture

AutoSFT is the first implementation of RSI-Service: a research service between a human objective and the code that trains and evaluates a model. The current research policy changes SFT training data while the base model, trainer and evaluation protocol stay fixed. The service owns objectives, resource limits, durable experiments and agent coordination; data optimization is its first domain.

## Package boundaries

The Python package lives in `src/autosft`. `core` contains shared paths and artifact utilities. `config` resolves objectives, resource limits, backends and independently selected agent components. `orchestration` owns experiment history, submission records, round gates and completion evidence. `agents` owns sessions, task dispatch and executor handoffs. `data` owns pool preparation, recipes, token accounting and retrospective intervention analysis. `backends` contains local task supervision and the public Megatron integration. `benchmarks` contains scoring and evaluation artifacts. `synthesis` owns generation, validation and publication.

The main commands are `autosft run`, `autosft datasets` and `autosft check`; `autosft round` runs a standalone pipeline, and `autosft tools` groups individual stages and analysis utilities. Shell launchers and environment setup live in `scripts`. Python workload implementations live under the corresponding package backends and data modules. Experiment presets live in `configs`, and runnable fixtures live in `examples`. Downloaded pools, model weights, environments and run artifacts are local assets excluded from Git. Skills contain procedures and references: shared row-analysis, pool-rewrite and recorded-call materials are under `skills/shared`, while role procedures are grouped under `skills/meta`, `skills/executor` and `skills/analysis`. The synthesis worker's model calls use the recorded batch runner; the shared recorded-call reference under `skills/shared/batch-inference/references/` is also used by the shared data operators.

## Research policy and roles

The meta agent owns the final decision and assembled training data. An optional executor runs training, checkpoint conversion and evaluation in its own session, returning artifact paths and a short report. Optional analysis subagents examine training data, failure cases, learning dynamics and research history. They use the same proposal contract and differ in their evidence source. An optional synthesis subagent produces supervision for undersupplied capabilities. Its deliveries are checked before publication, and the meta agent decides which accepted rows enter training.

Executor, analysis and synthesis are independent components in one configuration. A baseline has none; adding the executor does not enable analysis. Synthesis does not change the execution or analysis setting. The enabled components, fixed objective, training protocol and evaluation protocol are recorded for each run.

## Data and evidence

A recipe points to the realized ChatML JSONL dataset and records the data decision and provenance. It does not prescribe an action vocabulary. Mixture, pruning, repair, resupervision, synthesis and ordering are retrospective labels for research analysis, available through a separate annotation command. These labels are not included in agent prompts or used as an admission gate.

Preparation validates all rows, records token exposure and enforces the configured token cap. Normal rounds initialize from the same base checkpoint. Synthesis publication requires valid unique rows, benchmark decontamination and a digest-bound quality inspection. Hypotheses precede training; outcomes and lessons remain durable records, with score variation distinguished from evidence for a proposed mechanism.

## Backend and domain extensions

Future optimization domains should implement their own proposal artifacts and preparation checks while using the same objective, run history and task lifecycle. A trainer or evaluation harness is a configured backend with explicit input artifacts and output receipts. New domains should not introduce implicit SFT-specific restrictions into the orchestration layer. The first release ships the SFT domain and does not implement speculative weight, algorithm or harness optimization engines.

Megatron-LM is fetched from a pinned public upstream commit. AutoSFT's dataset adapter and Qwen3 checkpoint conversion remain in its own backend package; upstream source is neither silently modified nor presented as AutoSFT-owned code. Third-party benchmark code retains its attribution and licenses.
