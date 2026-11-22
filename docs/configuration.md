# Configuration

An experiment is one JSON file composed from named blocks: objective, resources, training, evaluation, data, agent, providers and components. A file may `extends` another file relative to its own location; nested objects merge, and lists replace the inherited list. `autosft run --config ... --dry-run` prints the fully resolved public protocol before anything is launched.

## Component composition

`components` is a list drawn from `executor`, `analysis` and `synthesis`. An empty list is the single-agent baseline. Each component is independent: enabling the executor does not enable analysis, and enabling synthesis does not change either of them. The same selection reaches startup, continuation, task dispatch, permission checks and synthesis publication. At runtime it is stored as `AUTO_SFT_COMPONENTS`; use the config or `--components` at launch to select it.

```bash
bash scripts/run.sh --config configs/paper/analysis.json --components none --dry-run
bash scripts/run.sh --config configs/paper/analysis.json --components executor --dry-run
bash scripts/run.sh --config configs/paper/analysis.json --components executor,analysis --dry-run
```

The objective contains a natural-language description and target benchmark scores. Resources contain a training-token cap, a maximum round count and the number of training GPUs. Training contains the backend, base model, epochs, sequence length, optimizer and batch geometry. Evaluation binds the benchmark set, generation backend, decoding protocol and repeat count. The data block selects a local staged pool. Future optimization domains can add their own preparation and backend implementation while retaining this configuration and orchestration boundary.

## Paper settings

The shared block in `configs/paper/shared.json` follows Table 5: Qwen3-1.7B-Base; eight training GPUs; three epochs; global/micro batch 16/1; Adam with betas 0.9/0.95, epsilon 1e-8 and weight decay 0.1; learning rate 1e-4 to 1e-7 with cosine decay and 5% warmup; and zero-shot evaluation at temperature 0.7. Paper presets use local vLLM generation with OpenCompass instruction-following scorers or EvalPlus code scoring. `evaluation.backend` selects `opencompass_vllm` for instruction following and `evalplus_vllm` for paper code experiments; custom code experiments may use `evalplus_sglang`. Unsupported benchmark/backend combinations and prompting settings are rejected before launch.

| Comparison | Configurations | Token cap | Sequence length | Evaluation |
| --- | --- | ---: | ---: | --- |
| Executor (§5.1) | `executor-baseline.json`, `executor.json` | 10M | 32,768 | IFEval and IFBench |
| Analysis (§5.2) | `analysis-baseline.json`, `analysis.json` | 3M | 32,768 | IFEval, avg@10 |
| Synthesis (§5.3) | `synthesis-baseline.json`, `synthesis.json` | 100M | 32,768 | IFEval and IFBench, avg@10 |
| Code (§5.4) | `code-baseline.json`, `code-analysis.json`, `code-synthesis.json` | 100M | 8,192 | MBPP+, avg@3 |

The analysis, synthesis and code rows correspond to Table 6. The executor row uses the pilot settings in §3.1; its repeat count in this release is ten. Configured maximum round counts bound runnable searches. The paper also compares snapshots under a common elapsed time, so these caps alone do not reproduce the snapshot selection. Reported paper numbers are shown in the README; the repository does not bundle the original search traces or data snapshots.

## Local assets

Copy `.env.example` to `.env` and supply the local HF model directory, a matching Megatron base checkpoint and your API settings. Activate your Megatron environment; preparation, training, conversion and evaluation use that Python by default. Optional stage-specific interpreter overrides are described in [runtime setup](local-runtime.md). `scripts/run.sh` loads the ignored file before resolving the experiment. The standalone `autosft round` command imports its base checkpoint automatically. Agent runs reuse a prepared base checkpoint.

A run stores the resolved protocol in `experiment.json`, fixed runtime settings in owner-only `task.env`, and durable datasets, receipts and scores under `runs/`. Do not publish a run directory directly: it contains source data and may contain credentials in agent session state. Change the protocol by starting a new run, rather than editing its fixed settings.

## Validation configuration

`configs/examples/smoke-agent.json` binds a two-round, one-GPU check to the bundled four-row pool. Its `evaluation.max_seq_len` and `evaluation.max_out_len` override the corresponding OpenCompass generation limits; paper presets retain their existing defaults. Use the complete procedure in [agent run](agent-run.md#a-bounded-agent-validation-run) to inspect workload completion separately from the benchmark target.
