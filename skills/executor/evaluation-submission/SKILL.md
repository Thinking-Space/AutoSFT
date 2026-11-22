---
name: evaluation-submission
description: Evaluate a complete converted checkpoint using the run's fixed benchmarks and repeat count.
---

# Evaluation submission

Confirm training and conversion have completed, recover existing evaluation task receipts, and use the fixed benchmark set. Do not select benchmarks, inspect private answers or alter decoding settings to influence the run.

```bash
autosft tools evaluate --model-path "$AUTO_SFT_RUN_DIR/ckpt_eval/round_1/step_FINAL_STEP/hf_model" --benchmark "$AUTO_SFT_BENCHMARKS" --run-name round_1 --repeats "$AUTO_SFT_EVAL_REPEATS"
```

The model path must contain complete local HF assets. Each benchmark receives one task covering all required repeats. Use the job lifecycle skill to monitor and collect every task. A round is complete only after all configured benchmarks have full results; limited smoke evaluations do not qualify for finalization. Return their artifact paths and lifecycle status to the meta agent.
