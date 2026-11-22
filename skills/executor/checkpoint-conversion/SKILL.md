---
name: checkpoint-conversion
description: Export the complete final Qwen3 Megatron checkpoint to a local Hugging Face model.
---

# Checkpoint conversion

Collect the training task and confirm its final checkpoint marker. Conversion requires the canonical final step, the complete iteration directory and local HF template assets matching the fixed base model.

```bash
autosft tools convert --checkpoint "$AUTO_SFT_RUN_DIR/sft/round_1/checkpoints" --ckpt-step FINAL_STEP --output "$AUTO_SFT_RUN_DIR/ckpt_eval/round_1/step_FINAL_STEP/hf_model" --hf-template "$AUTO_SFT_SFT_TOKENIZER_MODEL"
```

Use the configured conversion interpreter. Conversion writes a hash-bound manifest and checks the exported model's required assets. Reuse an existing conversion only when its input identity matches and its model assets are complete. A failed or interrupted conversion must be recovered before evaluation. See [runtime setup](../../../docs/local-runtime.md) for dependencies.
