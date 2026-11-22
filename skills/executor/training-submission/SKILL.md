---
name: training-submission
description: Submit fixed-protocol SFT from the meta agent's realized data recipe.
---

# Training submission

Read the run's formal recipe and existing submission receipts. The meta agent owns data and interpretation; the executor operates the fixed training backend. Never change the data, base checkpoint, training parameters, token cap or resources during execution.

```bash
autosft tools train --recipe "$AUTO_SFT_RUN_DIR/data_rounds/round_1/data-recipe.yaml" --run-name round_1 --run-dir "$AUTO_SFT_RUN_DIR"
```

Use a canonical `round_<n>` run name matching the recipe. Preparation validates every row, provenance, earlier-round disposition and token accounting. Normal rounds initialize from the configured base checkpoint and train for three epochs. Recover a live or completed matching task instead of submitting another one.

Read [job lifecycle](../job-lifecycle/SKILL.md) to monitor and collect the task. Convert only its complete final checkpoint, then evaluate it with the fixed benchmark protocol. Return artifact paths and a short execution report rather than raw logs or data recommendations.
