# Skill library

Skills are procedures grouped by the role that owns them, following paper Table 3. The agent chooses their sequence. Runtime implementations live in the package and scripts; skill text describes evidence, boundaries and artifact contracts.

| Role | Skills |
| --- | --- |
| Meta agent | [Recipe authoring](meta/recipe-authoring/SKILL.md), [synthesis handbook](meta/synthesis-handbook/SKILL.md) |
| Executor | [Training submission](executor/training-submission/SKILL.md), [job lifecycle](executor/job-lifecycle/SKILL.md), [checkpoint conversion](executor/checkpoint-conversion/SKILL.md), [evaluation submission](executor/evaluation-submission/SKILL.md) |
| Analysis | [Training-data examination](analysis/training-data/SKILL.md), [failure-case analysis](analysis/failure-case/SKILL.md), [learning-dynamics analysis](analysis/learning-dynamics/SKILL.md), [history analysis](analysis/history/SKILL.md) |
| Shared data operators | [Row analysis](shared/row-analysis/SKILL.md), [pool rewrite](shared/pool-rewrite/SKILL.md) |
| Synthesis | [Batch generation](synthesis/batch-generation/SKILL.md) |

The row-analysis and pool-rewrite operators are grouped under `skills/shared` because they document shared model-call transport and data operations. They are available to the four analysis agents within their own task directories, while the meta agent decides whether to promote their candidates into the pool or formal training data. The synthesis worker's generation, review and repair calls use `skills/synthesis/batch-generation`, which documents the recorded batch runner. The shared recorded-call transport is documented in `skills/shared/batch-inference/references/recorded-calls.md`. Skills do not provide the retrospective data-action taxonomy as an agent operator menu. See [role dispatch](meta/recipe-authoring/references/role-dispatch.md) for task identities and proposal contracts.
