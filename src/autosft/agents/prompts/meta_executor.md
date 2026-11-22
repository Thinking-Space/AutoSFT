# Executor component

Delegate training, checkpoint conversion and evaluation to the executor using a round-specific `executor.task_context.md` and the durable handoff. Read `skills/executor/training-submission/SKILL.md` and the role dispatch reference. The harness launches and resumes the executor; it returns artifact paths and a short report, keeping raw execution logs in its own context. You retain the data decision and interpretation of results.
