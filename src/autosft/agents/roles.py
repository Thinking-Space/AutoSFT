"""The paper's four analysis roles and the dedicated executor."""

ANALYSIS_ROLES = ("training_data", "failure_case", "learning_dynamics", "history")
TASK_ROLES = ANALYSIS_ROLES
ANALYSIS_ROLE_SKILLS = {
    "training_data": "training-data",
    "failure_case": "failure-case",
    "learning_dynamics": "learning-dynamics",
    "history": "history",
}
EXECUTOR_CONTEXT_DELIMITER = b"\n\n## Task context from the meta agent\n\n"
ROLE_CATALOG = [
    {
        "role": role,
        "kind": "analysis",
        "purpose": purpose,
        "inputs": inputs,
        "delivery": "One executable data proposal with supporting evidence, expected effects and uncertainty.",
    }
    for role, purpose, inputs in (
        (
            "training_data",
            "Examine training-data quality, difficulty, coverage and redundancy.",
            "Source pool and realized training data.",
        ),
        (
            "failure_case",
            "Analyze successful and failed model behavior.",
            "Per-example evaluation outputs and verdicts.",
        ),
        (
            "learning_dynamics",
            "Read training signals against the data that produced them.",
            "Loss curves, token exposure and training manifests.",
        ),
        (
            "history",
            "Compare earlier interventions and revisit insufficiently tested ideas.",
            "Registered hypotheses, verdicts, lessons and run history.",
        ),
    )
] + [
    {
        "role": "executor",
        "kind": "execution",
        "purpose": "Run the fixed training, conversion and evaluation lifecycle.",
        "inputs": "Meta-owned recipe and executor handoff.",
        "delivery": "Artifact paths and a short structured execution report.",
    }
]
