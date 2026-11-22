"""One independent component selection shared by every agent lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

COMPONENT_NAMES = ("executor", "analysis", "synthesis")


@dataclass(frozen=True)
class AgentComponents:
    executor: bool = False
    analysis: bool = False
    synthesis: bool = False

    @classmethod
    def parse(cls, value: str) -> AgentComponents:
        names = [] if not value or value == "none" else value.split(",")
        if len(names) != len(set(names)) or any(
            name not in COMPONENT_NAMES for name in names
        ):
            raise ValueError(
                "components must contain unique names from executor, analysis, synthesis, or none"
            )
        return cls(**{name: name in names for name in COMPONENT_NAMES})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in COMPONENT_NAMES if getattr(self, name))

    @property
    def name(self) -> str:
        return "_".join(self.names) or "baseline"


def components() -> AgentComponents:
    return AgentComponents.parse(os.environ.get("AUTO_SFT_COMPONENTS", "none"))


def composition_name() -> str:
    return components().name


def executor_enabled() -> bool:
    return components().executor


def analysis_enabled() -> bool:
    return components().analysis


def synthesis_enabled() -> bool:
    return components().synthesis


def root_prompt(repo_root: Path) -> str:
    selected = components()
    prompts = repo_root / "src/autosft/agents/prompts"
    parts = [
        (prompts / "meta.md").read_text(encoding="utf-8"),
        "Objective: "
        + os.environ.get(
            "AUTO_SFT_OBJECTIVE",
            "Improve the configured benchmarks through SFT data research.",
        ),
        "Fixed targets: "
        + os.environ.get(
            "AUTO_SFT_TARGET_SCORES", "Read the run protocol before acting."
        ),
    ]
    for name in selected.names:
        parts.append((prompts / f"meta_{name}.md").read_text(encoding="utf-8"))
    if not selected.executor:
        parts.append(
            "Executor is disabled, so you are the single agent responsible for the full training, task-lifecycle, checkpoint-conversion and evaluation workflow. Before each corresponding stage, read these skills: "
            f"{repo_root / 'skills/executor/training-submission/SKILL.md'}, "
            f"{repo_root / 'skills/executor/job-lifecycle/SKILL.md'}, "
            f"{repo_root / 'skills/executor/checkpoint-conversion/SKILL.md'}, and "
            f"{repo_root / 'skills/executor/evaluation-submission/SKILL.md'}. "
            "Read their artifacts before the next data decision."
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    from autosft.core.common import REPO_ROOT

    print(root_prompt(REPO_ROOT))
