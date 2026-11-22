"""Load the realized dataset envelope and validate fixed run constraints.

The recipe names the assembled ChatML data and records the decision and
provenance. It does not define an operation menu. Retrospective data-action
labels live in data.action_analysis and never gate recipe preparation.
Canonical digests bind the executor, ledger and policy to the same artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from autosft.core.common import load_structured
from autosft.core.round_ids import RoundIdError, logical_round_id


# Fields in the realized-data envelope; the launcher owns the token cap.
MINIMAL_RECIPE_KEYS = frozenset(
    {
        "recipe_id",
        "round_id",
        "datasets",
        "execution_summary",
        "rewrote_pool",
        "rewritten_row_ids_file",
    }
)
# Where a round's rewritten-row-ids file must live. Duplicated from
# ledger.RECIPE_HISTORY_DIRNAME rather than imported to keep this module free
# of the ledger's dependency tree.
_RECIPE_HISTORY_DIRNAME = "recipe_history"
# The launcher owns the round token cap; a recipe naming its own budget would
# move the budget gate itself, so the rejection message calls that out.
_BUDGET_KEY = "token_budget"


class RecipeValidationError(ValueError):
    """Raised when a minimal recipe violates the strict open-pool schema."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def recipe_sha256(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(recipe).encode("utf-8")).hexdigest()


def resolve_dataset_path(recipe: dict[str, Any], *, run_dir: Path) -> Path:
    """Resolve the recipe's single dataset path against the run directory.

    Shared by validation and preparation so path confinement and the
    materialization step can never disagree about which file the recipe
    names. Assumes the recipe already passed ``validate_minimal_recipe``.
    """
    raw = recipe["datasets"][0]["path"]
    path = Path(os.path.expandvars(str(raw))).expanduser()
    if not path.is_absolute():
        path = Path(run_dir) / path
    return path.resolve(strict=False)


def _validate_datasets(datasets: Any, *, run_dir: Path | None) -> None:
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise RecipeValidationError(
            "recipe.datasets must contain exactly one entry naming the "
            "round's final_data.jsonl"
        )
    entry = datasets[0]
    if not isinstance(entry, dict):
        raise RecipeValidationError("recipe.datasets[0] must be a mapping")
    extra = sorted(set(entry) - {"path"})
    if extra:
        raise RecipeValidationError(
            "recipe.datasets[0] must have exactly one key (path); "
            "remove: " + ", ".join(extra)
        )
    path = entry.get("path")
    if not isinstance(path, str) or not path.strip():
        raise RecipeValidationError(
            "recipe.datasets[0].path must be a non-empty path to the "
            "assembled final_data.jsonl"
        )
    if run_dir is not None:
        run_root = Path(run_dir).expanduser().resolve(strict=False)
        resolved = resolve_dataset_path({"datasets": [entry]}, run_dir=run_root)
        if run_root not in resolved.parents:
            raise RecipeValidationError(
                f"recipe.datasets[0].path must resolve under the run dir "
                f"{run_root}; got {resolved}"
            )


def resolve_rewritten_row_ids_file(
    recipe: dict[str, Any], *, run_dir: Path
) -> Path | None:
    """Resolve the recipe's rewritten-row-ids file against the run directory.

    Returns None when the round rewrote nothing. Assumes the recipe already
    passed ``validate_minimal_recipe``.
    """
    raw = recipe.get("rewritten_row_ids_file")
    if raw is None:
        return None
    path = Path(os.path.expandvars(str(raw))).expanduser()
    if not path.is_absolute():
        path = Path(run_dir) / path
    return path.resolve(strict=False)


def _validate_rewrite_fields(recipe: dict[str, Any], *, run_dir: Path | None) -> None:
    """Validate the per-round rewrite record.

    ``rewrote_pool`` is required on every recipe: the round states whether it
    rewrote pool rows. When true, ``rewritten_row_ids_file`` must name a JSON
    file under ``recipe_history/`` holding the rewritten rows' row ids; when
    false, the pointer must be absent.
    """
    if "rewrote_pool" not in recipe:
        raise RecipeValidationError(
            "recipe.rewrote_pool is required: state whether this round "
            "rewrote pool rows (true) or not (false)"
        )
    rewrote = recipe.get("rewrote_pool")
    if not isinstance(rewrote, bool):
        raise RecipeValidationError("recipe.rewrote_pool must be a boolean")
    pointer = recipe.get("rewritten_row_ids_file")
    if not rewrote:
        if pointer is not None:
            raise RecipeValidationError(
                "recipe.rewritten_row_ids_file must be omitted when "
                "rewrote_pool is false"
            )
        return
    if not isinstance(pointer, str) or not pointer.strip():
        raise RecipeValidationError(
            "recipe.rewritten_row_ids_file is required when rewrote_pool is "
            "true: a path to the JSON file under recipe_history/ holding the "
            "rewritten rows' row ids"
        )
    if run_dir is None:
        return
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    resolved = resolve_rewritten_row_ids_file(recipe, run_dir=run_root)
    assert resolved is not None
    history_root = (run_root / _RECIPE_HISTORY_DIRNAME).resolve(strict=False)
    if history_root not in resolved.parents:
        raise RecipeValidationError(
            f"recipe.rewritten_row_ids_file must resolve under "
            f"{history_root}; got {resolved}"
        )
    if resolved.suffix.lower() != ".json":
        raise RecipeValidationError(
            f"recipe.rewritten_row_ids_file must be a .json file; got {resolved}"
        )
    if not resolved.is_file():
        raise RecipeValidationError(
            f"recipe.rewritten_row_ids_file does not exist: {resolved}. "
            "Write the row-id file before the recipe that points at it."
        )
    try:
        content = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecipeValidationError(
            f"recipe.rewritten_row_ids_file is not valid JSON: {resolved}: {exc}"
        ) from exc
    if not content:
        raise RecipeValidationError(
            f"recipe.rewritten_row_ids_file is empty: {resolved}. A round "
            "that rewrote nothing records rewrote_pool: false instead."
        )


def validate_minimal_recipe(
    recipe: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> None:
    """Enforce the strict minimal-recipe schema (open-pool contract).

    ``run_dir`` additionally confines the dataset path to the run directory;
    pass it whenever the run root is known.
    """
    if not isinstance(recipe, dict):
        raise RecipeValidationError("recipe must be a mapping")
    if _BUDGET_KEY in recipe:
        raise RecipeValidationError(
            "recipe must not set token_budget; the launcher-owned "
            "AUTO_SFT_SFT_TOKEN_BUDGET is the only budget source"
        )
    unknown = sorted(set(recipe) - MINIMAL_RECIPE_KEYS)
    if unknown:
        raise RecipeValidationError(
            "recipe contains unsupported top-level keys: "
            + ", ".join(unknown)
            + "; the minimal schema allows only "
            + ", ".join(sorted(MINIMAL_RECIPE_KEYS))
        )
    recipe_id = recipe.get("recipe_id")
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise RecipeValidationError(
            "recipe.recipe_id must be a non-empty string, unique per attempt"
        )
    try:
        logical_round_id(recipe.get("round_id"), label="recipe.round_id")
    except RoundIdError as exc:
        raise RecipeValidationError(str(exc)) from exc
    _validate_datasets(recipe.get("datasets"), run_dir=run_dir)
    summary = recipe.get("execution_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RecipeValidationError(
            "recipe.execution_summary must be non-empty free text describing "
            "what was done to the data and why"
        )
    _validate_rewrite_fields(recipe, run_dir=run_dir)


def load_minimal_recipe(path: Path) -> dict[str, Any]:
    """Load one recipe file and validate it against the minimal schema.

    Path confinement needs the run root, so callers that know it should
    additionally run ``validate_minimal_recipe(recipe, run_dir=...)``.
    """
    recipe = load_structured(Path(path))
    validate_minimal_recipe(recipe)
    return recipe
