#!/usr/bin/env python3
"""Cross-iteration experiment ledger for AutoSFT agent runs.

The guarded run dir reuses single-file artifacts (`data-recipe.yaml`,
`data_manifest.json`, `run_manifest.json`) while each round writes its own
`sft/round_<id>/checkpoints/`. The append/merge ledger pairs each iteration's
**recipe** with its **result**:

- `recipe_history/iter_<NNN>.yaml` + `.data_manifest.json` — an immutable copy
  of the exact recipe used for each round (the live `data-recipe.yaml` keeps
  being overwritten; these snapshots do not).
- `iterations.jsonl` — one JSON object per iteration, filled progressively:
  recipe summary + hyperparameters + fixed base initialization at SFT submit time,
  training health + eval scores at eval-collect time, and an optional free-text
  `rationale`, `expectation`, and `evidence_plan` supplied by the meta agent, with
  append-only note and review histories.
- `iterations.md` — a compact, human/agent-readable table regenerated from
  `iterations.jsonl` so the agent can eyeball the whole trajectory before
  choosing the next recipe.

The objective fields are populated automatically by `src/autosft/backends/megatron/train.py` and
`autosft tools jobs eval collect`; a subjective `rationale`, when used, is added by
the agent via `note`. A note written before the first SFT submission is staged
and automatically attached when that iteration is opened. Reading is via
`summarize`.

`hypothesis_history.jsonl` keeps optional immutable statements and append-only
updates and judgments. Timestamps and phases describe recorded ledger state,
not proof of preregistration. Missing notes, reviews, and hypotheses never
prevent submission or executor recovery.

CLI:
    python -m autosft.orchestration.ledger summarize  --run-dir <run_dir>
    python -m autosft.orchestration.ledger record-eval --run-dir <run> --eval-manifest <eval_run>/eval_successful_submissions.jsonl
    python -m autosft.orchestration.ledger note --run-dir <run_dir> --rationale "..." \
      --expectation "..." --evidence-plan "..." --deviation "..."
    python -m autosft.orchestration.ledger review --run-dir <run_dir> --iteration 1 --review "..."
    python -m autosft.orchestration.ledger hypothesis record --run-dir <run_dir> \
      --round-id <round> --hypothesis "..." --planned-action "..." \
      --reason "..." --baseline-round <round> --expected-direction up|flat|down|inconclusive \
      --expected-effect "..." --mechanism-observable "..."
    python -m autosft.orchestration.ledger hypothesis judge --run-dir <run_dir> \
      --round-id <round> --score-verdict up|flat|down|inconclusive --score-basis "..." \
      --mechanism-verdict observed|not_observed|mixed|inconclusive \
      --mechanism-basis "..." --lesson "..." --verdict-round <round>
    python -m autosft.orchestration.ledger hypothesis update --run-dir <run_dir> \
      --hypothesis-id <id> --status open|supported|refuted|retired --evidence "..." --at-round 1
    python -m autosft.orchestration.ledger hypothesis status --run-dir <run_dir>
    python -m autosft.orchestration.ledger best-scores --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from autosft.benchmarks.evalplus import CODE_BENCHMARKS, EVALPLUS_EXPECTED_DETAILS

import autosft.benchmarks.aggregate as eval_aggregate
from autosft.core.common import append_jsonl, utc_now, write_json
from autosft.agents.context_audit import write_context_audit
from autosft.core.round_ids import (
    RoundIdError,
    logical_id_from_storage,
    logical_round_id,
    storage_round_id,
)
from autosft.orchestration.submission_registry import (
    model_identity as submission_model_identity,
    record_repeat_indices,
    record_repeats,
    submission_task,
)

LEDGER_NAME = "iterations.jsonl"
RECIPE_HISTORY_DIRNAME = "recipe_history"
SUMMARY_NAME = "iterations.md"
PENDING_RATIONALE_NAME = "pending_rationale.json"
HYPOTHESIS_HISTORY_NAME = "hypothesis_history.jsonl"

# The two verdict vocabularies of `hypothesis judge`. The score verdict says
# what the benchmark mean did relative to the round's own baseline, read
# against the eval-repeat spread; the mechanism verdict says whether the
# fine-grained phenomenon the round predicted actually showed up in the eval
# artifacts. The two are judged independently: a round can move the score
# without its mechanism (luck or a confound) or show its mechanism without
# moving the score (the targeted slice was too small, or losses elsewhere
# cancelled the gain), and both half-results are exactly what the next
# round needs to know. `inconclusive` records insufficient evidence, including
# a planned intervention that was not realized in the training data.
SCORE_VERDICTS = ("up", "flat", "down", "inconclusive")
MECHANISM_VERDICTS = ("observed", "not_observed", "mixed", "inconclusive")
HYPOTHESIS_STATUSES = ("open", "supported", "refuted", "retired")

# Hypotheses can predict a cost or seek information without a score prediction.
EXPECTED_DIRECTIONS = ("up", "flat", "down", "inconclusive")


def ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / LEDGER_NAME


def recipe_history_dir(run_dir: Path) -> Path:
    return Path(run_dir) / RECIPE_HISTORY_DIRNAME


def summary_path(run_dir: Path) -> Path:
    return Path(run_dir) / SUMMARY_NAME


def pending_rationale_path(run_dir: Path, round_id: str | int | None = None) -> Path:
    if round_id is not None:
        return (
            recipe_history_dir(run_dir)
            / f"round_{logical_round_id(round_id)}.pending_rationale.json"
        )
    return Path(run_dir) / PENDING_RATIONALE_NAME


def hypothesis_history_path(run_dir: Path) -> Path:
    return Path(run_dir) / HYPOTHESIS_HISTORY_NAME


def default_run_dir() -> Path | None:
    value = os.environ.get("AUTO_SFT_RUN_DIR")
    return Path(value) if value else None


def _require_ledger_writer() -> None:
    roles = [
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    ]
    if (
        os.environ.get("AUTO_SFT_TASK_DIR")
        or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")
    ) or any(role and role not in {"meta", "executor"} for role in roles):
        raise ValueError("task sessions cannot write the meta-owned experiment ledger")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


# --------------------------------------------------------------------------- #
# Ledger read / write
# --------------------------------------------------------------------------- #
def read_iterations(run_dir: Path) -> list[dict[str, Any]]:
    path = ledger_path(run_dir)
    if not path.exists():
        return []
    iterations: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in experiment ledger {path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"experiment ledger record must be an object: {path}:{line_number}"
            )
        iterations.append(record)
    iterations.sort(key=lambda it: it.get("iteration", 0))
    return iterations


def write_iterations(run_dir: Path, iterations: list[dict[str, Any]]) -> None:
    _require_ledger_writer()
    path = ledger_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            for record in sorted(iterations, key=lambda it: it.get("iteration", 0)):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _next_index(iterations: list[dict[str, Any]]) -> int:
    return max((it.get("iteration", 0) for it in iterations), default=0) + 1


def _find_target(
    iterations: list[dict[str, Any]], iteration: int | None
) -> dict[str, Any] | None:
    if not iterations:
        return None
    if iteration is not None:
        for record in iterations:
            if record.get("iteration") == iteration:
                return record
        return None
    return max(iterations, key=lambda it: it.get("iteration", 0))


def _recording_phase(record: dict[str, Any] | None) -> str:
    """Describe recorded state, not proof of when an idea was conceived."""
    if record is None:
        return "before_submission"
    if record.get("eval") or record.get("training_health"):
        return "after_result"
    return "after_submission"


def rounds_awaiting_eval(
    run_dir: Path,
    *,
    benchmarks: list[str] | None = None,
    repeats: int | None = None,
) -> list[str]:
    """Logical ids of submitted rounds whose evaluation is not yet recorded.

    Every configured benchmark, repeat and aggregate must be present. The
    first benchmark to arrive must not release a partially evaluated round.

    An unreadable ledger or invalid configuration raises: silently treating
    either as no pending work could advance the run past missing results.
    """
    from autosft.orchestration.round_gates import evaluation_problems

    iterations = read_iterations(run_dir)
    pending: set[str] = set()
    for record in iterations:
        if not evaluation_problems(
            record, run_dir, benchmarks=benchmarks, repeats=repeats
        ):
            continue
        try:
            round_id = record_round_id(record)
        except (ValueError, RoundIdError):
            continue
        if round_id is not None:
            pending.add(round_id)
    return sorted(pending, key=int)


def record_round_id(record: dict[str, Any]) -> str | None:
    """Resolve a logical round independently of its submission-order iteration."""
    recipe = record.get("recipe")
    values = [record.get("round_id")]
    if isinstance(recipe, dict):
        values.append(recipe.get("round_id"))
    identities = {logical_round_id(value) for value in values if value is not None}
    if record.get("run_name") is not None:
        identities.add(logical_id_from_storage(record["run_name"], label="run_name"))
    if len(identities) > 1:
        raise ValueError(
            f"conflicting round identifiers in iteration {record.get('iteration')}"
        )
    return next(iter(identities), None)


def _find_round(
    iterations: list[dict[str, Any]], round_id: str
) -> dict[str, Any] | None:
    matches = [record for record in iterations if record_round_id(record) == round_id]
    if len(matches) > 1:
        raise ValueError(f"multiple ledger iterations identify round {round_id}")
    return matches[0] if matches else None


def _submission_note(
    run_dir: Path, round_id: str, iteration: int
) -> tuple[Path, dict[str, Any]] | None:
    # A round-bound plan must never be consumed by another round submitted first.
    for path in (
        pending_rationale_path(run_dir, round_id),
        pending_rationale_path(run_dir),
    ):
        try:
            pending = _read_json_if_exists(path)
        except (OSError, ValueError) as exc:
            print(
                f"ledger: optional pending note was not attached: {exc}",
                file=sys.stderr,
            )
            continue
        if pending is None:
            continue
        if path == pending_rationale_path(run_dir, round_id):
            if str(pending.get("round_id")) != round_id:
                print(
                    f"ledger: pending note has a mismatched round identity: {path}",
                    file=sys.stderr,
                )
                continue
        else:
            if (
                pending.get("round_id") is not None
                and str(pending["round_id"]) != round_id
            ):
                continue
            if pending.get("iteration", iteration) != iteration:
                continue
        if path != pending_rationale_path(run_dir):
            legacy_path = pending_rationale_path(run_dir)
            if legacy_path.exists():
                print(
                    f"ledger: round-bound note takes precedence; separate iteration note remains at {legacy_path}",
                    file=sys.stderr,
                )
        return path, pending
    return None


# --------------------------------------------------------------------------- #
# Recipe snapshotting
# --------------------------------------------------------------------------- #
def _recipe_summary(data_manifest: dict[str, Any] | None) -> dict[str, Any]:
    manifest = data_manifest or {}
    datasets = []
    for spec in manifest.get("datasets", []) or []:
        if not isinstance(spec, dict):
            continue
        datasets.append(
            {
                "path": spec.get("path"),
                "rows": spec.get("rows"),
                "total_tokens": spec.get("total_tokens"),
                "sha256": spec.get("sha256"),
            }
        )
    return {
        "recipe_id": manifest.get("recipe_id"),
        "round_id": manifest.get("round_id"),
        "execution_summary": manifest.get("execution_summary"),
        "rewrote_pool": manifest.get("rewrote_pool"),
        "rewritten_row_ids_file": manifest.get("rewritten_row_ids_file"),
        "token_budget": manifest.get("token_budget"),
        "unused_budget_tokens": manifest.get("unused_budget_tokens"),
        "rows": manifest.get("rows"),
        # Token accounting: how much data this round actually trained on and
        # how much launcher cap was unused. Realized volumes can differ
        # between rounds with the same row count; a later reader should not
        # attribute such a difference to the mix alone.
        "total_tokens": manifest.get("total_tokens"),
        "token_accounting": manifest.get("token_accounting"),
        "overlong_rows": manifest.get("overlong_rows"),
        "realized_iters_per_epoch": manifest.get("realized_iters_per_epoch"),
        # The trainer's row order is a seeded shuffle applied at prepare
        # time, never the agent's file order; the seed says which draw.
        "shuffle": manifest.get("shuffle"),
        "datasets": datasets,
    }


def archive_recipe(
    run_dir: Path,
    index: int,
    recipe_path: Path | None,
    data_manifest: dict[str, Any] | None,
) -> dict[str, str]:
    _require_ledger_writer()
    """Immutably snapshot the recipe + data manifest for one iteration."""
    history = recipe_history_dir(run_dir)
    history.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    if recipe_path is not None and Path(recipe_path).exists():
        dest = history / f"iter_{index:03d}.yaml"
        shutil.copy2(recipe_path, dest)
        archived["recipe_yaml"] = str(dest)
    if data_manifest is not None:
        dest = history / f"iter_{index:03d}.data_manifest.json"
        write_json(dest, data_manifest)
        archived["data_manifest"] = str(dest)
    return archived


# --------------------------------------------------------------------------- #
# Record SFT submission (objective recipe side)
# --------------------------------------------------------------------------- #
def record_sft(
    run_dir: Path,
    *,
    run_name: str,
    recipe_path: Path | None,
    data_manifest: dict[str, Any] | None,
    train_iters: int,
    seq_length: int,
    hyperparams: dict[str, Any] | None,
    sft_output_dir: str | None,
    base_model: str | None = None,
    initialization: dict[str, Any] | None = None,
    train_epochs: int | None = None,
    iters_per_epoch: int | None = None,
) -> int:
    """Open a new ledger iteration for one SFT round and return its index."""
    _require_ledger_writer()
    run_dir = Path(run_dir)
    try:
        storage_round_id(run_name, label="run_name")
    except RoundIdError as exc:
        raise SystemExit(f"ledger: {exc}") from exc
    iterations = read_iterations(run_dir)
    for existing in iterations:
        if existing.get("run_name") == run_name:
            index = existing.get("iteration")
            if isinstance(index, int):
                return index
    index = _next_index(iterations)
    archived = archive_recipe(run_dir, index, recipe_path, data_manifest)
    pending_note = _submission_note(run_dir, logical_id_from_storage(run_name), index)
    pending_path, pending = pending_note if pending_note is not None else (None, None)
    staged = pending or {}

    record = {
        "iteration": index,
        "created_at": utc_now(),
        "run_name": run_name,
        "kind": "fresh",
        "base_model": base_model,
        "initialization": initialization,
        "recipe": _recipe_summary(data_manifest),
        "recipe_archive": archived,
        "hyperparams": {
            "train_epochs": train_epochs,
            "iters_per_epoch": iters_per_epoch,
            "train_iters": train_iters,
            "seq_length": seq_length,
            **(hyperparams or {}),
        },
        "sft_output_dir": sft_output_dir,
        "training_health": None,
        "eval": {},
        "rationale": staged.get("rationale"),
        "expectation": staged.get("expectation"),
        "evidence_plan": staged.get("evidence_plan"),
        "note_history": _existing_note_history(
            staged, legacy_phase="before_submission"
        ),
        "reviews": [],
        "deviations": staged.get("deviations"),
    }
    iterations.append(record)
    write_iterations(run_dir, iterations)
    if pending_path is not None:
        pending_path.unlink(missing_ok=True)
    try:
        summarize(run_dir, write_md=True, do_print=False)
    except Exception:  # pragma: no cover - summary is best-effort
        pass
    return index


# --------------------------------------------------------------------------- #
# Record eval results (objective result side)
# --------------------------------------------------------------------------- #
def _accuracy(scores: dict[str, Any], key: str) -> float | None:
    value = scores.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    metrics = scores.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _compact_scores(
    scores: dict[str, Any],
    *,
    benchmark: str | None = None,
    repeat: int | None = None,
) -> dict[str, Any]:
    eval_backend = scores.get("eval_backend")
    if not isinstance(eval_backend, str) or not eval_backend:
        raise ValueError("scores.json is missing eval_backend")
    try:
        code = eval_aggregate.code_score_summary(
            scores, benchmark=benchmark, repeat=repeat
        )
    except eval_aggregate.EvalAggregateError as exc:
        raise ValueError(str(exc)) from exc
    categories = []
    metrics = scores.get("metrics") if isinstance(scores.get("metrics"), dict) else {}
    entries = (
        scores.get("top_failure_categories")
        or (metrics.get("top_failure_categories") if code else [])
        or []
    )
    for entry in entries[:3]:
        if isinstance(entry, dict):
            fields = (
                ("category", "count", "total", "rate")
                if code
                else ("category", "strict_fail", "total")
            )
            categories.append({key: entry.get(key) for key in fields})
    if code is not None:
        return {
            **code,
            "eval_backend": eval_backend,
            "top_failure_categories": categories,
        }
    return {
        "eval_backend": eval_backend,
        **{
            key: scores.get(key, metrics.get(key))
            for key in (
                "limit",
                "num_prompts",
                "full_num_prompts",
                "is_full_benchmark",
            )
            if key in scores or key in metrics
        },
        "primary_score": scores.get("primary_score"),
        "strict_prompt_accuracy": _accuracy(scores, "strict_prompt_accuracy"),
        "loose_prompt_accuracy": _accuracy(scores, "loose_prompt_accuracy"),
        "strict_instruction_accuracy": _accuracy(scores, "strict_instruction_accuracy"),
        "loose_instruction_accuracy": _accuracy(scores, "loose_instruction_accuracy"),
        "num_failures": scores.get("num_failures"),
        "top_failure_categories": categories,
    }


def _training_health(sft_output_dir: str | None) -> dict[str, Any] | None:
    if not sft_output_dir:
        return None
    path = Path(sft_output_dir) / "loss_summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    loss_tags = summary.get("loss_tags") or []
    tags = summary.get("tags") or {}
    if not loss_tags:
        return None

    def rank(tag: str) -> int:
        lower = tag.lower()
        if lower == "loss":
            return 0
        if "lm loss" in lower or "lm_loss" in lower:
            return 1
        if "loss" in lower:
            return 2
        return 3

    primary = sorted(loss_tags, key=rank)[0]
    stats = tags.get(primary, {})
    return {
        "loss_tag": primary,
        "last": stats.get("last"),
        "min": stats.get("min"),
        "last_pct_slope": stats.get("last_pct_slope"),
        "last_step": stats.get("last_step"),
        "has_nan": any(tags.get(t, {}).get("has_nan") for t in loss_tags),
        "has_inf": any(tags.get(t, {}).get("has_inf") for t in loss_tags),
    }


def _ckpt_step_from_model(model_path: str) -> str:
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("eval submission is missing model_path")
    manifest = Path(model_path) / "conversion_manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"eval model is missing a valid conversion manifest: {manifest}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"conversion manifest must be an object: {manifest}")
    step = payload.get("ckpt_step")
    if not isinstance(step, str) or not step.isdigit() or step.startswith("0"):
        raise ValueError(
            f"conversion manifest ckpt_step must be a canonical positive integer: {manifest}"
        )
    return step


def _eval_success_manifest_path(run_dir: Path, eval_manifest: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    resolved = Path(eval_manifest).resolve()
    eval_runs = run_dir / "eval_runs"
    if (
        resolved.name != "eval_successful_submissions.jsonl"
        or resolved.parent.parent != eval_runs
    ):
        raise ValueError(
            "eval manifest must be "
            f"{eval_runs}/<name>/eval_successful_submissions.jsonl; got {resolved}"
        )
    return resolved


def _expected_repeats(success_manifest: Path) -> int:
    """Repeat count the eval run was submitted with (1 for legacy runs)."""
    request = Path(success_manifest).parent / "eval_request.json"
    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    value = payload.get("repeats") if isinstance(payload, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return 1


def _check_eval_round_binding(
    target: dict[str, Any],
    eval_manifest: Path,
    *,
    explicit_iteration: int | None,
) -> None:
    """Refuse to attach one round's eval scores to a different round.

    ``_find_target`` falls back to the newest SFT iteration, which is wrong
    whenever a later round has already been submitted while an earlier
    round's eval is still being collected: the scores land under the wrong
    ``run_name`` and every later analysis reasons about a data change that
    did not produce them. The eval run directory is named for the round it
    evaluated, so the two names must agree.

    Nothing is written when this fires, so re-running with the matching
    ``--iteration`` recovers the round.
    """
    eval_round = Path(eval_manifest).parent.name
    request_path = Path(eval_manifest).parent / "eval_request.json"
    if request_path.is_file():
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if isinstance(request, dict) and request.get("retry_of_task"):
            eval_round = storage_round_id(
                request.get("run_name"), label="retry original round"
            )
    target_round = target.get("run_name")
    if not isinstance(target_round, str) or not target_round:
        return
    try:
        storage_round_id(eval_round, label="eval run directory")
    except RoundIdError:
        # Legacy or custom eval directory names carry no round claim.
        return
    if eval_round == target_round:
        return
    chosen = (
        f"--iteration {explicit_iteration}"
        if explicit_iteration is not None
        else "the newest SFT iteration (no --iteration was passed)"
    )
    raise ValueError(
        f"eval/round mismatch — these results come from {eval_manifest.parent} "
        f"({eval_round}), but they would be attached to ledger iteration "
        f"{target.get('iteration')} whose run_name is {target_round}, because "
        f"this command selected {chosen}. Recording them would credit "
        f"{eval_round}'s scores to {target_round}'s data, so every later "
        f"analysis would read the wrong cause for this change.\n"
        f"Nothing was written and the eval artifacts are intact. Fix and re-run:\n"
        f"  1. find the ledger iteration whose run_name is {eval_round} "
        f"(python -m autosft.orchestration.ledger summarize, or read iterations.jsonl)\n"
        f"  2. re-run this record-eval with --iteration <that iteration number>\n"
        f"If {eval_round} has no ledger iteration yet, its SFT round was never "
        f"recorded; record the SFT round first, then attach these scores."
    )


def record_eval(
    run_dir: Path,
    eval_manifest: Path,
    iteration: int | None = None,
) -> int:
    """Attach eval scores from an eval-run manifest to a ledger iteration.

    Joins each eval submission to an iteration by index (``--iteration``) or, by
    default, to the most recent submitted SFT iteration. Initialization metadata
    is recorded on the SFT side, while eval collection attaches the unique
    final-checkpoint scores to the round that produced it.
    Returns the number of benchmark results recorded.
    """
    _require_ledger_writer()
    run_dir = Path(run_dir)
    try:
        eval_manifest = _eval_success_manifest_path(run_dir, eval_manifest)
    except ValueError as exc:
        raise SystemExit(f"ledger: {exc}") from exc
    iterations = read_iterations(run_dir)
    if not iterations:
        raise ValueError(
            f"ledger: no SFT iterations recorded under {run_dir}; recover the SFT record before collecting evaluation"
        )
    if not eval_manifest.exists():
        print(f"ledger: eval manifest not found: {eval_manifest}")
        return 0

    recorded = 0
    expected_repeats = _expected_repeats(eval_manifest)
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for line_number, line in enumerate(
        eval_manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in eval success manifest {eval_manifest}:{line_number}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"eval success manifest record must be an object: "
                f"{eval_manifest}:{line_number}"
            )
        if entry.get("event"):
            raise ValueError(
                f"eval success manifest must contain submission records only: "
                f"{eval_manifest}:{line_number}"
            )
        if submission_task(entry) is None:
            raise ValueError("eval submission is missing a canonical task")
        model_path_value = entry.get("model_path")
        if not isinstance(model_path_value, str) or not model_path_value:
            raise ValueError("eval submission is missing model_path")
        step = _ckpt_step_from_model(model_path_value)
        try:
            expected_model_identity = submission_model_identity(model_path_value)
        except ValueError as exc:
            raise ValueError(
                f"eval submission has invalid model provenance: {exc}"
            ) from exc
        recorded_model_identity = entry.get("model_identity")
        if not isinstance(recorded_model_identity, str) or not recorded_model_identity:
            raise ValueError("eval submission is missing model_identity")
        if recorded_model_identity != expected_model_identity:
            raise ValueError(
                "eval submission model_identity does not match its conversion manifest"
            )
        output_dir = entry.get("output_dir")
        benchmark = entry.get("benchmark")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("eval submission is missing output_dir")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("eval submission is missing benchmark")
        # One record covers every repeat its task produced: repeat k's
        # scores live in output_dir/rep_<k>/ (plain output_dir when the task
        # ran once). Legacy per-repeat records carry a single index and point
        # straight at their own directory.
        repeat_indices = record_repeat_indices(entry)
        repeat_dirs = (
            eval_aggregate.repeat_dirs(Path(output_dir), record_repeats(entry))
            if "repeats" in entry
            else [Path(output_dir)]
        )
        repeats_for_benchmark = grouped.setdefault(benchmark, {})
        for repeat, rep_dir in zip(repeat_indices, repeat_dirs):
            scores_file = rep_dir / "scores.json"
            if not scores_file.exists():
                raise ValueError(
                    f"eval submission is missing scores.json: {scores_file}"
                )
            try:
                scores = json.loads(scores_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"invalid scores artifact: {scores_file}") from exc
            if not isinstance(scores, dict):
                raise ValueError(f"scores artifact must be an object: {scores_file}")

            if repeat in repeats_for_benchmark:
                raise ValueError(
                    f"duplicate eval submission for benchmark {benchmark!r} repeat "
                    f"{repeat}: {eval_manifest}:{line_number}"
                )
            summary = _compact_scores(scores, benchmark=benchmark, repeat=repeat)
            summary["repeat"] = repeat
            summary["eval_run_name"] = str(entry.get("run_name") or "")
            summary["model_path"] = model_path_value
            summary["ckpt_step"] = step
            summary["output_dir"] = str(rep_dir)
            temperature = entry.get("temperature")
            if isinstance(temperature, (int, float)) and not isinstance(
                temperature, bool
            ):
                summary["temperature"] = float(temperature)
            repeats_for_benchmark[repeat] = summary

    for benchmark, repeats_for_benchmark in grouped.items():
        expected_set = set(range(1, expected_repeats + 1))
        present_set = set(repeats_for_benchmark)
        if present_set != expected_set:
            missing = sorted(expected_set - present_set)
            extra = sorted(present_set - expected_set)
            detail = []
            if missing:
                detail.append(f"missing repeats {missing}")
            if extra:
                detail.append(f"unexpected repeats {extra}")
            raise ValueError(
                f"benchmark {benchmark!r} has an incomplete repeat set "
                f"({'; '.join(detail)}); a round is recorded only when every "
                f"one of its {expected_repeats} repeat(s) has succeeded"
            )

        ordered = [repeats_for_benchmark[k] for k in sorted(repeats_for_benchmark)]
        raw_scores: list[float] = []
        for repeat_summary in ordered:
            score = repeat_summary.get("primary_score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError(
                    f"benchmark {benchmark!r} repeat "
                    f"{repeat_summary.get('repeat')} has no numeric primary_score"
                )
            raw_scores.append(float(score))

        target = _find_target(iterations, iteration)
        if target is None:
            raise ValueError("requested ledger iteration does not exist")
        _check_eval_round_binding(
            target,
            eval_manifest,
            explicit_iteration=iteration,
        )
        first = ordered[0]
        benchmark_dir = Path(first["output_dir"])
        if expected_repeats > 1:
            benchmark_dir = benchmark_dir.parent
        summary = {
            "eval_backend": first.get("eval_backend"),
            **{
                key: first[key]
                for key in (
                    "limit",
                    "num_prompts",
                    "full_num_prompts",
                    "is_full_benchmark",
                )
                if key in first
            },
            "primary_score": sum(raw_scores) / len(raw_scores),
            "primary_scores": raw_scores,
            "primary_score_sd": eval_aggregate.sample_sd(raw_scores),
            "n_repeats": expected_repeats,
            "repeats": ordered,
            "num_failures": first.get("num_failures")
            if expected_repeats == 1
            else None,
            "top_failure_categories": (
                first.get("top_failure_categories") if expected_repeats == 1 else None
            ),
            "eval_run_name": first.get("eval_run_name"),
            "model_path": first.get("model_path"),
            "ckpt_step": first.get("ckpt_step"),
            "result_artifacts": {
                "aggregate": str(benchmark_dir / "aggregate.json"),
                "repeat_dirs": [item["output_dir"] for item in ordered],
                "submission_manifest": str(eval_manifest),
            },
            "recorded_at": utc_now(),
        }
        code = first.get("primary_metric") == "plus_pass_at_1"
        if code:
            if any(
                any(
                    item.get(key) != first.get(key)
                    for key in (
                        "benchmark",
                        "primary_metric",
                        "num_problems",
                        "model_path",
                        "ckpt_step",
                        "eval_backend",
                    )
                )
                for item in ordered
            ):
                raise ValueError(
                    "code repeats have inconsistent model, benchmark, or metric identities"
                )
            try:
                # Validate every repeat's failure evidence as well as its score
                # before attaching this checkpoint's result to the ledger.
                eval_aggregate.aggregate_benchmark(benchmark_dir, expected_repeats)
            except eval_aggregate.EvalAggregateError as exc:
                raise ValueError(str(exc)) from exc
            summary["primary_metric"] = "plus_pass_at_1"
            summary["num_problems"] = first["num_problems"]
            summary["is_full_benchmark"] = first["is_full_benchmark"]
            if expected_repeats == 1:
                summary["num_base_pass"] = first["num_base_pass"]
                summary["num_plus_pass"] = first["num_plus_pass"]
        if "temperature" in first:
            summary["temperature"] = first["temperature"]
        metric_keys = (
            ("base_pass_at_1", "plus_pass_at_1")
            if code
            else (
                "strict_prompt_accuracy",
                "loose_prompt_accuracy",
                "strict_instruction_accuracy",
                "loose_instruction_accuracy",
            )
        )
        for key in metric_keys:
            values = [
                repeat_summary.get(key)
                for repeat_summary in ordered
                if isinstance(repeat_summary.get(key), (int, float))
            ]
            summary[key] = sum(values) / len(values) if values else None

        target.setdefault("eval", {})[benchmark] = summary
        if not target.get("training_health"):
            health = _training_health(target.get("sft_output_dir"))
            if health:
                target["training_health"] = health
        recorded += 1

    write_iterations(run_dir, iterations)
    try:
        summarize(run_dir, write_md=True, do_print=False)
    except Exception:  # pragma: no cover - summary is best-effort
        pass
    try:
        audit = write_context_audit(
            run_dir,
            trigger="record-eval",
            related_manifest=str(eval_manifest),
        )
        print(f"ledger: context audit written to {audit['latest_markdown_path']}")
    except (
        Exception
    ) as exc:  # context audit is best-effort; never block eval collection
        print(f"WARN: failed to write context audit: {exc}", file=sys.stderr)
    print(f"ledger: recorded {recorded} eval result(s) into {ledger_path(run_dir)}")
    return recorded


# --------------------------------------------------------------------------- #
# Agent-authored rationale
# --------------------------------------------------------------------------- #
def _validate_deviations(deviations: list[str] | None) -> list[str]:
    """Normalize the note's repeatable ``--deviation`` values."""
    if not deviations:
        return []
    if any(not isinstance(item, str) or not item.strip() for item in deviations):
        raise SystemExit("ledger: note deviations must be non-empty strings")
    return [item.strip() for item in deviations]


def _optional_text(value: Any, *, field: str) -> str | None:
    return None if value is None else _required_text(value, field=field)


def _existing_note_history(
    record: dict[str, Any], *, legacy_phase: str | None = None
) -> list[dict[str, Any]]:
    history = record.get("note_history")
    if isinstance(history, list):
        return list(history)
    if not any(
        record.get(field) is not None
        for field in ("rationale", "expectation", "evidence_plan")
    ):
        return []
    # Older rows lack note timing. Preserve their text without inventing it.
    return [
        {
            "recorded_at": record.get("created_at") if legacy_phase else None,
            "phase": legacy_phase,
            "rationale": record.get("rationale"),
            "expectation": record.get("expectation"),
            "evidence_plan": record.get("evidence_plan"),
            "deviations": record.get("deviations") or [],
        }
    ]


def add_note(
    run_dir: Path,
    rationale: str,
    iteration: int | None = None,
    *,
    round_id: str | int | None = None,
    expectation: str | None = None,
    evidence_plan: str | None = None,
    deviations: list[str] | None = None,
    force_restage: bool = False,
) -> None:
    _require_ledger_writer()
    run_dir = Path(run_dir)
    rationale = _required_text(rationale, field="rationale")
    expectation = _optional_text(expectation, field="expectation")
    evidence_plan = _optional_text(evidence_plan, field="evidence_plan")
    validated_deviations = _validate_deviations(deviations)
    iterations = read_iterations(run_dir)
    if round_id is not None and iteration is not None:
        raise SystemExit("ledger: choose --round-id or --iteration, not both")
    canonical_round = logical_round_id(round_id) if round_id is not None else None
    # An unnumbered note is ambiguous once a round exists, and the ambiguity is
    # not resolvable from here: agents write both prospective notes ("round N
    # will restore the broad stride", meant for the round being assembled) and
    # retrospective ones ("round N regressed", meant for the round that just
    # ran), through this same command and in comparable numbers. Either default
    # silently misfiles the other genre onto a neighbouring round, which is
    # worse than an empty cell because the row then reads as evidence about a
    # round it does not describe. Ask instead of guessing.
    if iteration is None and canonical_round is None and iterations:
        raise SystemExit(
            f"ledger: --round-id or --iteration is required once a round exists under {run_dir}; "
            "a note is filed against the round it explains, and that is "
            f"--iteration {_next_index(iterations)} for the round you are "
            f"assembling (staged until it is submitted) or --iteration "
            f"{_next_index(iterations) - 1} for the round that just ran."
        )
    target = (
        _find_round(iterations, canonical_round)
        if canonical_round is not None
        else _find_target(iterations, iteration)
    )
    note = {
        "recorded_at": utc_now(),
        "phase": _recording_phase(target),
        "rationale": rationale,
        "expectation": expectation,
        "evidence_plan": evidence_plan,
        "deviations": validated_deviations,
    }
    if target is None:
        # Round-bound plans can coexist before submission order is known.
        # An iteration-bound plan still names only the next submission index.
        if (
            canonical_round is not None
            or iteration is None
            or iteration == _next_index(iterations)
        ):
            pending_path = pending_rationale_path(run_dir, canonical_round)
            existing = _read_json_if_exists(pending_path)
            if isinstance(existing, dict) and not force_restage:
                raise SystemExit(
                    "ledger: a staged rationale from "
                    f"{existing.get('created_at') or 'an earlier note'} is still "
                    "waiting for its SFT submission; staging again would "
                    "change its active rationale. Re-run with --force-restage "
                    "to revise it while preserving the earlier note in history."
                )
            staged = dict(existing or {})
            staged.update(
                {
                    **(
                        {"round_id": canonical_round}
                        if canonical_round is not None
                        else {"iteration": _next_index(iterations)}
                    ),
                    "rationale": rationale,
                    "deviations": validated_deviations,
                    "created_at": note["recorded_at"],
                    "note_history": _existing_note_history(
                        staged, legacy_phase="before_submission"
                    )
                    + [note],
                }
            )
            for field, value in (
                ("expectation", expectation),
                ("evidence_plan", evidence_plan),
            ):
                if value is not None or field not in staged:
                    staged[field] = value
            pending_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(pending_path, staged)
            print(
                f"ledger: staged rationale for round {canonical_round}"
                if canonical_round is not None
                else f"ledger: staged rationale for the next SFT iteration ({_next_index(iterations)})"
            )
            return
        raise SystemExit(
            f"ledger: no iteration {iteration} found under {run_dir}; "
            f"recorded iterations run to {_next_index(iterations) - 1} and the "
            f"next one will be {_next_index(iterations)}. Name a recorded "
            "iteration to amend it, or the next one to stage this note for the "
            "round you are about to submit."
        )
    target["note_history"] = _existing_note_history(target) + [note]
    target["rationale"] = rationale
    if expectation is not None or "expectation" not in target:
        target["expectation"] = expectation
    if evidence_plan is not None or "evidence_plan" not in target:
        target["evidence_plan"] = evidence_plan
    if validated_deviations:
        target["deviations"] = validated_deviations
    write_iterations(run_dir, iterations)
    try:
        summarize(run_dir, write_md=True, do_print=False)
    except Exception:  # pragma: no cover - summary is best-effort
        pass
    print(f"ledger: set rationale on iteration {target.get('iteration')}")


def review_evidence_digest(record: dict[str, Any]) -> str:
    """Hash objective ledger facts; notes and collection timestamps are not evidence."""
    fields = (
        "iteration",
        "run_name",
        "kind",
        "base_model",
        "initialization",
        "recipe",
        "recipe_archive",
        "hyperparams",
        "sft_output_dir",
        "training_health",
        "eval",
    )

    def normalized(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in value.items()
                if key not in {"recorded_at", "created_at"}
            }
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    evidence = normalized({field: record.get(field) for field in fields})
    if "round_id" in record:
        evidence["round_id"] = normalized(record["round_id"])
    encoded = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_review(
    run_dir: Path,
    *,
    iteration: int | None = None,
    round_id: str | int | None = None,
    review: str,
) -> dict[str, Any]:
    """Append a reflection bound to currently recorded, possibly partial evidence."""
    _require_ledger_writer()
    review = _required_text(review, field="review")
    if (iteration is None) == (round_id is None):
        raise SystemExit("ledger: choose exactly one of --round-id or --iteration")
    iterations = read_iterations(run_dir)
    canonical_round = logical_round_id(round_id) if round_id is not None else None
    target = (
        _find_round(iterations, canonical_round)
        if canonical_round is not None
        else _find_target(iterations, iteration)
    )
    if target is None:
        identity = (
            f"round {canonical_round}"
            if canonical_round is not None
            else f"iteration {iteration}"
        )
        raise SystemExit(f"ledger: no {identity} found under {run_dir}")
    digest = review_evidence_digest(target)
    reviews = target.setdefault("reviews", [])
    for previous in reviews[-1:]:
        if (
            previous.get("review") == review
            and previous.get("evidence_digest") == digest
        ):
            return previous
    event = {"recorded_at": utc_now(), "review": review, "evidence_digest": digest}
    reviews.append(event)
    write_iterations(run_dir, iterations)
    try:
        summarize(run_dir, write_md=True, do_print=False)
    except Exception:  # pragma: no cover - summary is best-effort
        pass
    print(f"ledger: recorded review on iteration {target.get('iteration')}")
    return event


# --------------------------------------------------------------------------- #
# Hypothesis history
# --------------------------------------------------------------------------- #
def read_hypothesis_history(run_dir: Path) -> list[dict[str, Any]]:
    """Read optional hypotheses in recorded order, including legacy entries."""
    path = hypothesis_history_path(run_dir)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in hypothesis history {path}:{line_number}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"hypothesis history record must be an object: {path}:{line_number}"
            )
        entry.setdefault("hypothesis_id", f"round_{entry.get('round_id')}")
        entry.setdefault("status", "open")
        entry.setdefault("updates", [])
        if "verdict_history" not in entry:
            entry["verdict_history"] = (
                [_verdict_snapshot(entry)]
                if entry.get("score_verdict") is not None
                else []
            )
        if (
            not isinstance(entry["hypothesis_id"], str)
            or not entry["hypothesis_id"].strip()
        ):
            raise ValueError(
                f"invalid hypothesis_id in hypothesis history {path}:{line_number}"
            )
        if entry["status"] not in HYPOTHESIS_STATUSES:
            raise ValueError(
                f"invalid status in hypothesis history {path}:{line_number}"
            )
        for field in ("updates", "verdict_history"):
            if not isinstance(entry[field], list) or any(
                not isinstance(item, dict) for item in entry[field]
            ):
                raise ValueError(
                    f"invalid {field} in hypothesis history {path}:{line_number}; "
                    "expected a list of objects"
                )
        entries.append(entry)
    return entries


def _write_hypothesis_history(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Rewrite the whole history atomically, preserving the recorded order."""
    _require_ledger_writer()
    path = hypothesis_history_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _hypothesis_round(value: Any, *, label: str) -> str:
    try:
        return logical_round_id(value, label=label)
    except RoundIdError as exc:
        raise SystemExit(f"ledger: {exc}") from exc


def _find_hypothesis(
    entries: list[dict[str, Any]],
    round_id: str | None = None,
    *,
    hypothesis_id: str | None = None,
) -> dict[str, Any] | None:
    matches = [
        entry
        for entry in entries
        if (round_id is None or str(entry.get("round_id")) == round_id)
        and (hypothesis_id is None or entry.get("hypothesis_id") == hypothesis_id)
    ]
    if len(matches) > 1:
        raise SystemExit(
            f"ledger: round {round_id} has multiple hypotheses; specify --hypothesis-id"
        )
    return matches[0] if matches else None


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"ledger: {field} must be a non-empty string")
    return value.strip()


def _baseline_round(value: Any, *, round_id: int) -> int:
    """Validate the baseline round: 0 (base model) or an earlier round."""
    if value == "0" or (
        isinstance(value, int) and not isinstance(value, bool) and value == 0
    ):
        baseline = 0
    else:
        baseline = int(_hypothesis_round(value, label="baseline_round"))
    if baseline < 0 or baseline >= round_id:
        raise SystemExit(
            f"ledger: baseline_round {baseline} must be 0 or an earlier "
            f"round than {round_id}; the score verdict is read against it"
        )
    return baseline


def record_hypothesis(
    run_dir: Path,
    *,
    round_id: Any,
    hypothesis: str,
    planned_action: str,
    reason: str,
    baseline_round: Any = None,
    expected_direction: str | None = None,
    expected_effect: str | None = None,
    mechanism_observable: str | None = None,
    hypothesis_id: str | None = None,
) -> dict[str, Any]:
    """Append an immutable statement; recorded phase is not preregistration proof."""
    _require_ledger_writer()
    run_dir = Path(run_dir)
    canonical = _hypothesis_round(round_id, label="hypothesis round_id")
    if expected_direction is not None and expected_direction not in EXPECTED_DIRECTIONS:
        raise SystemExit(
            "ledger: expected_direction must be one of "
            + ", ".join(EXPECTED_DIRECTIONS)
        )
    entry = {
        "hypothesis_id": (
            f"round_{canonical}"
            if hypothesis_id is None
            else _required_text(hypothesis_id, field="hypothesis_id")
        ),
        "round_id": int(canonical),
        "recorded_at": utc_now(),
        "phase": _recording_phase(_find_round(read_iterations(run_dir), canonical)),
        "hypothesis": _required_text(hypothesis, field="hypothesis"),
        "planned_action": _required_text(planned_action, field="planned_action"),
        "reason": _required_text(reason, field="reason"),
        "baseline_round": (
            _baseline_round(baseline_round, round_id=int(canonical))
            if baseline_round is not None
            else None
        ),
        "expected_direction": expected_direction,
        "expected_effect": _optional_text(expected_effect, field="expected_effect"),
        "mechanism_observable": _optional_text(
            mechanism_observable, field="mechanism_observable"
        ),
        "status": "open",
        "updates": [],
        "verdict_history": [],
        "score_verdict": None,
        "score_basis": None,
        "mechanism_verdict": None,
        "mechanism_basis": None,
        "verdict_round": None,
        "verdict_at": None,
    }
    entries = read_hypothesis_history(run_dir)
    if _find_hypothesis(entries, hypothesis_id=entry["hypothesis_id"]) is not None:
        raise SystemExit(
            f"ledger: already has a hypothesis with id {entry['hypothesis_id']!r}; "
            "original statements are immutable; use a new id for a different hypothesis"
        )
    append_jsonl(hypothesis_history_path(run_dir), entry)
    print(f"ledger: recorded the hypothesis for round {canonical}")
    return entry


def update_hypothesis(
    run_dir: Path, *, hypothesis_id: str, status: str, evidence: str, at_round: Any
) -> dict[str, Any]:
    """Append the meta agent's explicit judgment, including reopening a hypothesis."""
    _require_ledger_writer()
    hypothesis_id = _required_text(hypothesis_id, field="hypothesis_id")
    evidence = _required_text(evidence, field="evidence")
    canonical = _hypothesis_round(at_round, label="at_round")
    if status not in HYPOTHESIS_STATUSES:
        raise SystemExit(
            "ledger: status must be one of " + ", ".join(HYPOTHESIS_STATUSES)
        )
    entries = read_hypothesis_history(run_dir)
    target = _find_hypothesis(entries, hypothesis_id=hypothesis_id)
    if target is None:
        raise SystemExit(f"ledger: no hypothesis entry with id {hypothesis_id!r}")
    if int(canonical) < int(target["round_id"]):
        raise SystemExit("ledger: at_round cannot precede the hypothesis round")
    target["updates"].append(
        {
            "recorded_at": utc_now(),
            "status": status,
            "evidence": evidence,
            "at_round": int(canonical),
        }
    )
    target["status"] = status
    _write_hypothesis_history(run_dir, entries)
    print(f"ledger: updated hypothesis {hypothesis_id} to {status}")
    return target


def _verdict_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        field: entry.get(field)
        for field in (
            "score_verdict",
            "score_basis",
            "mechanism_verdict",
            "mechanism_basis",
            "lesson",
            "verdict_round",
            "verdict_at",
        )
    }


def judge_hypothesis(
    run_dir: Path,
    *,
    round_id: Any,
    score_verdict: str,
    score_basis: str,
    mechanism_verdict: str,
    mechanism_basis: str,
    lesson: str,
    verdict_round: Any = None,
    hypothesis_id: str | None = None,
) -> dict[str, Any]:
    """Append a two-axis assessment, without inferring causality or status.

    Same-round judgment requires recorded results. Later reconsideration keeps
    earlier verdicts and lessons in ``verdict_history``; top-level verdict
    fields expose the latest assessment for compatible readers.
    """
    _require_ledger_writer()
    run_dir = Path(run_dir)
    canonical = _hypothesis_round(round_id, label="hypothesis round_id")
    if score_verdict not in SCORE_VERDICTS:
        raise SystemExit(
            "ledger: score verdict must be one of " + ", ".join(SCORE_VERDICTS)
        )
    if mechanism_verdict not in MECHANISM_VERDICTS:
        raise SystemExit(
            "ledger: mechanism verdict must be one of " + ", ".join(MECHANISM_VERDICTS)
        )
    score_basis = _required_text(score_basis, field="score_basis")
    mechanism_basis = _required_text(mechanism_basis, field="mechanism_basis")
    lesson = _required_text(lesson, field="lesson")
    entries = read_hypothesis_history(run_dir)
    if hypothesis_id is not None:
        hypothesis_id = _required_text(hypothesis_id, field="hypothesis_id")
    target = _find_hypothesis(entries, canonical, hypothesis_id=hypothesis_id)
    if target is None:
        raise SystemExit(
            f"ledger: round {canonical} has no hypothesis entry under {run_dir}; "
            "record a hypothesis before judging it"
        )
    if verdict_round is None:
        verdict_round = canonical
    judged_by = _hypothesis_round(verdict_round, label="verdict_round")
    if int(judged_by) < int(canonical):
        raise SystemExit(
            f"ledger: verdict_round {judged_by} may not judge round "
            f"{canonical}; a verdict cannot precede its hypothesis round"
        )
    if (
        judged_by == canonical
        and _recording_phase(_find_round(read_iterations(run_dir), canonical))
        != "after_result"
    ):
        raise SystemExit(
            f"ledger: same-round judgment of round {canonical} requires recorded results"
        )
    target["score_verdict"] = score_verdict
    target["score_basis"] = score_basis
    target["mechanism_verdict"] = mechanism_verdict
    target["mechanism_basis"] = mechanism_basis
    target["lesson"] = lesson
    target["verdict_round"] = int(judged_by)
    target["verdict_at"] = utc_now()
    target["verdict_history"].append(_verdict_snapshot(target))
    _write_hypothesis_history(run_dir, entries)
    print(
        f"ledger: judged round {canonical} score={score_verdict} "
        f"mechanism={mechanism_verdict} (by round {judged_by})"
    )
    return target


def hypothesis_status(run_dir: Path, *, do_print: bool = True) -> dict[str, Any]:
    """Show full histories; legacy ``undecided`` means no two-axis assessment."""
    run_dir = Path(run_dir)
    entries = read_hypothesis_history(run_dir)
    undecided = [
        str(entry.get("round_id"))
        for entry in entries
        if entry.get("score_verdict") is None
    ]
    open_entries = [entry for entry in entries if entry.get("status") == "open"]
    status = {"entries": entries, "undecided": undecided, "open": open_entries}
    if not do_print:
        return status
    lines = [f"# Hypothesis history — {Path(run_dir).name}", ""]
    if not entries:
        lines.append("_No hypothesis recorded yet._")
        lines.append("")
    for entry in entries:
        lines.append(
            f"round {entry.get('round_id')} [{entry['hypothesis_id']}]: "
            f"{entry['status'].upper()} (recorded {entry.get('recorded_at')})"
        )
        if entry.get("score_verdict") is None:
            lines.append("  No detailed two-axis assessment recorded (optional).")
        else:
            lines.append(
                f"  score {entry.get('score_verdict')} "
                f"vs round {entry.get('baseline_round')}, mechanism "
                f"{entry.get('mechanism_verdict')} (judged by round "
                f"{entry.get('verdict_round')} at {entry.get('verdict_at')})"
            )
        for field in (
            "hypothesis_id",
            "status",
            "phase",
            "hypothesis",
            "planned_action",
            "reason",
            "baseline_round",
            "expected_direction",
            "expected_effect",
            "mechanism_observable",
        ):
            lines.append(f"  {field}: {entry.get(field)}")
        if entry.get("updates"):
            lines.append("  updates (recorded order):")
            for index, update in enumerate(entry["updates"], start=1):
                lines.append(
                    f"    update {index}: {update.get('status')} "
                    f"at round {update.get('at_round')} "
                    f"(recorded {update.get('recorded_at')})"
                )
                lines.append(f"      evidence: {update.get('evidence')}")
        if entry.get("verdict_history"):
            lines.append("  verdict_history (recorded order):")
            for index, verdict in enumerate(entry["verdict_history"], start=1):
                lines.append(
                    f"    verdict {index}: score {verdict.get('score_verdict')}, "
                    f"mechanism {verdict.get('mechanism_verdict')} "
                    f"(judged by round {verdict.get('verdict_round')} "
                    f"at {verdict.get('verdict_at')})"
                )
                for field in ("score_basis", "mechanism_basis", "lesson"):
                    lines.append(f"      {field}: {verdict.get(field)}")
        lines.append("")
    decided = [e for e in entries if e.get("score_verdict") is not None]
    if decided:
        lines.append(
            "## LESSONS (each is the judging round's own attribution — "
            "possibly wrong; earlier assessments remain in verdict_history)"
        )
        for entry in decided:
            lines.append(
                f"- round {entry.get('round_id')} "
                f"[score {entry.get('score_verdict')}, mechanism "
                f"{entry.get('mechanism_verdict')}]: {entry.get('lesson')}"
            )
        lines.append("")
    lines.append(
        "rounds without optional two-axis assessment: "
        + (", ".join(undecided) if undecided else "none")
    )
    lines.append(
        "open hypotheses: "
        + (
            ", ".join(entry["hypothesis_id"] for entry in open_entries)
            if open_entries
            else "none"
        )
    )
    print("\n".join(lines))
    return status


def verify_hypothesis_recorded(recipe: dict[str, Any], run_dir: Path) -> bool:
    """Report whether a round has an optional prediction, without gating it.

    Kept for compatibility with callers of the former submission gate. Both
    architectures may record hypotheses; an absent entry is informational.
    Explicit history inspection still reports malformed ledger data.
    """
    canonical = _hypothesis_round(recipe.get("round_id"), label="recipe.round_id")
    return any(
        str(entry.get("round_id")) == canonical
        for entry in read_hypothesis_history(run_dir)
    )


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #
def _cell(text: Any) -> str:
    if text is None:
        return "—"
    return str(text).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _kind_cell(record: dict[str, Any]) -> str:
    # Every current SFT row is an independent base-model round.
    return "fresh"


def _data_cell(record: dict[str, Any]) -> str:
    recipe = record.get("recipe") or {}
    datasets = recipe.get("datasets") or []
    if not datasets:
        return "—"
    parts = [
        f"{Path(str(d.get('path') or '?')).name}×{d.get('rows')} rows"
        f"@{d.get('total_tokens')} tok"
        for d in datasets
    ]
    cell = " + ".join(parts)
    # Realized share of the launcher-owned token budget. Two rounds with the
    # same row count can train on very different volumes, so the summary must
    # show what the round actually consumed rather than only its size.
    utilization = (recipe.get("token_accounting") or {}).get("budget_utilization")
    if isinstance(utilization, (int, float)) and not isinstance(utilization, bool):
        cell += f" ({utilization:.0%} of budget)"
    return _cell(cell)


def _ckpt_cell(record: dict[str, Any], run_dir: Path) -> str:
    """Return this round's final-checkpoint root, compacted relative to run dir."""
    sft_output_dir = record.get("sft_output_dir")
    if not sft_output_dir:
        return "—"
    ckpt = Path(sft_output_dir) / "checkpoints"
    try:
        return _cell(str(ckpt.resolve().relative_to(run_dir.resolve())))
    except (ValueError, OSError):
        return _cell(str(ckpt))


def _initialization_cell(record: dict[str, Any]) -> str:
    initialization = record.get("initialization") or {}
    if initialization.get("type") == "base":
        model = initialization.get("model") or record.get("base_model")
        return _cell(f"base:{model}") if model else "base"
    # The current schema has no parent-checkpoint lineage: every round starts
    # from the configured base model. Keep the column stable for summaries
    # generated from an incomplete record.
    return "base"


def _loss_cell(record: dict[str, Any]) -> str:
    health = record.get("training_health")
    if not health:
        return "—"
    last = health.get("last")
    slope = health.get("last_pct_slope")
    last_s = f"{last:.3f}" if isinstance(last, (int, float)) else "—"
    slope_s = f"{slope:.1e}" if isinstance(slope, (int, float)) else "—"
    flag = " (!)" if health.get("has_nan") or health.get("has_inf") else ""
    return f"{last_s}/{slope_s}{flag}"


def _bench_cell(record: dict[str, Any], benchmark: str) -> str:
    final = (record.get("eval") or {}).get(benchmark)
    if final and isinstance(final.get("primary_score"), (int, float)):
        step = final.get("ckpt_step")
        cell = f"{final['primary_score']:.4f}"
        sd = final.get("primary_score_sd")
        if isinstance(sd, (int, float)):
            cell += f" ±{sd:.4f}"
        raw_scores = final.get("primary_scores")
        if (
            isinstance(raw_scores, list)
            and len(raw_scores) > 1
            and all(isinstance(v, (int, float)) for v in raw_scores)
        ):
            cell += " (" + "/".join(f"{v:.4f}" for v in raw_scores) + ")"
        if step is not None:
            cell += f"@{step}"
        return cell
    return "—"


def _collect_benchmarks(iterations: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for record in iterations:
        for benchmark in record.get("eval") or {}:
            if benchmark not in seen:
                seen.append(benchmark)
    return seen or ["ifeval", "ifbench"]


def summarize(run_dir: Path, write_md: bool = True, do_print: bool = True) -> str:
    run_dir = Path(run_dir)
    iterations = read_iterations(run_dir)
    benchmarks = _collect_benchmarks(iterations)

    lines: list[str] = []
    lines.append(f"# Iteration ledger — {run_dir.name}")
    lines.append("")
    if not iterations:
        lines.append("_No iterations recorded yet._")
        text = "\n".join(lines) + "\n"
        _write_summary_md(run_dir, text, write_md)
        if do_print:
            print(text, end="")
        return text

    header = [
        "iter",
        "kind",
        "initialization",
        "recipe_id",
        "final ckpt dir (rel. run dir)",
        "data (file × rows @ tokens)",
        "epochs",
        "iters/epoch",
        "train_iters",
        "loss last/slope",
    ]
    header += [b.upper() for b in benchmarks]
    header += ["rationale", "expectation", "evidence_plan", "latest review"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for record in iterations:
        row = [
            str(record.get("iteration")),
            _kind_cell(record),
            _initialization_cell(record),
            _cell((record.get("recipe") or {}).get("recipe_id")),
            _ckpt_cell(record, run_dir),
            _data_cell(record),
            _cell((record.get("hyperparams") or {}).get("train_epochs")),
            _cell((record.get("hyperparams") or {}).get("iters_per_epoch")),
            _cell((record.get("hyperparams") or {}).get("train_iters")),
            _loss_cell(record),
        ]
        row += [_bench_cell(record, b) for b in benchmarks]
        row.append(_cell(record.get("rationale")))
        row.append(_cell(record.get("expectation")))
        row.append(_cell(record.get("evidence_plan")))
        reviews = record.get("reviews") or []
        row.append(_cell(reviews[-1].get("review") if reviews else None))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "_Each row keeps that round's final checkpoint for evaluation and final-model "
        "selection. The exact recipe behind each row is archived at "
        "`recipe_history/iter_<NNN>.yaml` (`.data_manifest.json`), where `<NNN>` is "
        "the zero-padded iteration index._"
    )

    text = "\n".join(lines) + "\n"
    _write_summary_md(run_dir, text, write_md)
    if do_print:
        print(text, end="")
    return text


def _write_summary_md(run_dir: Path, text: str, write_md: bool) -> None:
    if not write_md:
        return
    _require_ledger_writer()
    path = summary_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def best_scores(run_dir: Path) -> dict[str, float]:
    """Best per-benchmark round score across all recorded iterations.

    Each iteration's ``eval[benchmark].primary_score`` is that round's score —
    the mean of its repeat evaluations when the run evaluates each benchmark
    more than once. Returns the max of those round scores per benchmark.
    Benchmarks with no numeric score yet are omitted. These independent maxima
    are display-only; target_checkpoint decides whether one checkpoint qualifies.
    """
    best: dict[str, float] = {}
    for record in read_iterations(run_dir):
        evals = record.get("eval") or {}
        for benchmark, summary in evals.items():
            if not isinstance(summary, dict):
                continue
            score = summary.get("primary_score")
            if isinstance(score, (int, float)) and (
                benchmark not in best or float(score) > best[benchmark]
            ):
                best[benchmark] = float(score)
    return best


def _full_code_benchmark(summary: dict[str, Any], benchmark: str) -> bool:
    """Partial code evaluations are diagnostics, never paper target evidence."""
    return (
        summary.get("primary_metric") == "plus_pass_at_1"
        and summary.get("is_full_benchmark") is True
        and type(summary.get("num_problems")) is int
        and summary["num_problems"] == EVALPLUS_EXPECTED_DETAILS[benchmark]
    )


def target_checkpoint(
    run_dir: Path,
    targets: dict[str, float],
    *,
    benchmarks: list[str] | None = None,
    repeats: int | None = None,
    required_model_identity: str | None = None,
) -> dict[str, Any] | None:
    """Find one evaluated checkpoint whose complete means meet every target.

    Per-benchmark historical maxima are useful for display, but cannot be
    combined into a model. Eligibility uses the configured repeat coverage and
    verifies every benchmark against the same conversion provenance.
    """
    if not targets:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in targets.values()
    ):
        raise ValueError("target scores must be finite numbers")
    if benchmarks is None:
        configured = (
            os.environ.get("AUTO_SFT_BENCHMARKS")
            or os.environ.get("AUTO_SFT_EVAL_BENCHMARK")
            or os.environ.get("AUTO_SFT_BENCHMARK")
            or ""
        )
        benchmarks = configured.replace(",", " ").split()
    required = set(benchmarks) | set(targets)
    if repeats is None:
        repeats = int(os.environ.get("AUTO_SFT_EVAL_REPEATS", "1"))
    if type(repeats) is not int or repeats < 1:
        raise ValueError("evaluation repeats must be a positive integer")
    eligible: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for record in read_iterations(run_dir):
        from autosft.orchestration.completion_evidence import (
            evaluation_evidence_problems,
            model_assets_complete,
        )

        if evaluation_evidence_problems(
            record, run_dir, benchmarks=sorted(required), repeats=repeats
        ):
            continue
        evals = record.get("eval") or {}
        scores: dict[str, float] = {}
        provenance: tuple[str, str, str] | None = None
        model_path = None
        for benchmark in sorted(required):
            summary = evals.get(benchmark)
            if (
                not isinstance(summary, dict)
                or type(summary.get("n_repeats")) is not int
                or summary["n_repeats"] != repeats
            ):
                break
            if benchmark in CODE_BENCHMARKS and not _full_code_benchmark(
                summary, benchmark
            ):
                break
            from autosft.orchestration.round_gates import if_coverage_problem

            if if_coverage_problem(summary, benchmark):
                break
            values = summary.get("primary_scores")
            if (
                not isinstance(values, list)
                or len(values) != repeats
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                )
            ):
                break
            mean = sum(values) / repeats
            recorded_mean = summary.get("primary_score")
            if (
                isinstance(recorded_mean, bool)
                or not isinstance(recorded_mean, (int, float))
                or not math.isfinite(recorded_mean)
                or not math.isclose(mean, recorded_mean, rel_tol=1e-12, abs_tol=1e-12)
            ):
                break
            try:
                current_model = str(summary["model_path"])
                if not Path(current_model).resolve().is_relative_to(
                    Path(run_dir).resolve()
                ) or not model_assets_complete(Path(current_model)):
                    break
                identity = submission_model_identity(current_model)
                conversion = json.loads(
                    (Path(current_model) / "conversion_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                checkpoint = str(Path(conversion["checkpoint"]).resolve())
                step = conversion["ckpt_step"]
                expected_root = str(
                    (Path(record["sft_output_dir"]) / "checkpoints").resolve()
                )
                current = (identity, checkpoint, step)
                if checkpoint != expected_root or summary.get("ckpt_step") != step:
                    break
                if provenance is not None and provenance != current:
                    break
                repeat_records = summary.get("repeats")
                if repeat_records is not None and (
                    not isinstance(repeat_records, list)
                    or len(repeat_records) != repeats
                    or any(
                        not isinstance(item, dict)
                        or item.get("ckpt_step") != step
                        or submission_model_identity(item.get("model_path")) != identity
                        or (
                            benchmark in CODE_BENCHMARKS
                            and not _full_code_benchmark(item, benchmark)
                        )
                        or if_coverage_problem(item, benchmark)
                        for item in repeat_records
                    )
                ):
                    break
            except (OSError, ValueError, TypeError, KeyError):
                break
            provenance = current
            model_path = current_model
            scores[benchmark] = mean
        if set(scores) != required or provenance is None:
            continue
        margins = [
            scores[benchmark] - threshold for benchmark, threshold in targets.items()
        ]
        if min(margins) < 0:
            continue
        identity, checkpoint, step = provenance
        if required_model_identity is not None and identity != required_model_identity:
            continue
        eligible.append(
            (
                (min(margins), sum(margins)),
                {
                    "iteration": record.get("iteration"),
                    "run_name": record.get("run_name"),
                    "checkpoint_root": checkpoint,
                    "ckpt_step": step,
                    "model_path": model_path,
                    "model_identity": identity,
                    "scores": scores,
                    "n_repeats": repeats,
                },
            )
        )
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def finalization_matches_target(
    run_dir: Path,
    targets: dict[str, float],
    *,
    benchmarks: list[str] | None = None,
    repeats: int | None = None,
) -> bool:
    """Require the finalized model's provenance to match a qualifying model."""
    run_dir = Path(run_dir).resolve()
    try:
        summary = json.loads(
            (run_dir / "final_model_summary.json").read_text(encoding="utf-8")
        )
        if not isinstance(summary, dict):
            return False
        output = (
            summary.get("final_model_dir") or summary.get("model_dir") or "final-model"
        )
        if not isinstance(output, str):
            return False
        model = Path(output)
        model = (model if model.is_absolute() else run_dir / model).resolve()
        if not model.is_relative_to(run_dir) or not model.is_dir():
            return False
        from autosft.orchestration.completion_evidence import model_assets_complete

        if not model_assets_complete(model):
            return False
        identity = submission_model_identity(model)
    except (OSError, ValueError, TypeError):
        return False
    return (
        target_checkpoint(
            run_dir,
            targets,
            benchmarks=benchmarks,
            repeats=repeats,
            required_model_identity=identity,
        )
        is not None
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoSFT cross-iteration experiment ledger."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_cmd = subparsers.add_parser(
        "summarize", help="Print the recipe→result table."
    )
    summarize_cmd.add_argument("--run-dir", type=Path, default=default_run_dir())
    summarize_cmd.add_argument(
        "--no-md", action="store_true", help="Do not (re)write iterations.md."
    )

    record_cmd = subparsers.add_parser(
        "record-eval", help="Attach eval scores to an iteration."
    )
    record_cmd.add_argument("--run-dir", type=Path, required=True)
    record_cmd.add_argument("--eval-manifest", type=Path, required=True)
    record_cmd.add_argument("--iteration", type=int, default=None)

    note_cmd = subparsers.add_parser("note", help="Attach a rationale to an iteration.")
    note_cmd.add_argument("--run-dir", type=Path, default=default_run_dir())
    note_cmd.add_argument("--rationale", required=True)
    note_cmd.add_argument("--expectation", default=None)
    note_cmd.add_argument("--evidence-plan", default=None)
    note_target = note_cmd.add_mutually_exclusive_group()
    note_target.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Submission-order ledger index, not the data round ID.",
    )
    note_target.add_argument(
        "--round-id",
        type=logical_round_id,
        default=None,
        help="Logical data round; a pre-submission note waits for this exact round.",
    )
    note_cmd.add_argument(
        "--force-restage",
        action="store_true",
        help=(
            "Revise an already-staged rationale while keeping earlier notes in history. "
            "Without this flag, staging over a pending note is an error."
        ),
    )
    note_cmd.add_argument(
        "--deviation",
        action="append",
        default=[],
        help=("Material deviation from what this round set out to do; repeatable."),
    )

    review_cmd = subparsers.add_parser(
        "review", help="Append a reflection on the iteration's current evidence."
    )
    review_cmd.add_argument("--run-dir", type=Path, default=default_run_dir())
    review_target = review_cmd.add_mutually_exclusive_group(required=True)
    review_target.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Submission-order ledger index, not the data round ID.",
    )
    review_target.add_argument(
        "--round-id",
        type=logical_round_id,
        default=None,
        help="Logical data round of an existing ledger record.",
    )
    review_cmd.add_argument("--review", required=True)

    hypothesis_cmd = subparsers.add_parser(
        "hypothesis",
        help="Maintain the run's cross-round hypothesis history.",
    )
    hypothesis_sub = hypothesis_cmd.add_subparsers(
        dest="hypothesis_command", required=True
    )

    record_hypothesis_cmd = hypothesis_sub.add_parser(
        "record",
        help="Record an optional immutable hypothesis statement.",
    )
    record_hypothesis_cmd.add_argument(
        "--run-dir", type=Path, default=default_run_dir()
    )
    record_hypothesis_cmd.add_argument("--round-id", required=True)
    record_hypothesis_cmd.add_argument("--hypothesis-id", default=None)
    record_hypothesis_cmd.add_argument(
        "--hypothesis",
        required=True,
        help="A consequential uncertain judgment that guides data decisions.",
    )
    record_hypothesis_cmd.add_argument(
        "--planned-action",
        required=True,
        help="What will be done to the data.",
    )
    record_hypothesis_cmd.add_argument(
        "--reason",
        required=True,
        help="Why this action, given the evidence read this round.",
    )
    record_hypothesis_cmd.add_argument(
        "--baseline-round",
        default=None,
        help=(
            "The evaluated round this round's score change is read against "
            "(0 for the bootstrap round's base model)."
        ),
    )
    record_hypothesis_cmd.add_argument(
        "--expected-direction",
        default=None,
        choices=list(EXPECTED_DIRECTIONS),
        help="What the benchmark mean should do relative to the baseline round.",
    )
    record_hypothesis_cmd.add_argument(
        "--expected-effect",
        default=None,
        help="Predicted rough size, as calibration context — not a criterion.",
    )
    record_hypothesis_cmd.add_argument(
        "--mechanism-observable",
        default=None,
        help=(
            "The fine-grained phenomenon the eval artifacts should show if "
            "the hypothesis is right — any recorded quantity the next round "
            "can look up, not a restated score."
        ),
    )

    judge_hypothesis_cmd = hypothesis_sub.add_parser(
        "judge",
        help="Append a detailed score/mechanism assessment to a recorded hypothesis.",
    )
    judge_hypothesis_cmd.add_argument("--run-dir", type=Path, default=default_run_dir())
    judge_hypothesis_cmd.add_argument(
        "--round-id", required=True, help="The round being judged."
    )
    judge_hypothesis_cmd.add_argument("--hypothesis-id", default=None)
    judge_hypothesis_cmd.add_argument(
        "--score-verdict", required=True, choices=list(SCORE_VERDICTS)
    )
    judge_hypothesis_cmd.add_argument(
        "--score-basis",
        required=True,
        help=(
            "The observed means vs the baseline round's, weighed against "
            "the eval-repeat spread."
        ),
    )
    judge_hypothesis_cmd.add_argument(
        "--mechanism-verdict",
        required=True,
        choices=list(MECHANISM_VERDICTS),
        help="Use inconclusive when evidence cannot establish the mechanism.",
    )
    judge_hypothesis_cmd.add_argument(
        "--mechanism-basis",
        required=True,
        help=("The fine-grained numbers for the recorded mechanism_observable."),
    )
    judge_hypothesis_cmd.add_argument(
        "--lesson",
        required=True,
        help=(
            "Your own takeaway from the two verdicts — what this outcome "
            "means for later rounds, in your own words. It may push back on "
            "an earlier round's lesson."
        ),
    )
    judge_hypothesis_cmd.add_argument(
        "--verdict-round",
        default=None,
        help=(
            "The round whose root is judging. Defaults to the round being judged, "
            "which requires recorded results. An earlier round is not allowed."
        ),
    )

    update_hypothesis_cmd = hypothesis_sub.add_parser(
        "update", help="Append an explicit status judgment and its evidence."
    )
    update_hypothesis_cmd.add_argument(
        "--run-dir", type=Path, default=default_run_dir()
    )
    update_hypothesis_cmd.add_argument("--hypothesis-id", required=True)
    update_hypothesis_cmd.add_argument(
        "--status", required=True, choices=list(HYPOTHESIS_STATUSES)
    )
    update_hypothesis_cmd.add_argument("--evidence", required=True)
    update_hypothesis_cmd.add_argument("--at-round", required=True)

    status_hypothesis_cmd = hypothesis_sub.add_parser(
        "status",
        help="Print complete judgment history and which hypotheses remain open.",
    )
    status_hypothesis_cmd.add_argument(
        "--run-dir", type=Path, default=default_run_dir()
    )

    scores_cmd = subparsers.add_parser(
        "best-scores", help="Print the best primary_score per benchmark as JSON."
    )
    scores_cmd.add_argument("--run-dir", type=Path, default=default_run_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        if args.run_dir is None:
            raise SystemExit("summarize requires --run-dir or AUTO_SFT_RUN_DIR")
        if not Path(args.run_dir).exists():
            raise SystemExit(f"run dir not found: {args.run_dir}")
        summarize(args.run_dir, write_md=not args.no_md, do_print=True)
        return
    if args.command == "record-eval":
        record_eval(args.run_dir, args.eval_manifest, iteration=args.iteration)
        return
    if args.command == "note":
        if args.run_dir is None:
            raise SystemExit("note requires --run-dir or AUTO_SFT_RUN_DIR")
        add_note(
            args.run_dir,
            args.rationale,
            iteration=args.iteration,
            round_id=args.round_id,
            expectation=args.expectation,
            evidence_plan=args.evidence_plan,
            deviations=args.deviation,
            force_restage=args.force_restage,
        )
        return
    if args.command == "review":
        if args.run_dir is None:
            raise SystemExit("review requires --run-dir or AUTO_SFT_RUN_DIR")
        record_review(
            args.run_dir,
            iteration=args.iteration,
            round_id=args.round_id,
            review=args.review,
        )
        return
    if args.command == "hypothesis":
        if args.run_dir is None:
            raise SystemExit("hypothesis requires --run-dir or AUTO_SFT_RUN_DIR")
        if args.hypothesis_command == "record":
            record_hypothesis(
                args.run_dir,
                round_id=args.round_id,
                hypothesis=args.hypothesis,
                planned_action=args.planned_action,
                reason=args.reason,
                baseline_round=args.baseline_round,
                expected_direction=args.expected_direction,
                expected_effect=args.expected_effect,
                mechanism_observable=args.mechanism_observable,
                hypothesis_id=args.hypothesis_id,
            )
            return
        if args.hypothesis_command == "judge":
            judge_hypothesis(
                args.run_dir,
                round_id=args.round_id,
                score_verdict=args.score_verdict,
                score_basis=args.score_basis,
                mechanism_verdict=args.mechanism_verdict,
                mechanism_basis=args.mechanism_basis,
                lesson=args.lesson,
                verdict_round=args.verdict_round,
                hypothesis_id=args.hypothesis_id,
            )
            return
        if args.hypothesis_command == "update":
            update_hypothesis(
                args.run_dir,
                hypothesis_id=args.hypothesis_id,
                status=args.status,
                evidence=args.evidence,
                at_round=args.at_round,
            )
            return
        if args.hypothesis_command == "status":
            hypothesis_status(args.run_dir)
            return
        return
    if args.command == "best-scores":
        if args.run_dir is None:
            raise SystemExit("best-scores requires --run-dir or AUTO_SFT_RUN_DIR")
        print(json.dumps(best_scores(args.run_dir), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
