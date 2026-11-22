"""Incremental experiment facts alongside clearly separated author notes.

Only small ledger and artifact summaries are read. Unchanged round summaries
are reused by content digest; raw training rows and model predictions are not
repeatedly scanned. This memory never gates submission or chooses a recipe.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import autosft.orchestration.ledger as ledger
from autosft.orchestration.ledger import read_iterations
from autosft.core.round_ids import logical_round_id
from autosft.agents.search_tasks import list_tasks


MEMORY_SCHEMA_VERSION = "2.0"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _artifact(run_dir: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        return {"path": str(path), "status": "outside_run"}
    if not resolved.is_file():
        return {"path": str(resolved), "status": "missing"}
    raw = resolved.read_bytes()
    result = {"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
    except (ValueError, UnicodeDecodeError):
        return {**result, "status": "invalid"}
    return {**result, "status": "observed", "value": value}


def _health(artifact: dict[str, Any], fallback: Any) -> dict[str, Any]:
    payload = artifact.get("value", {})
    tags = payload.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    series = {
        name: values
        for name, values in tags.items()
        if isinstance(values, dict) and " vs samples" not in name
    }
    if not series and isinstance(fallback, dict) and fallback:
        series = {str(fallback.get("loss_tag", "aggregate")): fallback}
    anomalies = []
    for name, values in series.items():
        if values.get("has_nan") or values.get("has_inf"):
            anomalies.append({"series": name, "reason": "nonfinite_values"})
        for key in ("first", "last", "min", "max", "mean"):
            if values.get(key) is not None and not _finite(values[key]):
                anomalies.append({"series": name, "reason": f"invalid_{key}"})
    return {
        "status": "anomaly" if anomalies else "observed" if series else "unknown",
        "anomalies": anomalies,
        "series": {
            name: {
                key: value
                for key, value in values.items()
                if key
                in {
                    "num_points",
                    "first",
                    "last",
                    "min",
                    "max",
                    "mean",
                    "last_step",
                    "last_pct_slope",
                    "has_nan",
                    "has_inf",
                }
            }
            for name, values in series.items()
        },
        "limitation": "Aggregate training loss does not identify slice difficulty, "
        "saturation, interference, or generalization.",
    }


def _scores(record: dict[str, Any]) -> dict[str, Any]:
    output = {}
    evaluations = record.get("eval") or {}
    if not isinstance(evaluations, dict):
        return output
    for benchmark, entry in evaluations.items():
        if not isinstance(entry, dict):
            continue
        scores = entry.get("primary_scores")
        mean = entry.get("primary_score")
        expected = entry.get("n_repeats")
        complete = (
            isinstance(scores, list)
            and bool(scores)
            and all(_finite(score) for score in scores)
            and type(expected) is int
            and expected == len(scores)
            and _finite(mean)
        )
        measured_mean = sum(scores) / len(scores) if complete else None
        if complete and not math.isclose(mean, measured_mean, abs_tol=1e-10):
            complete = False
        output[benchmark] = {
            "status": "complete" if complete else "incomplete_or_unknown",
            "primary_score": mean if _finite(mean) else None,
            "primary_scores": scores,
            "primary_score_sd": entry.get("primary_score_sd"),
            "n_repeats": expected,
            "model_path": entry.get("model_path"),
            "metrics": {
                key: value
                for key, value in entry.items()
                if key.endswith("accuracy") and _finite(value)
            },
        }
    return output


def _exposure(artifact: dict[str, Any], manifest: dict[str, Any]) -> tuple[Any, str]:
    value = artifact.get("value")
    if value is None:
        return None, artifact["status"]
    pinned = manifest.get("candidate_exposure")
    if isinstance(pinned, dict) and pinned.get("sha256") != artifact.get("sha256"):
        return None, "stale"
    datasets = manifest.get("datasets") or []
    digests = {item.get("sha256") for item in datasets if isinstance(item, dict)}
    if digests and value.get("data_sha256") not in digests:
        return None, "stale"
    # Row-by-row inclusion stays in the referenced artifact, not every prompt.
    return {
        key: item for key, item in value.items() if key != "candidate_outcomes"
    }, "observed"


def _evaluation_configuration(warnings: list[str]) -> dict[str, Any]:
    configured = (
        os.environ.get("AUTO_SFT_BENCHMARKS")
        or os.environ.get("AUTO_SFT_EVAL_BENCHMARK")
        or os.environ.get("AUTO_SFT_BENCHMARK")
        or ""
    )
    benchmarks = sorted(set(configured.replace(",", " ").split()))
    raw_repeats = os.environ.get("AUTO_SFT_EVAL_REPEATS")
    repeats = None
    valid = True
    if raw_repeats is not None:
        try:
            repeats = int(raw_repeats)
            if repeats < 1:
                raise ValueError
        except ValueError:
            valid = False
            warnings.append(
                "Invalid configured evaluation repeat count; review readiness is unknown."
            )
    return {"benchmarks": benchmarks, "repeats": repeats, "valid": valid}


def _evaluation_complete(
    record: dict[str, Any], scores: dict[str, Any], configuration: dict[str, Any]
) -> bool:
    evaluations = record.get("eval")
    if not configuration["valid"] or not isinstance(evaluations, dict):
        return False
    required = configuration["benchmarks"] or list(evaluations)
    return bool(required) and all(
        scores.get(benchmark, {}).get("status") == "complete"
        and (
            configuration["repeats"] is None
            or scores[benchmark]["n_repeats"] == configuration["repeats"]
        )
        for benchmark in required
    )


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _decision_notes(
    record: dict[str, Any], label: str, warnings: list[str]
) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    for key in ("rationale", "expectation", "evidence_plan"):
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            warnings.append(
                f"{label}: optional {key} is not text; treated as unavailable."
            )
            value = None
        notes[key] = value
    for key in ("note_history", "reviews"):
        value = record.get(key, [])
        if not isinstance(value, list):
            warnings.append(
                f"{label}: optional {key} is not a list; treated as unavailable."
            )
            value = []
        notes[key] = [item for item in value if isinstance(item, dict)]
        if len(notes[key]) != len(value):
            warnings.append(
                f"{label}: malformed entries in optional {key} were ignored."
            )
    return notes


def _pending_decision(
    run_dir: Path, warnings: list[str], *, round_id: str | None = None
) -> dict[str, Any] | None:
    path = ledger.pending_rationale_path(run_dir, round_id)
    try:
        artifact = _artifact(run_dir, path)
    except OSError as exc:
        warnings.append(f"Optional pending decision is unavailable at {path}: {exc}")
        return None
    if artifact["status"] == "missing":
        return None
    if artifact["status"] != "observed":
        warnings.append(
            f"Optional pending decision is unavailable at {path}: {artifact['status']}"
        )
        return None
    record = artifact["value"]
    recorded_round = record.get("round_id")
    if recorded_round is not None or round_id is not None:
        try:
            recorded_round = logical_round_id(recorded_round)
        except (TypeError, ValueError):
            warnings.append(f"Pending plan has an invalid round identity at {path}.")
            return None
        if round_id is not None and recorded_round != round_id:
            warnings.append(
                f"Pending plan round identity does not match its path: {path}."
            )
            return None
    iteration = record.get("iteration")
    if iteration is not None and (type(iteration) is not int or iteration < 1):
        warnings.append(
            "Pending plan has an invalid iteration; its intended iteration is unknown."
        )
        iteration = None
    notes = _decision_notes(record, "Pending plan", warnings)
    return {
        "source_path": artifact["path"],
        "iteration": iteration,
        **({"round_id": recorded_round} if recorded_round is not None else {}),
        **{
            key: notes[key]
            for key in ("rationale", "expectation", "evidence_plan", "note_history")
        },
    }


def _round_pending_decisions(
    run_dir: Path, warnings: list[str]
) -> list[dict[str, Any]]:
    rounds = []
    for path in ledger.recipe_history_dir(run_dir).glob(
        "round_*.pending_rationale.json"
    ):
        try:
            rounds.append(
                logical_round_id(
                    path.name.removeprefix("round_").removesuffix(
                        ".pending_rationale.json"
                    )
                )
            )
        except (TypeError, ValueError):
            warnings.append(f"Pending plan has an invalid round filename: {path}.")
    decisions = [
        _pending_decision(run_dir, warnings, round_id=round_id)
        for round_id in sorted(rounds, key=int)
    ]
    return [decision for decision in decisions if decision is not None]


def _hypotheses(
    run_dir: Path, warnings: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        status = ledger.hypothesis_status(run_dir, do_print=False)
        entries, opened = status["entries"], status["open"]
        if (
            not isinstance(entries, list)
            or not isinstance(opened, list)
            or any(not isinstance(item, dict) for item in [*entries, *opened])
        ):
            raise ValueError("expected hypothesis entry lists")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        warnings.append(f"Optional hypothesis history is unavailable: {exc}")
        return [], []
    reminders = []
    for entry in opened:
        updates = entry.get("updates")
        latest = updates[-1] if isinstance(updates, list) and updates else None
        reminders.append(
            {
                "hypothesis_id": entry.get("hypothesis_id"),
                "round_id": entry.get("round_id"),
                "claim": entry.get("hypothesis"),
                "status": "open",
                "latest_update": {
                    key: latest[key]
                    for key in ("recorded_at", "status", "evidence", "at_round")
                    if key in latest
                }
                if isinstance(latest, dict)
                else None,
            }
        )
    return entries, reminders


def _learning_reminders(
    rounds: list[dict[str, Any]], opened: list[dict[str, Any]], warnings: list[str]
) -> dict[str, list[dict[str, Any]]]:
    reminders: dict[str, list[dict[str, Any]]] = {
        "missing_expectations": [],
        "unreviewed_rounds": [],
        "stale_reviews": [],
        "open_hypotheses": opened,
    }
    for item in rounds:
        identity = {"round_id": item["round_id"], "iteration": item["iteration"]}
        notes = item["decision_notes"]
        missing = [
            key
            for key in ("expectation", "evidence_plan")
            if not _has_text(notes.get(key))
        ]
        if missing:
            reminders["missing_expectations"].append({**identity, "missing": missing})
        if not item["evaluation_complete"]:
            continue
        reviews = [
            review for review in notes["reviews"] if _has_text(review.get("review"))
        ]
        if not reviews:
            reminders["unreviewed_rounds"].append(
                {**identity, "evidence_digest": item["review_evidence_digest"]}
            )
            continue
        latest = reviews[-1]
        previous_digest, current_digest = (
            latest.get("evidence_digest"),
            item["review_evidence_digest"],
        )
        if not _has_text(previous_digest):
            warnings.append(
                f"Round {item['round_id']}: latest review lacks an evidence digest; freshness is unknown."
            )
        elif current_digest is not None and previous_digest != current_digest:
            reminders["stale_reviews"].append(
                {
                    **identity,
                    "review_recorded_at": latest.get("recorded_at"),
                    "review_evidence_digest": previous_digest,
                    "current_evidence_digest": current_digest,
                }
            )
    return reminders


def format_learning_reminders(memory: dict[str, Any], limit: int = 5) -> str:
    """Bound each reminder kind by limit, prioritizing recently active hypotheses."""
    if not isinstance(memory, dict) or type(limit) is not int or limit < 1:
        return ""
    reminders = memory.get("learning_reminders")
    if not isinstance(reminders, dict):
        reminders = {}

    def compact(value: Any, maximum: int = 120) -> str:
        text = " ".join(str(value or "unknown").split())
        return text if len(text) <= maximum else text[: maximum - 3] + "..."

    def round_label(item: dict[str, Any]) -> str:
        return f"round {compact(item.get('round_id'), 24)} (iteration {compact(item.get('iteration'), 24)})"

    def hypothesis_recency(item: dict[str, Any]) -> tuple[int, int]:
        def number(value: Any) -> int:
            try:
                return int(value) if not isinstance(value, bool) else 0
            except (TypeError, ValueError, OverflowError):
                return 0

        original = number(item.get("round_id"))
        latest = item.get("latest_update")
        updated = number(latest.get("at_round")) if isinstance(latest, dict) else 0
        return updated or original, original

    lines = []
    pending = memory.get("pending_decision")
    pending_plans = [pending] if isinstance(pending, dict) else []
    round_pending = memory.get("pending_round_decisions", [])
    if isinstance(round_pending, list):
        pending_plans.extend(item for item in round_pending if isinstance(item, dict))
    for pending in pending_plans[:limit]:
        details = [
            f"{key}: {compact(pending.get(key))}"
            for key in ("rationale", "expectation", "evidence_plan")
        ]
        source = pending.get("source_path")
        if isinstance(source, str) and source:
            details.append(f"source: {source}")
        identity = (
            f"round {compact(pending['round_id'], 24)}"
            if pending.get("round_id") is not None
            else f"iteration {compact(pending.get('iteration'), 24)}"
        )
        lines.append(
            f"- Pending plan for {identity} (not submitted): {'; '.join(details)}."
        )
    if len(pending_plans) > limit:
        lines.append(
            f"- {len(pending_plans) - limit} additional pending plans are recorded in experiment memory."
        )
    for key, title in (
        ("missing_expectations", "Missing plan notes"),
        ("unreviewed_rounds", "Complete evaluation without a review"),
        ("stale_reviews", "Objective evidence changed since the latest review"),
        ("open_hypotheses", "Open author hypotheses"),
    ):
        entries = reminders.get(key, [])
        if not isinstance(entries, list):
            continue
        entries = [item for item in entries if isinstance(item, dict)]
        if not entries:
            continue
        if key == "open_hypotheses":
            entries = sorted(entries, key=hypothesis_recency, reverse=True)
        parts = []
        for item in entries[:limit]:
            if key == "open_hypotheses":
                text = f"{compact(item.get('hypothesis_id'), 48)} [round {compact(item.get('round_id'), 24)}]: {compact(item.get('claim'))}"
                latest = item.get("latest_update")
                if isinstance(latest, dict):
                    text += f" (latest {compact(latest.get('status'), 24)} at round {compact(latest.get('at_round'), 24)})"
            else:
                text = round_label(item)
                if key == "missing_expectations":
                    missing = item.get("missing", [])
                    if not isinstance(missing, list):
                        missing = []
                    text += ": " + ", ".join(
                        key
                        for key in ("expectation", "evidence_plan")
                        if key in missing
                    )
            parts.append(text)
        suffix = f"; +{len(entries) - limit} more" if len(entries) > limit else ""
        lines.append(f"- {title}: {'; '.join(parts)}{suffix}.")
    if not lines:
        return ""
    return "\n".join(
        [
            "Learning reminders are advisory and never block submission, recovery, or finalization.",
            *lines,
            "Keep retrospective notes distinct from prior predictions; no new hypothesis is required.",
        ]
    )


def build_memory(
    run_dir: Path, previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    old_rounds = (
        (previous or {}).get("rounds", [])
        if (previous or {}).get("schema_version") == MEMORY_SCHEMA_VERSION
        else []
    )
    cached = (
        {item.get("round_id"): item for item in old_rounds if isinstance(item, dict)}
        if isinstance(old_rounds, list)
        else {}
    )
    rounds = []
    warnings = []
    configuration = _evaluation_configuration(warnings)
    for record in read_iterations(run_dir):
        recipe = record.get("recipe") or {}
        if not isinstance(recipe, dict):
            warnings.append("A ledger iteration has no usable recipe object.")
            continue
        try:
            round_id = ledger.record_round_id(record)
            if round_id is None:
                raise ValueError("missing round ID")
        except (ValueError, TypeError):
            warnings.append(f"Invalid round ID in iteration {record.get('iteration')}")
            continue
        archive = record.get("recipe_archive") or {}
        if not isinstance(archive, dict):
            archive = {}
        paths = {
            "data_manifest": Path(
                archive.get("data_manifest")
                or run_dir
                / "recipe_history"
                / f"iter_{record.get('iteration', 0):03d}.data_manifest.json"
            ),
            "loss": run_dir / "sft" / f"round_{round_id}" / "loss_summary.json",
            "exposure": run_dir / "data_rounds" / f"round_{round_id}" / "exposure.json",
            "assembly": run_dir
            / "data_rounds"
            / f"round_{round_id}"
            / "candidate_assembly.json",
        }
        artifacts = {
            name: _artifact(run_dir, path if path.is_absolute() else run_dir / path)
            for name, path in paths.items()
        }
        fingerprint = _digest(
            {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "record": record,
                "artifacts": artifacts,
                "evaluation_configuration": configuration,
            }
        )
        notes = _decision_notes(record, f"Round {round_id}", warnings)
        if cached.get(round_id, {}).get("input_digest") == fingerprint:
            summary = cached[round_id]
        else:
            manifest = artifacts["data_manifest"].get("value", {})
            exposure, exposure_status = _exposure(artifacts["exposure"], manifest)
            scores = _scores(record)
            summary = {
                "round_id": round_id,
                "iteration": record.get("iteration"),
                "input_digest": fingerprint,
                "recipe_id": recipe.get("recipe_id"),
                "execution_summary": recipe.get("execution_summary"),
                "datasets": manifest.get("datasets", recipe.get("datasets", [])),
                "rows": manifest.get("rows", recipe.get("rows")),
                "total_tokens": manifest.get(
                    "total_tokens", recipe.get("total_tokens")
                ),
                "train_iters": (record.get("hyperparams") or {}).get("train_iters"),
                "scores": scores,
                "evaluation_complete": _evaluation_complete(
                    record, scores, configuration
                ),
                "review_evidence_digest": ledger.review_evidence_digest(record),
                "decision_notes": notes,
                "training_health": _health(
                    artifacts["loss"], record.get("training_health")
                ),
                "exposure": exposure,
                "exposure_status": exposure_status,
                "assembly": artifacts["assembly"].get("value"),
                "evidence": {
                    name: {key: value for key, value in item.items() if key != "value"}
                    for name, item in artifacts.items()
                },
            }
        rounds.append(summary)
    best = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for item in rounds:
        for benchmark, score in item["scores"].items():
            if score["status"] != "complete":
                continue
            if (
                benchmark not in best
                or score["primary_score"] > best[benchmark]["primary_score"]
            ):
                best[benchmark] = {"round_id": item["round_id"], **score}
            curves.setdefault(benchmark, []).append(
                {
                    "iteration": item["iteration"],
                    "round_id": item["round_id"],
                    "score": score["primary_score"],
                    "best_observed": best[benchmark]["primary_score"],
                }
            )
    hypothesis_history, open_hypotheses = _hypotheses(run_dir, warnings)
    reminders = _learning_reminders(rounds, open_hypotheses, warnings)
    pending_decision = _pending_decision(run_dir, warnings)
    pending_round_decisions = _round_pending_decisions(run_dir, warnings)
    return _json_safe(
        {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "rounds": rounds,
            "pending_decision": pending_decision,
            "pending_round_decisions": pending_round_decisions,
            "hypothesis_history": hypothesis_history,
            "learning_reminders": reminders,
            "task_index": [
                {
                    key: value
                    for key, value in task.items()
                    if key
                    in {
                        "round_id",
                        "task_id",
                        "role",
                        "status",
                        "detail",
                        "dependencies",
                        "input_digest",
                        "task_dir",
                        "report_path",
                        "report_sha256",
                    }
                }
                for task in list_tasks(run_dir)
            ],
            "best_observed": best,
            "score_curves": curves,
            "warnings": warnings,
            "limitations": [
                "Best observed scores are adaptively selected, not independent confirmation.",
                "Repeat spread measures decoding variation, not architecture uncertainty.",
                "Absent exposure is unknown, not evidence that an operation was tested.",
                "Per-benchmark best rounds need not be the same checkpoint.",
                "Task reports are discoverable evidence, not proof that a proposal entered training.",
                "Decision notes, reviews and hypothesis history are meta-authored interpretations, not objective observations.",
                "Learning reminders are advisory and never gate submission, recovery or finalization.",
                "Without benchmark configuration, review readiness covers only recorded benchmarks; missing benchmarks cannot be inferred.",
                "Without a configured repeat count, review readiness uses each recorded benchmark's declared repeat count.",
            ],
        }
    )


def refresh_memory(run_dir: Path) -> Path:
    roles = {
        os.environ.get(name)
        for name in ("AUTO_SFT_ROLE", "AUTO_SFT_TASK_ROLE", "AUTO_SFT_SEARCH_ROLE")
    } - {None, "", "meta"}
    if (
        roles
        or os.environ.get("AUTO_SFT_TASK_DIR")
        or os.environ.get("AUTO_SFT_SEARCH_TASK_DIR")
    ):
        raise ValueError(
            "task roles read experiment memory; only the meta agent or the harness refreshes it"
        )
    run_dir = Path(run_dir).resolve()
    path = run_dir / "experiment_memory.json"
    with (run_dir / ".experiment_memory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = None
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                pass
        if not isinstance(previous, dict):
            previous = None
        payload = build_memory(run_dir, previous)
        if previous == payload:
            return path
        fd, temporary = tempfile.mkstemp(prefix=".experiment_memory-", dir=run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh", action="store_true", help="Persist the incremental memory"
    )
    args = parser.parse_args()
    if args.refresh:
        print(refresh_memory(args.run_dir))
    else:
        print(json.dumps(build_memory(args.run_dir), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
