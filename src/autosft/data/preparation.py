"""Prepare one round's assembled open-pool training data for Local SFT.

``prepare_recipe`` is the single trust boundary between the agent's free-form
data work and the fixed training pipeline: it strictly validates the minimal
recipe, validates every ChatML row of the referenced ``final_data.jsonl``
against the shared converter's acceptance rule, materializes the training
JSONL (in a seeded random order, so the agent's row order never reaches the
trainer) and public token arrays, and enforces the launcher-owned token budget before any
GPU job can be submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from autosft.data.training_data import TokenDataset

from autosft.core.common import REPO_ROOT, load_structured, utc_now, write_json
from autosft.data.recipe import (
    RecipeValidationError,
    resolve_dataset_path,
    validate_minimal_recipe,
)
from autosft.orchestration.round_gates import (
    assert_data_preparation_safe,
    verify_previous_round_complete,
)
from autosft.orchestration.submission_registry import submission_lock
from autosft.synthesis.usage import maybe_record_syn_usage, verify_syn_round_ready


DATASET_NAME = "final_data"
MAX_ROW_ERRORS = 10

TOKEN_BUDGET_ENV = "AUTO_SFT_SFT_TOKEN_BUDGET"
GLOBAL_BATCH_ENV = "AUTO_SFT_SFT_GLOBAL_BATCH_SIZE"
SEQ_LENGTH_ENV = "AUTO_SFT_SFT_SEQ_LENGTH"
SHUFFLE_SEED_ENV = "AUTO_SFT_SFT_SHUFFLE_SEED"
DEFAULT_SHUFFLE_SEED = 1234
SHUFFLE_METHOD = "random.Random(seed).shuffle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a minimal-recipe round dataset for Local SFT."
    )
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--tokenizer-code-dir", default=os.environ.get("AUTO_SFT_TOKENIZER_CODE_DIR")
    )
    parser.add_argument(
        "--tokenizer-python",
        default=os.environ.get("AUTO_SFT_TOKENIZER_PYTHON", "python"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _row_error(line_number: int, reason: str) -> str:
    return f"line {line_number}: {reason}"


def _invalid_row_reason(row: Any) -> str | None:
    """Why the shared converter (prepare_tokens.encode_row) would drop ``row``.

    Mirrors the converter's ChatML acceptance rule exactly: non-empty
    ``messages`` list, string role/content on every turn, at least one
    assistant turn with non-empty content. Extra keys are allowed and ignored.
    """
    if not isinstance(row, dict):
        return "row must be a JSON object"
    if "input" in row and "output" in row:
        return "ambiguous mixed ChatML and input/output: input/output keys override messages in the converter"
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "row must contain a non-empty messages list"
    has_assistant_content = False
    for index, turn in enumerate(messages):
        if not isinstance(turn, dict):
            return f"messages[{index}] must be an object"
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not role.strip():
            return f"messages[{index}].role must be a non-empty string"
        if not isinstance(content, str):
            return f"messages[{index}].content must be a string"
        if role == "assistant" and content:
            has_assistant_content = True
    if not has_assistant_content:
        return "row needs at least one assistant turn with non-empty content"
    return None


def validate_final_data(dataset_path: Path) -> int:
    """Validate every ChatML row of ``final_data.jsonl`` and return the row count.

    ANY invalid row fails the preparation with up to ``MAX_ROW_ERRORS``
    offending line numbers and reasons, so the agent can fix the data instead
    of discovering silently-dropped rows after an expensive conversion.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"final_data.jsonl does not exist: {dataset_path}")
    if dataset_path.suffix.lower() != ".jsonl":
        raise ValueError(f"recipe dataset path must be a .jsonl file: {dataset_path}")
    rows = 0
    errors: list[str] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row = None
                reason: str | None = f"invalid JSON: {exc}"
            else:
                reason = _invalid_row_reason(row)
            if reason is not None:
                if len(errors) < MAX_ROW_ERRORS:
                    errors.append(_row_error(line_number, reason))
                continue
            rows += 1
    if errors:
        raise ValueError(
            f"final_data.jsonl at {dataset_path} contains invalid rows "
            f"(showing up to {MAX_ROW_ERRORS}):\n- " + "\n- ".join(errors)
        )
    if rows == 0:
        raise ValueError(
            f"final_data.jsonl at {dataset_path} contains no rows; assemble "
            "the training data before preparing the recipe"
        )
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_exposure_for_data(
    dataset_path: Path, run_dir: Path
) -> dict[str, Any] | None:
    """Bind optional assembly evidence to the exact submitted dataset bytes."""
    path = dataset_path.parent / "exposure.json"
    if not path.exists():
        return None
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("candidate exposure must remain under the run directory")
    exposure = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(exposure, dict) or exposure.get("kind") != "candidate_exposure":
        raise ValueError("candidate exposure must be a candidate_exposure object")
    actual = _sha256_file(dataset_path)
    if exposure.get("data_sha256") != actual:
        raise ValueError(
            "candidate exposure is stale: final dataset changed after assembly; "
            "publish a new assembly version before submission"
        )
    summary_keys = (
        "requested_candidate_occurrences",
        "emitted_candidate_occurrences",
        "deduplicated_candidate_occurrences",
        "candidate_quality_rows",
        "base_retained_occurrences",
        "base_replaced_occurrences",
        "total_rows",
        "token_accounting",
        "assembly_manifest",
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "data_sha256": actual,
        **{key: exposure[key] for key in summary_keys if key in exposure},
    }


def resolve_shuffle_seed() -> int:
    """The launcher-owned row-order seed (fixed default; never agent-set)."""
    raw = os.environ.get(SHUFFLE_SEED_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_SHUFFLE_SEED
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{SHUFFLE_SEED_ENV} must be an integer; got {raw!r}") from exc


def materialize_dataset(
    dataset_path: Path,
    run_dir: Path,
    *,
    seed: int,
    dry_run: bool = False,
) -> Path:
    """Write the validated final_data.jsonl to the run's training JSONL location
    in a seeded uniformly random order.

    The trainer packs the resulting token stream in order. Shuffle with the
    launcher-owned seed before conversion so agent-authored row order does
    not control which examples share an optimizer step. The same data and
    seed produce the same stream in each epoch.
    """
    jsonl_dir = run_dir / "data" / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    out_path = jsonl_dir / f"{DATASET_NAME}.jsonl"
    if dry_run:
        return out_path
    with dataset_path.open("rb") as stream:
        rows = [line.rstrip(b"\r\n") for line in stream if line.strip()]
    random.Random(seed).shuffle(rows)
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("wb") as out:
        for row in rows:
            out.write(row)
            out.write(b"\n")
    os.replace(tmp_path, out_path)
    return out_path


def _read_total_tokens(token_file: Path) -> int | None:
    """Read a persisted token count written next to the dataset manifest."""
    try:
        return int(token_file.read_text().strip())
    except (OSError, ValueError):
        return None


def _converter_path(code_dir: str | None) -> Path:
    if not code_dir:
        raise RuntimeError(
            "AUTO_SFT_TOKENIZER_CODE_DIR or --tokenizer-code-dir is required"
        )
    converter = Path(code_dir) / "prepare_tokens.py"
    if not converter.exists():
        raise RuntimeError(f"Token dataset converter not found: {converter}")
    return converter


def tokenizer_subprocess_env() -> dict[str, str]:
    """Keep CPU tokenization independent of optional model backend ABIs."""
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                str(REPO_ROOT / "src"),
                str(REPO_ROOT / "third_party"),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "HF_HUB_OFFLINE": "1",
        "USE_TORCH": "0",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }


def prepare_token_dataset(
    jsonl_path: Path,
    run_dir: Path,
    code_dir: str | None,
    python_bin: str,
    dry_run: bool,
) -> tuple[str, int | None]:
    """Convert training JSONL to public token arrays.

    ``dataset_path`` keeps the submit-script convention ``"<weight> <dir>"``
    with the fixed weight 1.0 (one dataset per round; source-level mixing is
    the agent's job before final_data.jsonl exists).
    """
    tokenized_dir = run_dir / "data" / "tokenized"
    dataset_dir = tokenized_dir / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output = dataset_dir / "dataset.json"
    token_file = dataset_dir / "total_token.txt"
    (dataset_dir / "token_lengths.tsv").unlink(missing_ok=True)
    dataset_path = f"1.0 {dataset_dir}"
    if dry_run:
        return dataset_path, None
    source_hash_file = dataset_dir / "source_sha256.txt"
    source_hash = _sha256_file(jsonl_path) if jsonl_path.exists() else None
    if output.exists():
        recorded_hash = None
        try:
            recorded_hash = source_hash_file.read_text().strip() or None
        except OSError:
            pass
        if source_hash is not None and recorded_hash == source_hash:
            try:
                stored = TokenDataset(output)
                return dataset_path, stored.total_tokens
            except (OSError, ValueError, KeyError, TypeError):
                pass
        shutil.rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
    converter = _converter_path(code_dir)
    proc = subprocess.Popen(
        [
            python_bin,
            str(converter),
            "-i",
            str(jsonl_path),
            "-o",
            str(output),
            "-v",
            "qwen3",
            "--workers",
            os.environ.get("AUTO_SFT_SFT_CONVERT_WORKERS", "1"),
        ],
        cwd=code_dir,
        env=tokenizer_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    total_tokens = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        match = re.search(r"total_token\s+(\d+)", line)
        if match:
            total_tokens = int(match.group(1))
    if proc.wait() != 0:
        raise RuntimeError(
            f"Token dataset conversion failed for {jsonl_path} (exit code {proc.returncode})"
        )
    stored = TokenDataset(output)
    if total_tokens != stored.total_tokens:
        raise RuntimeError("Converter token count disagrees with the persisted dataset")
    if total_tokens is not None:
        token_file.write_text(str(total_tokens) + "\n", encoding="utf-8")
    if source_hash is not None:
        source_hash_file.write_text(source_hash + "\n", encoding="utf-8")
    return dataset_path, (
        total_tokens if total_tokens is not None else _read_total_tokens(token_file)
    )


def count_overlong_rows(
    jsonl_path: Path,
    run_dir: Path,
    code_dir: str | None,
    python_bin: str,
    *,
    seq_length: int,
) -> int | None:
    """Count rows whose tokenized length exceeds ``seq_length``.

    Uses the converter's ``--count-only`` mode (the same ``encode_row`` path
    as dataset conversion, so lengths match ``total_token`` exactly).
    Recorded, not fatal: a counting failure returns None with a warning
    instead of failing the round.

    Store diagnostic lengths separately from the training arrays.
    """
    lengths_file = run_dir / "data" / "token_lengths.tsv"
    try:
        converter = _converter_path(code_dir)
        lengths_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                python_bin,
                str(converter),
                "-i",
                str(jsonl_path),
                "-v",
                "qwen3",
                "--count-only",
                "--count-output",
                str(lengths_file),
                "--workers",
                os.environ.get("AUTO_SFT_SFT_CONVERT_WORKERS", "1"),
            ],
            cwd=code_dir,
            env=tokenizer_subprocess_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        overlong = 0
        with lengths_file.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and int(parts[1]) > seq_length:
                    overlong += 1
        return overlong
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.stderr.write(
            f"WARN: could not count overlong rows for {jsonl_path}: {exc}\n"
        )
        return None


def runtime_tokens_per_iter() -> int:
    """Return the fixed launcher's token throughput for one optimizer step.

    Preparation and submission must use the same runtime knobs. The launcher
    owns these values; this helper only reads them and rejects malformed values
    before a manifest can encode inconsistent step accounting.
    """
    raw_batch = os.environ.get(GLOBAL_BATCH_ENV, "16")
    raw_seq = os.environ.get(SEQ_LENGTH_ENV, "32768")
    try:
        global_batch = int(raw_batch)
        seq_length = int(raw_seq)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{GLOBAL_BATCH_ENV} and {SEQ_LENGTH_ENV} must be positive integers"
        ) from exc
    if global_batch <= 0 or seq_length <= 0:
        raise ValueError(
            f"{GLOBAL_BATCH_ENV} and {SEQ_LENGTH_ENV} must be positive integers"
        )
    return global_batch * seq_length


def runtime_seq_length() -> int:
    """The launcher-owned packed sequence length (for overlong-row accounting)."""
    raw_seq = os.environ.get(SEQ_LENGTH_ENV, "32768")
    try:
        seq_length = int(raw_seq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SEQ_LENGTH_ENV} must be a positive integer") from exc
    if seq_length <= 0:
        raise ValueError(f"{SEQ_LENGTH_ENV} must be a positive integer")
    return seq_length


def resolve_token_budget() -> int:
    """Resolve the round's token budget, which the LAUNCHER owns -- not the recipe.

    The budget gate is unconditional: AUTO_SFT_SFT_TOKEN_BUDGET must be set to
    a positive integer for every prepared round. The minimal recipe schema
    already rejects an inline ``token_budget``, so this is the only budget
    source.
    """
    env_value = os.environ.get(TOKEN_BUDGET_ENV)
    if env_value in (None, ""):
        raise ValueError(
            f"{TOKEN_BUDGET_ENV} is required: the launcher-owned token budget "
            "must be set to a positive integer before preparing round data"
        )
    try:
        budget = int(env_value)
    except ValueError:
        raise ValueError(
            f"{TOKEN_BUDGET_ENV} must be an integer token count; got {env_value!r}"
        ) from None
    if budget <= 0:
        raise ValueError(f"{TOKEN_BUDGET_ENV} must be positive; got {budget}")
    return budget


def verify_token_budget(
    manifest: dict[str, Any],
    *,
    tokens_per_iter: int | None = None,
) -> dict[str, Any]:
    """Check that the round stays within its launcher-owned token cap.

    The converter is what actually produces the trained tokens, so the check
    runs on the realized tokenized volume BEFORE any GPU job is submitted. A
    realized volume below the cap is valid and is recorded as unused budget;
    an over-budget dataset is rejected with the rework message.
    """
    budget = resolve_token_budget()
    if tokens_per_iter is None:
        tokens_per_iter = runtime_tokens_per_iter()
    if (
        isinstance(tokens_per_iter, bool)
        or not isinstance(tokens_per_iter, int)
        or tokens_per_iter <= 0
    ):
        raise ValueError("tokens_per_iter must be a positive integer")
    realized = manifest.get("total_tokens")
    if isinstance(realized, bool) or not isinstance(realized, int) or realized <= 0:
        raise ValueError(
            "the tokenized size of the prepared data is unknown; "
            "cannot verify the round's data volume against the token budget"
        )
    if realized > budget:
        raise ValueError(
            f"Token budget exceeded: {realized} tokens > {budget} cap. The "
            "assembled training data has been rejected; rework the data to "
            "fit the budget and resubmit."
        )
    planned_iters = max(-(-budget // tokens_per_iter), 1)
    realized_iters = max(-(-realized // tokens_per_iter), 1)
    accounting = {
        "token_budget": budget,
        "realized_tokens": realized,
        "deviation": realized - budget,
        "unused_budget_tokens": budget - realized,
        "budget_utilization": realized / budget,
        "tokens_per_iter": tokens_per_iter,
        "planned_iters_per_epoch": planned_iters,
        "realized_iters_per_epoch": realized_iters,
    }
    manifest["token_accounting"] = accounting
    # Keep the cap accounting easy for ledger consumers to read without
    # requiring them to unpack the nested object.
    manifest["unused_budget_tokens"] = accounting["unused_budget_tokens"]
    manifest["realized_iters_per_epoch"] = accounting["realized_iters_per_epoch"]
    return accounting


def record_row_counts(rows: int, run_dir: Path) -> dict[str, Any] | None:
    """Record the final and staged row counts for provenance, without a cap."""
    staging_path = run_dir / "data_pool" / "staging_manifest.json"
    if not staging_path.is_file():
        return None
    try:
        staging = json.loads(staging_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(staging, dict):
        return None
    known_rows_total = staging.get("known_rows_total")
    if isinstance(known_rows_total, bool) or not isinstance(known_rows_total, int):
        return None
    unknown_format_files = staging.get("unknown_format_files")
    unknown = (
        unknown_format_files
        if isinstance(unknown_format_files, int)
        and not isinstance(unknown_format_files, bool)
        else 0
    )
    if unknown > 0:
        return None
    return {
        "rows": rows,
        "known_rows_total": known_rows_total,
        "rows_delta": rows - known_rows_total,
        "enforced": False,
    }


def prepare_recipe(
    recipe_path: Path,
    run_dir: Path,
    *,
    tokenizer_code_dir: str | None = None,
    tokenizer_python: str = "python",
    dry_run: bool = False,
    _submission_locked: bool = False,
) -> dict[str, Any]:
    recipe_path = Path(recipe_path)
    run_dir = Path(run_dir)
    if not _submission_locked:
        # Direct preparation and the SFT entrypoint must serialize the whole
        # check/materialize/publish sequence with task submission.
        with submission_lock(run_dir):
            return prepare_recipe(
                recipe_path,
                run_dir,
                tokenizer_code_dir=tokenizer_code_dir,
                tokenizer_python=tokenizer_python,
                dry_run=dry_run,
                _submission_locked=True,
            )
    recipe = load_structured(recipe_path)
    try:
        validate_minimal_recipe(recipe, run_dir=run_dir)
    except RecipeValidationError as exc:
        raise ValueError(f"invalid recipe {recipe_path}: {exc}") from exc
    verify_previous_round_complete(recipe, run_dir)
    assert_data_preparation_safe(run_dir)
    verify_syn_round_ready(run_dir, int(recipe["round_id"]))
    # The unconditional budget gate: fail before any expensive work when the
    # launcher-owned budget is missing or malformed.
    token_budget = resolve_token_budget()

    dataset_path = resolve_dataset_path(recipe, run_dir=run_dir)
    rows = validate_final_data(dataset_path)
    candidate_exposure = candidate_exposure_for_data(dataset_path, run_dir)
    row_accounting = record_row_counts(rows, run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    recipe_copy = run_dir / "data-recipe.yaml"
    if recipe_path.resolve() != recipe_copy.resolve():
        shutil.copy2(recipe_path, recipe_copy)

    shuffle_seed = resolve_shuffle_seed()
    jsonl_path = materialize_dataset(
        dataset_path, run_dir, seed=shuffle_seed, dry_run=dry_run
    )
    dataset_tokens, total_tokens = prepare_token_dataset(
        jsonl_path,
        run_dir,
        tokenizer_code_dir,
        tokenizer_python,
        dry_run,
    )

    manifest: dict[str, Any] = {
        "prepared_at": utc_now(),
        "recipe_path": str(recipe_path),
        "recipe_id": recipe.get("recipe_id"),
        "round_id": recipe.get("round_id"),
        "execution_summary": recipe.get("execution_summary"),
        "rewrote_pool": recipe.get("rewrote_pool"),
        "rewritten_row_ids_file": recipe.get("rewritten_row_ids_file"),
        "token_budget": token_budget,
        "run_dir": str(run_dir),
        "dry_run": dry_run,
        "datasets": [
            {
                "path": str(dataset_path),
                "rows": rows,
                "total_tokens": total_tokens,
                "sha256": _sha256_file(dataset_path),
            }
        ],
        "rows": rows,
        "dataset_path": dataset_tokens,
        "overlong_rows": None,
        # The row order the trainer consumes is derived here, not authored by
        # the agent; a later reader comparing rounds needs the seed to know
        # whether two rounds' orders are comparable.
        "shuffle": {
            "seed": shuffle_seed,
            "method": SHUFFLE_METHOD,
            "materialized_sha256": (None if dry_run else _sha256_file(jsonl_path)),
        },
    }
    if row_accounting is not None:
        manifest["row_accounting"] = row_accounting
    if candidate_exposure is not None:
        manifest["candidate_exposure"] = candidate_exposure
    if total_tokens is not None:
        manifest["total_tokens"] = total_tokens
    if not dry_run:
        # Overlong rows are recorded, not fatal: a row longer than the packed
        # sequence length still trains, but the round's record should know it
        # was there.
        manifest["overlong_rows"] = count_overlong_rows(
            jsonl_path,
            run_dir,
            tokenizer_code_dir,
            tokenizer_python,
            seq_length=runtime_seq_length(),
        )
        verify_token_budget(manifest)
        syn_usage = maybe_record_syn_usage(dataset_path, manifest)
        if syn_usage is not None:
            manifest["syn_usage"] = syn_usage
    write_json(run_dir / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_recipe(
        args.recipe,
        args.run_dir,
        tokenizer_code_dir=args.tokenizer_code_dir,
        tokenizer_python=args.tokenizer_python,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
