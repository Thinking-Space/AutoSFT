#!/usr/bin/env python3
from __future__ import annotations
from autosft.config.components import executor_enabled

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from autosft.core.common import (
    REPO_ROOT,
    load_structured,
    run_command,
    utc_now,
    write_json,
)
from autosft.agents.executor_handoff import refresh_executor_handoff
from autosft.orchestration.ledger import record_sft
from autosft.backends.megatron.sft_records import (
    finish_sft_accounting,
    freeze_submission,
    load_submission,
    recipe_for_recovery,
)
from autosft.orchestration.policy_guard import (
    check_policy_or_exit,
    policy_enabled,
    policy_summary,
    record_policy_violation_and_exit,
)
from autosft.orchestration.blocked_rounds import assert_no_unresolved_blocked_round
from autosft.data.preparation import prepare_recipe
from autosft.core.round_ids import (
    RoundIdError,
    logical_id_from_storage,
    logical_round_id,
    storage_round_id,
)
from autosft.orchestration.submission_registry import (
    find_sft_submission,
    sft_submission_conflicts,
    submission_lock,
    submission_task,
)


from autosft.backends.megatron.model_profiles import (
    get_model_profile,
    resolve_model_assets,
)


BASE_MODEL = get_model_profile("1.7b").model_id
TRAIN_EPOCHS = 3
DEFAULT_TOKENIZER_CODE_DIR = REPO_ROOT / "src" / "autosft" / "data"
BASE_TOKENIZER_MODEL, BASE_LOAD_CKPT = resolve_model_assets(get_model_profile("1.7b"))


SFT_HYPERPARAM_ENV = {
    "lr": ("AUTO_SFT_SFT_LR", "1e-4"),
    "min_lr": ("AUTO_SFT_SFT_MIN_LR", "1e-7"),
    "global_batch_size": ("AUTO_SFT_SFT_GLOBAL_BATCH_SIZE", "16"),
    "micro_batch_size": ("AUTO_SFT_SFT_MICRO_BATCH_SIZE", "1"),
    "save_interval": ("AUTO_SFT_SFT_SAVE_INTERVAL", None),
}


def resolve_sft_run_dir(
    requested: Path | None,
    *,
    guarded: bool,
) -> tuple[Path, Path]:
    """Return the launcher-owned run root required by the current contract."""
    policy_value = os.environ.get("AUTO_SFT_RUN_DIR")
    if not policy_value:
        raise SystemExit("AUTO_SFT_RUN_DIR is required for SFT")
    if requested is None:
        raise SystemExit("--run-dir is required for SFT")
    policy_root = Path(policy_value)
    run_dir = Path(requested)
    if run_dir.resolve() != policy_root.resolve():
        message = (
            f"SFT output must equal AUTO_SFT_RUN_DIR: {policy_root}; got {run_dir}"
        )
        if guarded:
            record_policy_violation_and_exit(
                stage="sft",
                run_dir=policy_root,
                rule_id="sft_run_dir_mismatch",
                message=message,
                path=str(run_dir),
            )
        raise SystemExit(message)
    return run_dir, policy_root


def check_recipe_matches_round(
    recipe_path: Path,
    run_dir: Path,
    *,
    run_name: str,
) -> None:
    """Reject any recipe other than the round's canonical data-recipe.yaml.

    Run-root snapshots can contain another round's data or a rejected draft. Only the canonical path tracks the round's current staged content; re-submit with that path after correcting a rejected attempt.
    """
    expected = logical_id_from_storage(run_name, label="--run-name")
    round_recipe = run_dir / "data_rounds" / run_name / "data-recipe.yaml"

    def resolved(path: Path) -> Path:
        # resolve() can itself fail (symlink loops raise RuntimeError); a
        # broken path is an ordinary rejection, not a traceback.
        try:
            return Path(path).resolve()
        except (OSError, RuntimeError):
            return Path(path).absolute()

    try:
        recipe = load_structured(Path(recipe_path))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"sft: could not read --recipe {recipe_path}: {exc}\n"
            f"Fix: pass this round's own recipe, {round_recipe}, and submit again."
        ) from exc
    if not isinstance(recipe, dict):
        raise SystemExit(
            f"sft: --recipe {recipe_path} is not a recipe mapping.\n"
            f"Fix: pass this round's own recipe, {round_recipe}, and submit again."
        )
    try:
        actual = logical_round_id(recipe.get("round_id"), label="recipe.round_id")
    except RoundIdError as exc:
        raise SystemExit(
            f"sft: --recipe {recipe_path} has an unusable round_id: {exc}\n"
            f"Fix: correct round_id to {expected} in the round's recipe "
            f"({round_recipe}) and submit again."
        ) from exc
    if actual == expected:
        if resolved(recipe_path) == resolved(round_recipe):
            return
        snapshot_note = ""
        if resolved(recipe_path) == resolved(run_dir / "data-recipe.yaml"):
            snapshot_note = (
                " It is the run-level snapshot, frozen at the last "
                "prepare_recipe call; after an in-place rework of the round "
                "it still holds the pre-rework text."
            )
        missing_note = ""
        if not round_recipe.is_file():
            missing_note = (
                f"\nThat canonical file does not exist yet. Author the round "
                f"artifacts first (data_rounds/{run_name}/final_data.jsonl "
                f"plus its data-recipe.yaml), then submit."
            )
        raise SystemExit(
            f"sft: --recipe {recipe_path} declares round_id {actual} but is "
            f"not the round's canonical recipe.{snapshot_note} A copy that "
            f"agrees on the round can still disagree on content, and the "
            f"submitted file is what gets archived as the round's recipe.\n"
            f"Nothing was submitted and no local task was created; this "
            f"round is still open. Re-run this same command with "
            f"--recipe {round_recipe}{missing_note}"
        )
    snapshot_hint = ""
    if resolved(recipe_path) == resolved(run_dir / "data-recipe.yaml"):
        snapshot_hint = (
            f"\n{recipe_path} is the run-level snapshot of the last prepared "
            "round, not this round's recipe. It is rewritten on every "
            "submission, so it lags whenever a new round has been authored."
        )
    raise SystemExit(
        f"sft: recipe/round mismatch — --run-name {run_name} expects "
        f"recipe.round_id {expected}, but --recipe {recipe_path} declares "
        f"round_id {actual}. Submitting this way would train round {actual}'s "
        f"data under round {expected}'s name."
        f"{snapshot_hint}\n"
        f"Nothing was submitted and no local task was created; this round is "
        f"still open. Fix and re-submit:\n"
        f"  1. confirm {round_recipe} exists and its round_id is {expected}\n"
        f"  2. re-run this same command with --recipe {round_recipe}\n"
        f"If that file is missing, author the round artifacts first "
        f"(data_rounds/{run_name}/final_data.jsonl plus its data-recipe.yaml), "
        f"then submit."
    )


def recover_existing_sft_run(
    run_dir: Path,
    *,
    run_name: str,
    dry_run: bool,
) -> dict | None:
    """Return a complete local handoff when this run name was already submitted."""
    if dry_run:
        return None
    submission = find_sft_submission(
        run_dir / "sft_submissions.jsonl",
        run_name=run_name,
    )
    if submission is None:
        return None
    run_manifest = load_submission(run_dir, run_name, submission)
    record_submission_ledger(run_dir, run_manifest)
    run_manifest.update(sft_task=submission_task(submission), submission_reused=True)
    write_json(run_dir / "run_manifest.json", run_manifest)
    from autosft.backends.local.tasks import task_path

    task = submission_task(submission)
    if task.startswith("tasks/local-"):
        receipt = json.loads(task_path(run_dir, task).read_text())
        finish_sft_accounting(run_dir, receipt)
    return {
        "status": "existing_task_found",
        "run_name": run_name,
        "task": submission_task(submission),
        "submission_manifest": str(run_dir / "sft_submissions.jsonl"),
        "run_manifest": str(run_dir / "run_manifest.json"),
    }


def record_submission_ledger(run_dir: Path, manifest: dict) -> int:
    """Rebuild a missing ledger row from the frozen submission, never new inputs."""
    return record_sft(
        run_dir,
        run_name=manifest["run_name"],
        recipe_path=recipe_for_recovery(run_dir, manifest),
        data_manifest=manifest["data_manifest"],
        base_model=manifest.get("base_model"),
        initialization=manifest.get("initialization"),
        train_epochs=manifest.get("train_epochs"),
        iters_per_epoch=manifest.get("iters_per_epoch"),
        train_iters=manifest["train_iters"],
        seq_length=manifest["seq_length"],
        hyperparams=manifest.get("hyperparams", {}),
        sft_output_dir=manifest["sft_output_dir"],
    )


def collect_hyperparams() -> dict[str, str]:
    """Capture the SFT knobs (env overrides or their submit-script defaults) for the ledger."""
    params: dict[str, str] = {}
    for name, (env_key, default) in SFT_HYPERPARAM_ENV.items():
        value = os.environ.get(env_key, default)
        if value is not None:
            params[name] = value
    return params


def estimate_iters_per_epoch(args: argparse.Namespace, data_manifest: dict) -> int:
    """Estimate optimizer steps for one pass over the selected tokenized data."""
    total_tokens = (data_manifest or {}).get("total_tokens")
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens <= 0
    ):
        if args.dry_run:
            return 1
        raise SystemExit(
            "Cannot auto-compute --train-iters: tokenized data size is unknown; "
            "prepare the recipe before submitting the round."
        )
    global_batch_size = int(os.environ.get("AUTO_SFT_SFT_GLOBAL_BATCH_SIZE", "16"))
    tokens_per_iter = max(global_batch_size * args.seq_length, 1)
    return max(-(-int(total_tokens) // tokens_per_iter), 1)


def check_token_budget(data_manifest: dict, iters_per_epoch: int) -> dict | None:
    """Fail before submission if the round exceeds its token cap.

    The gate is unconditional: every prepared round carries the
    ``token_accounting`` written by ``prepare_recipe``'s budget verification,
    and a manifest without it must be re-prepared rather than silently
    submitted uncapped. A realized volume below the cap is valid (the
    optimizer-step count is intentionally derived from that actual volume);
    only a dry-run manifest — which skips the SSTable conversion and therefore
    has no realized token count — passes without accounting.
    """
    accounting = (data_manifest or {}).get("token_accounting")
    if not isinstance(accounting, dict):
        if (data_manifest or {}).get("dry_run"):
            return None
        raise SystemExit(
            "Token-budget accounting is missing from data_manifest.json; the "
            "budget gate is unconditional. Re-run prepare_recipe (without "
            "--dry-run) so the tokenized volume is verified against "
            "AUTO_SFT_SFT_TOKEN_BUDGET before submission."
        )
    required = {
        "token_budget",
        "realized_tokens",
        "deviation",
        "unused_budget_tokens",
        "budget_utilization",
        "tokens_per_iter",
        "planned_iters_per_epoch",
        "realized_iters_per_epoch",
    }
    if not required.issubset(accounting):
        raise SystemExit(
            "Token-budget accounting is missing a positive integer cap or "
            "required cap fields; "
            "re-prepare the data."
        )
    budget = accounting["token_budget"]
    realized = accounting["realized_tokens"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        or isinstance(realized, bool)
        or not isinstance(realized, int)
    ):
        raise SystemExit(
            "Token-budget accounting is missing a positive integer cap or "
            "realized token count."
        )
    if realized <= 0:
        raise SystemExit(
            "Token-budget round contains no trainable tokens; refusing to submit."
        )
    if realized > budget:
        raise SystemExit(
            f"Token-budget cap exceeded: {realized} realized tokens > {budget} cap."
        )
    unused = accounting["unused_budget_tokens"]
    deviation = accounting["deviation"]
    tokens_per_iter = accounting["tokens_per_iter"]
    planned_iters = accounting["planned_iters_per_epoch"]
    realized_iters = accounting["realized_iters_per_epoch"]
    utilization = accounting["budget_utilization"]
    if (
        isinstance(unused, bool)
        or not isinstance(unused, int)
        or unused != budget - realized
        or isinstance(deviation, bool)
        or not isinstance(deviation, int)
        or deviation != realized - budget
        or isinstance(tokens_per_iter, bool)
        or not isinstance(tokens_per_iter, int)
        or tokens_per_iter <= 0
        or isinstance(planned_iters, bool)
        or not isinstance(planned_iters, int)
        or planned_iters <= 0
        or isinstance(realized_iters, bool)
        or not isinstance(realized_iters, int)
        or realized_iters <= 0
        or isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(float(utilization))
        or not math.isclose(
            float(utilization), realized / budget, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise SystemExit(
            "Token-budget accounting is inconsistent; re-prepare the data."
        )
    expected_planned_iters = max(-(-budget // tokens_per_iter), 1)
    expected_realized_iters = max(-(-realized // tokens_per_iter), 1)
    if (
        planned_iters != expected_planned_iters
        or realized_iters != expected_realized_iters
    ):
        raise SystemExit(
            "Token-budget accounting has incorrect optimizer-step counts; "
            "re-prepare the data."
        )
    manifest_total = (data_manifest or {}).get("total_tokens")
    if manifest_total is not None and (
        isinstance(manifest_total, bool)
        or not isinstance(manifest_total, int)
        or manifest_total != realized
    ):
        raise SystemExit(
            "Token-budget accounting does not match the prepared total_tokens; "
            "re-prepare the data."
        )
    if realized_iters != iters_per_epoch:
        raise SystemExit(
            f"Token-budget accounting mismatch: preparation recorded "
            f"{realized_iters} iters/epoch but this submission computed "
            f"{iters_per_epoch}; re-prepare the data."
        )
    return accounting


def resolve_train_iters(args: argparse.Namespace, data_manifest: dict) -> int:
    """Resolve the optimizer-step count for this round.

    The step count is NEVER set by the caller — ``--train-iters`` is rejected:

    A round computes one epoch from its prepared data volume, then trains the
    selected data for exactly ``TRAIN_EPOCHS`` passes:

        TRAIN_EPOCHS * ceil(total_tokens / (global_batch_size * seq_length))

    ``total_tokens`` is the tokenized size recorded by ``prepare_recipe``; one
    step consumes ``global_batch_size`` sequences of ``seq_length`` packed
    tokens.
    """
    if args.train_iters and args.train_iters > 0:
        raise SystemExit(
            "--train-iters is not user-settable. A normal round computes it automatically from the "
            f"data volume for {TRAIN_EPOCHS} epochs. "
            "Omit --train-iters."
        )
    return TRAIN_EPOCHS * estimate_iters_per_epoch(args, data_manifest)


def resolve_checkpoint_initialization() -> tuple[str, dict]:
    """Return the fixed Qwen3 base initialization for every SFT round."""
    profile = get_model_profile()
    _, checkpoint = resolve_model_assets(profile)
    return checkpoint, {
        "type": "base",
        "model": profile.model_id,
        "checkpoint": checkpoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed SFT entrypoint for Local control-plane runs."
    )
    parser.add_argument(
        "--recipe", type=Path, required=True, help="Recipe YAML describing HF datasets."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Canonical storage round name: round_<positive integer>.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help="Local directory where the agent downloaded/processed recipe datasets.",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=0,
        help=(
            f"Not user-settable (passing a value errors). A normal round trains for {TRAIN_EPOCHS} "
            "epochs, with one epoch estimated as ceil(total_tokens / "
            "(global_batch_size * seq_length))."
        ),
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=int(os.environ.get("AUTO_SFT_SFT_SEQ_LENGTH", "8192")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_name = storage_round_id(args.run_name, label="--run-name")
        get_model_profile()
    except (RoundIdError, ValueError) as exc:
        raise SystemExit(f"sft: {exc}") from exc

    guarded = policy_enabled()
    run_dir, guard_run_dir = resolve_sft_run_dir(
        args.run_dir,
        guarded=guarded,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Publication shares this lock: one data version must span preparation,
    # task creation, and the recipe/data archive recorded for that task.
    with submission_lock(run_dir):
        _run_locked_round(args, run_name, run_dir, guard_run_dir)


def _run_locked_round(
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    guard_run_dir: Path,
) -> None:
    check_policy_or_exit(
        stage="sft",
        run_dir=guard_run_dir,
        datasets_dir=args.datasets_dir,
        recipe_path=args.recipe,
    )
    # Before recovery or submission: a recipe naming a different round than
    # --run-name means the wrong data would be trained under this round's name.
    check_recipe_matches_round(args.recipe, run_dir, run_name=run_name)
    # And before a NEW round starts: an earlier round whose executor gave up
    # unrecoverably must be accounted for first. Submission is the one
    # chokepoint every round passes, so the gate lives here rather than on
    # recipe authoring -- the meta agent stays free to draft, revise, and review.
    assert_no_unresolved_blocked_round(
        run_dir, requested_round=logical_id_from_storage(run_name)
    )
    recovered = recover_existing_sft_run(
        run_dir,
        run_name=run_name,
        dry_run=args.dry_run,
    )
    if recovered is not None:
        print(json.dumps(recovered, ensure_ascii=False, indent=2))
        if executor_enabled():
            refresh_executor_handoff(run_dir, run_name=run_name)
        return

    data_manifest = prepare_recipe(
        args.recipe,
        run_dir,
        tokenizer_code_dir=os.environ.get(
            "AUTO_SFT_TOKENIZER_CODE_DIR", str(DEFAULT_TOKENIZER_CODE_DIR)
        ),
        tokenizer_python=os.environ.get("AUTO_SFT_TOKENIZER_PYTHON", sys.executable),
        dry_run=args.dry_run,
        _submission_locked=True,
    )
    dataset_path = data_manifest.get("dataset_path")
    if not dataset_path:
        raise SystemExit("recipe preparation did not produce dataset_path")
    sft_output_dir = str(run_dir / "sft" / run_name)

    args.train_iters = resolve_train_iters(args, data_manifest)
    train_epochs = TRAIN_EPOCHS
    iters_per_epoch = estimate_iters_per_epoch(args, data_manifest)
    # Gate before any local submission: the launcher-owned budget is an
    # upper bound, while the actual token volume determines the step count.
    token_accounting = check_token_budget(data_manifest, iters_per_epoch)
    if token_accounting:
        print(
            f"Token budget OK: {token_accounting['realized_tokens']:,} tokens "
            f"(budget {token_accounting['token_budget']:,}, "
            f"unused {token_accounting.get('unused_budget_tokens', token_accounting['token_budget'] - token_accounting['realized_tokens']):,}), "
            f"{iters_per_epoch} iters/epoch."
        )

    env = {
        "AUTO_SFT_SFT_DATASET_PATH": str(dataset_path),
        "AUTO_SFT_SFT_OUTPUT_DIR": sft_output_dir,
        "AUTO_SFT_SFT_MANIFEST": str(run_dir / "sft_submissions.jsonl"),
        "AUTO_SFT_LOCAL_DRY_RUN": "1"
        if args.dry_run
        else os.environ.get("AUTO_SFT_LOCAL_DRY_RUN", "0"),
    }
    load_ckpt, initialization = resolve_checkpoint_initialization()
    env["AUTO_SFT_SFT_TOKENIZER_MODEL"] = resolve_model_assets()[0]
    env["AUTO_SFT_SFT_MODEL"] = get_model_profile().model_id
    # Always pass the configured base checkpoint explicitly, overriding any
    # inherited environment value so the submit helper cannot select a prior
    # round checkpoint.
    env["AUTO_SFT_SFT_LOAD_CKPT"] = load_ckpt

    print(
        f"Fresh fine-tune from {BASE_MODEL} for {train_epochs} epochs "
        f"({iters_per_epoch} iters/epoch, {args.train_iters} total iters)."
    )

    manifest = {
        "run_name": run_name,
        "created_at": utc_now(),
        "base_model": BASE_MODEL,
        "recipe": str(args.recipe),
        "run_dir": str(run_dir),
        "dataset_path": dataset_path,
        "train_epochs": train_epochs,
        "iters_per_epoch": iters_per_epoch,
        "train_iters": args.train_iters,
        "token_accounting": token_accounting,
        "seq_length": args.seq_length,
        "sft_output_dir": sft_output_dir,
        "sft_manifest": env["AUTO_SFT_SFT_MANIFEST"],
        "dry_run": args.dry_run,
        "initialization": initialization,
        "data_manifest": data_manifest,
        "hyperparams": collect_hyperparams(),
        "policy": policy_summary(guard_run_dir),
    }
    if not args.dry_run:
        manifest = freeze_submission(run_dir, manifest, Path(args.recipe))

    completed = None
    existing_submission = None
    try:
        existing_submission = find_sft_submission(
            Path(env["AUTO_SFT_SFT_MANIFEST"]),
            run_name=run_name,
        )
        if existing_submission is not None:
            conflicts = sft_submission_conflicts(
                existing_submission,
                output_dir=Path(sft_output_dir),
                dataset_path=Path(str(dataset_path)),
                train_iters=args.train_iters,
                seq_length=args.seq_length,
            )
            if conflicts:
                raise SystemExit(
                    "SFT run_name already has a different durable submission; "
                    "refusing to create another task:\n- " + "\n- ".join(conflicts)
                )
        else:
            completed = run_command(
                [
                    sys.executable,
                    "-m",
                    "autosft.backends.local.submission",
                    "sft",
                    run_name,
                    str(args.train_iters),
                    str(args.seq_length),
                ],
                env=env,
                cwd=REPO_ROOT,
                dry_run=False,
            )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
            if not exc.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if exc.stderr:
            sys.stderr.write(exc.stderr)
            if not exc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.write(
            f"ERROR: local SFT submission exited with code {exc.returncode}. "
            f"Inspect the submission manifest and local_tasks receipts before retrying.\n"
        )
        raise SystemExit(exc.returncode)
    submission_reused = existing_submission is not None
    if existing_submission is not None:
        print(
            json.dumps(
                {
                    "status": "existing_task_found",
                    "run_name": run_name,
                    "task": submission_task(existing_submission),
                    "submission_manifest": env["AUTO_SFT_SFT_MANIFEST"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif completed is not None:
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    durable_submission = existing_submission
    if durable_submission is None and not args.dry_run:
        durable_submission = find_sft_submission(
            Path(env["AUTO_SFT_SFT_MANIFEST"]),
            run_name=run_name,
        )

    manifest.update(
        sft_task=submission_task(durable_submission) if durable_submission else None,
        submission_reused=submission_reused,
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if not args.dry_run:
        try:
            iteration_index = record_submission_ledger(run_dir, manifest)
            print(
                f"Recorded SFT iteration {iteration_index} in the experiment ledger "
                f"({run_dir / 'iterations.jsonl'})."
            )
        except Exception as exc:
            raise SystemExit(
                f"SFT submission is durable, but its ledger recording failed: {exc}. "
                "Re-run this same src/autosft/backends/megatron/train.py command to repair accounting without resubmitting."
            ) from exc
        if executor_enabled():
            try:
                refresh_executor_handoff(run_dir, run_name=run_name)
            except OSError as exc:
                sys.stderr.write(f"WARN: failed to refresh executor handoff: {exc}\n")


if __name__ == "__main__":
    main()
