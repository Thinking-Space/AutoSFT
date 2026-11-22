"""Resolve experiment blocks into an immutable run protocol."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess

from autosft.config.components import AgentComponents, COMPONENT_NAMES
from autosft.core.common import REPO_ROOT


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key != "target_scores"
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.expanduser().resolve()
    if path in ancestors:
        raise ValueError("experiment inheritance contains a cycle")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be an object")
    parent = payload.pop("extends", None)
    return (
        _merge(load_config(path.parent / parent, (*ancestors, path)), payload)
        if parent
        else payload
    )


def validate_config(config: dict) -> None:
    allowed = {
        "name",
        "domain",
        "components",
        "objective",
        "resources",
        "training",
        "evaluation",
        "data",
        "agent",
        "providers",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown experiment blocks: {sorted(unknown)}")
    if config.get("domain") != "sft_data":
        raise ValueError("this release implements the sft_data optimization domain")
    names = config.get("components")
    if not isinstance(names, list) or any(
        not isinstance(name, str) or name not in COMPONENT_NAMES for name in names
    ):
        raise ValueError("components must be a list of names")
    AgentComponents.parse(",".join(names))
    training, evaluation = config["training"], config["evaluation"]
    optimizer = training["optimizer"]
    if optimizer.get("name") != "adam" or optimizer.get("schedule") != "cosine":
        raise ValueError(
            "the SFT adapter supports Adam with cosine learning-rate decay"
        )
    for key in ("lr", "min_lr", "eps", "weight_decay", "warmup_fraction"):
        value = optimizer.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"optimizer.{key} must be a finite non-negative number")
    if (
        optimizer["lr"] <= 0
        or optimizer["eps"] <= 0
        or optimizer["min_lr"] > optimizer["lr"]
        or optimizer["warmup_fraction"] >= 1
    ):
        raise ValueError("invalid optimizer learning-rate, epsilon or warmup settings")
    if (
        not isinstance(optimizer.get("betas"), list)
        or len(optimizer["betas"]) != 2
        or any(
            type(beta) not in (int, float) or not 0 <= beta < 1
            for beta in optimizer["betas"]
        )
    ):
        raise ValueError("Adam needs two beta values in [0, 1)")
    if training["model"] != "Qwen/Qwen3-1.7B-Base" or training["backend"] != "megatron":
        raise ValueError("this SFT backend supports Qwen3-1.7B-Base with Megatron")
    if training["epochs"] != 3:
        raise ValueError("the paper protocol uses three epochs per round")
    for block, fields in (
        (
            training,
            (
                "seq_length",
                "global_batch_size",
                "micro_batch_size",
                "context_parallel",
                "tensor_parallel",
            ),
        ),
        (config["resources"], ("token_budget", "max_rounds", "train_gpus")),
        (evaluation, ("repeats",)),
    ):
        for field in fields:
            if type(block[field]) is not int or block[field] <= 0:
                raise ValueError(f"{field} must be a positive integer")
    from autosft.benchmarks.catalog import CODE_BENCHMARKS, unsupported

    benchmarks = evaluation["benchmarks"]
    if (
        not isinstance(benchmarks, list)
        or not benchmarks
        or any(not isinstance(name, str) for name in benchmarks)
        or len(benchmarks) != len(set(benchmarks))
        or unsupported(benchmarks)
    ):
        raise ValueError("experiment names an unavailable benchmark")
    code = set(benchmarks) & CODE_BENCHMARKS
    if code and code != set(benchmarks):
        raise ValueError(
            "use separate experiments for code and instruction-following backends"
        )
    allowed_backends = (
        {"evalplus_vllm", "evalplus_sglang"} if code else {"opencompass_vllm"}
    )
    if evaluation.get("backend") not in allowed_backends:
        raise ValueError(
            f"selected benchmarks require one of {sorted(allowed_backends)}"
        )
    if type(evaluation.get("shots")) is not int or evaluation["shots"] != 0:
        raise ValueError("the evaluation adapters support zero-shot prompting")
    temperature = evaluation.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("evaluation.temperature must be finite and non-negative")
    for key in ("max_seq_len", "max_out_len"):
        if key in evaluation and (
            type(evaluation[key]) is not int or evaluation[key] <= 0
        ):
            raise ValueError(f"evaluation.{key} must be a positive integer")
    targets = config["objective"]["target_scores"]
    if not targets or set(targets) != set(evaluation["benchmarks"]):
        raise ValueError("every evaluated benchmark needs one target score")
    if any(
        type(value) not in (int, float) or not 0 <= value <= 1
        for value in targets.values()
    ):
        raise ValueError("target scores must be finite fractions in [0, 1]")
    gpus, cp, tp = (
        config["resources"]["train_gpus"],
        training["context_parallel"],
        training["tensor_parallel"],
    )
    if gpus % (cp * tp) or training["global_batch_size"] % (
        training["micro_batch_size"] * gpus // (cp * tp)
    ):
        raise ValueError("GPU, parallelism and batch geometry are incompatible")


def environment(config: dict, base: dict[str, str] | None = None) -> dict[str, str]:
    validate_config(config)
    env = dict(os.environ if base is None else base)
    t, e, r = config["training"], config["evaluation"], config["resources"]
    values = {
        "AUTO_SFT_COMPONENTS": ",".join(config["components"]) or "none",
        "AUTO_SFT_OBJECTIVE": config["objective"]["description"],
        "AUTO_SFT_LLM_MODEL": env.get("AUTO_SFT_LLM_MODEL")
        or config["providers"]["generation_model"],
        "AUTO_SFT_RESEARCH_MODEL": config["agent"]["model"],
        "AUTO_SFT_REASONING_EFFORT": config["agent"]["reasoning_effort"],
        "AUTO_SFT_DATA_MODE": "open_pool",
        "AUTO_SFT_SOURCE_DATASETS_DIR": str(REPO_ROOT / config["data"]["pool"]),
        "AUTO_SFT_SFT_TOKEN_BUDGET": r["token_budget"],
        "AUTO_SFT_MAX_ROUNDS": r["max_rounds"],
        "AUTO_SFT_SFT_MODEL": t["model"],
        "AUTO_SFT_SFT_SEQ_LENGTH": t["seq_length"],
        "AUTO_SFT_SFT_GLOBAL_BATCH_SIZE": t["global_batch_size"],
        "AUTO_SFT_SFT_MICRO_BATCH_SIZE": t["micro_batch_size"],
        "AUTO_SFT_SFT_CP_SIZE": t["context_parallel"],
        "AUTO_SFT_SFT_TP": t["tensor_parallel"],
        "AUTO_SFT_SFT_GPUS_PER_NODE": r["train_gpus"],
        "AUTO_SFT_SFT_LR": t["optimizer"]["lr"],
        "AUTO_SFT_SFT_MIN_LR": t["optimizer"]["min_lr"],
        "AUTO_SFT_SFT_ADAM_BETA1": t["optimizer"]["betas"][0],
        "AUTO_SFT_SFT_ADAM_BETA2": t["optimizer"]["betas"][1],
        "AUTO_SFT_SFT_ADAM_EPS": t["optimizer"]["eps"],
        "AUTO_SFT_SFT_WEIGHT_DECAY": t["optimizer"]["weight_decay"],
        "AUTO_SFT_SFT_WARMUP_FRACTION": t["optimizer"]["warmup_fraction"],
        "AUTO_SFT_SFT_DETERMINISTIC": int(t.get("deterministic", False)),
        "AUTO_SFT_BENCHMARKS": ",".join(e["benchmarks"]),
        "AUTO_SFT_EVAL_REPEATS": e["repeats"],
        "AUTO_SFT_OPENCOMPASS_TEMPERATURE": e["temperature"],
        "AUTO_SFT_EVAL_BACKEND": "opencompass",
        "AUTO_SFT_CODE_INFERENCE_BACKEND": "sglang"
        if e["backend"] == "evalplus_sglang"
        else "vllm",
        "AUTO_SFT_TARGET_SCORES": ",".join(
            f"{key}={value}"
            for key, value in config["objective"]["target_scores"].items()
        ),
        "AUTO_SFT_AGENT_RUNS_DIR": str(REPO_ROOT / "runs" / config["name"]),
        "AUTO_SFT_TOKENIZER_CODE_DIR": str(REPO_ROOT / "src/autosft/data"),
        "AUTO_SFT_EXPERIMENT_JSON": json.dumps(config, sort_keys=True),
    }
    env.update({key: str(value) for key, value in values.items()})
    for key, variable in (
        ("max_seq_len", "AUTO_SFT_OPENCOMPASS_MAX_SEQ_LEN"),
        ("max_out_len", "AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN"),
    ):
        if key in e:
            env[variable] = str(e[key])
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--components", help="Comma-separated components, or none")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved public experiment protocol without launching jobs",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.components is not None:
        config["components"] = list(AgentComponents.parse(args.components).names)
    env = environment(config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    return subprocess.call(
        ["bash", str(REPO_ROOT / "scripts/agents/launch.sh")], cwd=REPO_ROOT, env=env
    )


if __name__ == "__main__":
    raise SystemExit(main())
