"""Local IF inference matching the optimization reference's no-think protocol."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re

from autosft.core.common import write_json
from autosft.benchmarks.opencompass_bridge import _write_artifacts, repeat_output_dirs


def visible_answer(text: str) -> str:
    """Apply the reference model postprocessor before instruction checking."""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    text = text.strip()
    return "" if len(text) > 81920 else text


def run_if(record: dict, *, root: Path, score_writer) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    benchmark = record["benchmark"]
    if benchmark == "ifeval":
        from opencompass_local.IFEval.ifeval import IFEvaluator as Evaluator
    elif benchmark == "ifbench":
        from opencompass_local.IFBench.ifbench import IFBenchEvaluator as Evaluator
    else:
        raise ValueError(f"unsupported local IF benchmark: {benchmark}")
    limit, repeats = int(record["limit"]), int(record["repeats"])
    if limit < -1 or repeats < 1:
        raise ValueError("limit must be -1 or nonnegative and repeats must be positive")
    data_path = root / "assets/benchmarks" / benchmark / "input_data.jsonl"
    rows = [
        json.loads(line) for line in data_path.read_text().splitlines() if line.strip()
    ]
    full_num_prompts = len(rows)
    if limit != -1:
        rows = rows[:limit]
    if not rows:
        raise ValueError("evaluation needs at least one prompt")
    tokenizer = AutoTokenizer.from_pretrained(
        record["model_path"], local_files_only=True
    )
    template = (root / "src/autosft/benchmarks/templates/qwen3.jinja").read_text()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    config = json.loads((Path(record["model_path"]) / "config.json").read_text())
    generation_path = Path(record["model_path"]) / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    )
    model_output_cap = generation.get("max_new_tokens")
    if model_output_cap is not None and (
        type(model_output_cap) is not int or model_output_cap <= 0
    ):
        raise ValueError("model max_new_tokens must be a positive integer")
    repetition_penalty = float(generation.get("repetition_penalty", 1.0))
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("model repetition_penalty must be positive and finite")
    model_context = int(config.get("max_position_embeddings", 32768))
    max_seq = int(os.environ.get("AUTO_SFT_OPENCOMPASS_MAX_SEQ_LEN") or model_context)
    if not 0 < max_seq <= model_context:
        raise ValueError(
            "evaluation context must be positive and within the converted model context"
        )
    temperature = float(os.environ.get("AUTO_SFT_OPENCOMPASS_TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("AUTO_SFT_EVAL_TOP_P", "0.8"))
    top_k = int(os.environ.get("AUTO_SFT_EVAL_TOP_K", "20"))
    min_p = float(os.environ.get("AUTO_SFT_EVAL_MIN_P", "0"))
    if (
        not math.isfinite(temperature)
        or temperature < 0
        or not 0 < top_p <= 1
        or not 0 <= min_p <= 1
        or (top_k != -1 and top_k < 1)
    ):
        raise ValueError("invalid IF sampling settings")
    seed_value = os.environ.get("AUTO_SFT_EVAL_SEED", "").strip()
    seed = int(seed_value) if seed_value else None
    if seed is not None and seed < 0:
        raise ValueError("evaluation seed must be nonnegative")
    cap_value = os.environ.get("AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN", "").strip()
    output_cap = int(cap_value) if cap_value else None
    if output_cap is not None and output_cap <= 0:
        raise ValueError("output token cap must be positive")
    available = [
        max_seq - len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts
    ]
    if min(available) <= 0:
        raise ValueError("prompt exceeds configured context window")
    # vLLM's OpenAI server takes the minimum of remaining context, the request
    # cap and generation_config.max_new_tokens. The offline API does not apply
    # that model cap to explicit SamplingParams, so reproduce it here.
    caps = [
        min(
            [tokens]
            + [cap for cap in (output_cap, model_output_cap) if cap is not None]
        )
        for tokens in available
    ]
    options = dict(
        model=record["model_path"],
        tokenizer=record["model_path"],
        tensor_parallel_size=int(os.environ.get("AUTO_SFT_EVAL_TP_SIZE", "1")),
        gpu_memory_utilization=float(
            os.environ.get("AUTO_SFT_EVAL_GPU_MEMORY_UTILIZATION", "0.88")
        ),
        max_model_len=max_seq,
        trust_remote_code=False,
    )
    if (
        options["tensor_parallel_size"] < 1
        or not math.isfinite(options["gpu_memory_utilization"])
        or not 0 < options["gpu_memory_utilization"] <= 1
    ):
        raise ValueError(
            "evaluation requires positive tensor parallelism and GPU utilization in (0, 1]"
        )
    if seed is not None:
        options["seed"] = seed
    model = LLM(**options)
    evaluator = Evaluator()
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    for repeat, output in enumerate(
        repeat_output_dirs(Path(record["output_dir"]), repeats), 1
    ):
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=cap,
                repetition_penalty=repetition_penalty,
                seed=seed + (repeat - 1) * len(rows) + i if seed is not None else None,
                skip_special_tokens=True,
            )
            for i, cap in enumerate(caps)
        ]
        generated = model.generate(prompts, params)
        raw = [result.outputs[0].text for result in generated]
        if len(raw) != len(rows):
            raise ValueError("incomplete local inference results")
        predictions = [visible_answer(text) for text in raw]
        report = evaluator.score(
            predictions, copy.deepcopy(rows), [r["prompt"] for r in rows]
        )
        result_path = output / "opencompass/results" / f"{benchmark}.json"
        write_json(result_path, report)
        write_json(
            output / "predictions.json",
            [
                {
                    "key": row["key"],
                    "prompt": row["prompt"],
                    "raw_prediction": text,
                    "prediction": prediction,
                }
                for row, text, prediction in zip(rows, raw, predictions)
            ],
        )
        _write_artifacts(output, report, benchmark=benchmark, source_path=result_path)
        metrics = json.loads((output / "metrics.json").read_text())
        metrics.update(
            num_prompts=len(rows),
            full_num_prompts=full_num_prompts,
            is_full_benchmark=len(rows) == full_num_prompts,
        )
        write_json(output / "metrics.json", metrics)
        write_json(
            output / "generation_config.json",
            {
                "backend": "vllm",
                "backend_version": version,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_model_len": max_seq,
                "max_out_len": output_cap,
                "seed": seed,
                "model_max_new_tokens": model_output_cap,
                "effective_max_tokens": caps,
                "repetition_penalty": repetition_penalty,
                "repeat": repeat,
                "template": "qwen3_nothink_sft",
                "enable_thinking": False,
                "skip_special_tokens": True,
                "postprocessor": "strip_reasoning_overflow_81920",
                "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            },
        )
        score_writer(record, output, repeat)
