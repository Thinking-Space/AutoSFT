#!/usr/bin/env python3
"""Local SGLang/vLLM generation and guarded scoring for public EvalPlus datasets.

The dataset is staged by ``prepare_evalplus.py``.  This adapter only reads
the staged files and uses the pinned public EvalPlus sanitizer and execution
guard; it never contacts a model service or downloads a dataset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import pickle
import re
import os
import sys
import types
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Mapping

from autosft.core.common import REPO_ROOT as ROOT

LOCAL_DIR = Path(__file__).resolve().parent
from autosft.benchmarks.evalplus_data import (
    DATASET_SPECS,
    GENERATION_INPUT_REVISION,
    dataset_root,
    generation_task_prompt,
    load_staged_rows,
)

EVALPLUS_VERSION = "0.3.1"
EVALPLUS_COMPATIBILITY_FIXES = ["humaneval_find_zero_progress"]
CODE_PROTOCOL_REVISION = "qwen3-evalplus"
_SCORER_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_GROUNDTRUTH_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


def _external_module(name: str) -> Any:
    """Import public EvalPlus while hiding this adapter from ``sys.path``."""

    old_path = list(sys.path)
    local_dir = LOCAL_DIR.resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != local_dir]
    try:
        return importlib.import_module(name)
    finally:
        sys.path[:] = old_path


def _require_evalplus() -> None:
    module = _external_module("evalplus")
    version = importlib.metadata.version("evalplus")
    if version != EVALPLUS_VERSION:
        raise RuntimeError(
            f"this adapter requires evalplus=={EVALPLUS_VERSION}, found {version}"
        )
    if not getattr(module, "__path__", None):
        raise RuntimeError("the public EvalPlus package was shadowed by a local module")


def _evalplus_module(name: str) -> Any:
    _require_evalplus()
    return _external_module(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_raw_key(data_dir: str | Path, benchmark: str, task_ids: list[str]) -> str:
    spec = DATASET_SPECS[benchmark]
    raw_hash = _sha256(Path(data_dir).expanduser().resolve() / spec["file"])
    selected_hash = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()
    return f"{raw_hash}:{selected_hash}"


def _mbpp_inputs(rows: list[dict[str, Any]]) -> None:
    """Decode MBPP's JSON-safe tuple/set representations after validation."""

    if not rows:
        return
    module = _evalplus_module("evalplus.data.mbpp")
    for row in rows:
        row["base_input"] = module.mbpp_deserialize_inputs(
            row["task_id"], row["base_input"]
        )
        plus_input = row["plus_input"]
        row["plus_input"] = (
            []
            if plus_input == {}
            else module.mbpp_deserialize_inputs(row["task_id"], plus_input)
        )


def _make_public_task_prompt(task_prompt: str) -> str:
    return (
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:\n```\n" + task_prompt + "\n```\n"
    )


def _generation_prompts(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[str], bool]:
    """Match the OpenCompass API path's effective no-think messages."""

    template_path = (
        ROOT / "src" / "autosft" / "benchmarks" / "templates" / "qwen3.jinja"
    )
    template = template_path.read_text(encoding="utf-8")
    prompts: list[str] = []
    for row in rows:
        messages = [
            {
                "role": "user",
                "content": _make_public_task_prompt(generation_task_prompt(row)),
            },
        ]
        # The API template parser stops before BOT with generate=True, so
        # the dataset's configured response prefix is not sent to the model.
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(rendered)
    return prompts, False


def _extract_non_reasoning_content(text: str) -> str:
    """Remove Qwen reasoning markers like the public postprocessor."""
    if not isinstance(text, str):
        return ""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    text = text.strip()
    return "" if len(text) > 81920 else text


def _prune_to_code(text: str, *, max_chars: int = 20000, max_lines: int = 200) -> str:
    """Extract the first fenced code block before EvalPlus sanitization."""
    if not isinstance(text, str):
        return ""
    if "```" in text:
        blocks = re.findall(r"```[Pp]ython\s*\n?(.*?)```", text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r"```\w*\s*\n?(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    if "```" not in text:
        match = re.search(r"(^|\n)(import\s|from\s|def\s|class\s)", text)
        if match:
            text = text[match.start(2) :]
    if len(text) > max_chars:
        text = text[:max_chars]
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines])
    return text


def _sanitize_worker(code: str, entrypoint: str, queue: Any) -> None:
    """Run the public sanitizer in a disposable process.

    EvalPlus's code extraction path can spend a long time on malformed model
    output.  Keeping this worker top-level makes it usable with ``spawn``
    after the parent has initialized CUDA.
    """
    try:
        sanitizer = _evalplus_module("evalplus.sanitize").sanitize
        queue.put(("ok", sanitizer(code, entrypoint=entrypoint)))
    except BaseException:
        queue.put(("failed", ""))


def _cuda_is_initialized() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        return False


def _sanitize_with_timeout(code: str, entrypoint: str) -> str:
    """Sanitize one completion without allowing malformed text to hang a run."""
    if not code.strip():
        return ""
    # Import errors remain infrastructure failures. Every actual sanitizer
    # invocation is bounded, including short text that resembles a function.
    _evalplus_module("evalplus.sanitize")
    try:
        timeout = float(os.environ.get("EVALPLUS_SANITIZE_TIMEOUT", "60"))
    except ValueError:
        timeout = 60.0
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 60.0
    # Fork is safe before CUDA initialization and also supports callers that
    # invoke the adapter from a top-level test script. Once the model has
    # touched CUDA, spawn avoids inheriting the CUDA runtime into the worker.
    method = (
        "spawn"
        if _cuda_is_initialized()
        or "fork" not in multiprocessing.get_all_start_methods()
        else "fork"
    )
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    process = context.Process(target=_sanitize_worker, args=(code, entrypoint, queue))
    started = False
    try:
        try:
            process.start()
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("could not start the EvalPlus sanitizer worker") from exc
        started = True
        # Drain before joining: a Queue worker waits for its feeder thread at
        # exit, which blocks when a valid result fills the pipe buffer.
        try:
            status, result = queue.get(timeout=timeout)
        except (Empty, EOFError):
            return ""
        return result if status == "ok" and isinstance(result, str) else ""
    finally:
        if started:
            process.join(0.1)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        if not process.is_alive():
            process.close()
        queue.close()
        queue.join_thread()


class _LocalGenerator:
    """A reusable local model with explicit protocol metadata and cleanup."""

    def __init__(self, record: Mapping[str, Any], rows: list[dict[str, Any]]):
        _require_evalplus()
        settings, tokenizer, prompts, direct, token_caps = _prepare_generation(
            record, rows
        )
        self.record = record
        self.settings = settings
        self.seed = settings["seed"]
        self.max_model_len = settings["max_model_len"]
        self.temperature = settings["temperature"]
        self.max_out_len = settings["max_out_len"]
        self.top_p = settings["top_p"]
        self.top_k = settings["top_k"]
        self.min_p = settings["min_p"]
        self.tokenizer = tokenizer
        self.prompts, self.direct = prompts, direct
        self.token_caps = token_caps
        self.task_ids = [row["task_id"] for row in rows]
        self.model = None

    def _request_seed(self, repeat: int, index: int) -> int | None:
        return (
            None
            if self.seed is None
            else self.seed + (repeat - 1) * len(self.task_ids) + index
        )

    def _validate_repeat(self, rows: list[dict[str, Any]], repeat: int) -> None:
        if repeat < 1:
            raise ValueError("repeat must be positive")
        if [row["task_id"] for row in rows] != self.task_ids:
            raise ValueError("reused generator must receive the same ordered tasks")

    def metadata(self, repeat: int) -> dict[str, Any]:
        try:
            backend_version = importlib.metadata.version(
                self.settings["inference_backend"]
            )
        except importlib.metadata.PackageNotFoundError:
            backend_version = None
        return {
            **self.settings,
            "backend": f"{self.settings['inference_backend']}_evalplus",
            "backend_version": backend_version,
            "protocol_revision": CODE_PROTOCOL_REVISION,
            "repeat": repeat,
            "repeat_seed": self._request_seed(repeat, 0),
            "request_seed_policy": "unset"
            if self.seed is None
            else "seed_plus_repeat_task_offset",
            "deterministic_inference": self.seed is not None
            and self.settings["inference_backend"] == "sglang",
            "skip_special_tokens": True,
            "ignore_eos": False,
            "stop_strings": [],
            "extra_stop_token_ids": [],
            "eos_source": "model_config_and_generation_config_and_tokenizer",
            "enable_thinking": False,
            "template": "qwen3_nothink_sft",
            "prompt_protocol": "opencompass_evalplus_api_user_only",
            "generation_input_revision": GENERATION_INPUT_REVISION,
            "generation_prompts_sha256": hashlib.sha256(
                json.dumps(self.prompts, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "prompt_whitespace": "preserved",
            "assistant_response_prefix": False,
            "postprocessing": {
                "overflow_empty_above_chars": 81920,
                "tab_width": 4,
                "code_max_chars": 20000,
                "code_max_lines": 200,
            },
            "sanitizer": f"evalplus=={EVALPLUS_VERSION}",
        }

    def close(self) -> None:
        self.model = None

    def __enter__(self) -> "_LocalGenerator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _VLLMGenerator(_LocalGenerator):
    """Optional vLLM backend; protocol is shared, engine results may differ."""

    def __init__(self, record: Mapping[str, Any], rows: list[dict[str, Any]]):
        super().__init__(record, rows)
        from vllm import LLM

        settings = self.settings
        model_path = str(record["model_path"])
        # All assets, settings, and prompt lengths are checked before allocation.
        options = dict(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=settings["tensor_parallel_size"],
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            max_model_len=self.max_model_len,
            trust_remote_code=False,
            # Explicit parameters below implement the protocol; do not inherit
            # an accidental max_new_tokens cap from generation_config.json.
            generation_config="vllm",
        )
        if self.seed is not None:
            options["seed"] = self.seed
        previous = os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN")
        try:
            if settings["context_override"]:
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            self.model = LLM(**options)
        finally:
            if previous is None:
                os.environ.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
            else:
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = previous

    def __call__(
        self, rows: list[dict[str, Any]], *, repeat: int = 1
    ) -> list[dict[str, str]]:
        from vllm import SamplingParams

        self._validate_repeat(rows, repeat)
        params = [
            SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k or -1,
                min_p=self.min_p,
                # vLLM uses None as its context-bounded no-cap value.
                max_tokens=cap,
                seed=self._request_seed(repeat, index),
                skip_special_tokens=True,
                ignore_eos=False,
                stop_token_ids=self.settings["eos_token_ids"],
                repetition_penalty=self.settings["repetition_penalty"],
            )
            for index, cap in enumerate(self.token_caps)
        ]
        generated = self.model.generate(self.prompts, params, use_tqdm=False)
        if len(generated) != len(rows) or any(
            len(result.outputs) != 1 for result in generated
        ):
            raise ValueError("incomplete or duplicate local inference results")
        return _sanitize_predictions(
            rows, [result.outputs[0].text for result in generated], self.direct
        )


class _SGLangGenerator(_LocalGenerator):
    """SGLang matches the original local server's generation protocol."""

    def __init__(self, record: Mapping[str, Any], rows: list[dict[str, Any]]):
        super().__init__(record, rows)
        from autosft.backends.local.evaluation.sglang_generator import LocalSGLang

        self.model = LocalSGLang(
            str(record["model_path"]),
            context_length=self.max_model_len,
            tensor_parallel_size=self.settings["tensor_parallel_size"],
            memory_fraction=self.settings["gpu_memory_utilization"],
            seed=self.seed,
            allow_longer_context=self.settings["context_override"],
        )

    def __call__(
        self, rows: list[dict[str, Any]], *, repeat: int = 1
    ) -> list[dict[str, str]]:
        self._validate_repeat(rows, repeat)
        params = []
        for index, cap in enumerate(self.token_caps):
            item = dict(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k or -1,
                min_p=self.min_p,
                max_new_tokens=cap,
                skip_special_tokens=True,
                ignore_eos=False,
                repetition_penalty=self.settings["repetition_penalty"],
            )
            seed = self._request_seed(repeat, index)
            if seed is not None:
                item["sampling_seed"] = seed
            params.append(item)
        return _sanitize_predictions(
            rows, self.model.generate(self.prompts, params), self.direct
        )

    def close(self) -> None:
        if self.model is not None:
            model, self.model = self.model, None
            model.close()


def _sanitize_predictions(
    rows: list[dict[str, Any]], completions: list[str], direct_completion: bool
) -> list[dict[str, str]]:
    if len(completions) != len(rows):
        raise ValueError(
            f"generation returned {len(completions)} rows, expected {len(rows)}"
        )
    result: list[dict[str, str]] = []
    for row, raw in zip(rows, completions):
        if not isinstance(raw, str):
            raise TypeError(f"non-text generation for {row['task_id']}")
        # OpenCompass's public EvalPlus configs apply ``evalplus_tab2space``
        # before the evaluator. Keep local scoring byte-compatible for model
        # outputs that use tab indentation, while retaining the untouched
        # generation in ``raw_prediction`` for audit/debugging.
        processed = _extract_non_reasoning_content(raw)
        processed = processed.replace("\t", "    ")
        candidate = (
            str(row["prompt"]).strip() + "\n" + processed
            if direct_completion
            else processed
        )
        candidate = _prune_to_code(candidate)
        # Malformed model answers are failed samples, not evaluator
        # infrastructure failures.  Suspicious inputs are isolated with the
        # same bounded sanitizer behavior as the public OpenCompass adapter.
        solution = _sanitize_with_timeout(candidate, str(row["entry_point"]))
        result.append(
            {
                "task_id": str(row["task_id"]),
                "prediction": solution,
                "raw_prediction": raw,
            }
        )
    return result


def generate_predictions(
    rows: list[dict[str, Any]],
    record: Mapping[str, Any],
    *,
    generator: Callable[..., list[str]] | _LocalGenerator | None = None,
    repeat: int = 1,
) -> list[dict[str, str]]:
    """Generate and sanitize one completion per task."""

    if isinstance(generator, _LocalGenerator):
        return generator(rows, repeat=repeat)
    if generator is None:
        with create_generator(record, rows=rows) as owned_generator:
            return owned_generator(rows, repeat=repeat)
    temperature = float(
        record.get(
            "temperature", os.environ.get("AUTO_SFT_OPENCOMPASS_TEMPERATURE", "0.7")
        )
    )
    seed_value = record.get("seed", os.environ.get("AUTO_SFT_EVAL_SEED"))
    seed = (
        int(seed_value) if seed_value is not None and str(seed_value).strip() else None
    )
    cap = record.get("max_out_len", os.environ.get("AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN"))
    max_tokens = int(cap) if cap is not None and str(cap).strip() else None
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_out_len must be positive")
    completions = generator(
        [generation_task_prompt(row) for row in rows],
        [
            None if seed is None else seed + (repeat - 1) * max(1, len(rows)) + i
            for i in range(len(rows))
        ],
        max_tokens,
        temperature,
    )
    # Injected generators represent a complete script, which keeps the test
    # hook independent of a model tokenizer's direct/chat mode.
    return _sanitize_predictions(rows, completions, direct_completion=False)


def _expected_outputs(
    rows: list[dict[str, Any]], benchmark: str, cache_key: str
) -> dict[str, dict[str, Any]]:
    content_hash = hashlib.sha256(pickle.dumps(rows, protocol=4)).hexdigest()
    key = (benchmark, f"{cache_key}:{content_hash}")
    if key in _GROUNDTRUTH_CACHE:
        return _GROUNDTRUTH_CACHE[key]
    trusted_exec = _evalplus_module("evalplus.gen.util").trusted_exec
    oracle_module = _evalplus_module("evalplus.eval._special_oracle")
    not_none = (
        oracle_module.MBPP_OUTPUT_NOT_NONE_TASKS if benchmark == "mbpp_plus" else []
    )
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = str(row["prompt"]) + str(row["canonical_solution"])
        base, base_time = trusted_exec(
            code,
            row["base_input"],
            str(row["entry_point"]),
            record_time=True,
            output_not_none=str(row["entry_point"]) in not_none,
        )
        plus, plus_time = trusted_exec(
            code,
            row["plus_input"],
            str(row["entry_point"]),
            record_time=True,
            output_not_none=str(row["entry_point"]) in not_none,
        )
        expected[str(row["task_id"])] = {
            "base": base,
            "base_time": base_time,
            "plus": plus,
            "plus_time": plus_time,
        }
    _GROUNDTRUTH_CACHE[key] = expected
    return expected


def evaluate_predictions(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, str]],
    benchmark: str,
    *,
    cache_key: str = "in-memory",
) -> dict[str, Any]:
    """Run base and extra tests and emit a compact EvalPlus report."""

    if benchmark not in DATASET_SPECS:
        raise ValueError(f"unsupported EvalPlus benchmark: {benchmark}")
    if not rows:
        raise ValueError("evaluation needs at least one EvalPlus task")
    problems = {str(row["task_id"]): row for row in rows}
    if len(problems) != len(rows) or len(predictions) != len(rows):
        raise ValueError(
            "generated predictions do not exactly cover staged EvalPlus tasks"
        )
    prediction_by_id: dict[str, dict[str, str]] = {}
    for item in predictions:
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or task_id in prediction_by_id:
            raise ValueError(
                "generated predictions contain invalid or duplicate task IDs"
            )
        prediction_by_id[task_id] = item
    if set(problems) != set(prediction_by_id):
        raise ValueError(
            "generated predictions do not exactly cover staged EvalPlus tasks"
        )

    _require_evalplus()
    evaluator = _evalplus_module("evalplus.eval")
    expected = _expected_outputs(rows, benchmark, cache_key)
    dataset = "humaneval" if benchmark == "humaneval_plus" else "mbpp"
    details: list[dict[str, Any]] = []
    base_pass = plus_pass = 0
    for row in rows:
        task_id = str(row["task_id"])
        item = prediction_by_id[task_id]
        solution = item.get("prediction")
        if not isinstance(solution, str):
            raise TypeError(f"non-text prediction for {task_id}")
        test_statuses: dict[str, str] = {}
        for suite in ("base", "plus"):
            if not solution.strip():
                test_statuses[suite] = "fail"
                continue
            inputs = row[f"{suite}_input"]
            expected_values = expected[task_id][suite]
            ref_time = expected[task_id][f"{suite}_time"]
            # Mbpp/793 has no generated extra cases in the public release;
            # EvalPlus represents that suite as an empty list and considers it
            # passed once the base suite has passed.
            if suite == "plus" and not inputs:
                test_statuses[suite] = "pass"
                continue
            if (
                not inputs
                or len(inputs) != len(expected_values)
                or len(inputs) != len(ref_time)
            ):
                raise RuntimeError(f"invalid reference tests for {task_id}/{suite}")
            result = _untrusted_check(
                evaluator,
                dataset,
                solution,
                inputs,
                str(row["entry_point"]),
                expected=expected_values,
                atol=row["atol"],
                ref_time=ref_time,
                fast_check=True,
            )
            test_statuses[suite] = _checked_status(result, len(inputs), task_id, suite)
        base = test_statuses["base"] == "pass"
        plus = base and test_statuses["plus"] == "pass"
        base_pass += int(base)
        plus_pass += int(plus)
        details.append(
            {
                "task_id": task_id,
                "reference": task_id,
                "prompt": str(row["prompt"]),
                "prediction": solution,
                "raw_prediction": item.get("raw_prediction", solution),
                "base_result": "pass" if base else "fail",
                "plus_result": "pass" if plus else "fail",
                "base_test_status": test_statuses["base"],
                "plus_test_status": test_statuses["plus"],
            }
        )
    count = len(rows)
    return {
        "schema_version": "1.0",
        "plus_result_semantics": "combined",
        "benchmark": benchmark,
        "dataset_version": DATASET_SPECS[benchmark]["version"],
        "scoring_protocol": {
            "revision": CODE_PROTOCOL_REVISION,
            "evalplus_version": EVALPLUS_VERSION,
            "compatibility_fixes": EVALPLUS_COMPATIBILITY_FIXES,
            "maximum_memory_bytes": evaluator.query_maximum_memory_bytes(),
            "cpu_thread_settings": {
                name: os.environ.get(name) for name in _SCORER_THREAD_VARIABLES
            },
        },
        "base_pass@1": 100.0 * base_pass / count,
        "plus_pass@1": 100.0 * plus_pass / count,
        "details": details,
    }


def _function_with_globals(function: Any, **overrides: Any) -> Any:
    """Reuse public bytecode with local bindings; leave its module unchanged."""
    clone = types.FunctionType(
        function.__code__,
        {**function.__globals__, **overrides},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def _find_zero_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs: Any,
    expected: Any,
    time_limits: Any,
    atol: float,
    fast_check: bool,
    stat: Any,
    details: Any,
    progress: Any,
) -> None:
    """Fix EvalPlus 0.3.1's skipped progress after a valid polynomial root.

    The public executor continues immediately after its special oracle,
    bypassing successful-case bookkeeping. Record only roots that satisfy
    that exact oracle; the public guard and exception handling stay intact.
    """
    evaluator = _evalplus_module("evalplus.eval")
    polynomial = evaluator._poly

    def checked_polynomial(*args: Any) -> Any:
        residual = polynomial(*args)
        if abs(residual) <= atol:
            details[progress.value] = True
            progress.value += 1
        return residual

    executor = _function_with_globals(
        evaluator.unsafe_execute, _poly=checked_polynomial
    )
    executor(
        dataset,
        entry_point,
        code,
        inputs,
        expected,
        time_limits,
        atol,
        fast_check,
        stat,
        details,
        progress,
    )


def _untrusted_check(
    evaluator: Any,
    dataset: str,
    code: str,
    inputs: Any,
    entry_point: str,
    **settings: Any,
) -> Any:
    checker = evaluator.untrusted_check
    if dataset == "humaneval" and entry_point == "find_zero":
        checker = _function_with_globals(checker, unsafe_execute=_find_zero_execute)
    return checker(dataset, code, inputs, entry_point, **settings)


def _checked_status(result: Any, count: int, task_id: str, suite: str) -> str:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"invalid EvalPlus result for {task_id}/{suite}")
    status, details = result
    if status not in {"pass", "fail", "timeout"}:
        raise RuntimeError(f"unknown EvalPlus status for {task_id}/{suite}: {status!r}")
    if not isinstance(details, (list, tuple)):
        # Public releases can also return NumPy arrays.
        details = details.tolist() if hasattr(details, "tolist") else None
    if (
        details is None
        or len(details) > count
        or any(value not in (True, False) for value in details)
    ):
        raise RuntimeError(f"invalid EvalPlus test details for {task_id}/{suite}")
    if status == "pass" and (len(details) != count or not all(details)):
        raise RuntimeError(f"incomplete passing EvalPlus result for {task_id}/{suite}")
    return status


def _generation_settings(record: Mapping[str, Any]) -> dict[str, Any]:
    def configured(key: str, env: str, default: Any) -> Any:
        return record.get(key, os.environ.get(env, default))

    model_path = Path(str(record["model_path"]))
    config_path = model_path / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    generation_path = model_path / "generation_config.json"
    generation_config = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    context_fields = ("max_position_embeddings", "max_seq_len", "max_sequence_length")
    config_sections = [config]
    for section_name in ("text_config", "language_config"):
        section = config.get(section_name)
        if isinstance(section, dict):
            config_sections.append(section)
    model_context = next(
        (
            int(section[key])
            for section in config_sections
            for key in context_fields
            if section.get(key)
        ),
        131072,
    )
    # The original protocol has distinct client and server limits. The
    # client's 81920 does not enlarge the server's 65536-token window.
    requested_context = record.get(
        "max_model_len",
        os.environ.get("AUTO_SFT_CODE_CONTEXT_LENGTH")
        or os.environ.get("AUTO_SFT_OPENCOMPASS_SGLANG_CONTEXT_LENGTH")
        or 65536,
    )
    context = int(requested_context)
    if context < 1:
        raise ValueError("configured evaluation context must be positive")
    backend = str(
        configured("inference_backend", "AUTO_SFT_CODE_INFERENCE_BACKEND", "sglang")
    ).lower()
    if backend not in {"sglang", "vllm"}:
        raise ValueError("code inference backend must be sglang or vllm")
    temperature = float(
        configured("temperature", "AUTO_SFT_OPENCOMPASS_TEMPERATURE", 0.7)
    )
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("evaluation temperature must be finite and nonnegative")
    cap = configured("max_out_len", "AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN", "")
    cap = int(cap) if cap is not None and str(cap).strip() else None
    if cap is not None and cap < 1:
        raise ValueError("max_out_len must be positive")
    tp = int(configured("tensor_parallel_size", "AUTO_SFT_EVAL_TP_SIZE", 1))
    memory = float(
        record.get(
            "gpu_memory_utilization",
            os.environ.get("AUTO_SFT_CODE_GPU_MEMORY_UTILIZATION")
            or os.environ.get("AUTO_SFT_OPENCOMPASS_SGLANG_MEM_FRACTION")
            or 0.8,
        )
    )
    seed_value = configured("seed", "AUTO_SFT_EVAL_SEED", None)
    seed = (
        int(seed_value) if seed_value is not None and str(seed_value).strip() else None
    )
    top_p = float(configured("top_p", "AUTO_SFT_EVAL_TOP_P", 0.8))
    top_k = int(configured("top_k", "AUTO_SFT_EVAL_TOP_K", 20))
    min_p = float(configured("min_p", "AUTO_SFT_EVAL_MIN_P", 0.0))
    if tp < 1 or not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError(
            "evaluation requires positive tensor parallelism and GPU utilization in (0, 1]"
        )
    if seed is not None and seed < 0:
        raise ValueError("evaluation seed must be nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("evaluation top_p must be in (0, 1]")
    if top_k < 0 or not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError(
            "evaluation top_k must be nonnegative and min_p must be in [0, 1]"
        )
    eos_token_ids = set()
    for section in (config, generation_config):
        ids = section.get("eos_token_id")
        if ids is not None:
            eos_token_ids.update(
                [int(ids)] if isinstance(ids, int) else [int(value) for value in ids]
            )
    repetition_penalty = float(generation_config.get("repetition_penalty", 1.0))
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("model repetition_penalty must be positive and finite")
    return dict(
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_out_len=cap,
        max_model_len=context,
        model_context=model_context,
        context_override=context > model_context,
        inference_backend=backend,
        eos_token_ids=sorted(eos_token_ids),
        repetition_penalty=repetition_penalty,
        tensor_parallel_size=tp,
        gpu_memory_utilization=memory,
    )


def _prepare_generation(record: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple:
    """Validate the tokenizer and every prompt without loading model weights."""
    settings = _generation_settings(record)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(record["model_path"]), local_files_only=True
    )
    if getattr(tokenizer, "eos_token_id", None) is not None:
        settings["eos_token_ids"] = sorted(
            set(settings["eos_token_ids"]) | {tokenizer.eos_token_id}
        )
    prompts, direct = _generation_prompts(tokenizer, rows)
    token_caps = []
    for prompt in prompts:
        available = settings["max_model_len"] - len(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        if available < 1:
            raise ValueError("EvalPlus prompt exceeds configured context window")
        cap = settings["max_out_len"]
        # SGLang validates explicit input+output lengths strictly below its
        # context. None follows the engine's context-bounded no-cap path.
        if cap is not None and settings["inference_backend"] == "sglang":
            available -= 1
            if available < 1:
                raise ValueError(
                    "EvalPlus prompt leaves no room for an explicit output cap"
                )
        token_caps.append(min(available, cap) if cap is not None else None)
    return settings, tokenizer, prompts, direct, token_caps


def prepare_record(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Validate offline assets and execution settings without allocating a model."""
    benchmark = str(record["benchmark"]).lower()
    if benchmark not in DATASET_SPECS:
        raise ValueError(f"unsupported EvalPlus benchmark: {benchmark}")
    limit = int(record.get("limit", -1))
    if limit == 0 or limit < -1:
        raise ValueError("evaluation limit must be -1 or positive")
    if int(record.get("repeats", 1)) < 1 or int(record.get("repeat", 1)) < 1:
        raise ValueError("evaluation repeats must be positive")
    data_root = record.get("evalplus_data_dir") or dataset_root()
    rows = load_staged_rows(data_root, benchmark)
    if limit != -1:
        rows = rows[:limit]
    _generation_settings(record)
    for module in (
        "evalplus.sanitize",
        "evalplus.eval",
        "evalplus.gen.util",
        "evalplus.data.mbpp",
    ):
        _evalplus_module(module)
    # Version 0.3.1 compares this environment string directly with a number.
    # Reject the unsupported override before GPU allocation instead of failing
    # halfway through scoring; the public default is a 60-second task cap.
    if os.environ.get("EVALPLUS_TIMEOUT_PER_TASK") is not None:
        raise ValueError(
            "evalplus 0.3.1 does not support EVALPLUS_TIMEOUT_PER_TASK; unset it"
        )
    memory_limit = int(os.environ.get("EVALPLUS_MAX_MEMORY_BYTES", str(4 * 1024**3)))
    if memory_limit != -1 and memory_limit < 1:
        raise ValueError("EVALPLUS_MAX_MEMORY_BYTES must be positive or -1")
    source_key = _staged_raw_key(
        data_root, benchmark, [str(row["task_id"]) for row in rows]
    )
    # Rows remain JSON-encoded until the isolated scorer deserializes MBPP.
    return rows, source_key


def _initialize_scorer() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Apply before NumPy/BLAS imports: thread stacks can otherwise reserve most
    # of the public checker's 4 GiB address-space allowance before code runs.
    for name in _SCORER_THREAD_VARIABLES:
        os.environ[name] = "1"
    # Spawn a fresh interpreter from the GPU process, then let the public
    # EvalPlus guard fork only from this CPU-only worker on POSIX hosts.
    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)


def _prepare_scorer(rows: list[dict[str, Any]], benchmark: str, cache_key: str) -> None:
    if benchmark == "mbpp_plus":
        _mbpp_inputs(rows)
    _expected_outputs(rows, benchmark, cache_key)


def _score_in_worker(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, str]],
    benchmark: str,
    cache_key: str,
) -> dict[str, Any]:
    if benchmark == "mbpp_plus":
        _mbpp_inputs(rows)
    return evaluate_predictions(rows, predictions, benchmark, cache_key=cache_key)


class LocalScorer:
    """Reuse one CPU-only process and reference outputs across benchmark repeats."""

    def __enter__(self) -> "LocalScorer":
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_scorer,
        )
        return self

    def prepare(
        self, rows: list[dict[str, Any]], benchmark: str, cache_key: str
    ) -> None:
        self.pool.submit(_prepare_scorer, rows, benchmark, cache_key).result()

    def score(
        self,
        rows: list[dict[str, Any]],
        predictions: list[dict[str, str]],
        benchmark: str,
        cache_key: str,
    ) -> dict[str, Any]:
        return self.pool.submit(
            _score_in_worker, rows, predictions, benchmark, cache_key
        ).result()

    def __exit__(self, *exc: Any) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def run_record(
    record: Mapping[str, Any],
    *,
    generator: Callable[..., list[str]] | _LocalGenerator | None = None,
    repeat: int | None = None,
    prepared: tuple[list[dict[str, Any]], str] | None = None,
    scorer: LocalScorer | None = None,
) -> dict[str, Any]:
    benchmark = str(record["benchmark"]).lower()
    rows, source_key = prepared if prepared is not None else prepare_record(record)
    current_repeat = int(repeat if repeat is not None else record.get("repeat", 1))
    if current_repeat < 1:
        raise ValueError("repeat must be positive")
    if scorer is None:
        with LocalScorer() as owned_scorer:
            owned_scorer.prepare(rows, benchmark, source_key)
            return run_record(
                record,
                generator=generator,
                repeat=current_repeat,
                prepared=(rows, source_key),
                scorer=owned_scorer,
            )
    predictions = generate_predictions(
        rows, record, generator=generator, repeat=current_repeat
    )
    report = scorer.score(rows, predictions, benchmark, source_key)
    output = Path(record["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "opencompass" / "results" / f"{benchmark}.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from autosft.benchmarks.evalplus import write_evalplus_artifacts

    write_evalplus_artifacts(
        output,
        report,
        benchmark=benchmark,
        source_path=source,
        repeat=current_repeat,
        expected_count=len(rows),
        expected_problem_ids=[str(row["task_id"]) for row in rows],
    )
    return report


def create_generator(
    record: Mapping[str, Any], *, rows: list[dict[str, Any]] | None = None
) -> _LocalGenerator:
    """Construct the reusable local model after preflight checks."""
    if rows is None:
        rows, _ = prepare_record(record)
    backend = _generation_settings(record)["inference_backend"]
    return (_SGLangGenerator if backend == "sglang" else _VLLMGenerator)(record, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate data, settings, and runtime imports without allocating a GPU",
    )
    args = parser.parse_args()
    record = json.loads(args.record)
    if args.preflight:
        rows, cache_key = prepare_record(record)
        _prepare_generation(record, rows)
        importlib.import_module(_generation_settings(record)["inference_backend"])
        print(
            json.dumps(
                {
                    "benchmark": record["benchmark"],
                    "count": len(rows),
                    "dataset_sha256": cache_key.split(":", 1)[0],
                    "status": "ready",
                }
            )
        )
        return
    run_record(record)


if __name__ == "__main__":
    main()
