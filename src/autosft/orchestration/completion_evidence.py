"""Verify the model and on-disk evaluation evidence used to declare completion."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickletools
import struct
import zipfile
import zlib

from autosft.backends.megatron.model_profiles import get_model_profile


def _weights_present(path: Path, expected_names: set[str] | None = None) -> bool:
    """Check public weight containers without importing torch or unpickling data."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".safetensors":
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                return False
            size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < size <= min(100_000_000, path.stat().st_size - 8):
                return False
            header = json.loads(stream.read(size))
        if not isinstance(header, dict):
            return False
        if (
            expected_names is not None
            and set(header) - {"__metadata__"} != expected_names
        ):
            return False
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        payload_size = path.stat().st_size - 8 - size
        widths = {
            "BOOL": 1,
            "I8": 1,
            "U8": 1,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
            "I16": 2,
            "U16": 2,
            "F16": 2,
            "BF16": 2,
            "I32": 4,
            "U32": 4,
            "F32": 4,
            "I64": 8,
            "U64": 8,
            "F64": 8,
        }
        spans = []
        for item in tensors:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("shape"), list)
                or any(
                    type(dimension) is not int or dimension < 0
                    for dimension in item["shape"]
                )
                or item.get("dtype") not in widths
                or not isinstance(item.get("data_offsets"), list)
                or len(item["data_offsets"]) != 2
                or any(type(offset) is not int for offset in item["data_offsets"])
            ):
                return False
            start, end = item["data_offsets"]
            if (
                not 0 <= start <= end <= payload_size
                or end - start != math.prod(item["shape"]) * widths[item["dtype"]]
            ):
                return False
            spans.append((start, end))
        end = 0
        for start, following in sorted(spans):
            if start != end:
                return False
            end = following
        return bool(tensors) and payload_size > 0 and end == payload_size
    # The bundled exporter uses torch.save's ZIP format. Do not deserialize
    # its pickle from the lightweight control-plane interpreter.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith("/data.pkl")]
        if len(names) != len(set(names)) or len(metadata) != 1:
            return False
        metadata_info = archive.getinfo(metadata[0])
        if not 0 < metadata_info.file_size <= 64 * 1024 * 1024:
            return False
        # Reading this small member checks its CRC. Parse opcodes without
        # executing any pickle reducers or importing torch into this process.
        raw = archive.read(metadata_info)
        operations = list(pickletools.genops(raw))
        if (
            not operations
            or operations[-1][0].name != "STOP"
            or operations[-1][2] != len(raw) - 1
        ):
            return False
        if not any(
            op.name == "GLOBAL" and str(arg).startswith("torch._utils _rebuild_tensor")
            for op, arg, _ in operations
        ):
            return False
        if not any(op.name in {"BINPERSID", "PERSID"} for op, _, _ in operations):
            return False
        prefix = metadata[0].removesuffix("data.pkl") + "data/"
        storage = [archive.getinfo(name) for name in names if name.startswith(prefix)]
        if not storage or not any(info.file_size > 0 for info in storage):
            return False
        # Check local ZIP headers and non-overlapping data bounds; do not read
        # multi-GB tensor payloads on every handoff. Full loadability remains
        # the conversion/evaluation worker's responsibility.
        for info in storage:
            if (
                info.header_offset < 0
                or info.header_offset + 30 + info.compress_size > archive.start_dir
            ):
                return False
            with archive.open(info) as stream:
                stream.read(1)
        return True


def model_assets_complete(model: Path, profile=None) -> bool:
    """Check fixed-model config/tokenizer and structural weight completeness.

    This catches missing/truncated containers without loading tensors. It is
    not a numerical model validation or a replacement for loading the model.
    """
    model = Path(model)
    profile = profile or get_model_profile()
    try:
        config = json.loads((model / "config.json").read_text())
        tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
        tokenizer = json.loads((model / "tokenizer.json").read_text())
        if (
            not isinstance(config, dict)
            or any(config.get(key) != value for key, value in profile.hf_config.items())
            or not isinstance(tokenizer_config, dict)
            or not isinstance(tokenizer, dict)
            or not isinstance(tokenizer.get("model"), dict)
            or not tokenizer["model"].get("type")
            or not tokenizer["model"].get("vocab")
        ):
            return False
        for extension in ("safetensors", "bin"):
            index = (
                model
                / f"{'model' if extension == 'safetensors' else 'pytorch_model'}.{extension}.index.json"
            )
            if index.exists():
                mapping = json.loads(index.read_text()).get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    return False
                files = set(mapping.values())
                if any(
                    not isinstance(name, str) or Path(name).name != name
                    for name in files
                ):
                    return False
                return all(
                    _weights_present(
                        model / name,
                        {key for key, value in mapping.items() if value == name},
                    )
                    for name in files
                )
            weight = model / (
                "model.safetensors"
                if extension == "safetensors"
                else "pytorch_model.bin"
            )
            if weight.exists():
                return _weights_present(weight)
        return False
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        EOFError,
        RuntimeError,
        zipfile.BadZipFile,
        struct.error,
        zlib.error,
    ):
        return False


def evaluation_evidence_problems(
    record: dict,
    run: Path,
    *,
    benchmarks: list[str],
    repeats: int,
    expected_model_identity: str | None = None,
) -> list[str]:
    """Recheck persisted repeats and aggregate instead of trusting cached flags."""
    from autosft.benchmarks.aggregate import aggregate_benchmark, repeat_dirs
    from autosft.orchestration.round_gates import evaluation_problems
    from autosft.orchestration.submission_registry import model_identity

    problems = evaluation_problems(record, run, benchmarks=benchmarks, repeats=repeats)
    run = Path(run).resolve()
    for benchmark in benchmarks:
        summary = (record.get("eval") or {}).get(benchmark)
        if not isinstance(summary, dict):
            continue
        try:
            aggregate_path = Path(summary["result_artifacts"]["aggregate"])
            aggregate_path = (
                aggregate_path if aggregate_path.is_absolute() else run / aggregate_path
            ).resolve()
            if not aggregate_path.is_relative_to(run):
                raise ValueError("aggregate is outside the run")
            directories = repeat_dirs(aggregate_path.parent, repeats)
            recorded_dirs = summary["result_artifacts"].get("repeat_dirs")
            if (
                not isinstance(recorded_dirs, list)
                or len(recorded_dirs) != repeats
                or [Path(value).resolve() for value in recorded_dirs] != directories
            ):
                raise ValueError("recorded repeat directories disagree with aggregate")
            identity = model_identity(summary["model_path"])
            if (
                expected_model_identity is not None
                and identity != expected_model_identity
            ):
                raise ValueError("ledger evaluation belongs to a different checkpoint")
            for index, directory in enumerate(directories, 1):
                if not directory.resolve().is_relative_to(run):
                    raise ValueError("repeat directory is outside the run")
                for name in (
                    "metrics.json",
                    "scores.json",
                    "failures.jsonl",
                    "failure_breakdown.json",
                ):
                    if not (directory / name).is_file():
                        raise ValueError(f"repeat {index} is missing {name}")
                metrics = json.loads((directory / "metrics.json").read_text())
                scores = json.loads((directory / "scores.json").read_text())
                breakdown = json.loads(
                    (directory / "failure_breakdown.json").read_text()
                )
                if not isinstance(breakdown, dict):
                    raise ValueError(f"repeat {index} has invalid failure breakdown")
                if (
                    scores.get("benchmark") != benchmark
                    or type(scores.get("repeat")) is not int
                    or scores["repeat"] != index
                    or model_identity(scores.get("model_path")) != identity
                ):
                    raise ValueError(
                        f"repeat {index} has different model/benchmark/repeat identity"
                    )
                if (
                    not isinstance(metrics.get("score"), (int, float))
                    or isinstance(metrics["score"], bool)
                    or not math.isfinite(metrics["score"])
                    or not math.isclose(
                        metrics["score"],
                        scores["primary_score"],
                        abs_tol=1e-12,
                        rel_tol=1e-12,
                    )
                ):
                    raise ValueError(f"repeat {index} metrics disagree with scores")
                if scores.get("metrics") != metrics:
                    raise ValueError(f"repeat {index} scores contain stale metrics")
            rebuilt = aggregate_benchmark(aggregate_path.parent, repeats)
            persisted = json.loads(aggregate_path.read_text())
            if any(
                persisted.get(key) != value
                for key, value in rebuilt.items()
                if key != "generated_at"
            ):
                raise ValueError("aggregate disagrees with current repeat artifacts")
            if rebuilt["primary_scores"] != summary.get("primary_scores"):
                raise ValueError("repeat scores disagree with ledger")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as exc:
            problems.append(f"{benchmark}: {exc}")
    return problems
