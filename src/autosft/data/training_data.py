"""Portable token storage and deterministic SFT packing using NumPy."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np


FORMAT = "autosft-tokens-v1"
ARRAYS = {"tokens": "<i4", "excluded": "u1", "offsets": "<i8"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_token_dataset(
    manifest_path: Path,
    rows: Iterable[tuple[list[int], list[int]]],
    *,
    pad_token_id: int = 0,
) -> dict:
    """Publish token IDs, excluded-token masks, and row offsets atomically."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(f"Dataset already exists: {manifest_path}")
    total_tokens = 0
    total_rows = 0
    supervised_tokens = 0
    with tempfile.TemporaryDirectory(
        prefix=".tokens-", dir=manifest_path.parent
    ) as temporary:
        staging = Path(temporary)
        with (
            (staging / "tokens.bin").open("wb") as token_stream,
            (staging / "excluded.bin").open("wb") as mask_stream,
            (staging / "offsets.bin").open("wb") as offset_stream,
        ):
            offset_stream.write(np.asarray([0], dtype="<i8").tobytes())
            for total_rows, (tokens, excluded) in enumerate(rows, 1):
                token_array = np.asarray(tokens)
                mask_array = np.asarray(excluded)
                if (
                    token_array.ndim != 1
                    or len(tokens) < 2
                    or token_array.dtype.kind not in "iu"
                    or np.any(token_array < 0)
                    or np.any(token_array > np.iinfo(np.int32).max)
                    or mask_array.shape != token_array.shape
                    or not np.all((mask_array == 0) | (mask_array == 1))
                ):
                    raise ValueError(f"Invalid encoded training row {total_rows}")
                if not np.any(mask_array[1:] == 0):
                    raise ValueError(
                        f"Training row {total_rows} has no supervised targets"
                    )
                token_stream.write(token_array.astype("<i4").tobytes())
                mask_stream.write(mask_array.astype("u1").tobytes())
                total_tokens += len(tokens)
                supervised_tokens += int(np.count_nonzero(mask_array[1:] == 0))
                offset_stream.write(np.asarray([total_tokens], dtype="<i8").tobytes())
        if not total_rows:
            raise ValueError("Training dataset is empty")
        files = {}
        for name, dtype in ARRAYS.items():
            source = staging / f"{name}.bin"
            files[name] = {
                "path": source.name,
                "dtype": dtype,
                "bytes": source.stat().st_size,
                "sha256": file_digest(source),
            }
        manifest = {
            "format": FORMAT,
            "rows": total_rows,
            "total_tokens": total_tokens,
            "supervised_tokens": supervised_tokens,
            "pad_token_id": int(pad_token_id),
            "files": files,
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for spec in files.values():
            os.replace(staging / spec["path"], manifest_path.parent / spec["path"])
        os.replace(staged_manifest, manifest_path)
    return manifest


class TokenDataset:
    """Read a public token dataset without importing a model framework."""

    def __init__(self, path: Path, *, verify_hashes: bool = True):
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / "dataset.json"
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"Unsupported training dataset format: {self.path}")
        self.total_tokens = self.manifest["total_tokens"]
        self.rows = self.manifest["rows"]
        self.pad_token_id = self.manifest["pad_token_id"]
        if (
            type(self.total_tokens) is not int
            or self.total_tokens <= 0
            or type(self.rows) is not int
            or self.rows <= 0
        ):
            raise ValueError("Invalid dataset counts")
        for name, dtype in ARRAYS.items():
            spec = self.manifest["files"][name]
            expected_name = f"{name}.bin"
            array_path = self.path.parent / expected_name
            count = self.rows + 1 if name == "offsets" else self.total_tokens
            expected_bytes = count * np.dtype(dtype).itemsize
            if (
                spec["path"] != expected_name
                or spec["dtype"] != dtype
                or array_path.is_symlink()
                or spec["bytes"] != expected_bytes
                or array_path.stat().st_size != expected_bytes
            ):
                raise ValueError(f"Invalid dataset array: {name}")
            if verify_hashes and file_digest(array_path) != spec["sha256"]:
                raise ValueError(f"Dataset digest mismatch: {name}")
            setattr(
                self, name, np.memmap(array_path, dtype=dtype, mode="r", shape=(count,))
            )
        if (
            self.offsets[0] != 0
            or self.offsets[-1] != self.total_tokens
            or np.any(np.diff(self.offsets) < 2)
        ):
            raise ValueError("Invalid dataset row offsets")


class PackedSFTDataset:
    """Partition each epoch by real token volume; padding never consumes budget.

    ``sequence_length`` caps source tokens per sample, not its padded tensor
    length. Each row fragment is padded separately for context parallelism,
    so a packed tensor can be longer. This is a public packing contract, not
    a claim of equivalence to the unavailable reference training loader.
    """

    def __init__(
        self,
        source: TokenDataset,
        sequence_length: int,
        global_batch_size: int,
        num_samples: int,
        *,
        alignment: int = 1,
    ):
        for value in (sequence_length, global_batch_size, num_samples, alignment):
            if type(value) is not int or value <= 0:
                raise ValueError("Packing dimensions must be positive integers")
        if sequence_length % alignment:
            raise ValueError(
                "Sequence length must be divisible by the parallel alignment"
            )
        self.source = source
        self.sequence_length = sequence_length
        self.alignment = alignment
        self.global_batch_size = global_batch_size
        # A dataset item is one sequence window.  The Megatron sampler adds
        # global-batch padding at the sample-index level; multiplying the
        # number of windows here creates all-padding items when a smoke or
        # small real dataset fits in one window, which can produce NaN loss.
        self.samples_per_epoch = max(
            1, (source.total_tokens + sequence_length - 1) // sequence_length
        )
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def protocol(self) -> dict:
        """Cheap metadata for every training launch; does not read token arrays."""
        epochs, remaining = divmod(self.num_samples, self.samples_per_epoch)
        return {
            "format": "autosft-packing-v1",
            "dataset_manifest_sha256": file_digest(self.source.path),
            "dataset_files": self.source.manifest["files"],
            "rows": self.source.rows,
            "source_tokens_per_epoch": self.source.total_tokens,
            "supervised_targets_per_epoch": self.source.manifest["supervised_tokens"],
            "sequence_length": self.sequence_length,
            "sequence_length_counts": "source_tokens_before_fragment_padding",
            "padded_tensor_may_exceed_sequence_length": self.alignment > 1,
            "alignment": self.alignment,
            "global_batch_size": self.global_batch_size,
            "samples_per_epoch": self.samples_per_epoch,
            "num_samples": self.num_samples,
            "complete_epochs": epochs,
            "remaining_samples": remaining,
            "row_boundary_attention": "isolated_by_span",
            "long_rows": "split_at_source_token_window_boundary",
            "position_ids": "restart_at_zero_in_each_fragment",
            "fragment_boundary_target": "next_source_token_in_same_row",
            "padding_loss": 0,
            "reference_loader_parity": "unverified_original_loader_unavailable",
        }

    def sample_spans(self, index):
        """Yield (source start, real length, padded length) for one sample."""
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        cursor = (index % self.samples_per_epoch) * self.sequence_length
        stop = min(cursor + self.sequence_length, self.source.total_tokens)
        row = int(np.searchsorted(self.source.offsets, cursor, side="right") - 1)
        while cursor < stop:
            piece_stop = min(stop, int(self.source.offsets[row + 1]))
            length = piece_stop - cursor
            padded = (length + self.alignment - 1) // self.alignment * self.alignment
            yield cursor, length, padded
            cursor = piece_stop
            row += 1

    def __getitem__(self, index):
        pieces = {
            name: [] for name in ("tokens", "labels", "loss_mask", "position_ids")
        }
        for cursor, length, padded in self.sample_spans(index):
            row = int(np.searchsorted(self.source.offsets, cursor, side="right") - 1)
            row_stop = int(self.source.offsets[row + 1])
            tokens = np.full(padded, self.source.pad_token_id, dtype=np.int64)
            labels = tokens.copy()
            loss_mask = np.zeros(padded, dtype=np.float32)
            tokens[:length] = self.source.tokens[cursor : cursor + length]
            targets = min(length, row_stop - cursor - 1)
            labels[:targets] = self.source.tokens[cursor + 1 : cursor + 1 + targets]
            loss_mask[:targets] = (
                1 - self.source.excluded[cursor + 1 : cursor + 1 + targets]
            )
            pieces["tokens"].append(tokens)
            pieces["labels"].append(labels)
            pieces["loss_mask"].append(loss_mask)
            pieces["position_ids"].append(np.arange(padded, dtype=np.int64))
        if not pieces["tokens"]:
            return {
                "tokens": np.full(
                    self.alignment, self.source.pad_token_id, dtype=np.int64
                ),
                "labels": np.full(
                    self.alignment, self.source.pad_token_id, dtype=np.int64
                ),
                "loss_mask": np.zeros(self.alignment, dtype=np.float32),
                "position_ids": np.arange(self.alignment, dtype=np.int64),
            }
        return {name: np.concatenate(chunks) for name, chunks in pieces.items()}
