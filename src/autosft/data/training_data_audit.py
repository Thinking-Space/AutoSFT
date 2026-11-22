"""Inspect public SFT packing on CPU, including padding and tensor identities.

Run ``python -m autosft.data.training_data_audit --help`` from the checkout.
The report measures the public loader; it cannot establish reference-loader
equivalence without a matching report from that loader on identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from autosft.data.training_data import PackedSFTDataset, TokenDataset


def audit_packing(dataset: PackedSFTDataset, *, sample_count: int = 32) -> dict:
    """Inspect a prefix of one epoch; zero explicitly requests the full epoch."""
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    available = min(dataset.samples_per_epoch, len(dataset))
    count = min(sample_count or available, available)
    records = []
    real_tokens = supervised_targets = padded_tokens = empty_samples = 0
    max_tensor_length = overlength_samples = 0
    combined = {
        name: hashlib.sha256()
        for name in ("tokens", "labels", "loss_mask", "position_ids")
    }
    for index in range(count):
        sample = dataset[index]
        spans = list(dataset.sample_spans(index))
        real_length = sum(span[1] for span in spans)
        tensor_length = len(sample["tokens"])
        losses = int(np.count_nonzero(sample["loss_mask"]))
        real_tokens += real_length
        supervised_targets += losses
        padded_tokens += tensor_length - real_length
        empty_samples += int(real_length == 0)
        overlength_samples += int(tensor_length > dataset.sequence_length)
        max_tensor_length = max(max_tensor_length, tensor_length)
        tensor_hashes = {}
        for name, values in sample.items():
            # Canonical dtypes and little-endian bytes make reports comparable
            # across hosts; include lengths in the epoch stream to bind packing.
            data = np.asarray(
                values, dtype="<f4" if name == "loss_mask" else "<i8"
            ).tobytes()
            tensor_hashes[name] = hashlib.sha256(data).hexdigest()
            combined[name].update(tensor_length.to_bytes(8, "little"))
            combined[name].update(data)
        records.append(
            {
                "sample_index": index,
                "source_tokens": real_length,
                "supervised_targets": losses,
                "tensor_length": tensor_length,
                "padding_tokens": tensor_length - real_length,
                "source_spans": [list(span) for span in spans],
                "tensor_sha256": tensor_hashes,
            }
        )
    complete = count == dataset.samples_per_epoch
    return {
        "protocol": dataset.protocol(),
        "inspected_samples": count,
        "complete_epoch_inspected": complete,
        "source_tokens": real_tokens,
        "supervised_targets": supervised_targets,
        "padding_tokens": padded_tokens,
        "empty_samples": empty_samples,
        "max_tensor_length": max_tensor_length,
        "samples_exceeding_sequence_length": overlength_samples,
        "full_epoch_exposure_matches_manifest": (
            real_tokens == dataset.source.total_tokens
            and supervised_targets == dataset.source.manifest["supervised_tokens"]
        )
        if complete
        else None,
        "tensor_stream_sha256": {
            name: digest.hexdigest() for name, digest in combined.items()
        },
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Prepared dataset.json or its directory",
    )
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--cp-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=32,
        help="Inspect this many samples; 0 inspects one complete epoch",
    )
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    if min(args.seq_length, args.global_batch_size, args.cp_size, args.tp_size) <= 0:
        parser.error("Packing and parallel dimensions must be positive")
    if args.sample_count < 0:
        parser.error("--sample-count must be nonnegative")
    source = TokenDataset(args.dataset)
    step_tokens = args.seq_length * args.global_batch_size
    samples = (
        (source.total_tokens + step_tokens - 1) // step_tokens
    ) * args.global_batch_size
    alignment = math.lcm(
        2 * args.cp_size if args.cp_size > 1 else 1,
        args.tp_size if args.sequence_parallel else 1,
    )
    dataset = PackedSFTDataset(
        source, args.seq_length, args.global_batch_size, samples, alignment=alignment
    )
    report = (
        json.dumps(audit_packing(dataset, sample_count=args.sample_count), indent=2)
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
