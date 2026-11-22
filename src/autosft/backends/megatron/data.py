"""SFT data adapters for single-host Megatron data, tensor and context parallelism."""

from __future__ import annotations

from functools import partial
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from autosft.data.training_data import PackedSFTDataset, TokenDataset


def collate_samples(samples, *, pad_token_id):
    length = max(len(sample["tokens"]) for sample in samples)
    batch = {}
    for name in ("tokens", "labels", "loss_mask", "position_ids"):
        dtype = np.float32 if name == "loss_mask" else np.int64
        fill = pad_token_id if name in ("tokens", "labels") else 0
        array = np.full((len(samples), length), fill, dtype=dtype)
        for index, sample in enumerate(samples):
            size = len(sample[name])
            array[index, :size] = sample[name]
            if name == "position_ids":
                array[index, size:] = sample[name][-1] + np.arange(1, length - size + 1)
        batch[name] = torch.from_numpy(array)
    batch["dataset_id"] = torch.zeros(len(samples), dtype=torch.int64)
    return batch


def build_data_iterators(num_samples):
    from megatron.core import parallel_state
    from megatron.core.rerun_state_machine import RerunDataIterator
    from megatron.legacy.data.data_samplers import MegatronPretrainingSampler
    from megatron.training import get_args

    args = get_args()
    if (
        args.pipeline_model_parallel_size != 1
        or args.mtp_num_layers is not None
        or args.virtual_pipeline_model_parallel_size is not None
    ):
        raise ValueError(
            "Local SFT supports data/tensor/context parallelism with PP=1 and no MTP"
        )
    if args.dataloader_type != "external":
        raise ValueError("SFT requires the external data loader")
    if len(args.data_path or []) != 2 or float(args.data_path[0]) != 1.0:
        raise ValueError("Local SFT requires one prepared dataset with weight 1.0")
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    source = TokenDataset(Path(args.data_path[1]))
    alignment = math.lcm(
        2 * args.context_parallel_size if args.context_parallel_size > 1 else 1,
        args.tensor_model_parallel_size if args.sequence_parallel else 1,
    )
    dataset = PackedSFTDataset(
        source,
        args.seq_length,
        args.global_batch_size,
        num_samples[0],
        alignment=alignment,
    )
    # Record semantics and input identity once without scanning another epoch.
    # Full CPU inspection remains an explicit training_data_audit command.
    if os.environ.get("AUTO_SFT_SFT_OUTPUT_DIR") and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        output = Path(os.environ["AUTO_SFT_SFT_OUTPUT_DIR"]) / "packing_protocol.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(dataset.protocol(), indent=2) + "\n", encoding="utf-8"
        )
    sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=args.consumed_train_samples,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_samples, pad_token_id=source.pad_token_id),
        persistent_workers=args.num_workers > 0,
    )
    return RerunDataIterator(iter(loader)), None, None


build_data_iterators.is_distributed = True


def get_batch(data_iterator):
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    device = torch.device("cuda", torch.cuda.current_device())
    primary = parallel_state.get_tensor_model_parallel_rank() == 0
    distributed = parallel_state.get_tensor_model_parallel_world_size() > 1
    group = parallel_state.get_tensor_model_parallel_group()
    root = parallel_state.get_tensor_model_parallel_src_rank()
    data = next(data_iterator) if primary else None
    shape = torch.tensor(
        data["tokens"].shape if primary else [0, 0], dtype=torch.int64, device=device
    )
    if distributed:
        torch.distributed.broadcast(shape, root, group=group)
    dimensions = tuple(shape.tolist())
    batch = {}
    for name in ("tokens", "labels", "loss_mask", "position_ids", "dataset_id"):
        dtype = torch.float32 if name == "loss_mask" else torch.int64
        tensor_shape = dimensions[:1] if name == "dataset_id" else dimensions
        tensor = (
            data[name].to(device, non_blocking=True)
            if primary
            else torch.empty(
                tensor_shape,
                dtype=dtype,
                device=device,
            )
        )
        if distributed:
            torch.distributed.broadcast(tensor, root, group=group)
        batch[name] = tensor
    # THD packing preserves each conversation's causal boundary. The public
    # attention implementation consumes global cumulative sequence lengths;
    # each CP rank receives the paired head/tail partitions of every span.
    positions = batch["position_ids"].reshape(-1)
    boundaries = torch.nonzero(positions == 0).flatten()
    cu = torch.cat([boundaries, boundaries.new_tensor([positions.numel()])]).to(
        torch.int32
    )
    lengths = cu[1:] - cu[:-1]
    maximum = int(lengths.max().item())
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    indices = packed_partition_indices(cu.tolist(), cp_size, cp_rank)
    selected = torch.tensor(indices, device=device, dtype=torch.long)
    for name in ("tokens", "labels", "loss_mask", "position_ids"):
        batch[name] = batch[name].reshape(1, -1).index_select(1, selected)
    batch["packed_seq_params"] = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=maximum,
        max_seqlen_kv=maximum,
    )
    return batch


def packed_partition_indices(boundaries: list[int], size: int, rank: int) -> list[int]:
    """Match the head/tail partition used by public TE packed attention."""
    if size < 1 or not 0 <= rank < size:
        raise ValueError("invalid context-parallel rank")
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start or (size > 1 and (end - start) % (2 * size)):
            raise ValueError("packed spans must align to twice the CP degree")
        if size == 1:
            result.extend(range(start, end))
        else:
            chunk = (end - start) // (2 * size)
            result.extend(range(start + rank * chunk, start + (rank + 1) * chunk))
            result.extend(range(end - (rank + 1) * chunk, end - rank * chunk))
    return result
