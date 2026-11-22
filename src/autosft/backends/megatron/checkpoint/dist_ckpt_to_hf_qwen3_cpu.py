"""
CPU-only converter: Megatron torch_dist checkpoint -> HuggingFace Qwen3 format.
No CUDA or distributed process group is required for loading complete tensors.

Key differences from MiniCPM converter:
  - QK layernorm weights (q_norm, k_norm)
  - Tied embeddings by default (no lm_head.weight)
"""

import importlib.abc
import importlib.machinery
import sys


# DCP metadata unpickling triggers megatron.core -> transformer_engine imports.
# transformer_engine may be installed but broken on CPU (symbol errors like
# undefined C++ symbols).  Block all transformer_engine imports so megatron's
# own try/except ModuleNotFoundError handlers take the HAVE_TE=False path.
class _BlockTransformerEngine(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _BLOCKED = "transformer_engine"

    def find_module(self, fullname, path=None):
        if fullname == self._BLOCKED or fullname.startswith(self._BLOCKED + "."):
            return self
        return None

    def load_module(self, fullname):
        raise ModuleNotFoundError(f"Blocked by CPU converter: {fullname}")


# Remove any already-cached transformer_engine entries and install blocker.
for _k in list(sys.modules):
    if _k == "transformer_engine" or _k.startswith("transformer_engine."):
        del sys.modules[_k]
sys.meta_path.insert(0, _BlockTransformerEngine())

import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", required=True, help="Megatron checkpoint dir (containing iter_XXXX/)"
    )
    parser.add_argument(
        "--save", required=True, help="Output dir for HF pytorch_model.bin"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to load (default: latest)",
    )

    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=6144)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--kv-channels", type=int, default=128)
    parser.add_argument(
        "--untie-embeddings-and-output-weights", action="store_true", default=False
    )
    parser.add_argument("--qk-norm", action="store_true", default=True)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    return parser.parse_args()


def resolve_iteration(load_dir, iteration):
    if iteration is not None:
        return iteration
    latest_file = os.path.join(load_dir, "latest_checkpointed_iteration.txt")
    with open(latest_file) as f:
        return int(f.read().strip())


def load_state_dict_from_dcp(ckpt_path):
    reader = FileSystemReader(ckpt_path)
    metadata = reader.read_metadata()

    state_dict = {}
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if not hasattr(tensor_meta, "size"):
            continue
        if key.startswith("optimizer."):
            continue
        state_dict[key] = torch.empty(
            tensor_meta.size, dtype=tensor_meta.properties.dtype
        )

    # This exporter reconstructs the complete model in one CPU process.
    # Avoid initializing a process group or reserving a shared TCP port.
    dcp.load(state_dict, checkpoint_id=ckpt_path, no_dist=True)
    return state_dict


def convert(state_dict, args):
    num_layers = args.num_layers
    num_heads = args.num_attention_heads
    num_kv_groups = args.num_query_groups
    kv_channels = args.kv_channels
    num_q_per_group = num_heads // num_kv_groups

    hf = {}
    hf["model.embed_tokens.weight"] = state_dict["embedding.word_embeddings.weight"]
    hf["model.norm.weight"] = state_dict["decoder.final_layernorm.weight"]
    if args.untie_embeddings_and_output_weights:
        hf["lm_head.weight"] = state_dict["output_layer.weight"]

    for i in range(num_layers):
        hf[f"model.layers.{i}.input_layernorm.weight"] = state_dict[
            "decoder.layers.self_attention.linear_qkv.layer_norm_weight"
        ][i]

        if args.qk_norm:
            hf[f"model.layers.{i}.self_attn.q_norm.weight"] = state_dict[
                "decoder.layers.self_attention.q_layernorm.weight"
            ][i]
            hf[f"model.layers.{i}.self_attn.k_norm.weight"] = state_dict[
                "decoder.layers.self_attention.k_layernorm.weight"
            ][i]

        qkv = state_dict["decoder.layers.self_attention.linear_qkv.weight"][i]
        qkv_split = torch.split(qkv, kv_channels, dim=0)

        q_list, k_list, v_list = [], [], []
        for g in range(num_kv_groups):
            base = (num_q_per_group + 2) * g
            q_list.extend(qkv_split[base : base + num_q_per_group])
            k_list.append(qkv_split[base + num_q_per_group])
            v_list.append(qkv_split[base + num_q_per_group + 1])

        hf[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.cat(q_list, dim=0)
        hf[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.cat(k_list, dim=0)
        hf[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.cat(v_list, dim=0)
        hf[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[
            "decoder.layers.self_attention.linear_proj.weight"
        ][i]

        hf[f"model.layers.{i}.post_attention_layernorm.weight"] = state_dict[
            "decoder.layers.mlp.linear_fc1.layer_norm_weight"
        ][i]
        fc1 = state_dict["decoder.layers.mlp.linear_fc1.weight"][i]
        gate_proj, up_proj = torch.split(fc1, fc1.shape[0] // 2, dim=0)
        hf[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_proj
        hf[f"model.layers.{i}.mlp.up_proj.weight"] = up_proj
        hf[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[
            "decoder.layers.mlp.linear_fc2.weight"
        ][i]

    for k in hf:
        hf[k] = hf[k].clone().contiguous()
    return hf


def main():
    args = parse_args()

    iteration = resolve_iteration(args.load, args.iteration)
    ckpt_path = os.path.join(args.load, f"iter_{iteration:07d}")
    print(f"Loading checkpoint from: {ckpt_path}")

    state_dict = load_state_dict_from_dcp(ckpt_path)
    print(f"Loaded {len(state_dict)} tensors from DCP")

    hf_state_dict = convert(state_dict, args)
    print(f"Converted to {len(hf_state_dict)} HF keys")

    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(args.save, "pytorch_model.bin")
    torch.save(hf_state_dict, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
