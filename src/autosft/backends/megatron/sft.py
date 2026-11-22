"""Small SFT entrypoint using the unmodified public Megatron GPT model and loss."""

from __future__ import annotations
from functools import partial


def _prepare_runtime_compat() -> None:
    """Avoid importing an incompatible torch.compile/Triton JIT fuser.

    The pinned Megatron checkout uses ``torch.compile`` for its optional
    bias/dropout and SwiGLU fusion decorators.  Some NVIDIA PyTorch builds
    ship a newer system Triton whose compiler module no longer exposes the
    ``triton_key`` symbol expected by that torch build.  The failure happens
    while importing Megatron, before command-line fusion flags can be parsed.
    These fusions are optional, so make the decorators no-ops for this
    combination and keep the standard (unfused) PyTorch kernels.
    """
    import torch

    try:
        from triton.compiler import compiler
    except Exception:
        return
    if hasattr(compiler, "triton_key"):
        return

    original_compile = getattr(torch, "compile", None)
    if original_compile is None or getattr(original_compile, "__autosft_compat__", False):
        return

    def no_op_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    no_op_compile.__autosft_compat__ = True
    torch.compile = no_op_compile


_prepare_runtime_compat()


def forward_step(data_iterator, model):
    from autosft.backends.megatron.data import get_batch
    from pretrain_gpt import loss_func

    batch = get_batch(data_iterator)
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=batch["packed_seq_params"],
    )
    return output, partial(loss_func, batch["loss_mask"])


def main():
    from megatron.core.enums import ModelType
    from megatron.training import pretrain
    from pretrain_gpt import model_provider
    from autosft.backends.megatron.data import build_data_iterators

    pretrain(
        build_data_iterators, model_provider, ModelType.encoder_or_decoder, forward_step
    )


if __name__ == "__main__":
    main()
