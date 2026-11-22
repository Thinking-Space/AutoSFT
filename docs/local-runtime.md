# Local runtime

Activate a Python environment with a CUDA/PyTorch build that matches your GPU before running AutoSFT. The active Python is the default for preparation, training and conversion. Evaluation may use a separate interpreter through `AUTO_SFT_EVAL_PYTHON`; this is useful when the vLLM wheel and the training stack require different PyTorch builds.

Agent runs also require an authenticated Codex CLI on `PATH` with support for
`codex exec --json` and `codex resume`. Install the CLI using the instructions
for your Codex deployment and authenticate it before launching an agent run;
the model API credentials in `.env` do not authenticate Codex. Check the
executable before starting:

```bash
command -v codex
codex --version
```

The standalone `autosft round` pipeline does not start Codex sessions and can
be used to validate data preparation, training, conversion and evaluation
without the agent dependency.

## Public Megatron-LM

AutoSFT uses NVIDIA Megatron-LM `core_v0.14.0` at [commit `23e00ed0963c35382dfe8a5a94fb3cda4d21e133`](https://github.com/NVIDIA/Megatron-LM/commit/23e00ed0963c35382dfe8a5a94fb3cda4d21e133). Fetch the pinned source:

```bash
python scripts/setup/fetch_megatron.py
```

The fetcher checks out the exact commit into `third_party/megatron` and records its identity in `third_party/megatron.lock.json`. The adapter uses upstream `pretrain_gpt.model_provider`, `pretrain_gpt.loss_func`, `megatron.training.pretrain`, packed sequences and the external data loader. An explicit `--commit` can select another full upstream SHA. Set `AUTO_SFT_MEGATRON_DIR` only when using a checkout outside the default directory, such as an archive transferred to another machine:

```bash
export AUTO_SFT_MEGATRON_DIR=/path/to/Megatron-LM-23e00ed0963c35382dfe8a5a94fb3cda4d21e133
```

The directory must contain `pretrain_gpt.py`; keep its commit equal to the SHA in `third_party/megatron.lock.json`.

## Environment and assets

Install PyTorch and its CUDA runtime first, choosing the official build that
matches the host driver and GPU. AutoSFT does not choose this native pair for
you. The pinned Megatron checkout requires Transformer Engine
`>=2.6.0a0,<2.8.0`; install its PyTorch extra after selecting PyTorch. Install
FlashAttention against that exact same PyTorch/CUDA ABI. Do not mix a vendor or
nightly PyTorch build with wheels compiled for another PyTorch release.

Verify that the selected native extensions are importable before starting a
workload:

```bash
python - <<'PY'
import torch
import transformer_engine.pytorch
import flash_attn

print("torch", torch.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformer_engine", getattr(transformer_engine.pytorch, "__version__", "ok"))
print("flash_attn", getattr(flash_attn, "__version__", "ok"))
PY
pip install -e .
pip install -r scripts/local/requirements-train.txt
```

The requirements files intentionally do not install PyTorch, Transformer Engine or FlashAttention: those packages must be selected or built for the host CUDA stack first. The training requirements include NumPy 1.x, also required by the pinned Megatron release. Conversion uses that same environment and checks both directions of the NumPy/PyTorch tensor bridge.

`scripts/local/requirements-eval.txt` pins the Python packages to `vllm==0.11.0` and `transformers==4.57.3`, but it deliberately does not pin PyTorch or CUDA. vLLM's compiled extension must be built for the PyTorch and CUDA runtime already selected on the evaluation machine. Follow the [vLLM 0.11.0 GPU installation guide](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html) and its supported PyTorch/CUDA matrix before installing this requirements file; do not assume that pip will choose a compatible pair for a vendor or nightly PyTorch build.

Installing vLLM into an existing training environment can replace PyTorch or leave a compiled extension linked against a different PyTorch ABI. Prefer a dedicated evaluation environment when the training stack already contains a pinned PyTorch/Transformer Engine build:

```bash
python3 -m venv .venv-eval
. .venv-eval/bin/activate
# Install the CUDA/PyTorch build first. This is the combination validated on
# the H800 development host (CUDA 12.8 runtime):
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
# pip's resolver installs vLLM's pinned xformers dependency and the matching
# Triton dependency from the selected PyTorch 2.8.0 environment.
pip install -r scripts/local/requirements-eval.txt
deactivate
export AUTO_SFT_EVAL_PYTHON="$PWD/.venv-eval/bin/python"
```

For another GPU or CUDA release, select the corresponding official PyTorch
wheel and vLLM-supported pair instead of copying the CUDA 12.8 command. Do not
use `--system-site-packages` for this environment: a system Triton, xFormers
or PyTorch package can be imported ahead of the selected wheel and recreate an
ABI failure. Verify the actual files and native imports before starting a run:

```bash
"$AUTO_SFT_EVAL_PYTHON" - <<'PY'
import torch, vllm, triton, xformers
from vllm import LLM, SamplingParams
print("torch", torch.__version__, torch.version.cuda, torch.__file__)
print("vllm", vllm.__version__, vllm.__file__)
print("triton", triton.__version__, triton.__file__)
print("xformers", xformers.__version__, xformers.__file__)
PY
```

The validated output uses PyTorch `2.8.0+cu128`, vLLM `0.11.0`, Triton
`3.4.0` and xFormers `0.0.32.post1`. A `triton_key` import error during vLLM
startup means Triton came from a different environment; reinstall the Triton
version selected by the PyTorch/vLLM compatibility matrix before retrying.

The evaluation interpreter must be able to import `transformers`, `vllm` and the repository's local scorer modules. Run the runtime probe before starting a workload:

```bash
TRAIN_PYTHON="${AUTO_SFT_TRAIN_PYTHON:-$(command -v python)}"
CONVERT_PYTHON="${AUTO_SFT_CONVERT_PYTHON:-$TRAIN_PYTHON}"
EVAL_PYTHON="${AUTO_SFT_EVAL_PYTHON:-$(command -v python)}"
autosft check --runtime \
  --train-python "$TRAIN_PYTHON" \
  --convert-python "$CONVERT_PYTHON" \
  --eval-python "$EVAL_PYTHON" \
  --model "$AUTO_SFT_SFT_TOKENIZER_MODEL" \
  --checkpoint "$AUTO_SFT_SFT_BASE_CKPT"
```

If the probe reports `vllm/_C.abi3.so: undefined symbol` (or another native symbol error), the vLLM wheel is incompatible with that interpreter's PyTorch/CUDA build. Reinstall a vLLM build supported by the installed PyTorch/CUDA pair, or point `AUTO_SFT_EVAL_PYTHON` at a compatible evaluation environment; changing AutoSFT provider URLs or model settings cannot fix this linker error. Do not start training until the training probe passes, and do not start instruction-following evaluation until the evaluation probe passes.

Paper presets use vLLM for all four benchmarks. Scoring dependencies are included by `scripts/local/requirements-eval.txt`. Benchmark inputs are under `assets/benchmarks`; the included local NLTK resources support instruction-following scoring. To prepare or check additional NLTK resources, use `bash scripts/setup/bootstrap_verifier_deps.sh` in your environment. Custom SGLang code evaluation has its own [requirements](../scripts/local/requirements-eval-sglang.txt).

Keep the HF model's config, tokenizer and safetensors together. `AUTO_SFT_SFT_TOKENIZER_MODEL` points to that full base-model directory, such as `/path/to/Qwen3-1.7B-Base`. `AUTO_SFT_SFT_BASE_CKPT` points to a separate directory containing the same base weights in Megatron's checkpoint format. Import it once:

To obtain the HF directory online, use `hf download Qwen/Qwen3-1.7B-Base --local-dir /path/to/Qwen3-1.7B-Base`; for an offline host, copy that complete directory, including every weight shard, `config.json` and tokenizer files, before running the import command.

```bash
python -m autosft.backends.megatron.import_model \
  --model "$AUTO_SFT_SFT_TOKENIZER_MODEL" --output "$AUTO_SFT_SFT_BASE_CKPT"
```

`autosft round` performs this import automatically for its standalone round. Agent runs reuse the imported base checkpoint each round. Converted final checkpoints return to HF format with their matching tokenizer assets for evaluation.

When Megatron is stored outside this checkout (for example, copied from an offline machine), export its path before training or checking:

```bash
export AUTO_SFT_MEGATRON_DIR=/path/to/Megatron-LM-23e00ed0963c35382dfe8a5a94fb3cda4d21e133
test -f "$AUTO_SFT_MEGATRON_DIR/pretrain_gpt.py"
```

The public default still fetches the pinned NVIDIA GitHub commit into `third_party/megatron`; training and conversion use that checkout automatically. `AUTO_SFT_MEGATRON_DIR` is only a local checkout override and is not copied into the AutoSFT repository.

## Environment checks

`autosft check` checks that source files, benchmark inputs and optional runtimes are available. It does not launch an experiment:

```bash
autosft check --agents --synthesis
autosft check --runtime --gpu \
  --model "$AUTO_SFT_SFT_TOKENIZER_MODEL" --checkpoint "$AUTO_SFT_SFT_BASE_CKPT"
```

Use the smoke round in the README to exercise preparation, import, training, conversion and evaluation on your hardware. An agent run also requires the checkout's own Git working tree for recording code identity. A source archive should first become a local Git working tree; exclude weights, run state and credentials from its baseline.

## Optional interpreter overrides

If inference dependencies conflict with your training stack, use `AUTO_SFT_EVAL_PYTHON` for another inference environment. `AUTO_SFT_CODE_EVAL_PYTHON` defaults to that evaluation interpreter. `AUTO_SFT_TRAIN_PYTHON` and `AUTO_SFT_TOKENIZER_PYTHON` can override their corresponding stages. They are optional runtime choices, so the normal `.env.example` does not list them.

`AUTO_SFT_CONVERT_PYTHON` selects a conversion override. Without it, conversion tries the active interpreter first, then an existing local environment created by `bash scripts/setup/bootstrap_convert_env.sh`. The bootstrap is available for incompatible host environments and is unnecessary when the active Megatron environment passes the conversion check.

## Pipeline validation

The bundled smoke configuration validates the public entrypoint and agent lifecycle on a small dataset:

```bash
bash scripts/run.sh --config configs/examples/smoke-agent.json --dry-run
bash scripts/run.sh --config configs/examples/smoke-agent.json
```

Inspect the training and evaluation receipts, complete repeat artifacts and round ledger described in [the agent-run guide](agent-run.md). Syntax, import and command checks help validate implementation changes before running workloads.

Agent settings are passed explicitly by the launchers. The repository does not require a `.codex/config.toml`; local `.codex` files are ignored. Your Codex authentication remains in your own Codex home and is copied into owner-only run state for isolated sessions.
