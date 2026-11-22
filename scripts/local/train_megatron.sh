#!/bin/bash
# Execute the configured Qwen3 SFT protocol on this host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Reject unsupported model selections before creating output or launching ranks.
PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" "${AUTO_SFT_TRAIN_PYTHON:-python}" \
  -m autosft.backends.megatron.model_profiles --field model_id >/dev/null
TRAIN_ITERS="${1:?train iterations required}"
SEQ_LENGTH="${2:?sequence length required}"
SAVE_INTERVAL="${AUTO_SFT_SFT_SAVE_INTERVAL:-$TRAIN_ITERS}"
# Megatron also saves on normal completion when the final iteration is not
# divisible by this interval, so intermediate saves never replace the final.
if [[ ! "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo 'AUTO_SFT_SFT_SAVE_INTERVAL must be a positive integer' >&2; exit 2
fi
: "${AUTO_SFT_SFT_LOAD_CKPT:?local base checkpoint required}"
: "${AUTO_SFT_SFT_TOKENIZER_MODEL:?local base tokenizer required}"
: "${AUTO_SFT_SFT_OUTPUT_DIR:?output required}"
export DATASET_PATH="${AUTO_SFT_SFT_DATASET_PATH:?prepared token dataset required}"
read -r DATA_WEIGHT DATA_DIRECTORY <<< "$DATASET_PATH"
if [[ "$DATA_WEIGHT" != "1.0" || -z "$DATA_DIRECTORY" ]]; then
  echo "Prepared data must contain one weighted token dataset: 1.0 <directory>." >&2
  exit 2
fi
export CP_SIZE="${AUTO_SFT_SFT_CP_SIZE:-${AUTO_SFT_SFT_CP:-4}}"
if [[ -n "${AUTO_SFT_SFT_CP:-}" && "$AUTO_SFT_SFT_CP" != "$CP_SIZE" ]]; then
  echo 'Conflicting CP settings' >&2; exit 2
fi
export AUTO_SFT_SFT_GPUS_PER_NODE="${AUTO_SFT_SFT_GPUS_PER_NODE:-8}"
TP="${AUTO_SFT_SFT_TP:-1}"
PP="${AUTO_SFT_SFT_PP:-1}"
if [[ "$PP" != 1 ]]; then
  echo 'Local SFT supports data, tensor and context parallelism with PP=1' >&2; exit 2
fi
GLOBAL_BATCH="${AUTO_SFT_SFT_GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH="${AUTO_SFT_SFT_MICRO_BATCH_SIZE:-1}"
if (( AUTO_SFT_SFT_GPUS_PER_NODE % (TP * PP * CP_SIZE) != 0 )); then
  echo 'GPU count must be divisible by TP * PP * CP' >&2; exit 2
fi
DP=$(( AUTO_SFT_SFT_GPUS_PER_NODE / (TP * PP * CP_SIZE) ))
if (( GLOBAL_BATCH % (MICRO_BATCH * DP) != 0 )); then
  echo 'Global batch must be divisible by micro batch * data parallel degree' >&2; exit 2
fi
MEGATRON_DIR="${AUTO_SFT_MEGATRON_DIR:-$ROOT/third_party/megatron}"
if [[ ! -f "$MEGATRON_DIR/pretrain_gpt.py" ]]; then
  echo "Fetch the pinned public Megatron checkout with scripts/setup/fetch_megatron.py." >&2
  exit 2
fi
export PYTHONPATH="$MEGATRON_DIR:$ROOT/src:$ROOT/third_party:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Required by the pinned Megatron checkout for tensor/context parallelism on
# GPUs older than Blackwell; keep it fixed for this launcher.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DETERMINISM=()
if [[ "${AUTO_SFT_SFT_DETERMINISTIC:-1}" == 1 ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NCCL_ALGO="${NCCL_ALGO:-Ring}"
  DETERMINISM=(--deterministic-mode)
fi
WARMUP_ITERS="${AUTO_SFT_SFT_LR_WARMUP_ITERS:-$("${AUTO_SFT_TRAIN_PYTHON:-python}" -c 'import sys; print(int(int(sys.argv[1]) * float(sys.argv[2])))' "$TRAIN_ITERS" "${AUTO_SFT_SFT_WARMUP_FRACTION:-0.05}")}"
mkdir -p "$AUTO_SFT_SFT_OUTPUT_DIR/checkpoints" "$AUTO_SFT_SFT_OUTPUT_DIR/tensorboard"
# Select a local training virtualenv by PATH, without changing the host Python.
if [[ -n "${AUTO_SFT_TRAIN_PYTHON:-}" ]]; then
  export PATH="$(dirname "$AUTO_SFT_TRAIN_PYTHON"):$PATH"
fi
exec "${AUTO_SFT_TRAIN_PYTHON:-python}" -m torch.distributed.run \
  --nproc_per_node "$AUTO_SFT_SFT_GPUS_PER_NODE" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-6000}" \
  --module autosft.backends.megatron.sft \
  --num-layers 28 --hidden-size 2048 --ffn-hidden-size 6144 \
  --num-attention-heads 16 --group-query-attention --num-query-groups 8 --kv-channels 128 \
  --vocab-size 151936 --padded-vocab-size 151936 --make-vocab-size-divisible-by 1 --disable-bias-linear \
  --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm --swiglu \
  --position-embedding-type rope --rotary-base 1000000 --no-position-embedding \
  --attention-dropout 0 --hidden-dropout 0 --init-method-std 0.02 \
  --tokenizer-type HuggingFaceTokenizer --dataloader-type external \
  --data-path "$DATA_WEIGHT" "$DATA_DIRECTORY" --no-create-attention-mask-in-dataloader \
  --micro-batch-size "$MICRO_BATCH" --global-batch-size "$GLOBAL_BATCH" \
  --lr "${AUTO_SFT_SFT_LR:-1e-4}" --min-lr "${AUTO_SFT_SFT_MIN_LR:-1e-7}" \
  --train-iters "$TRAIN_ITERS" --lr-decay-iters "$TRAIN_ITERS" --lr-decay-style cosine \
  --lr-warmup-iters "$WARMUP_ITERS" \
  --optimizer adam --adam-beta1 "${AUTO_SFT_SFT_ADAM_BETA1:-0.9}" --adam-beta2 "${AUTO_SFT_SFT_ADAM_BETA2:-0.95}" --adam-eps "${AUTO_SFT_SFT_ADAM_EPS:-1e-8}" --weight-decay "${AUTO_SFT_SFT_WEIGHT_DECAY:-0.1}" --clip-grad 1 \
  --bf16 --use-distributed-optimizer --ckpt-format torch_dist \
  --save "$AUTO_SFT_SFT_OUTPUT_DIR/checkpoints" --tensorboard-dir "$AUTO_SFT_SFT_OUTPUT_DIR/tensorboard" \
  --save-interval "$SAVE_INTERVAL" --load "$AUTO_SFT_SFT_LOAD_CKPT" --finetune --exit-on-missing-checkpoint \
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --context-parallel-size "$CP_SIZE" \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --seq-length "$SEQ_LENGTH" --max-position-embeddings "$SEQ_LENGTH" \
  --tokenizer-model "$AUTO_SFT_SFT_TOKENIZER_MODEL" --log-interval 1 --log-throughput \
  --num-workers "${AUTO_SFT_SFT_NUM_WORKERS:-0}" --eval-iters 0 "${DETERMINISM[@]}"
