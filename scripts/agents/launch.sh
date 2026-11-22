#!/bin/bash
set -euo pipefail

# Create one run per benchmark set and launch its meta-agent controller.
# GPU workloads are supervised locally on this host.

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
# Helper function: sets variable to default if unset or "UNDEFINED"
set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""
    
    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

set_default HF_HOME "$HOME/.cache/huggingface"
set_default AUTO_SFT_JOB_SCHEDULER "local"
set_default AUTO_SFT_RUNTIME "host"
set_default AUTO_SFT_REPO_ROOT "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set_default AUTO_SFT_LOCAL_WORKERS "1"
set_default AUTO_SFT_EVAL_BACKEND "opencompass"

# Run-level token budget is supplied by the launcher.
# The agent may read it but must never author or override it.
# The budget is a hard cap: assembled training data that exceeds it is
# rejected and must be reworked, so realized volume is always <= the cap.
# train_iters follows the realized token volume, and the ledger must record
# volume/steps for comparability.
# The launcher should choose a value with sufficient margin from an iteration boundary.

export PYTHONNOUSERSITE=1

# The launcher owns this value; callers may provide or override it before launch.
if [[ ! "$(printenv AUTO_SFT_SFT_TOKEN_BUDGET 2>/dev/null)" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AUTO_SFT_SFT_TOKEN_BUDGET must be supplied by the launcher as a positive integer." >&2
    return 2 2>/dev/null || exit 2
fi
export AUTO_SFT_SFT_TOKEN_BUDGET
# Resolve the SFT manifest after local overrides. Eval runs remain per-request;
# src/autosft/benchmarks/submit.py supplies an exact eval_runs/<name>/eval_submissions.jsonl path.
if [ -n "${AUTO_SFT_RUN_DIR:-}" ]; then
    set_default AUTO_SFT_SFT_MANIFEST "${AUTO_SFT_RUN_DIR}/sft_submissions.jsonl"
fi

# Each entry is one agent run; an entry may list multiple comma-separated benchmarks
# that are all evaluated during that run's eval phase (e.g. "ifeval,ifbench").
BENCHMARKS=(${AUTO_SFT_BENCHMARKS:-ifeval,ifbench})
export AUTO_SFT_COMPONENTS="${AUTO_SFT_COMPONENTS:-none}"
COMPOSITION="$(python -c 'from autosft.config.components import composition_name; print(composition_name())')"
agent=meta
export AUTO_SFT_AGENT_PROMPT="$(python -m autosft.config.components)"
ROOT_DIR="$(pwd)"
# Source archives omit Git history, but the experiment policy guard needs an
# independent working tree to track changes against its launch-time baseline.
# A parent repository would report paths relative to the wrong checkout.
if ! _AUTOSFT_GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || [ "${_AUTOSFT_GIT_ROOT}" != "$(pwd -P)" ]; then
  echo "ERROR: the agent workflow requires this checkout to be its own Git working tree." >&2
  echo "For a source archive, create a fresh local Git baseline as described in docs/local-runtime.md before launching." >&2
  exit 1
fi
unset _AUTOSFT_GIT_ROOT
RUNS_DIR="${AUTO_SFT_AGENT_RUNS_DIR:-${ROOT_DIR}/runs}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://huggingface.co}"
TOKENIZER_MODEL="${AUTO_SFT_SFT_TOKENIZER_MODEL:-${AUTO_SFT_BASE_MODEL_PATH:-${ROOT_DIR}/models/Qwen3-1.7B-Base}}"
export AUTO_SFT_SFT_TOKENIZER_MODEL="${TOKENIZER_MODEL}"
export AUTO_SFT_SFT_BASE_CKPT="${AUTO_SFT_SFT_BASE_CKPT:-${AUTO_SFT_BASE_CHECKPOINT:-${ROOT_DIR}/models/qwen3_1.7b_base.tp1}}"
export AUTO_SFT_BASE_MODEL_PATH="${AUTO_SFT_SFT_TOKENIZER_MODEL}"
export AUTO_SFT_BASE_CHECKPOINT="${AUTO_SFT_SFT_BASE_CKPT}"
# Resolve supported identity before any run directory or dataset is created.
if ! AUTO_SFT_SFT_MODEL="$(python -m autosft.backends.megatron.model_profiles --field model_id)"; then
  echo "ERROR: model selection failed; this release supports Qwen3-1.7B-Base." >&2
  exit 1
fi
export AUTO_SFT_SFT_MODEL
SFT_SEQ_LENGTH="${AUTO_SFT_SFT_SEQ_LENGTH:-8192}"
SFT_CP_SIZE="${AUTO_SFT_SFT_CP_SIZE:-4}"
SFT_DETERMINISTIC="${AUTO_SFT_SFT_DETERMINISTIC:-1}"
if [ "${SFT_DETERMINISTIC}" = "1" ]; then
  SFT_CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  SFT_NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
  SFT_NCCL_ALGO="${NCCL_ALGO:-Ring}"
else
  # Opting out means giving the kernels their fast paths back. Leaving
  # NVTE_ALLOW_NONDETERMINISTIC_ALGO at 0 here would keep Transformer Engine on
  # the memory-hungry attention backward without buying any reproducibility —
  # an OOM at long sequence lengths for nothing.
  SFT_CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-}"
  SFT_NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
  SFT_NCCL_ALGO="${NCCL_ALGO:-}"
fi
# Score-based stop condition: the runner keeps relaunching the agent until every
# benchmark reaches its target primary_score. This is REQUIRED and must be passed
# on the launch command — there is no built-in default. Format is
# "benchmark=score,benchmark=score" (e.g. AUTO_SFT_TARGET_SCORES=ifeval=0.8,ifbench=0.5).
if [ -z "${AUTO_SFT_TARGET_SCORES:-}" ]; then
  echo "ERROR: AUTO_SFT_TARGET_SCORES is required; pass it on the launch command, e.g.:" >&2
  echo "  AUTO_SFT_TARGET_SCORES=ifeval=0.8,ifbench=0.5 bash scripts/agents/launch.sh" >&2
  exit 1
fi
# Every benchmark named in the stop condition must actually be evaluated:
# a target for a benchmark outside AUTO_SFT_BENCHMARKS can never be met, so
# the runner would relaunch the agent forever. Fail at launch instead.
for _target_entry in ${AUTO_SFT_TARGET_SCORES//,/ }; do
  _target_name="${_target_entry%%=*}"
  case ",$(IFS=,; echo "${BENCHMARKS[*]}")," in
    *",${_target_name},"*) ;;
    *)
      echo "ERROR: AUTO_SFT_TARGET_SCORES names '${_target_name}' but AUTO_SFT_BENCHMARKS=(${BENCHMARKS[*]}) does not evaluate it; the run could never terminate." >&2
      echo "  Add it to AUTO_SFT_BENCHMARKS or drop it from AUTO_SFT_TARGET_SCORES." >&2
      exit 1
      ;;
  esac
done
unset _target_entry _target_name
# ...and every benchmark must be one this checkout can actually evaluate.
# The check above only proves the targets and AUTO_SFT_BENCHMARKS agree with
# each other; both can still name a benchmark with no scorer, no inputs, or
# no OpenCompass mapping here. That used to surface at eval submission — one
# round past the point where a full SFT had already been trained, with the
# executor reporting unsupported_benchmark and no evaluation task created.
# Reject it before the run exists.
_ALL_BENCHMARKS=()
for _entry in "${BENCHMARKS[@]}"; do
  IFS=',' read -r -a _split <<< "${_entry}"
  _ALL_BENCHMARKS+=("${_split[@]}")
done
if ! python -m autosft.benchmarks.catalog check --runtime --model "${TOKENIZER_MODEL}" "${_ALL_BENCHMARKS[@]}"; then
  echo "  AUTO_SFT_BENCHMARKS=${AUTO_SFT_BENCHMARKS:-<default>} cannot be evaluated by this checkout." >&2
  exit 1
fi
if [ "${AUTO_SFT_DATA_MODE:-open_pool}" != "open_pool" ]; then
  echo "ERROR: this branch supports only AUTO_SFT_DATA_MODE=open_pool." >&2
  exit 1
fi
# Source datasets dir: staged wholesale into each run's data_pool. It must
# EXIST but may be empty (placeholder; the user adds data later).
AUTO_SFT_SOURCE_DATASETS_DIR="${AUTO_SFT_SOURCE_DATASETS_DIR:-${ROOT_DIR}/data_pool}"
if [ ! -d "${AUTO_SFT_SOURCE_DATASETS_DIR}" ]; then
  echo "ERROR: AUTO_SFT_SOURCE_DATASETS_DIR is not a directory: ${AUTO_SFT_SOURCE_DATASETS_DIR}" >&2
  exit 1
fi
export AUTO_SFT_SOURCE_DATASETS_DIR
export AUTO_SFT_ROLE=meta
AGENT_TARGET_SCORES="${AUTO_SFT_TARGET_SCORES}"
AGENT_RETRY_MAX="${AUTO_SFT_AGENT_RETRY_MAX:-0}"
AGENT_SUCCESS_RETRY_DELAY_SECONDS="${AUTO_SFT_AGENT_SUCCESS_RETRY_DELAY_SECONDS:-60}"
# A round waiting for local GPU work has no work for the root: the executor
# owns it to its terminal report and the harness polls the task. The launcher
# sleeps this long between parked checks instead of starting a root session,
# so a wait costs the run's transcript nothing.
PARKED_IDLE_SECONDS="${AUTO_SFT_PARKED_IDLE_SECONDS:-300}"
# Evaluation repeat count is a fixed run setting like the backend: each
# benchmark is evaluated this many times per round (one task per benchmark
# sampling every prompt N times) and the round's benchmark score is the mean
# of the repeats.
EVAL_REPEATS="${AUTO_SFT_EVAL_REPEATS:-4}"
MAX_ROUNDS="${AUTO_SFT_MAX_ROUNDS:-5}"
if ! [[ "${MAX_ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AUTO_SFT_MAX_ROUNDS must be a positive integer; got ${MAX_ROUNDS}" >&2
  exit 1
fi
if ! [[ "${EVAL_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AUTO_SFT_EVAL_REPEATS must be a positive integer; got ${EVAL_REPEATS}" >&2
  exit 1
fi
EVAL_BACKEND="${AUTO_SFT_EVAL_BACKEND}"
case "${EVAL_BACKEND}" in
  opencompass) ;;
  *)
    echo "ERROR: AUTO_SFT_EVAL_BACKEND must be opencompass." >&2
    exit 1
    ;;
esac

SYN_ENABLED="$(python -c 'from autosft.config.components import synthesis_enabled; print(int(synthesis_enabled()))')"
# Verify the selected trainer source before runtime probing.  The override is
# also needed by the training import probe when Megatron was copied offline.
export AUTO_SFT_MEGATRON_DIR="${AUTO_SFT_MEGATRON_DIR:-${ROOT_DIR}/third_party/megatron}"
if [ ! -f "${AUTO_SFT_MEGATRON_DIR}/pretrain_gpt.py" ]; then
  echo "ERROR: public Megatron-LM checkout is missing: ${AUTO_SFT_MEGATRON_DIR}" >&2
  echo "Fetch it with python scripts/setup/fetch_megatron.py, or set AUTO_SFT_MEGATRON_DIR." >&2
  exit 2
fi

# Resolve the conversion interpreter before the full preflight.  Conversion
# may use the dedicated NumPy 1.x environment even when the controller and
# trainer use another interpreter.
if ! CONVERT_PYTHON="$(python -m autosft.backends.megatron.conversion_environment resolve)"; then
  echo "ERROR: checkpoint conversion environment check failed." >&2
  exit 1
fi

# Check public sources, matching local weights, and every selected runtime
# before creating a run.  This prevents spending a training round before a
# missing vLLM/SGLang or native ABI problem is discovered at evaluation time.
PREFLIGHT_ARGS=(
  --agents --runtime
  --train-python "${AUTO_SFT_TRAIN_PYTHON:-python}"
  --convert-python "${CONVERT_PYTHON}"
  --eval-python "${AUTO_SFT_EVAL_PYTHON:-python}"
  --code-eval-python "${AUTO_SFT_CODE_EVAL_PYTHON:-${AUTO_SFT_EVAL_PYTHON:-python}}"
  --benchmarks "${_ALL_BENCHMARKS[@]}"
  --model "${TOKENIZER_MODEL}" --checkpoint "${AUTO_SFT_SFT_BASE_CKPT}"
)
if [ "${SYN_ENABLED}" = "1" ]; then
  PREFLIGHT_ARGS+=(--synthesis)
fi
python -B -m autosft.orchestration.preflight "${PREFLIGHT_ARGS[@]}"
unset _entry _split _ALL_BENCHMARKS

mkdir -p "${RUNS_DIR}"

for benchmark in "${BENCHMARKS[@]}"; do
    bench_label="${benchmark//,/+}"
    run_id="${agent}_${bench_label}_$(date +%Y%m%d%H%M%S)"
    run_dir="${RUNS_DIR}/${run_id}"
    mkdir -p "${run_dir}/datasets" "${run_dir}/logs"
    # The run dir accumulates codex session rollouts and shell snapshots that
    # replicate task.env secrets with default (world-readable) modes we don't
    # control; deny non-owner traversal at the top instead.
    chmod 700 "${run_dir}"

    python - "${run_dir}/experiment.json" <<'EXPERIMENT'
import json, os, pathlib, sys
payload = os.environ.get('AUTO_SFT_EXPERIMENT_JSON')
if payload:
    pathlib.Path(sys.argv[1]).write_text(json.dumps(json.loads(payload), indent=2) + '\n')
EXPERIMENT
    # Pin the code version this run is launched against. The checkout is
    # shared and writable, so this baseline is what later attempts compare
    # themselves with; without it, a mid-run edit changes the experiment with
    # nothing to show which rounds ran on which version.
    if ! python -m autosft.core.code_identity record \
        --run-dir "${run_dir}" > "${run_dir}/logs/code_identity.txt"; then
      echo "ERROR: could not pin the run's code identity for ${run_dir}" >&2
      exit 1
    fi

    # Stage a full copy of the source datasets dir into this run's open data
    # pool. The staged copy plus staging_manifest.json is the frozen baseline
    # record; the pool itself is fully writable by the agent. One staging per
    # run -- rounds within the run reuse it (pool_staging is idempotent on an
    # existing staging_manifest.json).
    echo "[commit] staging source datasets into ${run_dir}/data_pool ..."
    if ! python -m autosft.data.pool_staging \
        --source-dir "${AUTO_SFT_SOURCE_DATASETS_DIR}" \
        --run-dir "${run_dir}" \
        > "${run_dir}/logs/pool_staging.json"; then
      echo "ERROR: source dataset staging failed for ${run_dir}" >&2
      exit 1
    fi
    if [ "${SYN_ENABLED}" = "1" ]; then
      python - "${run_dir}/syn_data_source" > "${run_dir}/logs/syn_data_source_init.json" <<'SYN_INIT'
import json
import sys
from autosft.synthesis import ensure_data_source

print(json.dumps(ensure_data_source(sys.argv[1]), indent=2))
SYN_INIT
    fi

    # task.env carries plaintext credentials (HF tokens); it must never be
    # world-readable. Create it 600 before any secret line lands.
    rm -f "${run_dir}/task.env"
    ( umask 077 && cat > "${run_dir}/task.env" <<EOF
AUTO_SFT_SFT_SEQ_LENGTH=${SFT_SEQ_LENGTH}
AUTO_SFT_SFT_CP_SIZE=${SFT_CP_SIZE}
AUTO_SFT_SFT_DETERMINISTIC=${SFT_DETERMINISTIC}
NVTE_ALLOW_NONDETERMINISTIC_ALGO=${SFT_NVTE_ALLOW_NONDETERMINISTIC_ALGO}
AUTO_SFT_CONVERT_PYTHON=${CONVERT_PYTHON}
AUTO_SFT_BENCHMARK=${benchmark}
AUTO_SFT_BENCHMARKS=${benchmark}
AUTO_SFT_EVAL_DATA_DIR=${ROOT_DIR}/assets/benchmarks
AUTO_SFT_RUN_ID=${run_id}
AUTO_SFT_RUN_DIR=${run_dir}
AUTO_SFT_DATASETS_DIR=${run_dir}/datasets
AUTO_SFT_DATA_POOL_DIR=${run_dir}/data_pool
AUTO_SFT_SOURCE_DATASETS_DIR=${AUTO_SFT_SOURCE_DATASETS_DIR}
AUTO_SFT_TARGET_SCORES=${AGENT_TARGET_SCORES}
AUTO_SFT_SFT_TOKEN_BUDGET=${AUTO_SFT_SFT_TOKEN_BUDGET}
AUTO_SFT_AGENT_RETRY_MAX=${AGENT_RETRY_MAX}
AUTO_SFT_AGENT_SUCCESS_RETRY_DELAY_SECONDS=${AGENT_SUCCESS_RETRY_DELAY_SECONDS}
AUTO_SFT_PARKED_IDLE_SECONDS=${PARKED_IDLE_SECONDS}
AUTO_SFT_EVAL_BACKEND=${EVAL_BACKEND}
AUTO_SFT_EVAL_REPEATS=${EVAL_REPEATS}
AUTO_SFT_MAX_ROUNDS=${MAX_ROUNDS}
AUTO_SFT_COMPONENTS=${AUTO_SFT_COMPONENTS}
AUTO_SFT_POLICY_GUARD=1
AUTO_SFT_POLICY_BASELINE=${run_dir}/policy_baseline.json
AUTO_SFT_POLICY_FEEDBACK=${run_dir}/policy_feedback.json
AUTO_SFT_AGENT_RUNS_DIR=${RUNS_DIR}
AUTO_SFT_AGENT_RUNS_ROOT=${AUTO_SFT_AGENT_RUNS_ROOT:-${ROOT_DIR}/runs}
HF_ENDPOINT=${HF_ENDPOINT_VALUE}
EOF
    )
    python - "${run_dir}/task.env" "${run_dir}" "${SYN_ENABLED}" <<'LOCAL_ENV'
import os, shlex, sys
keys = ("AUTO_SFT_ROLE", "AUTO_SFT_RESEARCH_MODEL", "AUTO_SFT_REASONING_EFFORT", "AUTO_SFT_OBJECTIVE", "AUTO_SFT_LLM_BASE_URL", "AUTO_SFT_LLM_MODEL", "AUTO_SFT_LLM_API_KEY",
        "AUTO_SFT_LLM_ENABLE_THINKING",
        "AUTO_SFT_MEGATRON_DIR", "AUTO_SFT_SFT_MODEL", "AUTO_SFT_SFT_TOKENIZER_MODEL", "AUTO_SFT_SFT_BASE_CKPT",
        "AUTO_SFT_SFT_MICRO_BATCH_SIZE", "AUTO_SFT_SFT_GLOBAL_BATCH_SIZE",
        "AUTO_SFT_SFT_TP", "AUTO_SFT_SFT_PP", "AUTO_SFT_SFT_CP",
        "AUTO_SFT_SFT_ADAM_BETA1", "AUTO_SFT_SFT_ADAM_BETA2", "AUTO_SFT_SFT_ADAM_EPS", "AUTO_SFT_SFT_WEIGHT_DECAY", "AUTO_SFT_SFT_WARMUP_FRACTION", "AUTO_SFT_SFT_LR", "AUTO_SFT_SFT_MIN_LR", "AUTO_SFT_SFT_LR_WARMUP_ITERS",
        "AUTO_SFT_SFT_RECOMPUTE_NUM_LAYERS", "AUTO_SFT_SFT_NUM_WORKERS",
        "AUTO_SFT_SFT_SHUFFLE_SEED", "AUTO_SFT_SFT_SAVE_INTERVAL",
        "AUTO_SFT_BASE_MODEL_PATH", "AUTO_SFT_BASE_CHECKPOINT", "AUTO_SFT_TRAIN_PYTHON",
        "AUTO_SFT_EVAL_PYTHON", "AUTO_SFT_CODE_EVAL_PYTHON", "AUTO_SFT_TOKENIZER_PYTHON", "AUTO_SFT_SFT_GPUS_PER_NODE",
        "AUTO_SFT_CODE_INFERENCE_BACKEND", "AUTO_SFT_CODE_CONTEXT_LENGTH",
        "AUTO_SFT_CODE_GPU_MEMORY_UTILIZATION",
        "AUTO_SFT_OPENCOMPASS_SGLANG_CONTEXT_LENGTH", "AUTO_SFT_OPENCOMPASS_SGLANG_MEM_FRACTION",
        "AUTO_SFT_EVAL_TP_SIZE", "AUTO_SFT_LOCAL_GPU_LOCK", "AUTO_SFT_EVAL_SEED",
        "AUTO_SFT_OPENCOMPASS_TEMPERATURE", "AUTO_SFT_OPENCOMPASS_MAX_SEQ_LEN",
        "AUTO_SFT_OPENCOMPASS_MAX_OUT_LEN", "AUTO_SFT_EVAL_GPU_MEMORY_UTILIZATION",
        "AUTO_SFT_EVALPLUS_DATA_DIR", "AUTO_SFT_EVAL_TOP_P", "AUTO_SFT_EVAL_TOP_K",
        "AUTO_SFT_EVAL_MIN_P", "AUTO_SFT_NLTK_DATA_DIR", "NLTK_DATA",
        "EVALPLUS_MAX_MEMORY_BYTES", "EVALPLUS_SANITIZE_TIMEOUT")
with open(sys.argv[1], "a") as stream:
    for key in keys:
        if key in os.environ:
            stream.write(f"{key}={shlex.quote(os.environ[key])}\n")
    stream.write("AUTO_SFT_JOB_SCHEDULER=local\n")
    if os.environ.get("AUTO_SFT_AGENT_PROMPT"):
        from pathlib import Path
        Path(sys.argv[2], "agent_prompt.txt").write_text(
            os.environ["AUTO_SFT_AGENT_PROMPT"], encoding="utf-8"
        )
    if sys.argv[3] == "1":
        for key in (
            "AUTO_SFT_SYN_WORKER_CONCURRENCY",
            "AUTO_SFT_SYN_ROWS_PER_TARGET",
            "AUTO_SFT_SYN_MAX_RESUMES", "AUTO_SFT_SYN_RESUME_BACKOFF_SECONDS",
            "AUTO_SFT_SYN_ATTEMPT_TIMEOUT_SECONDS",
            "AUTO_SFT_SYN_BATCH_REFERENCE",
        ):
            if key in os.environ:
                stream.write(f"{key}={shlex.quote(os.environ[key])}\n")
        stream.write(
            "AUTO_SFT_SYN_DATA_SOURCE_DIR="
            + shlex.quote(os.path.join(sys.argv[2], "syn_data_source"))
            + "\n"
        )
LOCAL_ENV
    if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
      echo "HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}" >> "${run_dir}/task.env"
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
      echo "HF_TOKEN=${HF_TOKEN}" >> "${run_dir}/task.env"
    fi
    if [ -n "${AUTO_SFT_DATA_MODE:-open_pool}" ]; then
      echo "AUTO_SFT_DATA_MODE=${AUTO_SFT_DATA_MODE}" >> "${run_dir}/task.env"
    fi
    if [ -n "${AUTO_SFT_TOKENIZER_CODE_DIR:-}" ]; then
      echo "AUTO_SFT_TOKENIZER_CODE_DIR=${AUTO_SFT_TOKENIZER_CODE_DIR}" >> "${run_dir}/task.env"
    fi
    # Only pinned when deterministic training is on; opting out leaves these
    # unset so the kernels choose their own (faster, nondeterministic) paths.
    if [ -n "${SFT_CUBLAS_WORKSPACE_CONFIG}" ]; then
      echo "CUBLAS_WORKSPACE_CONFIG=${SFT_CUBLAS_WORKSPACE_CONFIG}" >> "${run_dir}/task.env"
    fi
    if [ -n "${SFT_NCCL_ALGO}" ]; then
      echo "NCCL_ALGO=${SFT_NCCL_ALGO}" >> "${run_dir}/task.env"
    fi

cat > "${run_dir}/agent_runner.sh" <<EOF
#!/bin/bash
set -euo pipefail
cd "${ROOT_DIR}"
set -a
source "${run_dir}/task.env"
set +a
write_terminal_status() {
  local status="\$1"
  local code="\$2"
  python - "\${AUTO_SFT_RUN_DIR}/agent_terminal_status.json" "\${status}" "\${code}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "last_returncode": int(sys.argv[3]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY
}
trap 'code=\$?; if [ "\${code}" -ne 0 ] && [ ! -f "\${AUTO_SFT_RUN_DIR}/agent_terminal_status.json" ]; then write_terminal_status runner_failed "\${code}"; fi; rm -f "\${AUTO_SFT_RUN_DIR}/agent.pid"' EXIT
# A bare SIGTERM/SIGINT skips the EXIT trap in bash; route them through exit
# so the pid file and terminal status are still written on operator kills.
trap 'exit 143' TERM
trap 'exit 130' INT
python -m autosft.orchestration.policy_guard snapshot \\
  --run-dir "\${AUTO_SFT_RUN_DIR}" \\
  --datasets-dir "\${AUTO_SFT_DATASETS_DIR}" \\
  --recipe "\${AUTO_SFT_RUN_DIR}/data-recipe.yaml"
python -m autosft.agents.runner "${agent}"
EOF
    chmod +x "${run_dir}/agent_runner.sh"

    echo "Starting local agent: agent=${agent} benchmark=${benchmark} run_id=${run_id}"
    # Keep the durable controller alive after the launching terminal closes.
    # The controller can then relaunch the root after local training or
    # evaluation completes.
    nohup setsid "${run_dir}/agent_runner.sh" > "${run_dir}/logs/agent.log" 2>&1 < /dev/null &
    echo "$!" > "${run_dir}/agent.pid"
done

echo "Submitted local agent runs. Logs are under ${RUNS_DIR}/*/logs/agent.log"
