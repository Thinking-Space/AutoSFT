#!/bin/bash
set -euo pipefail

# Launch a round-scoped AutoSFT session for the executor lifecycle.
#
# The harness invokes this launcher after the meta agent writes the executor context.
# Analysis subagents use search_tasks and their own task IDs instead.
# Each session has durable input and output:
#   input:   ${AUTO_SFT_RUN_DIR}/role_io/<round>/<role>.prompt.md  (task context)
#   receipt: ${AUTO_SFT_RUN_DIR}/role_io/<round>/<role>.receipt.json (final message)
#   events:  ${AUTO_SFT_RUN_DIR}/role_io/<round>/<role>.events.jsonl
#   report:  executor writes it via executor_reports persist-executor
# Agent settings are passed by the launchers. Each role may additionally use
# Codex built-in subagents within its own role contract.
#
# Re-running the launcher for the same round is safe:
#   - receipt present, delegation finished -> no-op; the receipt path is
#     printed again. The receipt alone proves nothing — a round writes one at
#     every stage boundary — so it only counts when the persisted executor
#     report is terminal for the consumed task context. Changed context
#     content reopens even a succeeded job; an identical-content touch rearms
#     only a failed/blocked attempt.
#   - interrupted session      -> resumed by its own session id, provided the
#     task context is byte-identical to what that session was started with.
#   - changed task context     -> the stale session is abandoned (its events
#     log is rotated aside) and a fresh session starts.
# The resume id comes ONLY from this round's own events log — never from
# --last, another round, or a directory scan — so a resume can never continue
# a different agent or conversation.
#
# Usage:
#   run_role.sh <role> <round_id> <task-context-file>
# <role> must be executor.
# <task-context-file> is a run-local file authored by the caller carrying the
# round-specific paths/artifacts. It is appended after the role instructions.
# The launcher prints the receipt path on success; the receipt file holds the
# role's final message (its persisted-report receipt JSON).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CODEX_PROFILE="autosft"
cd "${ROOT_DIR}"

# Role prompts name every skill file as ${AUTO_SFT_REPO_ROOT}/skills/..., and
# every role persists its report through `python -m autosft....`.
# See the same two exports in run_codex.sh for why neither survives on a bare
# relative path once a session changes its working directory.
export AUTO_SFT_REPO_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! python3 -c 'from autosft.config.components import components; components()'; then
  exit 2
fi
if ! python3 -c 'from autosft.config.components import executor_enabled; raise SystemExit(not executor_enabled())'; then
  echo "ERROR: the executor component is disabled." >&2
  exit 2
fi


ROLE="${1:-}"
ROUND_ID="${2:-}"
TASK_CONTEXT="${3:-}"

case "${ROLE}" in
  executor) ;;
  *)
    echo "ERROR: unknown role '${ROLE}'. Expected executor; analysis subagents use search_tasks." >&2
    exit 2
    ;;
esac

export AUTO_SFT_ROLE=executor

if [ -z "${ROUND_ID}" ]; then
  echo "ERROR: round id is required (arg 2)" >&2
  exit 2
fi

if [ -z "${TASK_CONTEXT}" ] || [ ! -f "${TASK_CONTEXT}" ]; then
  echo "ERROR: task context file is required (arg 3) and must exist" >&2
  exit 2
fi

if [ -z "${AUTO_SFT_RUN_DIR:-}" ]; then
  echo "ERROR: AUTO_SFT_RUN_DIR is required" >&2
  exit 2
fi

if [ -z "${CODEX_HOME:-}" ]; then
  echo "ERROR: CODEX_HOME must point to the AutoSFT run-local Codex home" >&2
  exit 2
fi

if [ ! -f "${CODEX_HOME}/${CODEX_PROFILE}.config.toml" ]; then
  echo "ERROR: AutoSFT Codex profile was not prepared under CODEX_HOME=${CODEX_HOME}" >&2
  exit 2
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI is not installed or is not on PATH" >&2
  exit 2
fi

ROLE_PROMPT="${ROOT_DIR}/src/autosft/agents/prompts/executor.md"
if [ ! -f "${ROLE_PROMPT}" ]; then
  echo "ERROR: missing role prompt: ${ROLE_PROMPT}" >&2
  exit 2
fi

IO_DIR="${AUTO_SFT_RUN_DIR}/role_io/round_${ROUND_ID}"
mkdir -p "${IO_DIR}"

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for executor publication coordination" >&2
  exit 2
fi
# Keep this descriptor inherited by the detached session. Closing only the
# launcher must not permit publication while its executor is still alive.
exec {EXECUTOR_LIFECYCLE_FD}>"${IO_DIR}/.executor.lifecycle.lock"
if ! flock -n "${EXECUTOR_LIFECYCLE_FD}"; then
  echo "in progress: executor lifecycle or data publication holds the round lock; session not started" >&2
  exit 75
fi
if [ ! -f "${TASK_CONTEXT}" ]; then
  echo "in progress: executor task context was withdrawn before launch; session not started" >&2
  exit 75
fi

PROMPT_COPY="${IO_DIR}/${ROLE}.prompt.md"
RECEIPT="${IO_DIR}/${ROLE}.receipt.json"
LOG="${IO_DIR}/${ROLE}.events.jsonl"

# Executor completion is bound to consumed context and the prelaunch report
# identity by role_pump. Both prelaunch idempotency and postlaunch recovery
# use that same check, including when a receipt is missing or bogus.
executor_state() {
  python3 -m autosft.agents.role_pump executor-state \
    --run-dir "${AUTO_SFT_RUN_DIR}" --round-id "${ROUND_ID}" \
    --task-context "${TASK_CONTEXT}"
}
executor_state_field() {
  python3 -c '
import json, sys
value = json.load(sys.stdin)[sys.argv[1]]
if value is not None:
    print(json.dumps(value))
' "$1"
}
# Set when a terminal delegation is being reopened: the concluded
# session already gave its final answer, so it must never be resumed — the
# retry starts fresh from the rewritten (or re-affirmed) context.
REDELEGATED=0
EXECUTOR_STATE="$(executor_state)"
RECOVERED_RECEIPT="$(printf '%s' "${EXECUTOR_STATE}" | executor_state_field receipt)"
if [ -n "${RECOVERED_RECEIPT}" ]; then
  printf '%s\n' "${RECOVERED_RECEIPT}" > "${RECEIPT}"
  echo "${RECEIPT}"
  exit 0
fi
REOPEN="$(printf '%s' "${EXECUTOR_STATE}" | executor_state_field needs_fresh_session)"
if [ "${REOPEN}" = "true" ]; then
  # Rotate before the live-pid check: a concluded session must not become
  # resumable if this invocation exits while its wrapper is still alive.
  if [ -f "${LOG}" ]; then
    mv "${LOG}" "${LOG%.jsonl}.stale.$(date +%Y%m%d%H%M%S).jsonl"
  fi
  if [ -s "${RECEIPT}" ]; then
    { cat "${RECEIPT}"; echo; } >> "${RECEIPT%.json}.failures.jsonl"
    rm -f "${RECEIPT}"
  fi
  REDELEGATED=1
  if [ -f "${PROMPT_COPY}" ]; then
    echo "WARN: role 'executor' round ${ROUND_ID}: prior report does not close the current task context; re-delegating" >&2
  fi
elif [ -s "${RECEIPT}" ]; then
  # Keep every stage receipt: append it to the round's phases archive, then
  # clear the live receipt so the launcher can resume the session. A crash
  # between the append and the rm only re-appends the same line next time.
  { cat "${RECEIPT}"; echo; } >> "${RECEIPT%.json}.phases.jsonl"
  rm -f "${RECEIPT}"
  echo "WARN: role 'executor' round ${ROUND_ID}: receipt covered a finished stage, not the whole delegation; archived it and continuing the session" >&2
fi

# Liveness: the codex session is detached from this launcher (see below), so a
# caller-side tool timeout kills the launcher but not the session. If a prior
# invocation's session is still alive, do not start or resume anything on top
# of it — report in-progress and let the caller re-invoke later.
#
# "Still alive" is decided from /proc, not kill -0: kill -0 answers "does the
# pid exist", and a zombie exists. When a session dies and nothing reaps it
# (orphans here are adopted by an init that never waits), the recorded pid
# stays a zombie indefinitely, kill -0 keeps succeeding, and every re-invoke
# reports in-progress — which once pinned a whole round behind a
# session that had died at "model is at capacity", with the pump's only
# unstick move being this launcher. The pid is trusted only while it is a
# running (non-zombie) process whose command line still names this
# (role, round)'s receipt; the identity half also rejects an unrelated
# process that recycled the pid, whose existence kill -0 would equally
# mistake for the session's.
PIDFILE="${IO_DIR}/.${ROLE}.codex.pid"
role_session_is_alive() {
  local pid="$1"
  [ -n "${pid}" ] || return 1
  if [ -d /proc ]; then
    local stat state cmdline
    stat="$(cat "/proc/${pid}/stat" 2>/dev/null)" || return 1
    state="${stat##*) }"
    state="${state%% *}"
    case "${state}" in
      Z|X) return 1 ;;
    esac
    # A zombie's cmdline is empty and a recycled pid's lacks the receipt
    # path, so both fail the substring test. Read via command substitution
    # rather than a pipe into grep so a vanishing /proc entry can only make
    # the test fail, never trip pipefail.
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    case "${cmdline}" in
      *"${RECEIPT}"*) return 0 ;;
    esac
    return 1
  fi
  # No /proc on this platform: fall back to signal 0 and accept its zombie
  # blindness rather than failing closed on every live session.
  kill -0 "${pid}" 2>/dev/null
}
if [ -f "${PIDFILE}" ]; then
  LIVE_PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
  if role_session_is_alive "${LIVE_PID}"; then
    echo "in progress: role '${ROLE}' session (pid ${LIVE_PID}) is still running; receipt not ready at ${RECEIPT}" >&2
    exit 75
  fi
  echo "WARN: role '${ROLE}' round ${ROUND_ID}: recorded session (pid ${LIVE_PID:-unknown}) is dead or is not this role's session; clearing its pidfile" >&2
  rm -f "${PIDFILE}"
fi

# Role sessions get their own codex home under the run dir, seeded from the
# meta agent's run-local home. This keeps role rollouts out of meta-session discovery
# when the outer runner first binds meta_agent_session.json. Each role launched
# here is itself a top-level Codex session, with its own task recovery rules.
ROLE_HOME="${AUTO_SFT_RUN_DIR}/codex_home_roles"
mkdir -p "${ROLE_HOME}"
chmod 700 "${ROLE_HOME}"
for seed in auth.json config.toml "${CODEX_PROFILE}.config.toml"; do
  if [ -f "${CODEX_HOME}/${seed}" ]; then
    cp -f "${CODEX_HOME}/${seed}" "${ROLE_HOME}/${seed}"
    chmod 600 "${ROLE_HOME}/${seed}"
  fi
done

# Build this invocation's prompt to a staging file first: whether we may resume
# depends on comparing it against the prompt the interrupted session was
# actually started with, so the durable copy must not be overwritten before
# that comparison.
PROMPT_CANDIDATE="${IO_DIR}/.${ROLE}.prompt.next"
{
  cat "${ROLE_PROMPT}"
  python3 -c 'import sys; from autosft.agents.roles import EXECUTOR_CONTEXT_DELIMITER; sys.stdout.buffer.write(EXECUTOR_CONTEXT_DELIMITER)'
  cat "${TASK_CONTEXT}"
} > "${PROMPT_CANDIDATE}"

# Resume candidate: the FIRST thread.started id in this (role, round)'s own
# events log. The log path is keyed by role and round inside this run's
# role_io tree, so an id read from it can only ever name this role's own
# session for this round — other roles, other rounds, and other runs each
# write to different files. --last and directory scans are never used.
#
# A re-delegated terminal attempt never resumes. This includes a failure
# rearmed by byte-identical context: that session already CONCLUDED, and a
# resume nudge for interrupted work would invite it to reaffirm the failure
# instead of retrying against the repaired preconditions. The
# concluded session's log was already rotated when its receipt was archived
# (durably, in case this launcher dies before reaching here); the branch
# also skips the resume scan in case that rotation was lost.
RESUME_ID=""
if [ "${REDELEGATED}" = 1 ]; then
  if [ -f "${LOG}" ]; then
    mv "${LOG}" "${LOG%.jsonl}.stale.$(date +%Y%m%d%H%M%S).jsonl"
  fi
elif [ -f "${LOG}" ] && [ -f "${PROMPT_COPY}" ] && cmp -s "${PROMPT_CANDIDATE}" "${PROMPT_COPY}"; then
  RESUME_ID="$(python3 - "${LOG}" <<'PY' 2>/dev/null || true
import json
import sys

for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "thread.started" and event.get("thread_id"):
        print(event["thread_id"])
        break
PY
)"
  # Resume only a session whose rollout actually exists in the role home; a
  # cleaned home means the id is unrecoverable and a fresh start is correct.
  if [ -n "${RESUME_ID}" ]; then
    if ! find "${ROLE_HOME}/sessions" -name "*${RESUME_ID}*" -print -quit 2>/dev/null | grep -q .; then
      echo "WARN: role '${ROLE}' session ${RESUME_ID} has no rollout under ${ROLE_HOME}; starting fresh" >&2
      RESUME_ID=""
    fi
  fi
elif [ -f "${LOG}" ]; then
  # The task context changed since the interrupted attempt: that session was
  # started with different instructions and must never be resumed. Rotate its
  # log aside so its session id cannot be picked up by a later invocation.
  mv "${LOG}" "${LOG%.jsonl}.stale.$(date +%Y%m%d%H%M%S).jsonl"
fi

mv "${PROMPT_CANDIDATE}" "${PROMPT_COPY}"
INPUT_ARGS=()
if [ -n "${RESUME_ID}" ]; then INPUT_ARGS+=(--resume); fi
python3 -m autosft.agents.role_pump record-executor-input \
  --run-dir "${AUTO_SFT_RUN_DIR}" --round-id "${ROUND_ID}" \
  --task-context "${TASK_CONTEXT}" "${INPUT_ARGS[@]}"

POLICY_HOOK="${ROOT_DIR}/scripts/agents/policy_hook.py"
HOOK_CONFIG="hooks.PreToolUse=[{matcher=\"^(shell_command|exec_command|Bash|shell|exec|local_shell_call|apply_patch|Edit|Write|spawn_agent|Task)$\",hooks=[{type=\"command\",command=\"python3 ${POLICY_HOOK}\",timeout=30}]}]"

codex_args=(
  -C "${ROOT_DIR}"
  --profile "${CODEX_PROFILE}"
  --model "${AUTO_SFT_RESEARCH_MODEL:-gpt-5.5}"
  -c "model_reasoning_effort=\"${AUTO_SFT_REASONING_EFFORT:-medium}\""
  -c 'features.multi_agent=true'
  -c 'agents.interrupt_message=true'
  --dangerously-bypass-approvals-and-sandbox
  --dangerously-bypass-hook-trust
  --disable unified_exec
  -c "${HOOK_CONFIG}"
  -c 'web_search="disabled"'
  -c 'shell_environment_policy.ignore_default_excludes=true'
  exec
  --json
  --skip-git-repo-check
  --output-last-message "${RECEIPT}"
)

STATUS=0
# Exported for the setsid subshell below, which is a fresh bash and inherits
# only the environment.
TRIM_EVENTS="${ROOT_DIR}/scripts/agents/trim_events.py"
export TRIM_EVENTS LOG PROMPT_COPY RESUME_ID
# The codex session runs in its own process group (setsid) so a caller-side
# tool timeout that kills this launcher's group cannot kill the session: the
# meta agent's shell tool caps commands at ~1200s, and an executor mid SFT/eval
# polling legitimately outlives that. The pidfile lets a later invocation see
# the surviving session (exit 75 above) instead of double-launching, and the
# receipt/report recovery below still works once the session finishes.
# The event stream passes through a filter that shortens the raw command
# output codex inlines into each event (see trim_events.py). It runs inside
# the same setsid session so it dies with the role rather than outliving it,
# and it preserves per-line JSON so the thread.started recovery above still
# works. `wait` must still track codex itself, not the filter: the pidfile is
# the liveness signal and ${STATUS} gates receipt recovery, so the session's
# own exit code has to survive the pipeline.
if [ -n "${RESUME_ID}" ]; then
  RESUME_PROMPT="Your previous session for this task was interrupted and has been resumed. Before continuing, re-read the durable state on disk (the run-directory artifacts named in your original instructions, including the current \${AUTO_SFT_RUN_DIR}/executor_handoff.json) — it is more current than this conversation, since work may have progressed or finished during the interruption. Then continue the original task. Persistence and receipt contracts are unchanged: persist your report with the documented command, then end with only the receipt JSON as your final message."
  # A stage-scoped task context ends a session, not the delegation: the
  # harness resumes this same session for each further stage. Without this
  # the resumed session can conclude its scoped stage is already done and
  # stop, and the round never advances.
  RESUME_PROMPT="${RESUME_PROMPT} A task context scoped to a single stage does not end the delegation: after refreshing the handoff, whenever it names a further stage, continue with that stage's next_action under your fixed role instructions — persisting a report at each stage boundary — until the handoff is complete or blocked."
  export RESUME_PROMPT
  # Append: the original thread.started must stay first in the log, because it
  # is the authoritative session id for any later resume.
  CODEX_HOME="${ROLE_HOME}" setsid bash -c '
    codex "${@}" resume "${RESUME_ID}" "${RESUME_PROMPT}" 2>&1 </dev/null \
      | python3 "${TRIM_EVENTS}" >> "${LOG}"
    exit "${PIPESTATUS[0]}"
  ' _ "${codex_args[@]}" &
else
  CODEX_HOME="${ROLE_HOME}" setsid bash -c '
    codex "${@}" "$(cat "${PROMPT_COPY}")" 2>&1 </dev/null \
      | python3 "${TRIM_EVENTS}" > "${LOG}"
    exit "${PIPESTATUS[0]}"
  ' _ "${codex_args[@]}" &
fi
CODEX_PID=$!
echo "${CODEX_PID}" > "${PIDFILE}"
wait "${CODEX_PID}" || STATUS=$?
rm -f "${PIDFILE}"

# A session that persisted its report but never returned leaves no receipt:
# the receipt is written by codex --output-last-message, so anything that kills
# the session after persistence (notably a caller-side command timeout) loses
# it while the durable work survives. Rebuild the receipt from that report
# rather than reporting failure, because the launcher's only completion signal
# is the receipt and the meta agent would otherwise relaunch a finished role forever.
#
# Executor recovery below additionally requires this invocation's input and
# report identity, regardless of terminal status or any claimed receipt.
EXECUTOR_STATE="$(executor_state)"
RECOVERED="$(printf '%s' "${EXECUTOR_STATE}" | executor_state_field receipt)"
if [ -n "${RECOVERED}" ]; then
  if [ ! -s "${RECEIPT}" ]; then
    echo "WARN: role '${ROLE}' returned no receipt; recovered it from the persisted report" >&2
  fi
  printf '%s\n' "${RECOVERED}" > "${RECEIPT}"
  echo "${RECEIPT}"
  exit 0
fi
if [ "$(printf '%s' "${EXECUTOR_STATE}" | executor_state_field phase_complete)" = "true" ]; then
  echo "in progress: role 'executor' finished a stage; delegation is still pending" >&2
  exit 75
fi
if [ -s "${RECEIPT}" ]; then
  { cat "${RECEIPT}"; echo; } >> "${RECEIPT%.json}.failures.jsonl"
  rm -f "${RECEIPT}"
fi
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: role '${ROLE}' codex session exited with ${STATUS}; see ${LOG}" >&2
  exit "${STATUS}"
fi
echo "ERROR: role 'executor' finished without a current task report; see ${LOG}" >&2
exit 3
