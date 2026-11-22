#!/bin/bash
set -euo pipefail

# Codex launcher shared by every component composition. The harness starts
# analysis and executor sessions from durable requests and owns their polling.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CODEX_PROFILE="autosft"
cd "${ROOT_DIR}"

# The prompts name every skill file as ${AUTO_SFT_REPO_ROOT}/skills/... A bare
# `skills/...` is only correct while the session's cwd is this checkout, and a
# session that reads CODEX_HOME instead resolves it against the run-local
# codex home -- which really does contain a skills/ directory (codex's own
# .system skills), so the wrong guess looks plausible and fails silently.
export AUTO_SFT_REPO_ROOT="${ROOT_DIR}"
# `python -m autosft.*` is the documented way to persist reports and
# write the ledger, and it resolves `src` from the interpreter's path, not from
# a flag we could anchor. A session that cd's into the run dir loses it. Put
# the checkout on PYTHONPATH so those commands survive any working directory.
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! python3 -c 'from autosft.config.components import components; components()'; then
  exit 2
fi


if [ -z "${AUTO_SFT_AGENT_PROMPT:-}" ]; then
  echo "ERROR: AUTO_SFT_AGENT_PROMPT is required" >&2
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

# Agents reach for rg constantly. Without it they improvise a grep shim whose
# flags do not match, and the resulting errors get misread as real failures.
if ! command -v rg >/dev/null 2>&1; then
  echo "ERROR: ripgrep (rg) is not installed or is not on PATH" >&2
  exit 2
fi

POLICY_HOOK="${ROOT_DIR}/scripts/agents/policy_hook.py"
HOOK_CONFIG="hooks.PreToolUse=[{matcher=\"^(shell_command|exec_command|Bash|shell|exec|local_shell_call|apply_patch|Edit|Write|spawn_agent|Task)$\",hooks=[{type=\"command\",command=\"python3 ${POLICY_HOOK}\",timeout=30}]}]"

args=(
  codex
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
)

args+=(-c 'web_search="disabled"')
args+=(-c 'shell_environment_policy.ignore_default_excludes=true')

args+=(
  exec
  --json
  --skip-git-repo-check
)

# The harness binds one root ID in the run's meta_agent_session.json and passes
# it through AUTO_SFT_CODEX_RESUME_SESSION on every retry and runner restart.
# The variable is empty only before the first root session exists. This launcher
# never falls back to a new session when resume fails. `resume --last` could
# pick a subagent's newer rollout and continue the wrong thread.
if [ -n "${AUTO_SFT_CODEX_RESUME_SESSION:-}" ]; then
  args+=(resume "${AUTO_SFT_CODEX_RESUME_SESSION}")
fi

args+=("${AUTO_SFT_AGENT_PROMPT}")

exec "${args[@]}"
