#!/usr/bin/env bash
set -Eeuo pipefail

# Build the CPU checkpoint-conversion environment without touching the active
# environment until the replacement has passed its NumPy/PyTorch smoke test.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
ENV_DIR="${AUTO_SFT_CONVERT_ENV_DIR:-${REPO_ROOT}/scripts/setup/.convert_venv}"
PYTHON_BIN="${AUTO_SFT_CONVERT_BOOTSTRAP_PYTHON:-python3}"
BACKUP_ROOT="${AUTO_SFT_CONVERT_BACKUP_DIR:-${REPO_ROOT}/scripts/setup/.convert_venv_backups}"
REQUIREMENTS_FILE="${AUTO_SFT_CONVERT_REQUIREMENTS:-${SCRIPT_DIR}/requirements-convert.txt}"
OFFLINE=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup/bootstrap_convert_env.sh [--offline] [--check]

Options:
  --offline  Do not contact an index. Reuse a validated NumPy 1.x install from
             the existing target environment when one is available.
  --check    Validate the current target environment without creating or moving
             anything.

Environment overrides:
  AUTO_SFT_CONVERT_ENV_DIR             target venv (default: scripts/setup/.convert_venv)
  AUTO_SFT_CONVERT_BOOTSTRAP_PYTHON    Python used to create the venv (default: python3)
  AUTO_SFT_CONVERT_BACKUP_DIR          backup root (default: scripts/setup/.convert_venv_backups)
  AUTO_SFT_CONVERT_REQUIREMENTS        requirements file (default: scripts/setup/requirements-convert.txt)
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --offline)
      OFFLINE=1
      ;;
    --check)
      CHECK_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
  shift
done

if [[ "$ENV_DIR" != /* ]]; then
  ENV_DIR="${REPO_ROOT}/${ENV_DIR}"
fi
if [[ "$BACKUP_ROOT" != /* ]]; then
  BACKUP_ROOT="${REPO_ROOT}/${BACKUP_ROOT}"
fi
if [[ "$REQUIREMENTS_FILE" != /* ]]; then
  REQUIREMENTS_FILE="${REPO_ROOT}/${REQUIREMENTS_FILE}"
fi
[[ -f "$REQUIREMENTS_FILE" ]] || die "conversion requirements file not found: $REQUIREMENTS_FILE"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "bootstrap Python is not executable: $PYTHON_BIN"
  PYTHON_BIN="$(readlink -f -- "$PYTHON_BIN")"
else
  PYTHON_BIN="$(command -v -- "$PYTHON_BIN" || true)"
  [[ -n "$PYTHON_BIN" ]] || die "bootstrap Python was not found"
fi

validate_env() {
  local python_path="$1"
  [[ -x "$python_path" ]] || return 1
  "$PYTHON_BIN" "${REPO_ROOT}/src/autosft/backends/megatron/conversion_environment.py" \
    check "$python_path"
}

if [[ "$CHECK_ONLY" == 1 ]]; then
  validate_env "${ENV_DIR}/bin/python" || die "target environment failed validation: ${ENV_DIR}"
  exit 0
fi

parent_dir="$(dirname -- "$ENV_DIR")"
mkdir -p -- "$parent_dir"

# Keep a validated NumPy installation from the old environment as an offline
# recovery source. This is intentionally read before the target is moved.
old_python="${ENV_DIR}/bin/python"
old_numpy_version=""
old_numpy_file=""
if [[ -x "$old_python" ]]; then
  old_numpy_info="$(PYTHONNOUSERSITE=1 "$old_python" -c 'import numpy; print(numpy.__version__); print(numpy.__file__)' 2>/dev/null || true)"
  if [[ "$old_numpy_info" == *$'\n'* ]]; then
    old_numpy_version="${old_numpy_info%%$'\n'*}"
    old_numpy_file="${old_numpy_info#*$'\n'}"
  fi
fi

old_numpy_valid=0
if [[ -n "$old_numpy_version" && -f "$old_numpy_file" ]]; then
  if PYTHONNOUSERSITE=1 "$old_python" -c 'import numpy; raise SystemExit(int(numpy.__version__.split(".", 1)[0]) >= 2)' 2>/dev/null; then
    old_numpy_valid=1
  fi
fi

staging=""
cleanup() {
  if [[ -n "$staging" && -d "$staging" ]]; then
    # staging is created by mktemp with the target prefix; never remove an
    # arbitrary path if a caller changes variables while the script runs.
    case "$staging" in
      "${ENV_DIR}.staging."*) rm -rf -- "$staging" ;;
    esac
  fi
}
trap cleanup EXIT

staging="$(mktemp -d "${ENV_DIR}.staging.XXXXXX")"
staging_path="${staging}"
staging_name="${staging_path##*/}"
env_name="${ENV_DIR##*/}"
venv_args=(--clear)
if [[ "${AUTO_SFT_CONVERT_SYSTEM_SITE_PACKAGES:-1}" != 0 ]]; then
  # The host image supplies the CUDA/PyTorch build used by the converter. Keep
  # that binary, but shadow its incompatible system NumPy in this venv.
  venv_args+=(--system-site-packages)
fi
"$PYTHON_BIN" -m venv "${venv_args[@]}" "$staging"
staging_python="${staging}/bin/python"
staging_site="$($staging_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# venv records the creation command, including the staging path, in
# pyvenv.cfg. Rewrite that informational field before the directory is moved so
# diagnostics do not point at a temporary location that no longer exists.
venv_command="${PYTHON_BIN} -m venv ${venv_args[*]} ${ENV_DIR}"
"$PYTHON_BIN" - "$staging/pyvenv.cfg" "$venv_command" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
replacement = sys.argv[2]
lines = config_path.read_text(encoding="utf-8").splitlines()
updated = []
found = False
for line in lines:
    if line.startswith("command ="):
        updated.append(f"command = {replacement}")
        found = True
    else:
        updated.append(line)
if not found:
    updated.append(f"command = {replacement}")
config_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY

# Activation scripts also embed the absolute venv path and prompt name. Rewrite
# them before the staging directory is moved so shell activation is correct.
"$PYTHON_BIN" - "$staging_path" "$ENV_DIR" "$staging_name" "$env_name" \
    "$staging/bin/activate" "$staging/bin/activate.csh" "$staging/bin/activate.fish" <<'PY'
from pathlib import Path
import sys

old_path, new_path, old_name, new_name, *files = sys.argv[1:]
for raw_path in files:
    path = Path(raw_path)
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    text = text.replace(old_path, new_path).replace(old_name, new_name)
    path.write_text(text, encoding="utf-8")
PY

clear_staged_numpy() {
  rm -rf -- "${staging_site}/numpy" "${staging_site}/numpy.libs"
  for metadata in "${staging_site}"/numpy-*.dist-info "${staging_site}"/numpy-*.egg-info; do
    [[ -e "$metadata" ]] && rm -rf -- "$metadata"
  done
}

copy_old_numpy() {
  [[ "$old_numpy_valid" == 1 ]] || return 1
  local old_numpy_dir old_site metadata old_libs
  old_numpy_dir="$(dirname -- "$old_numpy_file")"
  old_site="$(dirname -- "$old_numpy_dir")"
  [[ -d "$old_numpy_dir" ]] || return 1
  clear_staged_numpy
  cp -a -- "$old_numpy_dir" "${staging_site}/"
  for metadata in "$old_site"/numpy-*.dist-info "$old_site"/numpy-*.egg-info; do
    [[ -e "$metadata" ]] && cp -a -- "$metadata" "${staging_site}/"
  done
  old_libs="${old_site}/numpy.libs"
  [[ -d "$old_libs" ]] && cp -a -- "$old_libs" "${staging_site}/"
  printf 'Reused NumPy %s from the previous target environment (offline fallback).\n' "$old_numpy_version"
}

if [[ "$OFFLINE" == 1 ]]; then
  copy_old_numpy || die "--offline requires an existing target with NumPy <2"
else
  # /dev/null disables an inherited global constraints file that could force
  # NumPy 2 back into this dedicated environment.
  if ! PIP_CONSTRAINT=/dev/null PIP_CONSTRAINTS= \
      "$staging_python" -m pip install \
      --disable-pip-version-check --no-input --no-cache-dir \
      --force-reinstall --no-deps -r "$REQUIREMENTS_FILE"; then
    printf 'WARN: could not download NumPy <2; trying the validated local copy.\n' >&2
    copy_old_numpy || die "NumPy <2 installation failed and no validated local copy exists"
  fi
fi

validate_env "$staging_python" || die "staged environment failed validation"

# Preserve the previous target instead of deleting/reusing it in place. The
# backup is intentionally outside the target so the final path remains stable.
backup_path=""
if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  mkdir -p -- "$BACKUP_ROOT"
  backup_path="${BACKUP_ROOT}/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mv -- "$ENV_DIR" "$backup_path"
fi

if ! mv -- "$staging" "$ENV_DIR"; then
  if [[ -n "$backup_path" ]]; then
    mv -- "$backup_path" "$ENV_DIR" || true
  fi
  die "could not install staged environment at ${ENV_DIR}"
fi
staging=""

restore_previous() {
  local reason="$1"
  if [[ -z "$backup_path" || ! -e "$backup_path" ]]; then
    die "$reason (no previous environment was available to restore)"
  fi
  local failed_path="${BACKUP_ROOT}/failed-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  if ! mv -- "$ENV_DIR" "$failed_path"; then
    die "$reason; could not move the failed replacement aside"
  fi
  if ! mv -- "$backup_path" "$ENV_DIR"; then
    mv -- "$failed_path" "$ENV_DIR" || true
    die "$reason; could not restore the previous environment"
  fi
  die "$reason; previous environment restored (failed replacement: ${failed_path})"
}

# venv records absolute paths in entry-point shebangs. Because the validated
# environment is installed by moving the staging directory, rewrite only the
# scripts generated by that venv so `pip` remains usable at the final location.
# Symlinks (notably `bin/python`) are deliberately skipped.
for script in "${ENV_DIR}/bin"/*; do
  [[ -f "$script" && ! -L "$script" && -x "$script" ]] || continue
  first_line="$(head -n 1 "$script" 2>/dev/null || true)"
  if [[ "$first_line" == "#!${staging_path}/bin/python"* ]]; then
    printf -v shebang '#!%s' "${ENV_DIR}/bin/python"
    sed -i "1c\\${shebang}" "$script" || restore_previous "could not repair script shebang: ${script}"
  fi
done

# Keep the diagnostic metadata aligned with the installed path as well.
if [[ -f "${ENV_DIR}/pyvenv.cfg" ]]; then
  sed -i "s#${staging_path}#${ENV_DIR}#g" "${ENV_DIR}/pyvenv.cfg" \
    || restore_previous "could not repair pyvenv.cfg"
fi

"${ENV_DIR}/bin/pip" --version >/dev/null \
  || restore_previous "installed venv pip entry point is broken"
validate_env "${ENV_DIR}/bin/python" \
  || restore_previous "installed environment failed validation"

printf 'Installed conversion environment: %s\n' "$ENV_DIR"
if [[ -n "$backup_path" ]]; then
  printf 'Previous environment preserved at: %s\n' "$backup_path"
fi
