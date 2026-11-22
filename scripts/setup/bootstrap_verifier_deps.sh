#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare public CPU scorer dependencies in the selected local environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${AUTO_SFT_EVAL_PYTHON:-python3}"
NLTK_TARGET="${AUTO_SFT_NLTK_DATA_DIR:-${HOME}/nltk_data}"
DEPS_DIR="${AUTO_SFT_VERIFIER_DEPS_DIR:-${HOME}/.cache/autosft/verifier_deps}"
CHECK_ONLY=0
OFFLINE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/bootstrap_verifier_deps.sh [--check] [--offline]

Install public scorer packages into the selected Python environment and
download NLTK data into the selected user-writable directory.

  --check    Check imports and NLTK resources without installing or downloading.
  --offline  Install from AUTO_SFT_VERIFIER_DEPS_DIR/packages and copy cached
             NLTK data from AUTO_SFT_VERIFIER_DEPS_DIR/nltk_data. No network.

Environment:
  AUTO_SFT_EVAL_PYTHON       Python in the local eval environment (default: python3)
  AUTO_SFT_NLTK_DATA_DIR     NLTK resource directory (default: ~/nltk_data)
  AUTO_SFT_VERIFIER_DEPS_DIR Offline cache (default: ~/.cache/autosft/verifier_deps)

Activate a virtual environment or set AUTO_SFT_EVAL_PYTHON before installation.
The script does not replace system libraries or create system-wide symlinks.
Keep AUTO_SFT_NLTK_DATA_DIR set when using a custom data directory for evaluation.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for option in "$@"; do
  case "$option" in
    --check) CHECK_ONLY=1 ;;
    --offline) OFFLINE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $option (use --help)" ;;
  esac
done

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "evaluation Python is not executable: $PYTHON_BIN"
else
  PYTHON_BIN="$(command -v -- "$PYTHON_BIN" || true)"
  [[ -n "$PYTHON_BIN" ]] || die "evaluation Python was not found"
fi
export NLTK_DATA="${NLTK_TARGET}${NLTK_DATA:+:${NLTK_DATA}}"

verify() {
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

missing = []
for name in ("absl", "emoji", "syllapy", "langdetect", "nltk"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name} ({exc.__class__.__name__}: {exc})")
if missing:
    print("verifier dependencies incomplete: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)

import nltk
for resource in (
    "tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
):
    try:
        nltk.data.find(resource)
    except LookupError:
        missing.append(f"nltk:{resource}")
if missing:
    print("verifier resources incomplete: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)

import langdetect
langdetect.DetectorFactory.seed = 0
if langdetect.detect("This is an English sentence used for the smoke test.") != "en":
    raise SystemExit("langdetect returned an unexpected language")

# Exercise the tokenizer and POS resources used by IFBench, including modern
# NLTK's punkt_tab dependency, instead of relying on directory presence alone.
nltk.sent_tokenize("One sentence. Another sentence.")
nltk.pos_tag(nltk.word_tokenize("Write a sentence."))
nltk.corpus.stopwords.words("english")
print(f"verifier dependencies ok: python={sys.executable}, nltk={nltk.__version__}")
PY
}

if [[ "$CHECK_ONLY" == 1 ]]; then
  verify
  exit 0
fi

# Check-only works in any environment. Installation requires a venv/Conda env
# so public setup cannot silently modify the host's Python packages.
"$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
    raise SystemExit(
        "Activate a virtual environment or set AUTO_SFT_EVAL_PYTHON to its "
        "Python before installing scorer dependencies."
    )
PY

if [[ "$OFFLINE" == 1 ]]; then
  [[ -d "${DEPS_DIR}/packages" ]] || die "offline packages not found: ${DEPS_DIR}/packages"
  "$PYTHON_BIN" -m pip install --no-index --find-links "${DEPS_DIR}/packages" \
    -r "${SCRIPT_DIR}/requirements-verifier.txt"
  if [[ -d "${DEPS_DIR}/nltk_data" ]]; then
    mkdir -p -- "$NLTK_TARGET"
    cp -an -- "${DEPS_DIR}/nltk_data/." "${NLTK_TARGET}/"
  fi
else
  "$PYTHON_BIN" -m pip install -r "${SCRIPT_DIR}/requirements-verifier.txt"
  mkdir -p -- "$NLTK_TARGET"
  "$PYTHON_BIN" - "$NLTK_TARGET" <<'PY'
import sys
import nltk
for name in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
    if not nltk.download(name, download_dir=sys.argv[1], raise_on_error=True):
        raise SystemExit(f"failed to download NLTK resource: {name}")
PY
fi

verify
printf 'NLTK resources: %s\n' "$NLTK_TARGET"
