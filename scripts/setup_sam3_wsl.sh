#!/usr/bin/env bash
# Candidate Linux bootstrap. It never runs sudo, changes drivers or downloads weights.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_sam3_wsl.sh --venv /absolute/linux/path [options]

Options:
  --venv PATH      Required absolute Linux venv path, preferably in your home directory.
                   An existing Python 3.12 venv can be reused; other directories are refused.
  --python PATH    Python 3.12 command or executable (default: python3.12).
  --gpu-check      Also run a small real CUDA BF16 matrix multiplication and synchronize.
  --dry-run        Print the installation/check commands without creating files or installing.
  --help          Show this help.

Run in Ubuntu 24.04 on Linux, or WSL2 after its GPU access works.
Required OS packages (install separately; this script never invokes sudo):
  python3.12 python3.12-venv python3-pip ffmpeg git build-essential libgl1 libglib2.0-0t64
For example, an administrator may install these with apt-get before running this script.
Do not install a Linux NVIDIA display driver inside WSL.

Example:
  bash scripts/setup_sam3_wsl.sh --venv "$HOME/.venvs/penguin-sam3" --gpu-check

The bootstrap installs a minimal smoke runtime, not the Streamlit/LLM/RAG application.
Direct candidate pins come from requirements-sam3-smoke.txt; transitive resolution is
recorded in <venv>/sam3-bootstrap-freeze.txt. It checks imports without loading models.
Hugging Face authentication/access and sam3.pt download are separate steps:
  <venv>/bin/hf auth login
Then run scripts/smoke_tracking.py with videos/8.mp4 in the prepared environment.
EOF
}

die() { printf 'Bootstrap error: %s\n' "$*" >&2; exit 1; }
VENV_PATH=""
PYTHON_COMMAND="python3.12"
GPU_CHECK=0
DRY_RUN=0
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --venv|--python)
            (($# >= 2)) || die "$1 needs a value."
            [[ -n "$2" && "$2" != --* ]] || die "$1 needs a value."
            if [[ "$1" == --venv ]]; then VENV_PATH="$2"; else PYTHON_COMMAND="$2"; fi
            shift 2 ;;
        --gpu-check) GPU_CHECK=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) die "Unknown option: $1 (see --help)." ;;
    esac
done
[[ "$VENV_PATH" == /* && "$VENV_PATH" != / ]] || die '--venv must be an absolute Linux path other than /.'
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
[[ "$VENV_PATH" != "$PROJECT_ROOT" ]] || die 'Use a separate venv directory, not the project root.'
REQUIREMENTS="$PROJECT_ROOT/requirements-sam3-smoke.txt"
PY="$VENV_PATH/bin/python"
SAM3_EXPECTED_SHA="847e1a3b15115a04c87c0760297f044f0555d970"

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then "$@"; fi
}

if (( DRY_RUN )); then
    printf 'Dry run only: Python must be 3.12, Linux prerequisites and pinned SAM3 source will be checked.\n'
    SAM3_SOURCE="${SAM3_ROOT:-$PROJECT_ROOT/sam3}"
else
    [[ "$(uname -s)" == Linux ]] || die 'Run installation in Linux/WSL2; --dry-run is available on other systems.'
    for prerequisite in git ffmpeg cc; do
        command -v "$prerequisite" >/dev/null || die "Missing $prerequisite. See --help for OS prerequisites."
    done
    command -v "$PYTHON_COMMAND" >/dev/null || die "Cannot find $PYTHON_COMMAND; install Python 3.12 and its venv package."
    "$PYTHON_COMMAND" -B -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
    cd -- "$PROJECT_ROOT"
    SAM3_SOURCE="$("$PYTHON_COMMAND" -B -c 'from tools.sam3_runtime import resolve_sam3_runtime; print(resolve_sam3_runtime().source_root)')"
    [[ "$(git -C "$SAM3_SOURCE" rev-parse HEAD)" == "$SAM3_EXPECTED_SHA" ]] || die 'SAM3 checkout is not at the pinned revision.'
    git -C "$SAM3_SOURCE" diff --quiet HEAD -- || die 'SAM3 tracked source has local changes; review them before installing.'
    if [[ -e "$VENV_PATH" ]]; then
        [[ -f "$VENV_PATH/pyvenv.cfg" && -x "$PY" ]] || die 'Existing --venv target is not a usable venv; choose a fresh path.'
    else
        run "$PYTHON_COMMAND" -m venv "$VENV_PATH"
    fi
    "$PY" -B -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing venv must use Python 3.12"; assert sys.prefix != sys.base_prefix, "An isolated venv is required"'
fi

if (( DRY_RUN )); then run "$PYTHON_COMMAND" -m venv "$VENV_PATH"; fi
run "$PY" -m pip install 'setuptools==80.9.0' wheel
run "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128'
# Avoid an isolated editable build selecting setuptools>=82 without pkg_resources.
run "$PY" -m pip install --no-build-isolation -r "$REQUIREMENTS" -e "$SAM3_SOURCE"
run "$PY" -m pip check
if (( DRY_RUN )); then
    printf '+ %q -m pip freeze --all > %q\n' "$PY" "$VENV_PATH/sam3-bootstrap-freeze.txt"
else
    "$PY" -m pip freeze --all > "$VENV_PATH/sam3-bootstrap-freeze.txt"
fi
run "$PY" -B "$PROJECT_ROOT/scripts/check_sam3.py"
PREFLIGHT=("$PY" -B "$PROJECT_ROOT/scripts/check_sam3_wsl_environment.py" --output "$VENV_PATH/sam3-bootstrap-environment.json")
if (( GPU_CHECK )); then PREFLIGHT+=(--gpu-check); fi
run "${PREFLIGHT[@]}"
if (( ! DRY_RUN )); then
    printf 'Bootstrap finished. Environment: %s\nResolved packages: %s\n' "$VENV_PATH/sam3-bootstrap-environment.json" "$VENV_PATH/sam3-bootstrap-freeze.txt"
fi
