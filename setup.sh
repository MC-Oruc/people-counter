#!/usr/bin/env bash
#
# SYNOPSIS
#   Interactive environment setup for People Counter (Linux/macOS, bash).
# DESCRIPTION
#   - Creates a Python virtual environment in .venv (if missing)
#   - Detects CUDA via nvidia-smi and offers CPU/GPU PyTorch installation
#   - Installs torch+torchvision for CPU or a selected CUDA version (12.9/12.8/12.6)
#   - Installs remaining dependencies from requirements.txt
#   - Runs a quick smoke test
# USAGE
#   bash ./setup.sh
#   or:  chmod +x setup.sh && ./setup.sh
# NOTES
#   Requires Python 3.10+ in PATH. For macOS/Linux shells.
#   On Apple Silicon, GPU acceleration uses MPS (Metal) and is enabled inside PyTorch automatically
#   if available; choose CPU here (CUDA is not applicable on macOS).

set -euo pipefail

note() { printf "\n==== %s ====\n\n" "$1"; }
warn() { printf "[warn] %s\n" "$1" >&2; }
err()  { printf "[error] %s\n" "$1" >&2; }

read_choice() {
  local title="$1"; shift
  local question="$1"; shift
  local -a choices=("$@")
  echo
  echo "$title"
  echo "$question"
  local i=1
  for c in "${choices[@]}"; do
    printf "  [%d] %s\n" "$i" "$c"
    i=$((i+1))
  done
  while true; do
    read -r -p "Selection (1-${#choices[@]}): " sel || true
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
      local idx=$((sel-1))
      if (( idx >= 0 && idx < ${#choices[@]} )); then
        echo "${choices[$idx]}"
        return 0
      fi
    fi
    echo "Invalid choice, try again."
  done
}

read_yesno() {
  local title="$1"; shift
  local question="$1"; shift
  local default_yes="${1:-false}"; shift || true
  echo
  echo "$title"
  local suffix="[y/N]"
  if [[ "$default_yes" == "true" ]]; then suffix="[Y/n]"; fi
  while true; do
    read -r -p "$question $suffix " ans || true
    if [[ -z "$ans" ]]; then
      if [[ "$default_yes" == "true" ]]; then echo "yes"; else echo "no"; fi
      return 0
    fi
    ans_lc=$(echo "$ans" | tr '[:upper:]' '[:lower:]')
    case "$ans_lc" in
      y|yes) echo "yes"; return 0;;
      n|no)  echo "no";  return 0;;
      *) echo "Please answer yes or no.";;
    esac
  done
}

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    err "Python 3.10+ is required in PATH."
    exit 1
  fi
  $PY - <<'PY'
import sys
ver = sys.version_info
ok = (ver.major == 3 and ver.minor >= 10) or ver.major > 3
sys.exit(0 if ok else 1)
PY
  if [[ $? -ne 0 ]]; then
    err "Python 3.10+ is required. Found: $($PY --version 2>&1)"
    exit 1
  fi
  echo "$PY"
}

init_venv() {
  local py_bin="$1"
  if [[ ! -d .venv ]]; then
    note "Creating virtual environment (.venv)"
    "$py_bin" -m venv .venv
  else
    echo "Venv already exists: .venv"
  fi
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  else
    err "Could not find venv python at .venv/bin/python"
    exit 1
  fi
}

pip_run() { local pyexe="$1"; shift; "$pyexe" -m pip "$@"; }

update_pip() {
  local pyexe="$1"
  note "Updating pip/setuptools/wheel"
  pip_run "$pyexe" install -U pip setuptools wheel
}

cuda_info() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "unavailable"
    return 0
  fi
  local out
  if ! out=$(nvidia-smi 2>/dev/null); then
    echo "unavailable"; return 0
  fi
  local ver
  ver=$(echo "$out" | grep -Eo 'CUDA Version:\s*[0-9]+\.[0-9]+' | head -n1 | awk '{print $3}') || true
  if [[ -n "$ver" ]]; then
    echo "$ver"
  else
    echo "available"
  fi
}

install_pytorch() {
  local pyexe="$1"
  note "PyTorch installation"
  local cudaver
  cudaver=$(cuda_info)
  local use_gpu="no"
  if [[ "$cudaver" != "unavailable" ]]; then
    echo "nvidia-smi detected."
    if [[ "$cudaver" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "CUDA Version: $cudaver"
    else
      echo "CUDA version could not be determined; GPU install may still work."
    fi
    local choice
    choice=$(read_choice "PyTorch mode" "Install with GPU (CUDA) support or CPU only?" "GPU (CUDA)" "CPU")
    if [[ "$choice" == "GPU (CUDA)" ]]; then use_gpu="yes"; fi
  else
    echo "nvidia-smi not found or CUDA unavailable. Will install CPU PyTorch."
  fi

  if [[ "$use_gpu" == "yes" ]]; then
    local idx_url=""
    case "$cudaver" in
      12.9*) idx_url="https://download.pytorch.org/whl/cu129" ;;
      12.8*) idx_url="https://download.pytorch.org/whl/cu128" ;;
      12.6*) idx_url="https://download.pytorch.org/whl/cu126" ;;
      *)
        local sel
        sel=$(read_choice "Select CUDA distribution" "Choose the CUDA wheel to install" "12.9" "12.8" "12.6" "Cancel and use CPU")
        case "$sel" in
          12.9) idx_url="https://download.pytorch.org/whl/cu129" ;;
          12.8) idx_url="https://download.pytorch.org/whl/cu128" ;;
          12.6) idx_url="https://download.pytorch.org/whl/cu126" ;;
          *) idx_url="" ;;
        esac
        ;;
    esac
    if [[ -n "$idx_url" ]]; then
      echo "Installing GPU PyTorch from: $idx_url"
      pip_run "$pyexe" install --force-reinstall torch torchvision --index-url "$idx_url"
      return 0
    else
      warn "GPU installation cancelled; falling back to CPU."
    fi
  fi
  echo "Installing CPU PyTorch"
  pip_run "$pyexe" install --force-reinstall torch torchvision
}

install_requirements() {
  local pyexe="$1"
  note "Installing project requirements (requirements.txt)"
  if [[ ! -f requirements.txt ]]; then
    err "requirements.txt not found."
    exit 1
  fi
  if [[ "${DO_UPGRADE:-no}" == "yes" ]]; then
    pip_run "$pyexe" install --upgrade -r requirements.txt
  else
    pip_run "$pyexe" install -r requirements.txt
  fi
}

run_smoke() {
  local pyexe="$1"
  note "Running quick smoke test"
  local tmp
  tmp=$(mktemp)
  cat >"$tmp" <<'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('torch:', getattr(torch, '__version__', 'n/a'))
    print('CUDA available:', getattr(getattr(torch, 'cuda', None), 'is_available', lambda: False)())
    if getattr(getattr(torch, 'cuda', None), 'is_available', lambda: False)():
        try:
            print('CUDA devices:', torch.cuda.device_count())
        except Exception:
            pass
except Exception as e:
    print('torch import error:', e)

import cv2, yaml, numpy
print('opencv:', getattr(cv2, '__version__', 'n/a'))
print('numpy:', getattr(numpy, '__version__', 'n/a'))
try:
    import supervision, ultralytics
    print('supervision:', getattr(supervision, '__version__', 'n/a'))
    print('ultralytics:', getattr(ultralytics, '__version__', 'n/a'))
except Exception as e:
    print('optional packages error:', e)
PY
  "$pyexe" "$tmp" || true
  rm -f "$tmp"
}

main() {
  note "People Counter Setup (Linux/macOS)"

  local host_os
  host_os=$(uname -s || true)
  echo "OS: $host_os"

  local PYBIN
  PYBIN=$(ensure_python)
  echo "System Python: $($PYBIN --version 2>/dev/null)"

  local VENV_PY
  VENV_PY=$(init_venv "$PYBIN")

  update_pip "$VENV_PY"

  local up
  up=$(read_yesno "Upgrade option" "Upgrade existing Python packages?" "no")
  if [[ "$up" == "yes" ]]; then DO_UPGRADE=yes; else DO_UPGRADE=no; fi

  if [[ "$host_os" == "Darwin" ]]; then
    echo "macOS detected. CUDA is not applicable; using CPU (and MPS if available in PyTorch)."
    pip_run "$VENV_PY" install --force-reinstall torch torchvision || true
  else
    install_pytorch "$VENV_PY"
  fi

  install_requirements "$VENV_PY"
  run_smoke "$VENV_PY"

  note "Setup completed."
  echo "To run the app:"
  echo "  $VENV_PY -m people_counter"
}

main "$@"
