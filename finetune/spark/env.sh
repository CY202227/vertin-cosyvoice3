#!/usr/bin/env bash
# Shared paths and helpers for this DGX Spark (GB10 / sm_121 / aarch64 / Python 3.12).
# Source this file from the other spark/*.sh scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FINETUNE_ROOT/.." && pwd)"
export SPARK_WORKSPACE="${SPARK_WORKSPACE:-$FINETUNE_ROOT}"
ROOT="$SPARK_WORKSPACE"

if command -v python3 >/dev/null 2>&1; then
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

export VENV_DIR="${VENV_DIR:-$ROOT/.venv-cosyvoice}"
export COSYVOICE_DIR="${COSYVOICE_DIR:-$REPO_ROOT/runtime/CosyVoice}"

_LOCAL_COSYVOICE3="/home/odb/model/Fun-CosyVoice3-0.5B-2512"
if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -f "$_LOCAL_COSYVOICE3/llm.pt" ]]; then
    MODEL_DIR="$_LOCAL_COSYVOICE3"
  else
    MODEL_DIR="$REPO_ROOT/runtime/pretrained_models/Fun-CosyVoice3-0.5B"
  fi
fi
export MODEL_DIR

# Proven on this box via ComfyUI: torch 2.14.0+cu130, torchaudio 2.11.0+cu130.
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
export TORCH_VERSION="${TORCH_VERSION:-2.14.0+cu130}"
export TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+cu130}"
export PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

export DATA_ROOT="${DATA_ROOT:-$ROOT/data/processed}"
export KALDI_DIR="${KALDI_DIR:-$DATA_ROOT/kaldi}"
export CLIPS_DIR="${CLIPS_DIR:-$DATA_ROOT/clips/vertin}"
export EXP_DIR="${EXP_DIR:-$ROOT/exp/cosyvoice3/llm/torch_ddp}"
export TB_DIR="${TB_DIR:-$ROOT/tensorboard/cosyvoice3/llm/torch_ddp}"

# flash-attn is unsupported on sm_121; keep PyTorch SDPA.
export TORCH_BACKENDS_CUDNN_ALLOW_TF32="${TORCH_BACKENDS_CUDNN_ALLOW_TF32:-1}"
export PYTHONPATH="${COSYVOICE_DIR}:${COSYVOICE_DIR}/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"


ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  # setuptools 82+ dropped pkg_resources; CosyVoice/pyworld still import it.
  python -m pip install -U pip 'setuptools<81' wheel >/dev/null
}


_torch_has_cuda() {
  python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    raise SystemExit(2)
import torch

print(
    "torch",
    torch.__version__,
    "cuda",
    torch.cuda.is_available(),
    getattr(torch.version, "cuda", None),
)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    raise SystemExit(0)
raise SystemExit(3)
PY
}


ensure_cuda_torch() {
  local rc=0
  _torch_has_cuda || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  echo "Installing CUDA 13 PyTorch for GB10 from $TORCH_INDEX_URL"
  python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
  _torch_has_cuda
}


pip_mirror() {
  python -m pip install "$@" \
    -i "$PIP_MIRROR" \
    --trusted-host=mirrors.aliyun.com \
    || python -m pip install "$@"
}
