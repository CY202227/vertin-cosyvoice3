#!/usr/bin/env bash
# Install CosyVoice3 on this DGX Spark (ARM64 Grace + GB10 / sm_121).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

mkdir -p "$ROOT"
cd "$ROOT"
ensure_venv
ensure_cuda_torch

if [[ ! -d "$COSYVOICE_DIR/.git" ]]; then
  echo "Cloning CosyVoice into $COSYVOICE_DIR"
  if ! git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"; then
    echo "GitHub clone failed, trying ModelScope mirror"
    git clone --recursive https://www.modelscope.cn/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
  fi
fi
cd "$COSYVOICE_DIR"
git submodule update --init --recursive

# Upstream requirements.txt pins torch==2.3.1 / cu121 / tensorrt. Use the Spark list.
echo "Installing CosyVoice Spark deps (not upstream torch/flash-attn/ttsfrd)"
pip_mirror -r "$SCRIPT_DIR/requirements-cosyvoice-spark.txt"
# Optional: pyworld often needs a local build on aarch64.
pip_mirror pyworld==0.3.4 || echo "pyworld skipped (not required for LLM SFT)"
python -c "import whisper" || pip_mirror openai-whisper
# Only needed so train.py can parse --deepspeed flags. torch_ddp does not use the engine.
# Skip CUDA ops on GB10 / sm_121; they often fail to compile.
DS_BUILD_OPS=0 pip_mirror deepspeed || echo "deepspeed skipped; torch_ddp does not need it"
ensure_cuda_torch

python "$SCRIPT_DIR/patch_cosyvoice_spark.py"
python - <<'PY'
import importlib

for name in ('onnxruntime', 'whisper', 'hyperpyyaml', 'librosa', 'onnx', 'pyarrow'):
    importlib.import_module(name)
    print('import ok', name)
PY

if [[ -f "$MODEL_DIR/llm.pt" && -f "$MODEL_DIR/campplus.onnx" && -d "$MODEL_DIR/CosyVoice-BlankEN" ]]; then
  echo "Using existing CosyVoice3 weights at $MODEL_DIR"
else
  mkdir -p "$(dirname "$MODEL_DIR")"
  python - <<PY
from modelscope import snapshot_download
snapshot_download(
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir="${MODEL_DIR}",
)
PY
fi

echo "CosyVoice ready at $COSYVOICE_DIR"
echo "Model ready at $MODEL_DIR"
echo "Next: run prepare_parquet.sh then train_llm.sh"
