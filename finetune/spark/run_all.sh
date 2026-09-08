#!/usr/bin/env bash
# Full Spark pipeline: Fun-ASR-Nano English ASR -> Kaldi -> parquet -> LLM SFT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cd "$ROOT"
ensure_venv
ensure_cuda_torch

echo "Installing FunASR / ModelScope into $VENV_DIR"
pip_mirror -U funasr modelscope soundfile numpy tqdm silero-vad imageio-ffmpeg
# FunASR's deps must not replace the GB10 CUDA wheel with a CPU torch.
ensure_cuda_torch

if [[ ! -f "$ROOT/data/raw_wav/source_map.jsonl" ]]; then
  echo "Missing data/raw_wav/source_map.jsonl. Unzip the full Spark pack first."
  exit 1
fi

echo "=== 1/4 Fun-ASR-Nano English ASR + cut ==="
python "$ROOT/scripts/02_asr_cut.py" \
  --backend funasr-nano \
  --language en

echo "=== 2/4 attach nonspeech tags ==="
python "$ROOT/scripts/03_attach_nonspeech.py"

echo "=== 3/4 export Kaldi lists ==="
python "$ROOT/scripts/04_export_kaldi.py"

echo "=== 4/4 CosyVoice3 parquet + LLM SFT ==="
bash "$SCRIPT_DIR/setup_cosyvoice3.sh"
bash "$SCRIPT_DIR/prepare_parquet.sh"
bash "$SCRIPT_DIR/train_llm.sh"

echo "Done. Check exp/ for llm checkpoints and data/processed/manifests/asr_cuts.jsonl"
