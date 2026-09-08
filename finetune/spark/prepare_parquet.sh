#!/usr/bin/env bash
# Build CosyVoice3 parquet lists from Kaldi files on this Spark box.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$COSYVOICE_DIR"

python "$SCRIPT_DIR/remap_wav_scp.py" \
  --rel-scp "$KALDI_DIR/train/wav.scp.rel" \
  --dest-scp "$KALDI_DIR/train/wav.scp" \
  --clips-root "$CLIPS_DIR"
python "$SCRIPT_DIR/remap_wav_scp.py" \
  --rel-scp "$KALDI_DIR/dev/wav.scp.rel" \
  --dest-scp "$KALDI_DIR/dev/wav.scp" \
  --clips-root "$CLIPS_DIR"

for split in train dev; do
  echo "Extract embeddings for $split"
  python tools/extract_embedding.py \
    --dir "$KALDI_DIR/$split" \
    --onnx_path "$MODEL_DIR/campplus.onnx"
  echo "Extract speech tokens for $split"
  python tools/extract_speech_token.py \
    --dir "$KALDI_DIR/$split" \
    --onnx_path "$MODEL_DIR/speech_tokenizer_v3.onnx"
  python "$SCRIPT_DIR/fix_embeddings.py" \
    "$KALDI_DIR/$split/utt2embedding.pt" \
    "$KALDI_DIR/$split/spk2embedding.pt"
  mkdir -p "$KALDI_DIR/$split/parquet"
  python tools/make_parquet_list.py \
    --num_utts_per_parquet 1000 \
    --num_processes 4 \
    --src_dir "$KALDI_DIR/$split" \
    --des_dir "$KALDI_DIR/$split/parquet"
done

echo "Parquet lists:"
echo "  $KALDI_DIR/train/parquet/data.list"
echo "  $KALDI_DIR/dev/parquet/data.list"
