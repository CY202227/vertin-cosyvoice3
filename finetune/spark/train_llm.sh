#!/usr/bin/env bash
# Single-GPU CosyVoice3 LLM SFT on DGX Spark. Do not train flow/hifigan first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

JOB_ID="${JOB_ID:-1986}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$COSYVOICE_DIR"

# Official yaml already comments "change to 1e-5 / constantlr during sft".
# Patch a local copy so we do not edit the upstream file in place.
CONF_SRC="$COSYVOICE_DIR/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml"
CONF_DST="$ROOT/conf/cosyvoice3_sft.yaml"
mkdir -p "$ROOT/conf"
python - <<PY
from pathlib import Path
src = Path(r"$CONF_SRC")
dst = Path(r"$CONF_DST")
text = src.read_text(encoding='utf-8')
text = text.replace("lr: 1e-4", "lr: 1e-5")
text = text.replace(
    "scheduler: warmuplr",
    "scheduler: constantlr # sft",
)
# 732 train utts: shuffle_size=1000 delays the first batch until the whole
# epoch is buffered. Keep buffers smaller than the dataset.
text = text.replace("shuffle_size: 1000", "shuffle_size: 200")
text = text.replace(
    "sort_size: 500  # sort_size should be less than shuffle_size",
    "sort_size: 100  # sort_size should be less than shuffle_size",
)
text = text.replace("max_epoch: 200", "max_epoch: 12")
text = text.replace(
    "use_spk_embedding: False # change to True during sft",
    "use_spk_embedding: True # change to True during sft",
)
dst.write_text(text, encoding='utf-8')
print("wrote", dst)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$EXP_DIR" "$TB_DIR"

torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id="$JOB_ID" --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  "$COSYVOICE_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config "$CONF_DST" \
  --train_data "$KALDI_DIR/train/parquet/data.list" \
  --cv_data "$KALDI_DIR/dev/parquet/data.list" \
  --qwen_pretrain_path "$MODEL_DIR/CosyVoice-BlankEN" \
  --onnx_path "$MODEL_DIR" \
  --model llm \
  --checkpoint "$MODEL_DIR/llm.pt" \
  --model_dir "$EXP_DIR" \
  --tensorboard_dir "$TB_DIR" \
  --ddp.dist_backend nccl \
  --num_workers 0 \
  --prefetch 4 \
  --use_amp

echo "LLM checkpoints are in $EXP_DIR"
echo "Listen to mid-run checkpoints every 1-2 epochs before picking a final llm.pt"
