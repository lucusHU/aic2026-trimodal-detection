#!/usr/bin/env bash

set -euo pipefail

# =========================
# 固定工作目录
# =========================

cd /home/newbie/bigdatan/jingsai/bisai


# =========================
# 只改这里
# =========================

GPU=4

CONFIG="configs/lr_1e4.yml"

WEIGHT="weights/dfine_l_obj365.pth"

OUTPUT_DIR="output/lr_1e4"

SEED=0


# =========================
# 建立输出目录和日志
# =========================

mkdir -p "$OUTPUT_DIR"

LOG="$OUTPUT_DIR/train.log"


# =========================
# 打印本次实验信息
# =========================

echo "============================================================"
echo "D-FINE Single GPU Training"
echo "GPU        : $GPU"
echo "Config     : $CONFIG"
echo "Weight     : $WEIGHT"
echo "Output     : $OUTPUT_DIR"
echo "Log        : $LOG"
echo "============================================================"


# =========================
# 单卡训练
# =========================

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

python -u ../D-FINE/train.py \
  -c "$CONFIG" \
  -t "$WEIGHT" \
  --use-amp \
  --seed="$SEED" \
  2>&1 | tee "$LOG"