#!/usr/bin/env bash

set -euo pipefail

# =========================
# 固定工作目录
# =========================

cd /home/newbie/bigdatan/jingsai/bisai


# =========================
# 只改这里
# =========================

GPU=0

CONFIG="configs/x_baseline_800_800.yml"

WEIGHT="weights/dfine_x_obj365.pth"

OUTPUT_DIR="output/x_baseline_800_800"

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

#!/usr/bin/env bash

set -euo pipefail

cd /home/newbie/bigdatan/jingsai/bisai

GPUS="0,1,2,3,4,5"
NUM_GPUS=6
PORT=29501

CONFIG="configs/x_baseline_800_800.yml"
WEIGHT="weights/dfine_x_obj365.pth"
OUTPUT_DIR="output/x_baseline_800_800"
SEED=0

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train.log"

echo "============================================================"
echo "D-FINE-X Multi GPU Training"
echo "GPUs       : $GPUS"
echo "Num GPUs   : $NUM_GPUS"
echo "Total batch: 16"
echo "Config     : $CONFIG"
echo "Weight     : $WEIGHT"
echo "Output     : $OUTPUT_DIR"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

torchrun \
  --master_port="$PORT" \
  --nproc_per_node="$NUM_GPUS" \
  ../D-FINE/train.py \
  -c "$CONFIG" \
  -t "$WEIGHT" \
  --use-amp \
  --seed="$SEED" \
  2>&1 | tee "$LOG"