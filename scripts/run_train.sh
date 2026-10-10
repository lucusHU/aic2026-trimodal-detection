#!/usr/bin/env bash
# =====================================================================
# bisai/run_train.sh —— tmux 后台 + 空闲 GPU 自动排队等待 + 自动启动
# 用法：
#   bash run_train.sh              # 训练（用预训练权重 -t）
#   bash run_train.sh resume       # 从 last.pth 断点续训
#   bash run_train.sh test [pth]   # 只测试（默认 best_stg2.pth）
# 特性（v2 修复 OOM）：
#   - 没有空闲 GPU 时不退出，每 5 分钟轮询一次，等到有空闲卡自动开跑
#   - 每卡固定 batch（PER_GPU），total_batch_size = PER_GPU * 卡数，启动前动态写入 config
#   - 不再限制卡数必须整除 24，抓到几卡就几卡（保证每卡显存安全）
#   - 强制使用 jingsai conda 环境（torchrun/torch 均来自该环境）
# 前提：在 bisai/ 目录下执行；D-FINE 仓库与 bisai 同级且零修改。
# =====================================================================
set -e
export PATH=/home/newbie/miniconda3/envs/jingsai/bin:$PATH

REPO=../D-FINE
CFG=configs/dfine_hgnetv2_l_bisai.yml
WEIGHT=weights/dfine_l_obj365.pth
OUT=output/dfine_l_test
PORT=7777
SESSION=test

# 每卡 batch（16GB 卡、640x640、D-FINE-L + denoising + aux loss，保守安全值）
# 想更高吞吐可调到 6/8，但 12 会 OOM，勿超 8
PER_GPU=4

# ---------- 内部模式：真正在 tmux 里执行的任务 ----------
if [ "$1" = "__inner" ]; then
  MODE=$2
  cd "$(dirname "$0")"
  mkdir -p output

  while true; do
    mapfile -t IDLE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 && $3 < 10 {print $1}')
    N=${#IDLE[@]}
    # 从大到小挑一个 <= 空闲卡数 的卡数（不再要求整除 24）
    PICK=0
    for c in 1; do
      if [ "$N" -ge "$c" ]; then PICK=$c; break; fi
    done
    if [ "$PICK" -ge 1 ]; then
      GPUS=$(IFS=,; echo "${IDLE[*]:0:$PICK}")
      TOTAL=$(( PER_GPU * PICK ))
      echo "$(date '+%F %T') 检测到 $PICK 张空闲 GPU: $GPUS，total_batch_size=$TOTAL (每卡 $PER_GPU)，启动训练"
      break
    fi
    echo "$(date '+%F %T') 当前空闲 GPU $N 张，5 分钟后重试..."
    sleep 300
  done

  # 启动前把 config 的 total_batch_size 动态改写为 PER_GPU*卡数，保证每卡显存安全
  python - "$CFG" "$TOTAL" <<'PYEOF'
import sys, re
cfg, total = sys.argv[1], sys.argv[2]
p = cfg
s = open(p, encoding="utf-8").read()
s2 = re.sub(r"total_batch_size: *\d+", f"total_batch_size: {total}", s)
if s2 != s:
    open(p, "w", encoding="utf-8").write(s2)
    print(f"config total_batch_size -> {total}")
else:
    print(f"total_batch_size already {total}")
PYEOF

  CMD="CUDA_VISIBLE_DEVICES=$GPUS python \
$REPO/train.py -c $CFG --use-amp --seed=0"

  case "$MODE" in
    train)  EXTRA="-t $WEIGHT" ;;
    resume) EXTRA="-r $OUT/last.pth" ;;
    test)   P=${3:-$OUT/best_stg2.pth}; EXTRA="--test-only -r $P" ;;
    *)      echo "未知模式: $MODE"; exit 1 ;;
  esac

  echo "命令: $CMD $EXTRA"
  eval $CMD $EXTRA 2>&1 | tee -a train.log
  exit 0
fi

# ---------- 外部模式：创建 tmux 会话后立即返回 ----------
MODE=${1:-train}
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash $(pwd)/run_train.sh __inner $MODE _"
echo "已在 tmux 会话 [$SESSION] 排队/启动（模式: $MODE, 每卡 batch=$PER_GPU）"
echo "查看: tmux attach -t $SESSION    退出会话(不杀训练): Ctrl+b 再按 d"
