#!/usr/bin/env bash
# =====================================================================
# bisai/run_infer.sh —— tmux 后台 + 空闲 GPU 自动排队等待 + 自动启动
# 用法：
#   bash run_infer.sh                    # 默认：对 test 图推理，生成 predictions.json（+可视化）
#   bash run_infer.sh infer [imgdir]     # 指定图片目录推理
#   bash run_infer.sh val                # 在 val 上评估 COCO mAP（--test-only）
# 特性：
#   - 没有空闲 GPU 时不退出，每 5 分钟轮询一次，等到有空闲卡自动开跑
#   - 单卡推理，避免多卡同步开销，降低排队门槛
#   - 强制使用 jingsai conda 环境
# 前提：在 bisai/ 目录下执行；D-FINE 仓库与 bisai 同级且零修改。
#   infer_bisai.py 需与本脚本同目录（bisai/）。
# =====================================================================
set -e
export PATH=/home/newbie/miniconda3/envs/jingsai/bin:$PATH
# 强制使用 jingsai 环境的 python，避免 conda / PATH 污染导致加载错环境
PYTHON=/home/newbie/miniconda3/envs/jingsai/bin/python

REPO=../D-FINE
CFG=configs/dfine_hgnetv2_l_bisai.yml
DEFAULT_WEIGHT=output/dfine_l_50ep/best_stg2.pth
DEFAULT_IMGDIR=dataset/data/val
OUT=submit
SESSION=infer
CONF=0.3

# ---------- 内部模式：真正在 tmux 里执行的任务 ----------
if [ "$1" = "__inner" ]; then
  MODE=$2
  IMGDIR=${3:-$DEFAULT_IMGDIR}
  WEIGHT=${4:-$DEFAULT_WEIGHT}
  cd "$(dirname "$0")"

  while true; do
    # 兼容 nvidia-smi 输出带千分位逗号（如 5,969 MiB -> 5969）
    mapfile -t IDLE < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | awk -F',' '{
        gsub(/[^0-9]/, "", $2); gsub(/[^0-9]/, "", $3);
        if ($2+0 < 1000 && $3+0 < 10) print $1
      }')
    N=${#IDLE[@]}
    if [ "$N" -ge 1 ]; then
      echo "$(date '+%F %T') 检测到 $N 张空闲 GPU: ${IDLE[*]}（模式: $MODE，权重: $WEIGHT），逐个尝试"
      break
    fi
    echo "$(date '+%F %T') 当前无空闲 GPU，5 分钟后重试..."
    sleep 300
  done

  # 逐张空闲卡尝试：某张卡挂了（OOM / 崩溃 / 被抢占）自动换下一张
  SUCCESS=0
  for GPU in "${IDLE[@]}"; do
    echo "$(date '+%F %T') ===== 尝试 GPU $GPU ====="

    case "$MODE" in
      val)
        # 评估模式：在 val 上算 COCO mAP
        CMD="CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
$REPO/train.py -c $CFG --test-only -r $WEIGHT"
        ;;
      infer|*)
        # 推理模式：对图片目录批量推理，输出 predictions.json + 可视化
        if [ ! -f "infer_bisai.py" ]; then
          echo "$(date '+%F %T') 错误：$(pwd)/infer_bisai.py 不存在，跳过 GPU $GPU"
          EXIT_CODE=1
          continue
        fi
        mkdir -p "$OUT"
        CMD="CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u \
infer_bisai.py -c $CFG -r $WEIGHT \
-i $IMGDIR -o $OUT --repo $REPO --conf $CONF --device cuda:0 --save-vis --amp"
        ;;
    esac

    echo "命令: $CMD"
    set +e
    eval $CMD 2>&1 | tee -a infer.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
      echo "$(date '+%F %T') GPU $GPU 执行成功"
      SUCCESS=1
      break
    else
      echo "$(date '+%F %T') GPU $GPU 执行失败（退出码 $EXIT_CODE），自动换下一张空闲卡"
    fi
  done

  if [ "$SUCCESS" -ne 1 ]; then
    echo "$(date '+%F %T') 所有空闲 GPU 均失败，退出"
    exit 1
  fi
  exit 0
fi

# ---------- 外部模式：创建 tmux 会话后立即返回 ----------
MODE=${1:-infer}
IMGDIR=${2:-$DEFAULT_IMGDIR}
WEIGHT=${3:-$DEFAULT_WEIGHT}
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash $(pwd)/run_infer.sh __inner $MODE $IMGDIR $WEIGHT"
echo "已在 tmux 会话 [$SESSION] 排队/启动（模式: $MODE，图片: $IMGDIR，权重: $WEIGHT）"
echo "查看: tmux attach -t $SESSION    退出会话(不杀推理): Ctrl+b 再按 d"
