#!/bin/bash
# usage: _infer_any.sh <config> <ckpt> <zipname> <gpu>
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1
CFG=$1; CK=$2; NAME=$3; GPU=${4:-cuda:4}
date "+%H:%M:%S"
echo "config=$CFG ckpt=$CK name=$NAME gpu=$GPU"
df -h /home/adlab7/ssd | tail -1
ls -la "$CK" || exit 1

/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py \
  -c "$CFG" -r "$CK" \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --aux dataset/data/AIC2026_PHASE_2_1000/infrared dataset/data/AIC2026_PHASE_2_1000/depth \
  --device "$GPU" --conf 0.001 --nms-iou 0.7 --max-det 100 \
  --size 800 1024 1088 --batch-size 4 \
  --submit-dir "submit/$NAME" \
  --zip-dir zip --zip-name "$NAME"
date "+%H:%M:%S"
ls -la "zip/$NAME.zip" 2>/dev/null
df -h /home/adlab7/ssd | tail -1
echo ===DONE===
