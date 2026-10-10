#!/bin/bash
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1
date "+%H:%M:%S"
echo "=== disk before ==="
df -h /home/adlab7/ssd | tail -1

/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py \
  -c configs/tri_final.yml \
  -r output/tri_final/best_stg1.pth \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --ir-dir dataset/data/AIC2026_PHASE_2_1000/infrared \
  --depth-dir dataset/data/AIC2026_PHASE_2_1000/depth \
  --device cuda:4 --conf 0.001 --nms-iou 0.7 --max-det 100 \
  --size 800 1024 1088 --batch-size 4 \
  --submit-dir submit/tri_scales_800_1024_1088 \
  --zip-dir zip --zip-name tri_scales_800_1024_1088

date "+%H:%M:%S"
echo "=== artifacts ==="
ls -la zip/tri_scales_800_1024_1088.zip 2>/dev/null
df -h /home/adlab7/ssd | tail -1
