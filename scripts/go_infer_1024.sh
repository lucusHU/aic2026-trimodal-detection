#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=4
/home/newbie/miniconda3/envs/jingsai/bin/python infer_best800.py \
  -c configs/baseline_800.yml -r output/seed1_800/best_stg2.pth \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --device cuda:0 --batch-size 8 --size 800 1024 \
  --nms-iou 0.7 --max-det 100 --conf 0.001 \
  --zip-name seed1_ms800_1024_nms07 2>&1 | tee logs/infer_1024.log
