#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_split.py \
  -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth \
  --device cuda:0 --sizes 800 928 1024 1088 1152 \
  --base-combo 800,928 \
  --combos 800,1024,1088 800,928,1024,1088 800,1024,1088,1152 \
  --nms-iou 0.7 --max-det 100 --conf 0.001 --folds 10 2>&1 | tee logs/split.log
