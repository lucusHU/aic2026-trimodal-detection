#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=4
/home/newbie/miniconda3/envs/jingsai/bin/python eval_brightness_strata.py \
  -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --aux dataset/tri/val/infrared dataset/tri/val/depth \
  --device cuda:0 --sizes 800 --strata 4 --batch-size 4 \
  --nms-iou 0.7 --max-det 100 --conf 0.001 2>&1 | tee logs/strata.log
