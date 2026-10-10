#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_ablate.py \
  -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --device cuda:0 --size 800 1024 1088 --nms-iou 0.7 --max-det 100 --batch-size 8 \
  2>&1 | tee logs/eval_tri3_scales.log
