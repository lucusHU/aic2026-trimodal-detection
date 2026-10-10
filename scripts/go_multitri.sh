#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri_multi.py \
  -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --aux dataset/tri/val/infrared dataset/tri/val/depth \
  --device cuda:0 --batch-size 4 --nms-iou 0.7 --max-det 100 --conf 0.001 \
  --sizes 800 992 1024 1088 1152 \
  --combos 800,1024,1088 800,1088 800,992,1024 800,992,1024,1088 800,1024,1088,1152 \
  --zero-ch 4 2>&1 | tee logs/multitri.log
