#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python ms_tta.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth \
  --device cuda:0 --batch 8 --conf 0.001 --nms 0.65 0.70 0.75 --maxdet 100 \
  --combos 800,1024,1088 800,1120 800,1024,1120 800,1088,1120 2>&1 | tee logs/ms_1120.log
