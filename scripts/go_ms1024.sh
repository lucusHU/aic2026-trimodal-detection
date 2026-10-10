#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python ms_tta.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth \
  --device cuda:0 --batch 8 --conf 0.001 --nms 0.7 0.8 --maxdet 100 \
  --combos 928 800,928 1024 800,1024 928,1024 800,928,1024 896,928,1024 2>&1 | tee logs/ms1024.log
