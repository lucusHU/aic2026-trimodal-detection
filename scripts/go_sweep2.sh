#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python ms_tta.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth \
  --device cuda:0 --batch 8 --conf 0.001 --nms 0.65 0.70 0.75 --maxdet 100 \
  --combos 800,1024 704,1024 832,1024 896,1024 960,1024 992,1024 800,992,1024 2>&1 | tee logs/ms1024b.log
