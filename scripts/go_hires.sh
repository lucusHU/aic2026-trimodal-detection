#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth --aux --device cuda:0 --size 1024 1280 --nms-iou 0.7 --max-det 100 --batch-size 2 --quiet 2>&1 | tee logs/eval_hires.log
