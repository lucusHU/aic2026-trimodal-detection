#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri.py -c configs/tri_final.yml -r output/tri_final/best_stg1.pth --device cuda:0 --size 800 928 --nms-iou 0.6 0.7 0.8 --max-det 100 --batch-size 4 --quiet 2>&1 | tee logs/eval_tri_best.log
