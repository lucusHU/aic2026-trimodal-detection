#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri.py -c configs/tri_final.yml -r output/tri_final/last.pth --device cuda:0 --size 800 --nms-iou 0.7 --max-det 100 --batch-size 4 --quiet 2>&1 | tee logs/eval_trifinal_e0.log
