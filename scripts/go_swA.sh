#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python sweep_post.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth --aux --device cuda:0 --size 800 928 --nms-iou 0.7 --max-det 100 --batch-size 8 --cache cache_A.pkl --out res_A.json --rounds 1 2>&1 | tee logs/sweep_A.log
