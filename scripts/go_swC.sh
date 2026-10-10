#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
CUDA_VISIBLE_DEVICES="" /home/newbie/miniconda3/envs/jingsai/bin/python sweep_post.py -c configs/seed1_800.yml -r output/seed1_800/best_stg2.pth --aux --device cpu --size 800 928 --nms-iou 0.7 --max-det 100 --soft 0.3 0.5 --rounds 0 --cache cache_A.pkl --out res_soft.json 2>&1 | tee logs/sweep_soft.log
