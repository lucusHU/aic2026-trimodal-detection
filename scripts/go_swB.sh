#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=5
/home/newbie/miniconda3/envs/jingsai/bin/python sweep_post.py -c configs/tri_final.yml -r output/tri_final/best_stg1.pth --device cuda:0 --size 800 928 --nms-iou 0.7 --max-det 100 --batch-size 8 --cache cache_B.pkl --out res_B.json --rounds 1 2>&1 | tee logs/sweep_B.log
