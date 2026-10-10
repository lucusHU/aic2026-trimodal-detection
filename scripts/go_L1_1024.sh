#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=0,1,2,3
/home/newbie/miniconda3/envs/jingsai/bin/torchrun --master_port=9411 --nproc_per_node=4 \
  ../D-FINE/train.py -c configs/seed1_1024.yml -t output/seed1_800/best_stg2.pth \
  --use-amp --seed=0 2>&1 | tee logs/tr_1024_v2.log
