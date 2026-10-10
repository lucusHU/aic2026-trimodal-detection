#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=0,1,2,3
/home/newbie/miniconda3/envs/jingsai/bin/torchrun --master_port=7931 --nproc_per_node=4 \
  ../D-FINE/train.py -c configs/tri_v3.yml -t output/tri_init_v3.pth --use-amp --seed=0 2>&1 \
  | tee logs/tr_tri_v3.log
