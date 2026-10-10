#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
CUDA_VISIBLE_DEVICES="" /home/newbie/miniconda3/envs/jingsai/bin/python vote_ms.py --cache cache_B.pkl --out vote_B.json 2>&1 | tee logs/vote_B.log
