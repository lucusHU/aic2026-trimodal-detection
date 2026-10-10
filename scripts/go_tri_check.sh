#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py \
  -c configs/tri_v3.yml -r output/tri_init_v3.pth \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --aux dataset/data/AIC2026_PHASE_2_1000/infrared dataset/data/AIC2026_PHASE_2_1000/depth \
  --device cuda:0 --size 800 928 --nms-iou 0.7 --max-det 100 --batch-size 8 \
  --submit-dir submit --zip-name tri_v3_initcheck --no-zip 2>&1 | tee logs/infer_tri_initcheck.log
