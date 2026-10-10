#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=5
/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py \
  -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --aux dataset/data/AIC2026_PHASE_2_1000/infrared dataset/data/AIC2026_PHASE_2_1000/depth \
  --device cuda:0 --conf 0.001 --nms-iou 0.7 --max-det 100 \
  --size 800 1024 1088 --batch-size 4 \
  --submit-dir submit/tri3_scales --zip-dir zip --zip-name tri3_scales_800_1024_1088 2>&1 | tee logs/tri3_pack.log
