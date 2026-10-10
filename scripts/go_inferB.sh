#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=7
/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py -c configs/tri_final.yml -r output/tri_final/best_stg1.pth -i dataset/data/AIC2026_PHASE_2_1000/visible --ir-dir dataset/data/AIC2026_PHASE_2_1000/infrared --depth-dir dataset/data/AIC2026_PHASE_2_1000/depth --device cuda:0 --size 800 928 --nms-iou 0.7 --max-det 100 --conf 0.001 --batch-size 8 --zip-name B_tri_ms800_928_nms70 2>&1 | tee logs/infer_B.log
