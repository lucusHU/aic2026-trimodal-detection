#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=4
echo "### tri_final3 @ 800+1024 (C ????????)"
/home/newbie/miniconda3/envs/jingsai/bin/python eval_ablate.py -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --device cuda:0 --size 800 1024 --nms-iou 0.7 --max-det 100 --batch-size 4 2>&1 | grep -E "info|RESULT"
echo "### tri_final3 @ 800+1024+1088 (????)"
/home/newbie/miniconda3/envs/jingsai/bin/python eval_ablate.py -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --device cuda:0 --size 800 1024 1088 --nms-iou 0.7 --max-det 100 --batch-size 4 2>&1 | grep -E "info|RESULT"
