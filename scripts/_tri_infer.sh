#!/bin/bash
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1

CK=output/tri_final/best_stg1.pth
CFG=configs/tri_final.yml

echo "########## STEP 1: tri-modal val under protocol B (800+928, nms 0.7, md 100) ##########"
date "+%H:%M:%S"
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri.py -c $CFG -r $CK --size 800 928 --nms-iou 0.7 --max-det 100 --device cuda:7 2>&1 | tail -20

echo
echo "########## STEP 2: tri-modal test-set inference -> submission zip ##########"
date "+%H:%M:%S"
/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py -c $CFG -r $CK \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --ir-dir dataset/data/AIC2026_PHASE_2_1000/infrared \
  --depth-dir dataset/data/AIC2026_PHASE_2_1000/depth \
  --device cuda:7 --conf 0.001 --nms-iou 0.7 --max-det 100 \
  --size 800 928 \
  --submit-dir submit/tri_final_ms800_928_nms07 \
  --zip-dir zip --zip-name tri_final_ms800_928_nms07 2>&1 | tail -25

echo
echo "########## STEP 3: result ##########"
date "+%H:%M:%S"
ls -la zip/tri_final_ms800_928_nms07.zip 2>/dev/null
echo "txt count = $(ls submit/tri_final_ms800_928_nms07 2>/dev/null | wc -l)"
