#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
CK=output/tri_final/best_stg1.pth
CFG=configs/tri_final.yml
date "+%H:%M:%S"
/home/newbie/miniconda3/envs/jingsai/bin/python infer_tri.py -c $CFG -r $CK \
  -i dataset/data/AIC2026_PHASE_2_1000/visible \
  --ir-dir dataset/data/AIC2026_PHASE_2_1000/infrared \
  --depth-dir dataset/data/AIC2026_PHASE_2_1000/depth \
  --device cuda:7 --conf 0.001 --nms-iou 0.7 --max-det 100 --size 800 928 \
  --submit-dir submit/tri_parity_ms800_928 --zip-dir zip --zip-name tri_parity_ms800_928_nms07 2>&1 | tail -20
date "+%H:%M:%S"
ls -la zip/tri_parity_ms800_928_nms07.zip 2>/dev/null
echo "txt count = $(ls submit/tri_parity_ms800_928 2>/dev/null | wc -l)"
