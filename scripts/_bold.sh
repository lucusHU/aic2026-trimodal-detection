#!/bin/bash
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1
date "+%H:%M:%S"

echo "=== A) launch bold run on GPU 0-3 (stop_epoch=epochs -> no crash transition) ==="
ps -eo pid,args | grep "[t]rain.py" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
cat > go_tri_bold.sh <<'EOF'
#!/bin/bash
cd /home/adlab7/ssd/bigdatan/jingsai/bisai
export CUDA_VISIBLE_DEVICES=0,1,2,3
/home/newbie/miniconda3/envs/jingsai/bin/torchrun --master_port=9301 --nproc_per_node=4 ../D-FINE/train.py -c configs/tri_bold.yml -t output/tri_init.pth --use-amp --seed=1 2>&1 | tee logs/tr_tri_bold.log
EOF
chmod +x go_tri_bold.sh
tmux kill-session -t tribold 2>/dev/null
tmux new-session -d -s tribold 'bash go_tri_bold.sh'
echo "bold run launched"

echo
echo "=== B) in parallel, eval tri_final3/best_stg1 at the validated best scales (GPU 4) ==="
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri.py -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --size 800 1024 1088 --nms-iou 0.7 --max-det 100 --device cuda:4 2>&1 | tail -6
echo "--- reference: tri_final @same scales = 0.5763 ; seed1 = 0.5752 ; base 800+928 = 0.5670 ---"
date "+%H:%M:%S"
echo ===DONE===
