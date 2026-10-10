#!/bin/bash
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1
date "+%H:%M:%S"

echo "=== build eval_tri_cx.py: codex's WBF merge (maximal score retained) ==="
cp -f eval_tri.py eval_tri_cx.py

/home/newbie/miniconda3/envs/jingsai/bin/python - <<'PY'
p = 'eval_tri_cx.py'
s = open(p).read()

WBF = '''

def _iou1(a, b):
    ix1, iy1 = torch.max(a[0], b[0]), torch.max(a[1], b[1])
    ix2, iy2 = torch.min(a[2], b[2]), torch.min(a[3], b[3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ua = ((a[2]-a[0]).clamp(min=0)*(a[3]-a[1]).clamp(min=0)
          + (b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0) - inter)
    return inter / ua.clamp(min=1e-6)


@torch.no_grad()
def wbf_merge(per_scale, num_classes=12, iou_thr=0.65, skip_thr=1e-3, max_det=100):
    dev = per_scale[0][1].device
    ldt = per_scale[0][0].dtype
    out_l, out_b, out_s = [], [], []
    for c in range(num_classes):
        bs, ss = [], []
        for lb, bx, sc in per_scale:
            m = (lb == c) & (sc >= skip_thr)
            if m.any():
                bs.append(bx[m].float().to(dev)); ss.append(sc[m].float().to(dev))
        if not bs:
            continue
        b, s = torch.cat(bs, 0), torch.cat(ss, 0)
        o = torch.argsort(s, descending=True); b, s = b[o], s[o]
        csum = b[0] * s[0]; cw = s[0].clone(); cb = b[0].clone(); cscore = s[0].clone()
        fb, fs = [], []
        for k in range(1, b.shape[0]):
            bk, sk = b[k], s[k]
            if _iou1(cb, bk) > iou_thr:
                csum = csum + bk * sk; cw = cw + sk; cb = csum / cw
                cscore = torch.maximum(cscore, sk)
            else:
                fb.append(cb); fs.append(cscore)
                csum, cw, cb, cscore = bk * sk, sk.clone(), bk.clone(), sk.clone()
        fb.append(cb); fs.append(cscore)
        out_b.append(torch.stack(fb)); out_s.append(torch.stack(fs))
        out_l.append(torch.full((len(fb),), c, dtype=ldt, device=dev))
    if not out_b:
        return (torch.zeros(0, dtype=ldt, device=dev), torch.zeros(0, 4, device=dev),
                torch.zeros(0, device=dev))
    lb, bx, sc = torch.cat(out_l), torch.cat(out_b), torch.cat(out_s)
    keep = torch.argsort(sc, descending=True)[:max_det]
    return lb[keep], bx[keep], sc[keep]

'''

anchor_imp = "from torchvision.ops import batched_nms"
assert anchor_imp in s, 'import anchor not found'
s = s.replace(anchor_imp, anchor_imp + WBF, 1)

old = "        k = batched_nms(bxb.float(), scb.float(), lbb, iou)[:maxdet]"
new = ("        lbb, bxb, scb = wbf_merge(list(zip(lbs, bxs, scs)), num_classes=12,\n"
       "                                  iou_thr=0.65, skip_thr=conf, max_det=maxdet)\n"
       "        k = torch.arange(lbb.shape[0])")
assert old in s, 'batched_nms anchor not found'
s = s.replace(old, new, 1)

# the old line `lbb = torch.cat(...)` becomes dead but harmless; drop it to avoid confusion
s = s.replace("        lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)\n", "", 1)

open(p, 'w').write(s)
print('PATCHED eval_tri_cx.py')
PY
/home/newbie/miniconda3/envs/jingsai/bin/python -m py_compile eval_tri_cx.py && echo "py_compile OK"

echo
echo "=== codex WBF merge on val @800+1024+1088 (baseline NMS = 0.5778) ==="
/home/newbie/miniconda3/envs/jingsai/bin/python eval_tri_cx.py -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
  --size 800 1024 1088 --nms-iou 0.65 --max-det 100 --device cuda:4 2>&1 | tail -8
date "+%H:%M:%S"
echo ===DONE===
