#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""框尺寸系统性偏差诊断 + 缩放系数扫描。
1) 统计"匹配上的预测框"与对应 GT 的 w/h 比值，看模型是否系统性把框预测小了。
2) 对预测框做 (w,h) 等比缩放（中心不变），扫描 scale 对 val mAP 的影响。
"""
import argparse, contextlib, io, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.ops import batched_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_state(path, use_ema=True):
    ck = torch.load(path, map_location='cpu')
    state = ck['ema']['module'] if (use_ema and 'ema' in ck) else (ck.get('model', ck))
    state = dict(state)
    state.pop('decoder.anchors', None)
    state.pop('decoder.valid_mask', None)
    return state


def build(args, state, size):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.core import YAMLConfig
    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.yaml_cfg['eval_spatial_size'] = [size, size]
    for key in ('HybridEncoder', 'DFINETransformer'):
        if key in cfg.yaml_cfg and 'eval_spatial_size' in cfg.yaml_cfg[key]:
            cfg.yaml_cfg[key]['eval_spatial_size'] = [size, size]
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.model.load_state_dict(state, strict=False)

    class M(torch.nn.Module):
        def __init__(self, model, post):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = post.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)
    return M(cfg.model, cfg.postprocessor).to(args.device).eval()


@torch.inference_mode()
def forward_all(model, items, img_dir, size, device, batch):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    out = []
    for i in range(0, len(items), batch):
        ts, origs = [], []
        for it in items[i:i + batch]:
            img = Image.open(os.path.join(img_dir, it['file_name'])).convert('RGB')
            ts.append(tf(img)); origs.append([img.size[0], img.size[1]])
        x = torch.stack(ts, 0).to(device)
        o = torch.tensor(origs, dtype=torch.float32, device=device)
        lb, bx, sc = model(x, o)
        for j in range(len(ts)):
            out.append((lb[j].cpu(), bx[j].cpu().float(), sc[j].cpu().float()))
    return out


def evaluate(coco, dt):
    if not dt:
        return [0.0] * 12
    cd = coco.loadRes(dt)
    E = COCOeval(coco, cd, iouType='bbox')
    E.evaluate(); E.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.summarize()
    return list(E.stats)


def iou_xyxy(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ua = ((a[:, None, 2] - a[:, None, 0]) * (a[:, None, 3] - a[:, None, 1])
          + (b[None, :, 2] - b[None, :, 0]) * (b[None, :, 3] - b[None, :, 1]) - inter)
    return inter / np.maximum(ua, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:3')
    ap.add_argument('--sizes', type=int, nargs='+', default=[800, 928])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--nms', type=float, default=0.7)
    ap.add_argument('--md', type=int, default=100)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--scales', type=float, nargs='+',
                    default=[1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5])
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    print('[data] val=%d' % len(items), flush=True)

    state = load_state(args.resume)
    per_size = {}
    for size in args.sizes:
        model = build(args, state, size)
        t0 = time.time()
        per_size[size] = forward_all(model, items, args.img_dir, size, args.device, args.batch)
        print('[fwd] size=%d %.0fs' % (size, time.time() - t0), flush=True)
        del model; torch.cuda.empty_cache()

    # NMS 后的框（原图 xyxy 像素域）
    raw = []
    for idx, it in enumerate(items):
        lbs, bxs, scs = [], [], []
        for size in args.sizes:
            lb, bx, sc = per_size[size][idx]
            m = sc >= args.conf
            if not bool(m.any()):
                continue
            lbs.append(lb[m]); bxs.append(bx[m]); scs.append(sc[m])
        if not lbs:
            raw.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
            continue
        lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)
        k = batched_nms(bxb.float(), scb.float(), lbb, args.nms)[:args.md]
        raw.append((bxb[k].numpy(), lbb[k].numpy(), scb[k].numpy()))

    # ---- 1) 匹配框尺寸比 ----
    rw, rh, rarea = [], [], []
    for it, (bx, lb, sc) in zip(items, raw):
        if bx.shape[0] == 0:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[it['id']]))
        if not anns:
            continue
        gb = np.array([[a['bbox'][0], a['bbox'][1],
                        a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns])
        gl = np.array([cat_ids.index(a['category_id']) for a in anns])
        I = iou_xyxy(bx, gb)
        for gi in range(gb.shape[0]):
            cand = np.where((I[:, gi] >= 0.5) & (lb == gl[gi]))[0]
            if cand.size == 0:
                continue
            bi = cand[np.argmax(I[cand, gi])]
            pw = bx[bi, 2] - bx[bi, 0]; ph = bx[bi, 3] - bx[bi, 1]
            gw = gb[gi, 2] - gb[gi, 0]; gh = gb[gi, 3] - gb[gi, 1]
            rw.append(pw / max(gw, 1e-6)); rh.append(ph / max(gh, 1e-6))
            rarea.append((pw * ph) / max(gw * gh, 1e-9))
    rw, rh, rarea = map(np.array, (rw, rh, rarea))
    print('\n[匹配框尺寸比] n=%d' % len(rw), flush=True)
    print('  w_pred/w_gt   中位=%.3f 均值=%.3f 分位[25,75]=%s' % (np.median(rw), rw.mean(), np.percentile(rw, [25, 75]).round(3)), flush=True)
    print('  h_pred/h_gt   中位=%.3f 均值=%.3f 分位[25,75]=%s' % (np.median(rh), rh.mean(), np.percentile(rh, [25, 75]).round(3)), flush=True)
    print('  面积比        中位=%.3f' % np.median(rarea), flush=True)

    # ---- 2) 缩放扫描 ----
    print('\n[scale 扫描] (中心不变, w/h 同比例放大)', flush=True)
    best = None
    for s in args.scales:
        dt = []
        for it, (bx, lb, sc) in zip(items, raw):
            if bx.shape[0] == 0:
                continue
            cx = (bx[:, 0] + bx[:, 2]) / 2.0
            cy = (bx[:, 1] + bx[:, 3]) / 2.0
            hw = (bx[:, 2] - bx[:, 0]) * s / 2.0
            hh = (bx[:, 3] - bx[:, 1]) * s / 2.0
            x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
            W, H = it['width'], it['height']
            x1 = np.clip(x1, 0, W); x2 = np.clip(x2, 0, W)
            y1 = np.clip(y1, 0, H); y2 = np.clip(y2, 0, H)
            for i in range(bx.shape[0]):
                dt.append({'image_id': it['id'], 'category_id': cat_ids[int(lb[i])],
                           'bbox': [float(x1[i]), float(y1[i]), float(max(0, x2[i] - x1[i])), float(max(0, y2[i] - y1[i]))],
                           'score': float(sc[i])})
        st = evaluate(coco, dt)
        print('  scale=%.2f  mAP=%.4f  AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
              % (s, st[0], st[1], st[2], st[3], st[4], st[5]), flush=True)
        if best is None or st[0] > best[1]:
            best = (s, st[0])
    print('\n[BEST] scale=%.2f mAP=%.4f' % best, flush=True)


if __name__ == '__main__':
    main()
