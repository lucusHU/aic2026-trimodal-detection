#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tiling 推理评估：把 16:9 原图切成若干重叠 tile，每块独立 resize 到方形输入，
框映射回原图坐标后跨 tile 合并 + NMS。对照 g1（整图拉伸 = 现有基线）。

思路：原图 1920x1080 整图压到 928x928 时，水平缩放 928/1920=0.48，小目标被压扁；
切成 2 块后每块 1200x1080 -> 928x928，水平缩放 0.77，目标更大且形变更小。
"""
import argparse, contextlib, io, os, sys, time
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.ops import batched_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_state(path, use_ema=True):
    ck = torch.load(path, map_location='cpu')
    if use_ema and 'ema' in ck:
        state = ck['ema']['module']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck
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
    base = cfg.model
    base.load_state_dict(state, strict=False)

    class M(torch.nn.Module):
        def __init__(self, model, post):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = post.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)
    return M(base, cfg.postprocessor).to(args.device).eval()


def split_1d(L, n, ov):
    if n <= 1:
        return [(0, L)]
    tw = min(int(round(L / n * (1.0 + ov))), L)
    stride = (L - tw) / (n - 1)
    return [(int(round(i * stride)), int(round(i * stride)) + tw) for i in range(n)]


def gen_crops(W, H, mode, ov):
    """返回 [(x0,y0,x1,y1)]，不含 global（global 单独拼）"""
    if mode == 'g1':
        return [(0, 0, W, H)]
    if mode == 't2':
        xs = split_1d(W, 2, ov)
        return [(x0, 0, x1, H) for x0, x1 in xs]
    if mode == 't3':
        xs = split_1d(W, 3, ov)
        return [(x0, 0, x1, H) for x0, x1 in xs]
    if mode == 't4':
        xs = split_1d(W, 2, ov)
        ys = split_1d(H, 2, ov)
        return [(x0, y0, x1, y1) for y0, y1 in ys for x0, x1 in xs]
    raise ValueError(mode)


@torch.inference_mode()
def forward_mode(model, items, img_dir, size, device, batch, mode, ov):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    per_img = [[] for _ in items]
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ts, metas = [], []
        for j, it in enumerate(chunk):
            img = Image.open(os.path.join(img_dir, it['file_name'])).convert('RGB')
            W, H = img.size
            for (x0, y0, x1, y1) in gen_crops(W, H, mode, ov):
                x1 = min(x1, W); y1 = min(y1, H)
                ts.append(tf(img.crop((x0, y0, x1, y1))))
                metas.append((i + j, x0, y0, x1 - x0, y1 - y0))
        if not ts:
            continue
        x = torch.stack(ts, 0).to(device)
        o = torch.tensor([[m[3], m[4]] for m in metas], dtype=torch.float32, device=device)
        lb, bx, sc = model(x, o)
        lb = lb.cpu(); bx = bx.cpu().float(); sc = sc.cpu().float()
        for k in range(len(ts)):
            ii, x0, y0, cw, ch = metas[k]
            b = bx[k].clone()
            b[:, 0::2] += x0
            b[:, 1::2] += y0
            per_img[ii].append((lb[k], b, sc[k]))
    return per_img


def evaluate(coco, dt):
    if not dt:
        return [0.0] * 12
    cd = coco.loadRes(dt)
    E = COCOeval(coco, cd, iouType='bbox')
    E.evaluate()
    E.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.summarize()
    return list(E.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--size', type=int, default=928)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--ov', type=float, default=0.25)
    ap.add_argument('--nms', type=float, nargs='+', default=[0.7, 0.8, 0.9])
    ap.add_argument('--maxdet', type=int, default=100)
    ap.add_argument('--conf', type=float, nargs='+', default=[0.001])
    ap.add_argument('--modes', nargs='+', default=['g1', 't2', 't3', 't4'])
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    if args.limit:
        items = items[:args.limit]
    print('[data] val=%d size=%d ov=%.2f conf=%s maxDet=%d'
          % (len(items), args.size, args.ov, args.conf, args.maxdet), flush=True)

    model = build(args, load_state(args.resume), args.size)
    cache = {}
    for mode in args.modes:
        t0 = time.time()
        cache[mode] = forward_mode(model, items, args.img_dir, args.size,
                                   args.device, args.batch, mode, args.ov)
        ntiles = sum(len(v) for v in cache[mode])
        print('[fwd] %-3s tiles=%d  %.0fs' % (mode, ntiles, time.time() - t0), flush=True)

    combos = [('g1', ['g1']), ('t2', ['t2']), ('t2g', ['t2', 'g1']),
              ('t3', ['t3']), ('t3g', ['t3', 'g1']),
              ('t4', ['t4']), ('t4g', ['t4', 'g1'])]

    results = []
    for cname, parts in combos:
        if any(p not in cache for p in parts):
            continue
        merged = []
        for idx in range(len(items)):
            lbs, bxs, scs = [], [], []
            for p in parts:
                for lb, bx, sc in cache[p][idx]:
                    lbs.append(lb); bxs.append(bx); scs.append(sc)
            if bxs:
                merged.append((torch.cat(lbs), torch.cat(bxs), torch.cat(scs)))
            else:
                merged.append((torch.zeros(0), torch.zeros(0, 4), torch.zeros(0)))
        for cf in args.conf:
          for ni in args.nms:
            dt = []
            for it, (lb, bx, sc) in zip(items, merged):
                keep0 = sc >= cf
                if not bool(keep0.any()):
                    continue
                lbb, bxb, scb = lb[keep0], bx[keep0], sc[keep0]
                k = batched_nms(bxb.float(), scb.float(), lbb, ni)[:args.maxdet]
                for idx in k:
                    x1, y1, x2, y2 = bxb[idx].tolist()
                    dt.append({
                        'image_id': it['id'],
                        'category_id': cat_ids[int(lbb[idx])],
                        'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        'score': float(scb[idx]),
                    })
            st = evaluate(coco, dt)
            results.append((st[0], cname, ni, st, cf))
            print('  %-4s nms=%.2f conf=%.3f mAP=%.4f AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f (%d det)'
                  % (cname, ni, cf, st[0], st[1], st[2], st[3], st[4], st[5], len(dt)), flush=True)

    print('\n===== RANK =====', flush=True)
    for r in sorted(results, reverse=True):
        print('mAP=%.4f %-4s nms=%.2f conf=%.3f | AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
              % (r[0], r[1], r[2], r[4], r[3][1], r[3][2], r[3][3], r[3][4], r[3][5]), flush=True)


if __name__ == '__main__':
    main()
