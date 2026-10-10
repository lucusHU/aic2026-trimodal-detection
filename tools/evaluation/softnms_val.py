#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""val 上对比 NMS / Soft-NMS(线性) / Soft-NMS(高斯)，叠加多尺度 TTA。
Soft-NMS 对密集场景通常有 0.5~1pp 收益，无需重训，成本低。
"""
import argparse, contextlib, io, os, sys, time
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
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


@torch.inference_mode()
def forward_all(model, items, img_dir, size, device, batch):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    out = []
    for i in range(0, len(items), batch):
        ts, origs = [], []
        for it in items[i:i + batch]:
            img = Image.open(os.path.join(img_dir, it['file_name'])).convert('RGB')
            ts.append(tf(img))
            origs.append([img.size[0], img.size[1]])
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
    E.evaluate()
    E.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.summarize()
    return list(E.stats)


def iou_matrix(b):
    x1 = torch.max(b[:, None, 0], b[None, :, 0])
    y1 = torch.max(b[:, None, 1], b[None, :, 1])
    x2 = torch.min(b[:, None, 2], b[None, :, 2])
    y2 = torch.min(b[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=1e-9)


def soft_nms(bx, sc, lb, iou_thr, sigma, method, score_thr, maxdet):
    """返回保留下来的索引（按分数降序）。method: linear | gaussian"""
    b = bx.clone()
    s = sc.clone()
    n = b.shape[0]
    if n == 0:
        return []
    order = torch.argsort(s, descending=True)
    b = b[order]
    s = s[order]
    l = lb[order]
    keep = []
    iou_thr_t = iou_thr
    while b.shape[0] > 0:
        i = 0
        keep.append(order[i].item() if False else i)
        # 记录当前最高分框的原始索引
        keep[-1] = int(order[i])
        if len(keep) >= maxdet:
            break
        cur = b[i:i + 1]
        if b.shape[0] > 1:
            rest = b[i + 1:]
            xx1 = torch.max(cur[:, 0], rest[:, 0])
            yy1 = torch.max(cur[:, 1], rest[:, 1])
            xx2 = torch.min(cur[:, 2], rest[:, 2])
            yy2 = torch.min(cur[:, 3], rest[:, 3])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            area_cur = (cur[:, 2] - cur[:, 0]).clamp(min=0) * (cur[:, 3] - cur[:, 1]).clamp(min=0)
            area_r = (rest[:, 2] - rest[:, 0]).clamp(min=0) * (rest[:, 3] - rest[:, 1]).clamp(min=0)
            iou = inter / (area_cur + area_r - inter).clamp(min=1e-9)
            if method == 'linear':
                w = torch.where(iou > iou_thr_t, (1.0 - iou).clamp(min=0.0),
                                torch.ones_like(iou))
            else:
                w = torch.exp(-(iou * iou) / (sigma * sigma))
            s[i + 1:] = s[i + 1:] * w
        # 去掉当前框
        b = torch.cat([b[:i], b[i + 1:]])
        s = torch.cat([s[:i], s[i + 1:]])
        order = torch.cat([order[:i], order[i + 1:]])
        if b.shape[0] == 0:
            break
        m = s >= score_thr
        if not bool(m.any()):
            break
        b, s, order = b[m], s[m], order[m]
        # 重新按分数排序
        o2 = torch.argsort(s, descending=True)
        b, s, order = b[o2], s[o2], order[o2]
    return keep[:maxdet]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--maxdet', type=int, default=100)
    ap.add_argument('--sizes', default='928,800,928')
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    print('[data] val=%d cats=%d' % (len(items), len(cat_ids)), flush=True)

    sizes = sorted({int(x) for x in args.sizes.split(',')})
    cache = {}
    for size in sizes:
        model = build(args, load_state(args.resume), size)
        t0 = time.time()
        cache[size] = forward_all(model, items, args.img_dir, size, args.device, args.batch)
        print('[fwd] size=%d  %.0fs' % (size, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    from torchvision.ops import batched_nms
    results = []
    for combo in [tuple(int(x) for x in args.sizes.split(','))]:
        for method, thr, sigma in [
            ('nms', 0.7, 0.0), ('nms', 0.8, 0.0),
            ('linear', 0.5, 0.0), ('linear', 0.6, 0.0), ('linear', 0.7, 0.0),
            ('gaussian', 0.5, 0.3), ('gaussian', 0.5, 0.5), ('gaussian', 0.5, 0.7),
        ]:
            dt = []
            for i, it in enumerate(items):
                lbs, bxs, scs = [], [], []
                for size in combo:
                    lb, bx, sc = cache[size][i]
                    m = sc >= args.conf
                    if not bool(m.any()):
                        continue
                    lbs.append(lb[m]); bxs.append(bx[m]); scs.append(sc[m])
                if not lbs:
                    continue
                lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)
                if method == 'nms':
                    k = batched_nms(bxb.float(), scb.float(), lbb, thr)[:args.maxdet]
                    k = k.tolist()
                else:
                    k = soft_nms(bxb.float(), scb.float(), lbb, thr, sigma,
                                 method, args.conf, args.maxdet)
                for idx in k:
                    x1, y1, x2, y2 = bxb[idx].tolist()
                    dt.append({
                        'image_id': it['id'],
                        'category_id': cat_ids[int(lbb[idx])],
                        'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        'score': float(scb[idx]),
                    })
            st = evaluate(coco, dt)
            tag = '%s thr=%.2f sig=%.2f' % (method, thr, sigma)
            results.append((st[0], tag, st))
            print('  %-22s mAP=%.4f AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f (%d det)'
                  % (tag, st[0], st[1], st[2], st[3], st[4], st[5], len(dt)), flush=True)

    print('\n===== RANK =====', flush=True)
    for r in sorted(results, reverse=True):
        print('mAP=%.4f  %-22s | AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
              % (r[0], r[1], r[2][1], r[2][2], r[2][3], r[2][4], r[2][5]), flush=True)


if __name__ == '__main__':
    main()
