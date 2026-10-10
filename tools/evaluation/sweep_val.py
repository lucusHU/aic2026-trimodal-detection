#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""val 上扫描：输入分辨率 x checkpoint(stg1/stg2/soup) x NMS iou
一次前向，多次 NMS + COCOeval，成本低。
用 pycocotools 默认 maxDets=[1,10,100]，与 D-FINE CocoEvaluator 口径一致。
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
    cfg = YAMLConfig(args.config, resume=args.resume[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', nargs='+', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--size', type=int, nargs='+', default=[800])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--nms', type=float, nargs='+', default=[0.5, 0.6, 0.7])
    ap.add_argument('--maxdet', type=int, nargs='+', default=[100])
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    if args.limit:
        items = items[:args.limit]
    print('[data] val images=%d cats=%s maxDets=default[1,10,100]' % (len(items), cat_ids), flush=True)

    states = [load_state(p, use_ema=not args.no_ema) for p in args.resume]
    combos = [(os.path.basename(p), s) for p, s in zip(args.resume, states)]
    if len(states) > 1:
        soup = {}
        for k in states[0]:
            acc = states[0][k].float()
            for s in states[1:]:
                acc = acc + s[k].float()
            soup[k] = (acc / len(states)).to(states[0][k].dtype)
        combos.append(('SOUP_avg%d' % len(states), soup))

    t_all = time.time()
    results = []
    for size in args.size:
        for name, state in combos:
            model = build(args, state, size)
            t0 = time.time()
            preds = forward_all(model, items, args.img_dir, size, args.device, args.batch)
            fwd = time.time() - t0
            raw = [(lb, bx[sc >= args.conf], sc[sc >= args.conf]) for lb, bx, sc in preds]
            print('[fwd] size=%d %s  %.0fs' % (size, name, fwd), flush=True)

            for ni in args.nms:
                for md in args.maxdet:
                    dt = []
                    for it, (lb, bx, sc) in zip(items, raw):
                        if bx.numel() == 0:
                            continue
                        k = batched_nms(bx.float(), sc.float(), lb, ni)[:md]
                        for idx in k:
                            x1, y1, x2, y2 = bx[idx].tolist()
                            dt.append({
                                'image_id': it['id'],
                                'category_id': cat_ids[int(lb[idx])],
                                'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                'score': float(sc[idx]),
                            })
                    st = evaluate(coco, dt)
                    results.append((st[0], size, name, ni, md, st))
                    print('  size=%-4d %-14s nms=%.2f md=%-3d mAP=%.4f AP50=%.4f AP75=%.4f '
                          'APs=%.4f APm=%.4f APl=%.4f (%d det)'
                          % (size, name, ni, md, st[0], st[1], st[2], st[3], st[4], st[5], len(dt)),
                          flush=True)
            del model
            torch.cuda.empty_cache()

    print('\n===== RANK (top 15) =====', flush=True)
    for r in sorted(results, reverse=True)[:15]:
        print('mAP=%.4f size=%-4d %-14s nms=%.2f md=%d | AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
              % (r[0], r[1], r[2], r[3], r[4], r[5][1], r[5][2], r[5][3], r[5][4], r[5][5]), flush=True)
    print('[done] %.0fs' % (time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
