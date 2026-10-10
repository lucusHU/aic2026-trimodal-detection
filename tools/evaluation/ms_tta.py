#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多尺度 TTA：val 上把同一张图在多个尺度分别前向，框合并后统一 NMS。
只用官方 val 标注评估，不产生任何标签，不涉及测试集。
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
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--nms', type=float, nargs='+', default=[0.7, 0.8, 0.9])
    ap.add_argument('--maxdet', type=int, default=100)
    # 逗号分隔的尺度组合，例如 "928 800,928 896,928"
    ap.add_argument('--combos', nargs='+',
                    default=['928', '800,928', '896,928', '832,928', '800,896,928', '928,992'])
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    print('[data] val=%d cats=%d' % (len(items), len(cat_ids)), flush=True)

    combos = [tuple(int(x) for x in c.split(',')) for c in args.combos]
    sizes = sorted({s for c in combos for s in c})

    cache = {}
    for size in sizes:
        model = build(args, load_state(args.resume), size)
        t0 = time.time()
        cache[size] = forward_all(model, items, args.img_dir, size, args.device, args.batch)
        print('[fwd] size=%d  %.0fs' % (size, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    results = []
    for combo in combos:
        for ni in args.nms:
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
            results.append((st[0], '+'.join(map(str, combo)), ni, st))
            print('  %-14s nms=%.2f mAP=%.4f AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f (%d det)'
                  % ('+'.join(map(str, combo)), ni, st[0], st[1], st[2], st[3], st[4], st[5], len(dt)),
                  flush=True)

    print('\n===== RANK =====', flush=True)
    for r in sorted(results, reverse=True):
        print('mAP=%.4f  scales=%-14s nms=%.2f | AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
              % (r[0], r[1], r[2], r[3][1], r[3][2], r[3][3], r[3][4], r[3][5]), flush=True)


if __name__ == '__main__':
    main()
