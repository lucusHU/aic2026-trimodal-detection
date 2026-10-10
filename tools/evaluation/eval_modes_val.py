#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 val 上对比不同输入方式(stretch / letterbox / aspect)的 mAP"""

import argparse, json, os, sys, time
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_state(path):
    ck = torch.load(path, map_location='cpu')
    if 'ema' in ck:
        state = ck['ema']['module']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck
    state = dict(state)
    state.pop('decoder.anchors', None)
    state.pop('decoder.valid_mask', None)
    return state


def build(args, hw, state):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.yaml_cfg['eval_spatial_size'] = list(hw)
    for key in ('HybridEncoder', 'DFINETransformer'):
        if key in cfg.yaml_cfg and 'eval_spatial_size' in cfg.yaml_cfg[key]:
            cfg.yaml_cfg[key]['eval_spatial_size'] = list(hw)
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


def prep(img, mode, size):
    w, h = img.size
    if mode == 'stretch':
        t = T.ToTensor()(img.resize((size, size), Image.BILINEAR))
        return t, (w, h), 1.0, 1.0
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = img.resize((nw, nh), Image.BILINEAR)
    if mode == 'aspect':
        return T.ToTensor()(r), (nw, nh), 1.0 / s, 1.0 / s
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(r, (0, 0))
    return T.ToTensor()(canvas), (size, size), 1.0 / s, 1.0 / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--modes', default='stretch800,stretch1280,letterbox800,letterbox1280,aspect800,aspect1280')
    ap.add_argument('--out', default='mode_compare.json')
    args = ap.parse_args()

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    items = coco.loadImgs(img_ids)
    state = load_state(args.resume)
    results = {}

    for spec in args.modes.split(','):
        spec = spec.strip()
        if not spec:
            continue
        mode = ''.join(ch for ch in spec if ch.isalpha())
        size = int(''.join(ch for ch in spec if ch.isdigit()))
        if mode == 'aspect':
            hw = [int(round(size * 9 / 16)), size]
        else:
            hw = [size, size]
        print('\n===== %s | eval_spatial_size=%s =====' % (spec, hw), flush=True)
        t0 = time.time()
        try:
            model = build(args, hw, state)
        except Exception as e:
            print('[skip] build failed: %s: %s' % (type(e).__name__, e), flush=True)
            continue

        preds = []
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            tensors, origs, sxs, syscales = [], [], [], []
            for it in chunk:
                p = os.path.join(args.img_dir, it['file_name'])
                img = Image.open(p).convert('RGB')
                t, osz, sx, sy = prep(img, mode, size)
                tensors.append(t)
                origs.append(osz)
                sxs.append(sx)
                syscales.append(sy)
            x = torch.stack(tensors, 0).to(args.device)
            o = torch.tensor(origs, dtype=torch.float32, device=args.device)
            with torch.inference_mode():
                labels, boxes, scores = model(x, o)
            for j, it in enumerate(chunk):
                lb = labels[j].cpu().numpy()
                bx = boxes[j].cpu().numpy()
                sc = scores[j].cpu().numpy()
                sx, sy = sxs[j], syscales[j]
                for k in range(len(sc)):
                    if sc[k] < args.conf:
                        continue
                    x1, y1, x2, y2 = bx[k]
                    x1 *= sx
                    x2 *= sx
                    y1 *= sy
                    y2 *= sy
                    preds.append({
                        'image_id': int(it['id']),
                        'category_id': int(lb[k]),
                        'bbox': [float(x1), float(y1),
                                 float(max(x2 - x1, 1.0)), float(max(y2 - y1, 1.0))],
                        'score': float(sc[k]),
                    })
            if i % (args.batch * 25) == 0:
                print('  %d/%d' % (i, len(items)), flush=True)

        if not preds:
            print('[skip] no preds', flush=True)
            continue

        tmp = '/tmp/_pred_%s.json' % spec
        json.dump(preds, open(tmp, 'w'))
        dt = coco.loadRes(tmp)
        E = COCOeval(coco, dt, 'bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        st = E.stats
        results[spec] = {
            'mAP': round(float(st[0]), 4), 'AP50': round(float(st[1]), 4),
            'AP75': round(float(st[2]), 4), 'APs': round(float(st[3]), 4),
            'APm': round(float(st[4]), 4), 'APl': round(float(st[5]), 4),
            'secs': int(time.time() - t0),
        }
        print('  => %s: mAP=%.4f AP50=%.4f AP75=%.4f APs=%.4f'
              % (spec, st[0], st[1], st[2], st[3]), flush=True)
        del model
        torch.cuda.empty_cache()

    print('\n===== SUMMARY =====')
    for k, v in sorted(results.items(), key=lambda x: -x[1]['mAP']):
        print('%-16s mAP=%.4f  AP50=%.4f  AP75=%.4f  APs=%.4f  APm=%.4f  APl=%.4f  (%ds)'
              % (k, v['mAP'], v['AP50'], v['AP75'], v['APs'], v['APm'], v['APl'], v['secs']))
    json.dump(results, open(args.out, 'w'), indent=2)
    print('saved -> %s' % args.out)


if __name__ == '__main__':
    main()
