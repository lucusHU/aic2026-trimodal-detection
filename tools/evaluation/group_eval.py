#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""val 分组诊断：一次前向，多组 COCO 评估。
目的：定位"val 分数高但榜单分数低"的原因。
分组维度: 整体 / 序列已见vs未见 / 1920vs640;  并对比 maxdet 100/300/1000。
"""
import argparse, contextlib, io, os, re, sys, time
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


def eval_subset(coco, dt, img_ids):
    if not dt:
        return None
    sub = [d for d in dt if d['image_id'] in img_ids]
    if not sub:
        return None
    cd = coco.loadRes(sub)
    E = COCOeval(coco, cd, iouType='bbox')
    E.params.imgIds = sorted(img_ids)
    E.evaluate()
    E.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.summarize()
    return list(E.stats)


def seq_of(name):
    m = re.match(r'(shuming_\d+)_', name)
    return m.group(1) if m else 'JPGSEQ'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', nargs='+', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--train-dir', default='dataset/images/train')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--sizes', type=int, nargs='+', default=[800, 928])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--nms', type=float, nargs='+', default=[0.7])
    ap.add_argument('--mds', type=int, nargs='+', default=[100, 300, 1000])
    ap.add_argument('--conf', type=float, default=0.001)
    args = ap.parse_args()

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    items = [coco.imgs[i] for i in sorted(coco.imgs.keys())]
    print('[data] val=%d cats=%d' % (len(items), len(cat_ids)), flush=True)

    train_seqs = {seq_of(p) for p in os.listdir(args.train_dir)}

    # 分组定义
    g_seen, g_unseen, g_1920, g_640 = [], [], [], []
    for it in items:
        iid = it['id']
        if seq_of(it['file_name']) in train_seqs:
            g_seen.append(iid)
        else:
            g_unseen.append(iid)
        (g_1920 if it['width'] == 1920 else g_640).append(iid)
    groups = [('ALL', [it['id'] for it in items]),
              ('序列已见(train同序列)', g_seen),
              ('序列未见', g_unseen),
              ('1920x1080', g_1920),
              ('640x360', g_640)]

    # GT 密度
    gtc = [len(coco.getAnnIds(imgIds=[it['id']])) for it in items]
    gtc = np.array(gtc)
    print('[GT密度] 每图GT数 mean=%.1f 中位=%.0f p90=%.0f max=%d  | >100框的图: %d张(%.1f%%)'
          % (gtc.mean(), np.median(gtc), np.percentile(gtc, 90), gtc.max(),
             (gtc > 100).sum(), 100 * (gtc > 100).mean()), flush=True)

    for ck in args.resume:
        state = load_state(ck)
        per_size = {}
        for size in args.sizes:
            model = build(args, state, size)
            t0 = time.time()
            per_size[size] = forward_all(model, items, args.img_dir, size, args.device, args.batch)
            print('[fwd] %s size=%d %.0fs' % (os.path.basename(ck), size, time.time() - t0), flush=True)
            del model
            torch.cuda.empty_cache()

        name = os.path.basename(os.path.dirname(ck))
        for ni in args.nms:
            for md in args.mds:
                dt = []
                ndet = []
                for idx_it, it in enumerate(items):
                    lbs, bxs, scs = [], [], []
                    for size in args.sizes:
                        lb, bx, sc = per_size[size][idx_it]
                        m = sc >= args.conf
                        if not bool(m.any()):
                            continue
                        lbs.append(lb[m]); bxs.append(bx[m]); scs.append(sc[m])
                    if not lbs:
                        ndet.append(0)
                        continue
                    lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)
                    k = batched_nms(bxb.float(), scb.float(), lbb, ni)[:md]
                    raw_k = batched_nms(bxb.float(), scb.float(), lbb, ni)
                    ndet.append(len(raw_k))
                    for idx in k:
                        x1, y1, x2, y2 = bxb[idx].tolist()
                        dt.append({'image_id': it['id'],
                                   'category_id': cat_ids[int(lbb[idx])],
                                   'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                   'score': float(scb[idx])})
                ndet = np.array(ndet)
                print('\n### %s | nms=%.2f md=%d | NMS后框数/图 mean=%.0f p90=%.0f max=%d | 被100截断的图 %.1f%%'
                      % (name, ni, md, ndet.mean(), np.percentile(ndet, 90), ndet.max(),
                         100 * (ndet > 100).mean()), flush=True)
                for gname, gids in groups:
                    st = eval_subset(coco, dt, set(gids))
                    if st is None:
                        print('   %-22s  无检测' % gname, flush=True)
                        continue
                    print('   %-22s n=%-4d mAP=%.4f  AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f'
                          % (gname, len(gids), st[0], st[1], st[2], st[3], st[4], st[5]), flush=True)
        del per_size
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
