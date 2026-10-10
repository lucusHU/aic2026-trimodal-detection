import torch, itertools
for p in ['weights/dfine_l_obj365.pth', 'weights/dfine_l_obj365_e25.pth']:
    try:
        ck = torch.load(p, map_location='cpu')
        keys = list(ck.keys())
        st = ck.get('ema', ck.get('model', ck))
        st = st['module'] if 'module' in st else st
        print(p)
        print('  top keys:', keys[:6])
        print('  n tensors:', len(st))
        for k in itertools.islice(st, 3):
            print('   ', k, tuple(st[k].shape))
        for k in st:
            if 'class' in k or 'score' in k or 'enc_score' in k:
                print('   [head]', k, tuple(st[k].shape))
    except Exception as e:
        print(p, 'FAIL', type(e).__name__, e)
