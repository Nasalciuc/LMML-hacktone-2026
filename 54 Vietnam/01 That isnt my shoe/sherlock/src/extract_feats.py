"""Extract frozen pooled features (orig + hflip) for candidate backbones.

usage: python -I extract_feats.py DATA_DIR WEIGHTS_DIR OUT_DIR name1 [name2 ...]
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os
import sys
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image

torch.set_num_threads(2)

CANDS = {
    # name: (timm arch, pretrained tag, weight file, input size)
    'mnv3_miil1k': ('mobilenetv3_large_100', 'miil_in21k_ft_in1k', 'mobilenetv3_large_100_1k_miil_78_0-66471c13.pth', 224),
    'mnv3_miil21k': ('mobilenetv3_large_100', 'miil_in21k', 'mobilenetv3_large_100_in21k_miil-d71cc17b.pth', 224),
    'effb0_ns': ('tf_efficientnet_b0', 'ns_jft_in1k', 'tf_efficientnet_b0_ns-c0e6a31c.pth', 224),
    'effb1_ns': ('tf_efficientnet_b1', 'ns_jft_in1k', 'tf_efficientnet_b1_ns-99dd0c41.pth', 240),
    'effb2_ns': ('tf_efficientnet_b2', 'ns_jft_in1k', 'tf_efficientnet_b2_ns-00306e48.pth', 260),
    'effb0_ra': ('efficientnet_b0', 'ra_in1k', 'efficientnet_b0_ra-3dd342df.pth', 224),
    'effb2_ra': ('efficientnet_b2', 'ra_in1k', 'efficientnet_b2_ra-bcdf34b7.pth', 256),
    'edgenext_s': ('edgenext_small', 'usi_in1k', 'edgenext_small_usi.pth', 256),
    'convnext_atto': ('convnext_atto', 'd2_in1k', 'convnext_atto_d2-01bb0f51.pth', 256),
    'convnext_femto': ('convnext_femto', 'd1_in1k', 'convnext_femto_d1-d71d5b4c.pth', 256),
    'convnext_pico': ('convnext_pico', 'd1_in1k', 'convnext_pico_d1-10ad7f0d.pth', 256),
    'tinynet_a': ('tinynet_a', 'in1k', 'tinynet_a.pth', 192),
    'mobilevit_s': ('mobilevit_s', 'cvnets_in1k', 'mobilevit_s-38a5a959.pth', 256),
    'regnety_008': ('regnety_008', 'pycls_in1k', 'regnety_008-dc900dbe.pth', 224),
    'effv2_b2': ('tf_efficientnetv2_b2', 'in1k', 'tf_efficientnetv2_b2-847de54e.pth', 256),
    'hardcorenas_f': ('hardcorenas_f', 'miil_green_in1k', 'hardcorenas_f_green_60ms_78_1-14b9e780.pth', 224),
}


def build(name, wdir, pretrained=True):
    arch, tag, wfile, size = CANDS[name]
    cfg = timm.models.get_pretrained_cfg(f'{arch}.{tag}')
    cfg = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
    ncls = cfg.get('num_classes', 1000)
    model = timm.create_model(arch, pretrained=False, num_classes=ncls)
    if pretrained:  # ImageNet weights are only needed for training; exported models load their own state_dict
        sd = torch.load(os.path.join(wdir, wfile), map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        # let timm adapt legacy key layouts when the model provides a filter
        mod = sys.modules[model.__class__.__module__]
        for fn_name in ('checkpoint_filter_fn', '_checkpoint_filter_fn'):
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                try:
                    sd = fn(sd, model)
                except TypeError:
                    sd = fn(sd)
                break
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f'  [{name}] missing={missing[:6]} ({len(missing)}) unexpected={unexpected[:6]} ({len(unexpected)})')
    model.reset_classifier(0)
    model.eval()
    mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    return model, mean, std, size, cfg


def load_images(data_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, 'train-output.csv'))))
    imgs = []
    for r in rows:
        im = Image.open(os.path.join(data_dir, 'train-input', r['filename'])).convert('RGB')
        imgs.append(np.asarray(im, dtype=np.uint8))
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,256,256
    labels = [r['label'] for r in rows]
    files = [r['filename'] for r in rows]
    return x, labels, files


@torch.no_grad()
def extract(model, mean, std, size, x_u8, bs=48):
    out_o, out_f = [], []
    for i in range(0, x_u8.shape[0], bs):
        xb = x_u8[i:i + bs].float() / 255.0
        if size != xb.shape[-1]:
            xb = F.interpolate(xb, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        xb = (xb - mean) / std
        out_o.append(model(xb))
        out_f.append(model(torch.flip(xb, dims=[3])))
    return torch.cat(out_o).numpy(), torch.cat(out_f).numpy()


def main():
    data_dir, wdir, out_dir = sys.argv[1:4]
    names = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    x_u8, labels, files = load_images(data_dir)
    np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels))
    for name in names:
        outp = os.path.join(out_dir, f'{name}.npz')
        if os.path.exists(outp):
            print('skip', name)
            continue
        t0 = time.time()
        model, mean, std, size, cfg = build(name, wdir)
        nparams = sum(p.numel() for p in model.parameters())
        fo, ff = extract(model, mean, std, size, x_u8)
        np.savez(outp, orig=fo, flip=ff)
        print(f'{name}: params(no head)={nparams/1e6:.2f}M dim={fo.shape[1]} size={size} '
              f'mean={tuple(round(float(v), 3) for v in mean.flatten())} time={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
