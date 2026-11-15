"""End-to-end numbers of an exported .pt on web photos it never saw, and (optional) agreement
with the fp32 checkpoint it was packed from.

python ko_eval_pt.py vietnam_model.pt [--ckpt results/ko_final_<tag>.pth]
-> results/ko_eval_pt.json
The file is driven exactly like a user would: uint8 batches into model.predict().
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import oe_lib as L
from ko_cv import TEST_BRANDS, load_all

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt"); ap.add_argument("--ckpt", default=""); ap.add_argument("--arch", default="efficientnet_b0")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pt = torch.jit.load(a.pt, map_location=dev)
    L.RES = pt.input_size
    xk, yk, xh, sets, meta = load_all()

    def run(x):
        idx, lg, sc = [], [], []
        for i in range(0, len(x), 100):
            xb = x[i:i + 100].to(dev)
            idx.append(pt.predict_index(xb).cpu())
            l, s = pt.logits_and_score(xb); lg.append(l.cpu()); sc.append(s.cpu())
        return torch.cat(idx).numpy(), torch.cat(lg), torch.cat(sc)

    res = {"file": a.pt, "score": pt.score_name, "thr": pt.thr}
    pk, lk, sk = run(xk)
    res["training_photos"] = dict(acc=float((pk == yk).mean()), rejected=float((pk == 3).mean()))
    print(f"training photos (seen by the network, optimistic): accuracy {res['training_photos']['acc']:.3f}, labelled unknown {res['training_photos']['rejected']:.3f}")
    ph, lh, sh = run(xh[sets["hk"]])
    res["web_known"] = dict(acc=float((ph == meta["hk_y"]).mean()), rejected=float((ph == 3).mean()),
                            by_brand={b: dict(acc=float((ph == meta["hk_y"])[meta["hk_y"] == i].mean()), rejected=float((ph == 3)[meta["hk_y"] == i].mean()))
                                      for i, b in enumerate(L.KNOWN)})
    print(f"web photos of adidas/converse/nike (1500, never trained on): accuracy {res['web_known']['acc']:.3f}, labelled unknown {res['web_known']['rejected']:.3f}",
          {b: round(v["acc"], 3) for b, v in res["web_known"]["by_brand"].items()})
    pu, lu, su = run(xh[sets["unk"]])
    res["rejection"] = {b: float((pu == 3)[meta["unk_brand"] == b].mean()) for b in TEST_BRANDS}
    print("rejection on web photos of the unseen brands:", {b: round(v, 3) for b, v in res["rejection"].items()})
    po, lo, so = run(xh[sets["oth"]])
    res["rejection_other"] = float((po == 3).mean())
    print(f"rejection on web photos of other brands (new balance, saucony, ...): {res['rejection_other']:.3f}")
    if a.ckpt:
        net = L.build(a.arch, n_out=3); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")); net = net.to(dev).eval()
        with torch.autocast("cuda", enabled=False):
            ref = []
            with torch.no_grad():
                for x in (xk, xh[sets["hk"]]):
                    for i in range(0, len(x), 100):
                        xb = L.to_input(x[i:i + 100].to(dev))
                        ref.append(((net(xb) + net(xb.flip(3))) / 2).cpu())
        ref = torch.cat(ref); new = torch.cat([lk, lh])
        res["vs_fp32"] = dict(max_logit_diff=float((ref - new).abs().max()), mean_logit_diff=float((ref - new).abs().mean()),
                              same_argmax=float((ref.argmax(1) == new.argmax(1)).float().mean()))
        print("packed file vs fp32 checkpoint (735 + 1500 photos):", res["vs_fp32"])
    json.dump(res, open(HERE / "results/ko_eval_pt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
