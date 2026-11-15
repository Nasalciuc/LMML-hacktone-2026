"""Continuous 'is the year of issue missing?' score for every stamp.

Most removals in the exhibits erase the small year of issue near an edge. The free read gives
only a binary 'no year read'; here we ask directly and keep P(No) so stamps can be ranked
within a page. One forward pass per stamp (no generation), batched.

    python yearq.py --what exhibits     -> yearq_exhibits.json
    python yearq.py --what albums       -> yearq_albums.json
"""
import argparse, json
from pathlib import Path

import torch

from vlm_fast import batch_inputs
from vlm_read import load, prep

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"

QUESTIONS = {
    "year": ("Postage stamps of Kazakhstan print the year of issue as small digits, usually near the "
             "bottom or side edge. Is the year of issue printed on this stamp? Answer Yes or No."),
    "small": ("Postage stamps usually carry small inscriptions near the edges: the year of issue, the "
              "designer's name, a printer's mark. Does this stamp have an empty area near an edge where "
              "such a small inscription seems to be missing? Answer Yes or No."),
}


@torch.no_grad()
def p_yes(model, proc, ims, prompt):
    x = batch_inputs(proc, ims, prompt)
    logits = model(**x).logits[:, -1].float()
    tok = proc.tokenizer
    y, n = tok.encode("Yes")[0], tok.encode("No")[0]
    return torch.softmax(logits[:, [y, n]], 1)[:, 0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["exhibits", "albums"], required=True)
    ap.add_argument("--bs", type=int, default=6)
    a = ap.parse_args()
    if a.what == "exhibits":
        items = [dict(name=p.stem, crop=str(p)) for p in sorted((DATA / "exhibits").glob("exhibit_??.jpg"))]
    else:
        items = [dict(name=c["crop"][:-4], crop=str(HERE / "crops" / c["crop"]))
                 for c in json.load(open(HERE / "crops.json"))]
    model, proc = load()
    proc.tokenizer.padding_side = "left"
    out = {}
    for b in range(0, len(items), a.bs):
        chunk = items[b:b + a.bs]
        ims = [prep(it["crop"]) for it in chunk]
        res = {k: p_yes(model, proc, ims, q) for k, q in QUESTIONS.items()}
        for j, it in enumerate(chunk):
            # year_missing = P(No) to "is the year printed"; small_missing = P(Yes) to "empty area"
            out[it["name"]] = dict(year_missing=1 - res["year"][j], small_missing=res["small"][j])
        print(b + len(chunk), "/", len(items), flush=True)
    json.dump(out, open(HERE / f"yearq_{a.what}.json", "w"), indent=0)


if __name__ == "__main__":
    main()
