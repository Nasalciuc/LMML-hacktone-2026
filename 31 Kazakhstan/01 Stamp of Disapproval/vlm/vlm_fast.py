"""Batched album pass: read + p_alt for every album crop (no box; the box was useless on exhibits).

Same prompts and model as vlm_read.py, only batched (left padding) so decoding is shared.

    python vlm_fast.py --what albums --bs 6
    python vlm_fast.py --what exhibits --bs 6     # re-check, writes vlm_exhibits_fast.json
"""
import argparse, json, time
from pathlib import Path

import torch
from PIL import Image

from vlm_read import ALT_PROMPT, READ_PROMPT, load, prep

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"


def batch_inputs(proc, ims, prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[text] * len(ims), images=ims, return_tensors="pt", padding=True).to("cuda:0")


@torch.no_grad()
def run_batch(model, proc, ims, n):
    x = batch_inputs(proc, ims, READ_PROMPT)
    out = model.generate(**x, max_new_tokens=n, do_sample=False)
    reads = proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
    x = batch_inputs(proc, ims, ALT_PROMPT)
    logits = model(**x).logits[:, -1].float()
    tok = proc.tokenizer
    y, no = tok.encode("Yes")[0], tok.encode("No")[0]
    p = torch.softmax(logits[:, [y, no]], 1)[:, 0].tolist()
    return reads, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["exhibits", "albums"], required=True)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--read_tokens", type=int, default=160)
    a = ap.parse_args()
    if a.what == "exhibits":
        items = [dict(crop=str(p), name=p.stem) for p in sorted((DATA / "exhibits").glob("exhibit_??.jpg"))]
        out_path = HERE / "vlm_exhibits_fast.json"
    else:
        items = json.load(open(HERE / "crops.json"))
        for it in items:
            it["name"] = it["crop"][:-4]
            it["crop"] = str(HERE / "crops" / it["crop"])
        out_path = HERE / "vlm_albums.json"
    done = {r["name"]: r for r in json.load(open(out_path, encoding="utf-8"))} if out_path.exists() else {}
    todo = [it for it in items if it["name"] not in done]
    model, proc = load()
    proc.tokenizer.padding_side = "left"
    t0 = time.time()
    for b in range(0, len(todo), a.bs):
        chunk = todo[b:b + a.bs]
        ims = [prep(it["crop"]) for it in chunk]
        reads, ps = run_batch(model, proc, ims, a.read_tokens)
        for it, rd, p in zip(chunk, reads, ps):
            r = dict(it); r["read"] = rd; r["p_alt"] = p
            done[r["name"]] = r
            print(r["name"], round(p, 3), rd.replace("\n", " ")[:150], flush=True)
        json.dump(list(done.values()), open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = b + len(chunk)
        print(f"## {n}/{len(todo)}  {(time.time() - t0) / n:.1f}s/stamp", flush=True)


if __name__ == "__main__":
    main()
