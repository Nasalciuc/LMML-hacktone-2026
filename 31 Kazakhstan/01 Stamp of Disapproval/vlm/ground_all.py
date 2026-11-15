"""Ground the candidate edit elements of every album stamp: face value and year of issue.

Batched Qwen2.5-VL grounding; boxes in crop pixels -> elements.json {stamp: {value: box, year: box}}.

    python ground_all.py
"""
import json
from pathlib import Path
import torch
from PIL import Image
from vlm_fast import batch_inputs
from vlm_read import load, parse_box, prep

HERE = Path(__file__).parent
Q = {
    "value": 'Locate the face value (denomination number) printed on this stamp. Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}.',
    "year": 'Locate the small year of issue printed on this stamp. Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}.',
}


@torch.no_grad()
def main():
    crops = json.load(open(HERE / "crops.json"))
    out_path = HERE / "elements.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    todo = [c for c in crops if c["crop"][:-4] not in out]
    model, proc = load()
    proc.tokenizer.padding_side = "left"
    bs = 6
    for b in range(0, len(todo), bs):
        chunk = todo[b:b + bs]
        paths = [str(HERE / "crops" / c["crop"]) for c in chunk]
        ims = [prep(p) for p in paths]
        res = {c["crop"][:-4]: {} for c in chunk}
        for k, q in Q.items():
            x = batch_inputs(proc, ims, q)
            o = model.generate(**x, max_new_tokens=40, do_sample=False)
            txt = proc.batch_decode(o[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
            for c, p, im, t in zip(chunk, paths, ims, txt):
                res[c["crop"][:-4]][k] = parse_box(t, im, Image.open(p).size)
        out.update(res)
        json.dump(out, open(out_path, "w"))
        print(b + len(chunk), "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
