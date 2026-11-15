"""Read every stamp with Qwen2.5-VL-7B (4-bit) and ask what looks wrong with it.

For each stamp crop (album crops from crops.py, and the 30 exhibits) it stores:
  read   : free-text JSON with country names, denominations, years, other text
  p_alt  : probability of "Yes" for "was this stamp altered?"
  box    : the region the model points at when asked where the alteration is

    python vlm_read.py --what exhibits      # 30 exhibits
    python vlm_read.py --what albums        # all album crops
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

READ_PROMPT = (
    "This is a postage stamp of Kazakhstan. Transcribe what is printed on it and answer in JSON with keys: "
    '"country" (list of the country-name inscriptions exactly as printed, e.g. ҚАЗАҚСТАН, KAZAKHSTAN), '
    '"denominations" (list of the face values printed, digits exactly as printed), '
    '"years" (list of 4-digit years printed), '
    '"other_text" (list of other inscriptions). Output only the JSON.'
)
ALT_PROMPT = (
    "A forger altered some postage stamps in one small region: he changed a digit of the face value or the year, "
    "duplicated a small element of the picture, or erased a small inscription such as the year of issue, the "
    "designer's name or part of the country name. Look carefully at the digits (font, colour, thickness, spacing) "
    "and at places where an inscription is expected but missing. Was this stamp altered? Answer Yes or No."
)
BOX_PROMPT = (
    "This stamp was altered in one small region: a digit changed, a small element duplicated, or a small inscription "
    "erased. Output the bounding box of the altered region as JSON: {\"bbox_2d\": [x1, y1, x2, y2]}."
)


def load():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model.eval(), proc


def prep(path, long_side=896):
    im = Image.open(path).convert("RGB")
    s = long_side / max(im.size)
    if s > 1:  # small crops (strips, sheets) are upscaled so the text is legible
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def inputs(proc, im, prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")


@torch.no_grad()
def generate(model, proc, im, prompt, n=256):
    x = inputs(proc, im, prompt)
    out = model.generate(**x, max_new_tokens=n, do_sample=False)
    return proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


@torch.no_grad()
def p_yes(model, proc, im, prompt):
    x = inputs(proc, im, prompt)
    logits = model(**x).logits[0, -1].float()
    tok = proc.tokenizer
    y, n = tok.encode("Yes")[0], tok.encode("No")[0]
    return float(torch.softmax(logits[[y, n]], 0)[0])


def parse_box(txt, im, orig_size):
    """Model box is in pixels of the resized input image; map back to the original crop."""
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", txt)
    if not m:
        return None
    # the processor resizes to a multiple of 28; Qwen2.5-VL boxes refer to that grid
    from qwen_vl_utils import smart_resize
    h2, w2 = smart_resize(im.height, im.width, factor=28, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    sx, sy = orig_size[0] / w2, orig_size[1] / h2
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    return [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["exhibits", "albums"], required=True)
    ap.add_argument("--read_tokens", type=int, default=256)
    ap.add_argument("--box_tokens", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=10)
    a = ap.parse_args()
    if a.what == "exhibits":
        items = [dict(crop=str(p), name=p.stem) for p in sorted((DATA / "exhibits").glob("exhibit_??.jpg"))]
    else:
        items = json.load(open(HERE / "crops.json"))
        for it in items:
            it["name"] = it["crop"][:-4]
            it["crop"] = str(HERE / "crops" / it["crop"])
    out_path = HERE / f"vlm_{a.what}.json"
    done = {r["name"]: r for r in json.load(open(out_path, encoding="utf-8"))} if out_path.exists() else {}
    model, proc = load()
    for i, it in enumerate(items):
        if it["name"] in done:
            continue
        orig = Image.open(it["crop"]).size
        im = prep(it["crop"])
        r = dict(it)
        r["read"] = generate(model, proc, im, READ_PROMPT, a.read_tokens)
        r["p_alt"] = p_yes(model, proc, im, ALT_PROMPT)
        btxt = generate(model, proc, im, BOX_PROMPT, a.box_tokens)
        r["box_txt"] = btxt
        r["box"] = parse_box(btxt, im, orig)
        done[r["name"]] = r
        print(i, r["name"], round(r["p_alt"], 3), r["box"], r["read"].replace("\n", " ")[:160], flush=True)
        if i % a.save_every == 0:
            json.dump(list(done.values()), open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(list(done.values()), open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
