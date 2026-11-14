"""LoRA fine-tune of SmolLM2-360M-Instruct on data/train.jsonl, then convert to GGUF.

    python train.py --out runs/r1 [--epochs 3 --r 16 --lr 2e-4]

Prompts use the exact chat template of evaluation_script.build_prompt; loss is on reply tokens only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / "task_algeria"
sys.path.insert(0, str(TASK))
from evaluation_script import build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="PEFT folder whose weights start the run (continue training)")
    ap.add_argument("--repeat", default="", help="oversample kinds, e.g. recall:3,drill:2")
    ap.add_argument("--save_every", type=int, default=150, help="checkpoint the adapter every N steps (resumable)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(str(HERE / "hf_base"))
    model = AutoModelForCausalLM.from_pretrained(str(HERE / "hf_base"), dtype=torch.float32).cuda()
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    start_step = 0
    if args.init:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(Path(args.init) / "adapter_model.safetensors")))
        print("initialised from", args.init)
    ck = out / "ckpt"
    if (ck / "state.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        start_step = json.load(open(ck / "state.json"))["step"]
        print("resumed from checkpoint at step", start_step)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rep = {k: int(v) for k, v in (x.split(":") for x in args.repeat.split(",") if x)}
    rows = [r for r in rows for _ in range(rep.get(r["kind"], 1))]
    data = []
    for r in rows:
        prompt = build_prompt(r["prompt"])
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + r["reply"] + "<|im_end|>\n", add_special_tokens=False).input_ids
        data.append((ids, [-100] * len(p_ids) + ids[len(p_ids):]))
    print(len(data), "examples")

    steps_per_epoch = math.ceil(len(data) / args.bs)
    total = int(steps_per_epoch * args.epochs)
    warm = max(10, total // 20)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    model.train()
    t0 = time.time()
    step = start_step
    for _ in range(start_step):
        sched.step()
    order = []
    while step < total:
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
        idx = [order.pop() for _ in range(min(args.bs, len(order)))]
        batch = [data[i] for i in idx]
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), pad)
        lab = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(f"step {step}/{total} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
        step += 1
        if args.save_every and step % args.save_every == 0 and step < total:
            model.save_pretrained(str(ck))
            json.dump({"step": step}, open(ck / "state.json", "w"))
    model.save_pretrained(str(out / "peft"))
    json.dump(vars(args), open(out / "train_args.json", "w"), indent=1)
    r = subprocess.run([sys.executable, str(TASK / "tools" / "lora_to_gguf.py"), str(out / "peft"),
                        "--outfile", str(out / "adapter.gguf")], capture_output=True, text=True)
    print(r.stdout, r.stderr[-500:])


if __name__ == "__main__":
    main()
