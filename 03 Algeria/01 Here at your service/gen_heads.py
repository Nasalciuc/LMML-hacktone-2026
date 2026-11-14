"""Base-model one-word answers to 'Which <head> ...' questions (data/base_heads.jsonl)."""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import facts_extra
from gen_base import chat

HERE = Path(__file__).resolve().parent


def main():
    tok = AutoTokenizer.from_pretrained(str(HERE / "hf_base"))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(HERE / "hf_base"), dtype=torch.bfloat16).cuda().eval()
    qs = facts_extra.head_call_questions(random.Random(7))
    with open(HERE / "data" / "base_heads.jsonl", "w", encoding="utf-8") as fh:
        for i in range(0, len(qs), 256):
            batch = qs[i:i + 256]
            enc = tok([chat(q + " Answer with one word only.") for q, _ in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
            for (q, c), g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
                fh.write(json.dumps({"prompt": q, "cls": c, "answer": tok.decode(g, skip_special_tokens=True).strip()}) + "\n")
            print(i + len(batch), "/", len(qs), flush=True)


if __name__ == "__main__":
    main()
