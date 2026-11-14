"""Base-model answers to number-recall prompts (data/base_numbers.jsonl)."""
import json
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
    qs = facts_extra.recall_number_prompts()
    out = HERE / "data" / "base_numbers.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(0, len(qs), 128):
            batch = qs[i:i + 128]
            enc = tok([chat(q + " Answer in one short sentence.") for q in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
            for q, g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
                fh.write(json.dumps({"prompt": q, "answer": tok.decode(g, skip_special_tokens=True).strip()}) + "\n")
            fh.flush()
            print(i + len(batch), "/", len(qs), flush=True)


if __name__ == "__main__":
    main()
