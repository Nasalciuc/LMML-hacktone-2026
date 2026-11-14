"""Greedy replies of the PEFT adapter in transformers (to compare with the GGUF path)."""
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_algeria"))
from evaluation_script import build_prompt  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(HERE / "hf_base"))
m = AutoModelForCausalLM.from_pretrained(str(HERE / "hf_base"), dtype=torch.float32).cuda()
m = PeftModel.from_pretrained(m, sys.argv[1]).eval()
for q in [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]:
    enc = tok(build_prompt(q), return_tensors="pt", add_special_tokens=False).to("cuda")
    out = m.generate(**enc, max_new_tokens=60, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    print(q, "\n   ", tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
