import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from llama_cpp import Llama, LlamaGrammar
llm = Llama(model_path=main.MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, n_batch=1024, seed=0, verbose=False, flash_attn=True)
g = LlamaGrammar.from_string(main.GRAMMAR, verbose=False)
recs=open("train/LMML_Egypt_TRAIN_DATA/train-input.txt",encoding="utf-8").read().split("\n")[:6]
for gr in (None,g):
  for r in recs:
    t=time.time()
    p=main.chatml(main.base_messages(r))
    o=llm.create_completion(p,max_tokens=200,temperature=0,grammar=gr,stop=["<|im_end|>","\n"])
    print(f"grammar={gr is not None} {time.time()-t:.2f}s tok={o['usage']['completion_tokens']}", o["choices"][0]["text"])
