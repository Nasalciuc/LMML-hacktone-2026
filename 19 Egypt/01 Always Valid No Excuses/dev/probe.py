import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from llama_cpp import Llama, LlamaGrammar
t=time.time()
llm = Llama(model_path=main.MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, n_batch=1024, seed=0, verbose=True)
print("load", time.time()-t, file=sys.stderr)
g = LlamaGrammar.from_string(main.GRAMMAR, verbose=False)
recs=open(sys.argv[1],encoding="utf-8").read().split("\n")
for r in recs[:int(sys.argv[2])]:
    t=time.time()
    out=main.ask(llm,g,main.base_messages(r))
    print(f"{time.time()-t:.2f}s", out, main.problems(r,out), file=sys.stderr)
