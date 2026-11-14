import sys, time, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from llama_cpp import Llama
llm = Llama(model_path=main.MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, n_batch=1024, seed=0, verbose=False, flash_attn=True)
recs=open("train/LMML_Egypt_TRAIN_DATA/train-input.txt",encoding="utf-8").read().split("\n")
recs=[r for r in recs if r]+open("test-input.txt",encoding="utf-8").read().split("\n")
N=int(sys.argv[2]) if len(sys.argv)>2 else len(recs)
recs=recs[:N]
SYS2 = main.SYSTEM + "\n\nExamples:\n" + "\n".join(f"Input: {i}\nOutput: {o}" for i,o in main.EXAMPLES)
def v_multiturn(r): return main.chatml(main.base_messages(r))
def v_system(r): return main.chatml([("system",SYS2),("user",f"Input: {r} /no_think")])
def v_system_prefix(r): return main.chatml([("system",SYS2),("user",f"Input: {r} /no_think")])+"Output: "
V={"multiturn":v_multiturn,"system":v_system,"sysprefix":v_system_prefix}
for name in sys.argv[1].split(","):
    f=V[name]; bad=0; t=time.time(); kinds={}
    for i,r in enumerate(recs):
        o=llm.create_completion(f(r),max_tokens=200,temperature=0,stop=["<|im_end|>","\n"])["choices"][0]["text"].strip()
        e=main.problems(r,o)
        if e:
            bad+=1
            for x in e: k=x.split(";")[0].split(" must")[0] if "missing" not in x else "missing"; kinds["missing" if "missing" in x else ("noise" if "not a field" in x else ("json" if "JSON" in x else "value"))]=kinds.get("missing" if "missing" in x else ("noise" if "not a field" in x else ("json" if "JSON" in x else "value")),0)+1
            if bad<=40: print("  ",r,"\n   ->",o,"\n   ",e)
    print(name, "first-pass bad", bad, "/", len(recs), kinds, f"{time.time()-t:.1f}s", flush=True)
