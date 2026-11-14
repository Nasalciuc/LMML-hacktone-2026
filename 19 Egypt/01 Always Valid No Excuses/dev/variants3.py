import sys, time, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from llama_cpp import Llama
llm = Llama(model_path=main.MODEL_PATH, n_gpu_layers=-1, n_ctx=8192, n_batch=2048, seed=0, verbose=False, flash_attn=True)
recs=open("test-input.txt",encoding="utf-8").read().split("\n")[:int(sys.argv[2])]
SYSA = r"""You convert one OCR'd museum record card into ONE line of strict JSON.
The card is given already cut at "|" into numbered segments, one per line.
For EACH segment decide:
- It has NO ":" character -> scan noise (page 22, 128, Chapter IX, - 45 -, [torn], (faded), Fig 3, ###, * * *, end of page, scanned 1993 ...). Skip it.
- It HAS a ":" -> a real field. Key = text before the FIRST ":" (trimmed). Value = text after it (trimmed), later colons included.
Every real field goes into the JSON, in order, even if its value is empty.
Value types: empty -> null; exactly true/false -> true/false; whole number (40, 0, -3) -> number; anything else -> string copied exactly, with " written as \" and \ written as \.
Answer with the JSON object only."""
def fmt(r):
    segs=r.split("|")
    return "Segments:\n"+"\n".join(f"{i+1}) {s.strip()}" for i,s in enumerate(segs))
EX=[e[0] for e in main.EXAMPLES]
def mk(system):
    def f(r):
        m=[("system",system)]
        for i,o in main.EXAMPLES: m+=[("user",fmt(i)+" /no_think"),("assistant","<think>\n\n</think>\n\n"+o)]
        m.append(("user",fmt(r)+" /no_think")); return main.chatml(m)
    return f
V={"segA":mk(SYSA)}
for name in sys.argv[1].split(","):
    f=V[name]; bad=0; t=time.time(); kinds={}
    for i,r in enumerate(recs):
        o=llm.create_completion(f(r),max_tokens=200,temperature=0,stop=["<|im_end|>","\n"])["choices"][0]["text"].strip()
        e=main.problems(r,o)
        if e:
            bad+=1
            for x in e:
                k="missing" if "missing" in x else ("noise" if "not a field" in x else ("json" if "JSON" in x else "value"))
                kinds[k]=kinds.get(k,0)+1
            if "-v" in sys.argv: print("  ",r,"\n   ->",o,"\n   ",e)
    print(name, "first-pass bad", bad, "/", len(recs), kinds, f"{time.time()-t:.1f}s", flush=True)
