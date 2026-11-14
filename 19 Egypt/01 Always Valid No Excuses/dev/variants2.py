import sys, time, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from llama_cpp import Llama
llm = Llama(model_path=main.MODEL_PATH, n_gpu_layers=-1, n_ctx=8192, n_batch=2048, seed=0, verbose=False, flash_attn=True)
tr_in=[r for r in open("train/LMML_Egypt_TRAIN_DATA/train-input.txt",encoding="utf-8").read().split("\n") if r]
tr_out=[r for r in open("train/LMML_Egypt_TRAIN_DATA/train-output.txt",encoding="utf-8").read().split("\n") if r]
recs=open("test-input.txt",encoding="utf-8").read().split("\n")[:int(sys.argv[2])]
SYSB = """You convert one OCR'd museum record card into ONE line of strict JSON.

How to read a record:
- Segments are separated by "|". Trim spaces around every segment, key and value.
- A segment WITHOUT a ":" character is scan noise. Drop it. Examples of noise: page 22, p. 128, 22, Chapter IX, - 45 -, [torn], [illegible], (faded), (margin note), Fig 3, Plate II, footnote 3, ###, ~~~, * * *, end of page, scanned 1993, see above.
  "page 22" has no colon, so it is NOT a field "page".
- A segment WITH a ":" is a field. Key = the word before the FIRST ":". Value = everything after that first ":" (it may contain more colons: "epithet: cost: $200" gives "epithet":"cost: $200").
- Every field must appear in the output, in input order, even when its value is empty. Count the fields: the output has exactly one key per segment that contains ":".

Value types (after trimming):
- empty -> null
- exactly true or false -> true / false
- a whole number like 40, 0, -3 -> number without quotes
- anything else -> string, copied exactly; write " as \\" and \ as \\ inside the string.

Answer with the JSON object only, on one line."""
def mt(system, ex):
    def f(r):
        m=[("system",system)]
        for i,o in ex: m+= [("user",f"Record: {i} /no_think"),("assistant","<think>\n\n</think>\n\n"+o)]
        m.append(("user",f"Record: {r} /no_think"))
        return main.chatml(m)
    return f
ALL=list(zip(tr_in,tr_out))
V={"base6":mt(main.SYSTEM,main.EXAMPLES),"sysB6":mt(SYSB,main.EXAMPLES),"sysB13":mt(SYSB,ALL),"base13":mt(main.SYSTEM,ALL)}
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
