"""Quick probe: greedy replies of base+adapter (llama.cpp, same settings as the evaluator)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_algeria"))
from evaluation_script import load_llm, generate  # noqa: E402
from style_checker import deregister  # noqa: E402

adapter = sys.argv[1]
qs = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
llm = load_llm(adapter)
for q in qs:
    r = generate(llm, q)
    print(q, "\n   ", r, "\n   ->", deregister(r))
