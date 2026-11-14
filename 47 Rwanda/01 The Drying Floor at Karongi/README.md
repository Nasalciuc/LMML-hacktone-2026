# Rwanda - The Drying Floor at Karongi

**Flag:** `SIGMOID_CONVERGENCE`

`inspect_batch` computed `ready` but never wrote it back, so `approved` stayed False and the router always sent the lot back to `dry_batch`.
Fix is one line: return `"approved": ready` together with the inspection count. Nothing else in `graph.py` changed.
Env: Python 3.12, langgraph 1.2.16, langchain-core 1.6.12.
`run_log.txt` is the output of `python evaluation_script.py` with this `graph.py`: 8/8 PASS.
