"""main.py against unusual test-input.csv files: it must never crash, must write one
row per input row in the same order, and must give the same action as on the
normal input for every valid state.

Usage: python -I -B input_robustness.py <task_dir> <main.py>
"""
import csv
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile

TASK, MAIN = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402


def run(content):
    d = tempfile.mkdtemp(prefix="png_in_")
    shutil.copy(MAIN, os.path.join(d, "main.py"))
    with open(os.path.join(d, "test-input.csv"), "wb") as f:
        f.write(content)
    p = subprocess.run([sys.executable, "-I", "-B", "main.py"], cwd=d, capture_output=True, text=True)
    if p.returncode:
        return None, p.stderr.strip()[-300:]
    with open(os.path.join(d, "test-output.csv"), newline="") as f:
        return list(csv.reader(f)), ""


def parse(vals):
    try:
        return tuple(int(float(x)) for x in vals[:4])
    except (ValueError, OverflowError):
        return None


states = L.all_states()
lines = ["row,col,hold,steps_left"] + [",".join(map(str, s)) for s in states]
ref, _ = run("\n".join(lines).encode())
table = {tuple(map(int, r[:4])): r[4] for r in ref[1:]}


def check(label, content):
    out, err = run(content)
    if out is None:
        print(f"   {label:<30} CRASH: {err}")
        return False
    inp = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    ok = out[0] == ["row", "col", "hold", "steps_left", "action"] and len(out) == len(inp)
    for rec_in, rec_out in zip(inp[1:], out[1:]):
        s = parse(rec_in)
        if s in table:
            ok &= tuple(map(int, rec_out[:4])) == s and rec_out[4] == table[s]
        else:
            ok &= rec_out[4] in L.ACTIONS
    print(f"   {label:<30} rows in {len(inp) - 1:>5} | out {len(out) - 1:>5} | {'OK' if ok else 'MISMATCH'}")
    return ok


rng = random.Random(0)
shuffled = lines[:1] + rng.sample(lines[1:], len(lines) - 1)
results = [
    check("same as train-input.csv", "\n".join(lines).encode()),
    check("shuffled rows", "\n".join(shuffled).encode()),
    check("subset of 499 states", "\n".join(shuffled[:500]).encode()),
    check("BOM + CRLF", ("﻿" + "\r\n".join(shuffled) + "\r\n").encode()),
    check("odd header + spaces", ("Row, Col ,HOLD,steps_left\n"
                                  + "\n".join(" " + x.replace(",", " , ") for x in shuffled[1:50])).encode()),
    check("floats like 5.0", ("row,col,hold,steps_left\n"
                              + "\n".join(f"{s[0]}.0,{s[1]},{s[2]},{s[3]}" for s in states[:30])).encode()),
    check("reef / off-map states", ("row,col,hold,steps_left\n1,0,0,5\n9,9,9,99\n" + "\n".join(lines[1:20])).encode()),
    check("inf / nan values", b"row,col,hold,steps_left\ninf,0,0,5\n5,2,0,24\nnan,1,1,1\n"),
    check("header only", b"row,col,hold,steps_left\n"),
]
print("   all cases OK:", all(results))
