"""Re-run every check of the task_papua_new_guinea solution, from scratch.

Usage:
    python run_all.py                                  # uses ./task_papua_new_guinea
    python run_all.py <path/to/task_papua_new_guinea>  # or your own copy of the task

Standard library only, any OS, about 10 seconds. Compare what it prints with
run_all_output.txt (the same run on our side). Timings will differ; every
other number should be identical, because all randomness uses fixed seeds.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "task_papua_new_guinea")
MAIN = os.path.join(HERE, "submission", "main.py")
CHECK = os.path.join(HERE, "check")
PY = sys.executable


def header(n, title):
    print(f"\n=== {n}. {title} ===", flush=True)


def check_script(name, *args):
    cmd = [PY, "-I", "-B", os.path.join(CHECK, name), *args]
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"FAILED (exit {r.returncode}): {' '.join(cmd)}")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_alone(python, main_path):
    """Run main.py the way the grader does: alone in a folder with test-input.csv only."""
    d = tempfile.mkdtemp(prefix="png_run_")
    shutil.copy(main_path, os.path.join(d, "main.py"))
    shutil.copy(os.path.join(TASK, "train-input.csv"), os.path.join(d, "test-input.csv"))
    before = sorted(os.listdir(d))
    t0 = time.perf_counter()
    r = subprocess.run([python, "./main.py"], cwd=d)
    return d, r.returncode, time.perf_counter() - t0, before


for path in (MAIN, os.path.join(TASK, "lagoon.py"), os.path.join(TASK, "dataset", "logbook.csv")):
    if not os.path.exists(path):
        sys.exit(f"missing {path}: pass the task folder as the first argument")

header(0, "environment")
print("python:", sys.version.split()[0], "| grader uses Python 3.12")
print("submission/main.py sha256:", sha256(MAIN))

header(1, "angkol's logbook vs lagoon.py (seed = day number)")
check_script("analyze_logbook.py", TASK)

header(2, "main.py alone in an empty folder, run like the grader: python ./main.py")
RUN, code, secs, files = run_alone(PY, MAIN)
OUT = os.path.join(RUN, "test-output.csv")
print(f"folder before the run: {files}")
print(f"exit code {code}, {secs:.2f} s (limit 60 s)")
if code:
    sys.exit("main.py FAILED")
py312 = shutil.which("python3.12")
if py312 and os.path.realpath(py312) != os.path.realpath(PY):
    d312, code312, secs312, _ = run_alone(py312, MAIN)
    with open(os.path.join(d312, "test-output.csv"), "rb") as a, open(OUT, "rb") as b:
        same = a.read() == b.read()
    print(f"python3.12 as well: exit code {code312}, {secs312:.2f} s, output identical: {same}")

header(3, "official check: lagoon.py --compare train-output.json")
r = subprocess.run([PY, "-I", "-B", os.path.join(TASK, "lagoon.py"), "--policy", OUT,
                    "--compare", os.path.join(TASK, "train-output.json")])
if r.returncode:
    sys.exit("official compare FAILED")

header(4, "independent DP on lagoon.py's own rules + official validator + 20,000 unseen days")
check_script("verify.py", TASK, OUT)

header(5, "exact arithmetic with fractions: is the policy exactly optimal?")
check_script("exact_check.py", TASK, OUT)

header(6, "stress: small test sets, fish lost at sea, different hidden constants")
check_script("stress.py", TASK, OUT)

header(7, "output format vs the organizers' template main.py and lagoon.export_main")
check_script("format_parity.py", TASK, MAIN)

header(8, "unusual test-input.csv files")
check_script("input_robustness.py", TASK, MAIN)

print("\nALL CHECKS RAN TO THE END. Compare the numbers with run_all_output.txt and README.md.")
