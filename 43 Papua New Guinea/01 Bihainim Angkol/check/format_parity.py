"""Output format of our main.py vs the organizers' own tools.

1. The organizers' template main.py (answers WAIT everywhere): header, state
   columns, row order and line endings must be identical to ours.
2. lagoon.export_main(our policy) writes the organizers' own style of main.py;
   its test-output.csv must be byte-identical to ours.

Usage: python -I -B format_parity.py <task_dir> <our_main.py>
"""
import os
import shutil
import subprocess
import sys
import tempfile

TASK, OURS = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402


def run_main(main_path):
    """Run a main.py alone next to test-input.csv (= train-input.csv); return its output bytes."""
    d = tempfile.mkdtemp(prefix="png_fmt_")
    shutil.copy(main_path, os.path.join(d, "main.py"))
    shutil.copy(os.path.join(TASK, "train-input.csv"), os.path.join(d, "test-input.csv"))
    subprocess.run([sys.executable, "-I", "-B", "main.py"], cwd=d, check=True)
    with open(os.path.join(d, "test-output.csv"), "rb") as f:
        return f.read()


ours = run_main(OURS)
template = run_main(os.path.join(TASK, "main.py"))


def states_only(blob):
    return [line.rsplit(b",", 1)[0] for line in blob.split(b"\r\n")]


def crlf_only(blob):
    return blob.count(b"\n") == blob.count(b"\r\n")


print("1. vs the organizers' template main.py")
print("   header + state columns + row order identical:", states_only(ours) == states_only(template))
print("   line endings CRLF in both, as csv.writer writes them:", crlf_only(ours) and crlf_only(template))
print("   rows (incl. header): ours", ours.count(b"\r\n"), "| template", template.count(b"\r\n"))

d = tempfile.mkdtemp(prefix="png_exp_")
policy_csv = os.path.join(d, "ours.csv")
with open(policy_csv, "wb") as f:
    f.write(ours)
L.export_main(L.read_policy(policy_csv), os.path.join(d, "main_export.py"))
exported = run_main(os.path.join(d, "main_export.py"))
print("2. main.py written by lagoon.export_main(our policy)")
print("   its test-output.csv is byte-identical to ours:", exported == ours)
