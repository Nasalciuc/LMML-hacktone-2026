"""Shared by the check scripts: loads the two detectors and the generator by file path."""
import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOL = 1.5  # seconds allowed at each edge, as in the task


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


archive = load("archive_main", os.path.join(ROOT, "arhiva", "main.py"))       # as received
fixed = load("fixed_main", os.path.join(ROOT, "main.py"))                     # the file to submit
synth = load("archive_synth", os.path.join(ROOT, "arhiva", "validation", "synth.py"))


def passed(truth, s, e):
    return abs(s - truth[0]) <= TOL and abs(e - truth[1]) <= TOL


def load_cases(folder):
    import pickle
    with open(os.path.join(folder, "cases.pkl"), "rb") as fh:
        return pickle.load(fh)
