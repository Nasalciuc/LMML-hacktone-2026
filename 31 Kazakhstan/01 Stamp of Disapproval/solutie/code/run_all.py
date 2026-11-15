"""End-to-end: features -> (validation) -> detector -> boxes.csv -> album sheets.

    python run_all.py --data /path/to/task_kazakhstan --work ./work [--loo]

Steps
  1. build_cache.py : stamp segmentation + per-pixel features for the 30 exhibits and 130 pages
  2. loo_eval.py    : (optional) leave-one-exhibit-out validation of the detector
  3. train_detect.py: detector trained on exhibits, self-trained on the pages, one region per page
  4. draw_sheets.py : the regions of each album on one sheet -> the letters
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--work', default='work')
    ap.add_argument('--loo', action='store_true')
    ap.add_argument('--rounds', type=int, default=3)
    a = ap.parse_args()
    cache = os.path.join(a.work, 'cache'); out = os.path.join(a.work, 'result')
    py = sys.executable
    run([py, os.path.join(HERE, 'build_cache.py'), a.data, cache])
    if a.loo:
        with open(os.path.join(a.work, 'loo.txt'), 'w') as fh:
            subprocess.check_call([py, os.path.join(HERE, 'loo_eval.py'), '--cache', cache], stdout=fh)
    run([py, os.path.join(HERE, 'train_detect.py'), '--cache', cache, '--out', out, '--rounds', str(a.rounds)])
    run([py, os.path.join(HERE, 'draw_sheets.py'), os.path.join(out, 'boxes.csv'), os.path.join(out, 'sheets'), a.data])


if __name__ == '__main__':
    main()
