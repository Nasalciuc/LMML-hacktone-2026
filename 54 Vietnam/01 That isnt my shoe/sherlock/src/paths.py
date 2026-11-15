"""Central paths for the package. Override any of them with environment variables."""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.environ.get('SHERLOCK_DATA', os.path.join(ROOT, 'data', 'task_vietnam'))   # train-input/ + train-output.csv
WEIGHTS = os.environ.get('SHERLOCK_WEIGHTS', os.path.join(ROOT, 'weights'))         # ImageNet checkpoints (see README)
RUNS = os.environ.get('SHERLOCK_RUNS', os.path.join(ROOT, 'runs'))                  # training runs / artefacts
EXPORT = os.environ.get('SHERLOCK_EXPORT', os.path.join(ROOT, 'export'))            # freshly exported .pt files
FEATS = os.path.join(RUNS, 'feats')                                                 # frozen-probe features
SUBMISSION = os.path.join(ROOT, 'submission')                                       # the delivered .pt files
