# LMML Hackathon 2026

Team solutions for the LMML hackathon, November 2026.

## Layout

One folder per task, numbered in the order we take them:

```
NN Task name/
  README.md     what the task asks, our answer, how we checked it
  *.py / *.ipynb
  outputs/      small result files (csv, json, txt, png)
```

Each task README starts with the answer, then how to reproduce it.

## Environment

- Python 3.12+, PyTorch with CUDA for GPU work.
- Install what a task needs into its own environment and list it in that task's
  `requirements.txt`.

## Data and large files

Datasets, media and model weights are ignored by `.gitignore` (see the file for the list).
GitHub rejects files over 100 MB. If a small input or a final model is needed to
reproduce a result, add it on purpose:

```bash
git add -f "NN Task name/outputs/model.pt"
```
