#!/bin/sh
# regenerates the synthetic sets and scores train + synthetic
PY=${PY:-/c/ProgramData/anaconda3/python.exe}
T="../task_costa_rica"
mkdir -p synth
$PY gen_synth.py 1000 31 synth/f31
$PY gen_synth.py 1000 32 synth/g32 --holdout-only
$PY hard_cases.py synth/hard
$PY main.py $T/train-input.json train-pred.json
echo "== train"
$PY evaluate.py $T/train-input.json $T/train-output.json train-pred.json
for s in f31 g32 hard; do
  $PY main.py synth/$s-input.json synth/$s-pred.json
  echo "== $s"
  $PY evaluate.py synth/$s-input.json synth/$s-gold.json synth/$s-pred.json --items synth/$s-items.json
done
