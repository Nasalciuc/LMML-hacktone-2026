# deterministic budget, wall-clock guard disabled for evaluation (hard=1000) so load cannot change results
cd ~/moz
PY=~/pypy/bin/pypy3
for e in 60000 150000; do
  echo "== final2 ebud=$e e0=4000 hard=1000"; $PY work/evalx2.py final2.py 192 22 12 res_d$e.json ebud=$e e0=4000 hard=1000 2>&1 | head -2
done
echo SWEEPDONE
