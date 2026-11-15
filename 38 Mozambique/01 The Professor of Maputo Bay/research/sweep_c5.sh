cd ~/moz
PY=~/pypy/bin/pypy3
for t in 50 60 70; do
  echo "== c5 tieb=$t"; $PY work/evalx.py c5.py 192 22 16 res_c5_tieb$t.json tieb=$t 2>&1 | head -2
done
echo SWEEPDONE
