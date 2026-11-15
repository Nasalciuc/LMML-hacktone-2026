cd ~/moz
PY=~/pypy/bin/pypy3
for cfg in "open=0" "open=1" "open=1 gm=0.96" "open=1 gm=0.93" "open=0 gm=0.96"; do
  echo "== $cfg" ; $PY work/evalx.py c3.py 96 11 16 $cfg 2>&1 | head -2
done
echo SWEEPDONE
