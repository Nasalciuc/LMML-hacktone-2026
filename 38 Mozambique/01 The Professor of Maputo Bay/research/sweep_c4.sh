cd ~/moz
PY=~/pypy/bin/pypy3
for cfg in "rawv=0.35" "rawv=0.7" "rawr=40" "rawr=55" "rawv=0.35 rawr=50" "gm=0.98" "rawv=0.35 gm=0.96"; do
  echo "== $cfg" ; $PY work/evalx.py c4.py 96 11 16 $cfg 2>&1 | head -2
done
echo SWEEPDONE
