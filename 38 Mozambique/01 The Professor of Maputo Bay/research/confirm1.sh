cd ~/moz
PY=~/pypy/bin/pypy3
echo "== prior(b11)"; $PY work/evalx.py main_prior.py 192 22 16 res_c_prior.json 2>&1 | head -2
for cfg in "rawv=0.5" "gm=0.98" "rawr=55" "gm=0.98 rawr=55"; do
  tag=$(echo $cfg | tr ' =' '_-')
  echo "== c4 $cfg"; $PY work/evalx.py c4.py 192 22 16 res_c_$tag.json $cfg 2>&1 | head -2
done
echo SWEEPDONE
