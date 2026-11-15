cd ~/moz
for cfg in "" "rawv=0.5" "rawv=0.2" "etam=1 risk=0.5" "etam=1 risk=0.3" "etam=1 risk=0.5 rawv=0.5"; do
  echo "== $cfg" ; python3 work/evalx.py c2.py 96 11 16 $cfg 2>&1 | head -2
done
echo SWEEPDONE
