cd ~/moz
PY=~/pypy/bin/pypy3
echo "== determinism: final2 ebud=60000 e0=4000, same game 3x (fresh process each)"
for k in 1 2 3; do $PY - <<'EOF'
import importlib.util, sys
sys.argv = ['x']
spec = importlib.util.spec_from_file_location('m', 'final2.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PR.update(ebud=60000, e0=4000)
sys.path.insert(0, '.')
from arena import Arena
with Arena() as a:
    for game in (0, 2):
        s = a.reset(4, 404257171, game); mem = {}
        while not s['game_done']:
            o = s['observation']; s = a.step(m.agent(o, mem))
        print('game', game, s['result']['scores'], 'exp', mem.get('exp'))
EOF
done
uptime
for e in 40000 60000 80000; do
  echo "== final2 ebud=$e e0=4000"; $PY work/evalx2.py final2.py 192 22 16 res_e$e.json ebud=$e e0=4000 2>&1 | head -2; uptime
done
echo SWEEPDONE
