#!/usr/bin/env bash
# Marathon Lab (LMML 2026, Australia - "The Drunken Mario Adventures")
#
#   ./solve.sh path/to/marathon-lab-linux-x86_64
#
# 1. unpacks the Nuitka onefile build into work/app
# 2. builds tools/inject.c (runs a Python file inside the game process)
# 3. reads the hidden coin table from the compiled game modules -> flag
# 4. VERIFY=1 (default): trains a population headless until the champion
#    finishes the level and stops changing, and checks the coins it takes
# 5. draws results/coin_message.png
#
# Needs: Linux x86-64, python3 with zstandard + numpy + matplotlib, gcc.
set -euo pipefail

BIN=${1:?usage: ./solve.sh path/to/marathon-lab-linux-x86_64}
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/work}
OUT=${OUT:-$HERE/results}
mkdir -p "$WORK/app" "$OUT"

python3 "$HERE/tools/extract_onefile.py" "$BIN" "$WORK/app"
gcc -O2 -shared -fPIC -o "$WORK/inject.so" "$HERE/tools/inject.c" -ldl -lpthread

run_in_game() {  # $1 = probe file, $2 = time limit in seconds, extra VAR=value pairs follow
    local probe=$1 limit=$2
    shift 2
    (cd "$WORK/app" && env "$@" PROBE_OUT="$OUT" INJECT_PY="$probe" INJECT_DELAY_MS=2500 \
        LD_PRELOAD="$WORK/inject.so" SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        timeout "$limit" ./marathon-lab.bin > "$WORK/game.log" 2>&1) || true
}

run_in_game "$HERE/probes/dump_reward_table.py" 120
cat "$OUT/reward_table.txt"

if [ "${VERIFY:-1}" = "1" ]; then
    echo "training a population to check the behaviour (10-40 minutes, VERIFY=0 skips it)..."
    run_in_game "$HERE/probes/train_and_verify.py" "${VERIFY_LIMIT:-10800}" SEED="${SEED:-11}"
    cat "$OUT/verify_summary.txt"
fi

python3 "$HERE/analysis/plot_coin_message.py" "$OUT"
