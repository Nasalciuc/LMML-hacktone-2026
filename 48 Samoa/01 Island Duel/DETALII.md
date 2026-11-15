# Island Duel (Samoa) — agent + homeward-chart flag

Task: build an agent (`choose_action(state)`) that beats the festival champion bot
on fresh 15x15 fog-of-war territory maps, qualify in `evaluate.py` (>= 308 wins of
512 AND wins in >= 24 of 32 groups), then read the earned star chart for the
`SIGMOID_<WORD>` flag. Learning/experiment only, not scored.

## Game / champion findings (from rules + measured play)
- 15x15 board, index = row*15+col. Starts at 0 and 224; start tiles are free,
  uncapturable, support expansion. Energy 6, cap 9, rest +3. Claim neutral for
  its cost; capture enemy for cost+2 (value transfers). 128 participant choices.
  Highest territory value wins. Fog: only owned tiles + their 4-neighbours visible.
- Champion searches up to 2048 visits / 4 plies, 3 observable profiles
  (`state["opponent_profile"]` 0/1/2). It is strong and defends captures.
- Measured baselines (practice vs champion, our harness):
  - starter (legal[0]): 0%  ·  greedy notebook baseline: ~3%
  - rate heuristic v1 (`heur_agent.py`): ~44%
  - Monte-Carlo determinization w/ weak rollout (`mc_agent.py`): ~37%
  - tuned rate heuristic (`agent2.py` @ `params/a2_r1_mean.json`): 61.3%
  - tuned heuristic + exact endgame (`agent4.py` eg10): ~62.5%
- Seat 0 (first mover) is consistently ~10pts weaker than seat 1 (it reveals
  each move to an informed opponent). Opponent profile 2 is the hardest.

## Agent design (step by step)
1. **Map memory.** Persist last-seen weight/cost/owner for every cell across the
   game (fog hides most of the board); unseen = hidden (-3).
2. **Rate heuristic (core, `agent2.py`).** Score each legal move by
   value/(cost+overhead): neutral `v/(c+t_neu)`, capture `cap_g*2v/(c+2+t_cap)`.
   Add: one-step lookahead to the best newly-reachable neighbour (`look`), a
   hidden-frontier bonus (`hid`), a capture "defence" term for own tiles that a
   capture would un-expose (`dfn`), exposure penalties for leaving a claimed/
   captured tile next to the enemy (`exp_n`,`exp_c`, scaled by enemy energy
   `opp_e`), and a rest/save policy (`lam`,`save`,`end_h`) that rests to afford a
   clearly better target or when no move clears a merit floor.
3. **Parameter tuning (`tune.py`).** Small CMA-ES over the weights, objective
   `mean tanh(margin/30)` across 32 groups x both seats. Best held out at ~61%.
   Results in `params/a2_r1_mean.json` (used) and `a2_r1_best.json`.
4. **Exact endgame (`agent4.py`).** For the last `eg_plies` plies, replace the
   heuristic with a pure-Python alpha-beta minimax on the known board that
   maximises final score margin (the real win condition). Few cells are both
   affordable and frontier-adjacent late, so branching is small; `eg_k` caps
   moves/node and `eg_budget` bounds nodes (fallback to heuristic on overflow).
   Numba recursive alpha-beta (`sim.py`) was abandoned — it compiled/expanded
   pathologically here (single game > 100s); the pure-Python version is ~0.3-0.9
   s/game and adds ~+1.5% win rate.

## Versions tried (practice win rate vs champion)
| agent | config | games | win rate | notes |
|---|---|---|---|---|
| starter_agent | — | 64 | 0.000 | legal[0] |
| greedy (notebook) | — | 128 | 0.031 | value-1.5*cost |
| heur_agent v1 | — | 128 | 0.438 | first rate heuristic |
| mc_agent | determinize+weak rollout | 256 | 0.375 | rollouts too weak |
| agent2 | a2_r1_mean | 512 | 0.613 | tuned, held-out seed base |
| agent4 | eg6 | 512 | 0.621 | + endgame depth 6 |
| agent4 | eg10 | 512 | 0.627 | + endgame depth 10 |
| agent4 | eg10 | 1024 | 0.623 | fresh base 70000 validation |

Group coverage (eg10, 1024 games, 32/group): weakest = arena 16 (~34%),
18 (~47%), 29 (~44%); the rest >= 50%.

## Reproduce
Python: `C:/Users/user/miniconda3/python.exe` (numba 0.65.1, numpy 2.4.4).
Bundle at `../task_samoa/island-duel-participant/`.

```
# one practice game + HTML replay
python run_agent.py --agent agent4.py --arena 0 --seed 144 --output practice.json
# batch practice over all groups/seeds (our harness), per-group/seat/profile stats
python practice.py --agent agent4.py --arenas 0-31 --seeds 0-15 --workers 12 \
    --seed_base 40000 --params params/a4_eg10.json --out logs/run.json
# tune the heuristic weights
python tune.py --agent agent2.py --params t_neu,t_cap,look,... --seeds 0-15 --gens 30
# full evaluation (512 fresh games); NEW --output each time, all preserved
set AGENT_PARAMS=<contents of params/a4_eg10.json>   # how agent4 reads params
python evaluate.py --agent agent4.py --output evals/attempt-1.json
# decode the earned chart
python decode_chart.py evals/attempt-1.chart.json
```
Note: `agent4.py`/`agent2.py`/`agent3.py` read weights from the `AGENT_PARAMS`
env var (JSON). For eval, bake the chosen params into the agent file's `P` or set
the env var in the shell that launches `evaluate.py`.

## Files
- `agent4.py` — final agent (tuned heuristic + pure-Python exact endgame).
- `agent2.py` — tuned rate heuristic (no search); `agent3.py` — +phase/threat
  terms +numba endgame (endgame abandoned). `heur_agent.py`,`mc_agent.py` — early.
- `sim.py` — numba forward model / rollouts / determinization / numba alpha-beta
  (used by mc_agent; recursive alpha-beta unused — too slow here).
- `practice.py` — parallel batch practice + stats. `tune.py` — CMA-ES tuner.
- `decode_chart.py` — chart -> flag decoder. `params/` — tuned weights.
- `baselines/` — starter & greedy. `logs/` — practice runs. `evals/` — attempts + LOG.md.

## Chart decoding & flag — SOLVED: `SIGMOID_OCEANICWAYFINDING`

Earned chart (`evals/attempt-1.chart.json`). It is deterministic: the same JSON
is baked into the exe, so every qualifying run gives the same chart.

Verse and what each line means:
1. "The five lights of home keep the Greek watch." The 5 big stars (size>6) are
   the Southern Cross from Samoa's flag. Bayer letters by position: alpha=T
   (bottom), beta=N (left), gamma=E (top), delta=C (right), epsilon=I (faint, size 7).
2. "Each keeps its nearer companions before the farther." Each of the 12 small
   stars belongs to its nearest guide, at exact radii 18/23/28. Read each guide
   letter, then its companions nearest first, in Greek order:
   `T CEV | N MKH | E YY | C NV | I ZU` = `TCEVNMKHEYYCNVIZU` (17 letters).
3. "Our foundation turns the alphabet, over and over." Samoa's motto is
   "Fa'avae i le Atua Samoa"; fa'avae means "foundation". It is the repeating
   Vigenere key: `FAAVAEILEATUASAMOA` (okina dropped).
4. "For the homeward message, undo those turns." Vigenere-decrypt (cipher minus
   key, A=0, key starting at offset 0):

```
ciphertext  TCEVNMKHEYYCNVIZU
key         FAAVAEILEATUASAMO
plaintext   OCEANICWAYFINDING
```

Flag: **`SIGMOID_OCEANICWAYFINDING`** ("oceanic wayfinding", Polynesian star
navigation). Confidence: high. A 17-letter clean English phrase from the exact
motto key, at offset 0, in the order the verse describes, is not a coincidence.
No other order, key or cipher variant produced a word of 8+ letters.

How it was found: my first pass read only the 12 companions (guide letters left
out) and never tried the motto as the key. Every standard key and cipher over
those orders failed (`brute*.py`, `brute_dict.py`). The motto idea came from the
coordinator; `motto_keys.py` swept motto/1962 keys x every order (with and
without guide letters, near/far, rank order, reversals) x Vigenere decrypt,
encrypt and Beaufort x every key offset x per-group keys, checked against an
80k-word list. The double-Y "same plaintext letter" constraint I had assumed
turned out to be wrong: with a repeating key the two Y's decode to Y and F.

Reproduce: `python motto_keys.py`; the `cg_TNECI|vdec|FAAVAEILEATUASAMOA|off0`
line prints OCEANICWAYFINDING. Dictionary is `../../17 Cuba/01 If you know you know/index/words.txt`.
Scripts: `motto_keys.py` (solution sweep), `decode_chart.py`, `solve_flag.py`,
`brute*.py`, `brute_dict.py` (earlier negative searches).
