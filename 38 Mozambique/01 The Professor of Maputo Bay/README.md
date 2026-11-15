# The Professor of Maputo Bay

Status: implemented; fresh local validation scored **52.08%**, below the 65% qualification threshold. The evaluator requires at least 65% map points. Each map counts by its **worst of five games**, so winning most individual games is insufficient.

## Explain it in one minute

“We control a shipping company with two boats and one action each round. We must choose between quick raw-material deliveries worth 5 points and a longer manufacturing chain worth 12 points. The difficult part is coordinating transport, warehouse inventory, workshop construction and production before the opponent takes the contracts.

Our agent uses beam search: simulate candidate actions, score the resulting positions, and keep only the best candidates at each depth. It executes the first action, then searches again after seeing the next real observation. This lets it revise plans when stock or contract ownership changes.

The score combines delivered points with estimated future value: products on boats, products waiting at the workshop, production in progress, and useful raw materials. Travel time, deadlines and estimated opponent arrival reduce that value. This is approximate planning, not a trained neural network and not an optimality guarantee.”

## Current implementation

- `Static`: precomputes shortest travel times and first hops, contract information and rough opponent delivery estimates.
- `Planner.encode`: converts the observation into a compact state containing cargo, travel timers, warehouses, workshops, jobs, stocks and closed contracts. Unseen stocks are estimated from remembered observations and regeneration.
- `actions` and `apply`: generate a pruned set of candidate actions and simulate our state transitions. The local model does not simulate the Professor's complete simultaneous action sequence.
- `evaluate`: values delivered points and future delivery potential. Carried products receive more credit than products in storage or production. Materials lose value with transport distance and as the game ends.
- `search`: retains a beam of 72 states for up to 22 search steps. The opening compares both manufacturing sites with width 96 and depth 26. Wall-clock limits may stop searches earlier; steps can fast-forward forced waiting.
- `decide`: updates remembered stocks and rough opponent arrival estimates, immediately takes available deliveries, and reuses the previous plan as a candidate in the next search.
- `agent`: checks the selected action against `legal_actions`, falling back to a legal wait/action. `serve` preserves the line-based JSON protocol and resets memory each game.

The nominal search budget is 6.8 seconds per game, with a 2-second opening allocation. This is a soft internal budget, not a guaranteed runtime bound. The supplied SDK permits 10 seconds per game plus first-process warmup; verify on the evaluation runtime (PyPy 3.10).

## What this task teaches

This is resource-constrained planning under partial observability and competition. Boats travel concurrently, but only one new action can be issued per round. A workshop is an investment: its construction consumes the same materials that could score raw contracts. Good planning must account for opportunity cost and the entire production-and-delivery chain.

Beam search trades completeness for speed. A narrow beam can discard a plan whose first moves look weak but whose later manufacturing payoff is strong. The heuristic gives partial plans credit so those investments can survive the search.

## Questions to defend

**Why not always choose the nearest contract?** Distance alone ignores cargo, production setup, deadlines and the opponent's likely arrival.

**Why not maximize manufactured goods?** Goods score only when delivered to an unclaimed contract before its deadline. Excess goods can waste actions and materials.

**Why replan each round?** Unknown stocks, opponent claims and simultaneous conflicts can invalidate an earlier plan.

**Is the opponent modeled exactly?** No. Arrival estimates adjust contract value, but they can be wrong. Enemy presence alone does not prove it carries the required goods.

**Does it build a third boat?** No. The current action generator omits that investment. Two boats can still travel in parallel; additional boats would also compete for the single action each round.

**What are the remaining weaknesses?** Approximate opponent and stock models, heuristic tuning, limited lookahead, pruned actions, and sensitivity to time-based search cutoffs. A single bad matchup can erase an otherwise strong map result.

## Reproduce validation

Download the official practice SDK and engine, place them together outside the submission directory, and run:

```sh
python3 arena.py "/absolute/path/to/main.py" --maps 24 --seed 11 --jobs 4
```

Repeat with other seeds and PyPy 3.10 before claiming qualification. Upload only `main.py`. No qualifying evaluator flag has been verified in this continuation.

## Verified local result (15 November 2026)

Official SDK, current `main.py`, CPython, 24 maps, seed 11, four workers:

```json
{"maps":24,"map_wins":8,"map_draws":9,"match_score":0.5208,"games_won":80,"slowest_game_seconds":6.73,"within_time_limit":true,"seconds":155.4}
```

That is 8 map wins, 9 draws and 7 losses across 120 games. The map score is `(8 + 9/2) / 24`, not `80 / 120`. Timing passed the SDK's limit on this machine. Python 3.10 syntax parsing and `git diff --check` also passed. PyPy is not installed here, so its runtime performance remains unverified. These practice results do not establish hidden-map qualification.

## Final version and checks (15 November 2026, PyPy 3.10, 24 maps per seed)

`main.py` is now the version with a fixed amount of search per move. It no longer stops on a clock, so a slow machine does not cut its planning short. It also compares both workshop sites in the opening, and once it has 17 points it sends the spare boat on quick raw-material deliveries.

| Version | Seed 777 | Seed 11 | Seed 2026 | Seed 5 | Seed 99 | Average |
|---|---|---|---|---|---|---|
| this `main.py` | 60.4% | 43.8% | 43.8% | 52.1% | 58.3% | 51.7% |
| previous `main.py` (clock-limited) | 39.6% | 52.1% (CPython) | not run | 52.1% | 50.0% | 48.5% |

Slowest game: 6.5 seconds, inside the 10-second limit. The numbers earlier in this README describe the previous version.

No seed reached 65%. The same version moves 10 to 15 points between seeds. The task is not solved.

`alternativa/` holds a teammate's different agent. It scored 0.27, 0.23 and 0.31 on the official arena.
