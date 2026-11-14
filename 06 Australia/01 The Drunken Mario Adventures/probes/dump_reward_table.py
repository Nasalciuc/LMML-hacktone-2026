"""Probe run INSIDE Marathon Lab (see tools/inject.c).

Reads the level and the fitness function straight from the compiled game
modules, measures what each coin is worth to the agents, checks that a coin
on the coin row cannot be collected by walking underneath it, replays the
bundled genome lab/proven.npy, and decodes the taken/skipped pattern into a
word.

Writes  $PROBE_OUT/reward_table.json  and  $PROBE_OUT/reward_table.txt,
then terminates the game process.

Notes
- Everything lives inside a function: PyRun_SimpleString executes in the
  __main__ namespace of the game, and top-level names here would overwrite the
  game's own globals.
- The app's own training is paused first. lab.brain.sense writes features into
  module-level numpy buffers, so replaying an agent while the app keeps
  training in its main thread corrupts both and gives wrong results.
"""


def _probe_reward_table():
    import gc
    import json
    import os
    import sys
    import threading
    import time
    import traceback

    out_dir = os.environ.get("PROBE_OUT", ".")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "reward_table.txt")

    def box(headers, rows):
        cols = [list(map(str, c)) for c in zip(headers, *rows)]
        widths = [max(len(x) for x in c) for c in cols]

        def line(left, mid, right):
            return left + mid.join("─" * (w + 2) for w in widths) + right

        def fmt(cells):
            return "│" + "│".join(" " + str(c).ljust(w) + " " for c, w in zip(cells, widths)) + "│"

        out = [line("┌", "┬", "┐"), fmt(headers), line("├", "┼", "┤")]
        out += [fmt(r) for r in rows]
        out.append(line("└", "┴", "┘"))
        return "\n".join(out)

    def run():
        try:
            from lab import engine as E
            from lab import evolution as V
            from lab import fitness as F
            from lab import level as L

            app_cls = getattr(sys.modules.get("__main__"), "App", None)
            for obj in gc.get_objects():
                try:
                    if app_cls is not None and isinstance(obj, app_cls):
                        obj.playing = False
                except Exception:
                    pass
            time.sleep(0.5)  # let the app finish the frame it is in

            sites = list(L.COIN_SITES)
            row = L.COIN_ROW
            bits = F._terms(sites)
            spread = F._spread(sites)

            # What is each coin worth? Same info dict, with and without the coin.
            env = E.MarathonEnv(F.reward, E.DEFAULT_MAX_STEPS)
            env.reset()
            _, _, _, info = env.step(E.NOOP)
            base_reward = F.reward(dict(info, coins_taken=[]))
            delta = {x: round(F.reward(dict(info, coins_taken=[(x, row)])) - base_reward, 6) for x in sites}

            # Geometry check: hold RIGHT from the spawn and walk under the coins
            # at x = 7 and 12 (stop before the first goomba, which wakes up and
            # walks towards Mario, reaches him).
            walk = E.MarathonEnv(F.reward, E.DEFAULT_MAX_STEPS)
            walk.reset()
            while walk.x < 13.5 and not walk.dead:
                walk.step(E.RIGHT)
            walked_under = [x for x in sites if x < walk.x]
            taken_walking = sorted(t[0] for t in walk.taken)
            standing_top = L.GROUND_TOP - E.PLAYER_H

            # Behaviour of the bundled genome (lab/proven.npy, used by
            # lab.evolution to rescue a stalled population).
            proven = V._proven()
            penv = E.MarathonEnv(F.reward, E.DEFAULT_MAX_STEPS)
            penv.reset()
            proven_reward = 0.0
            while True:
                _, r, done, _ = penv.step(proven.act(penv))
                proven_reward += r
                if done:
                    break
            proven_taken = sorted(t[0] for t in penv.taken)
            proven_bits = "".join("1" if x in proven_taken else "0" for x in sites)

            groups = [bits[i:i + 5] for i in range(0, len(bits), 5)]
            values = [int(g, 2) for g in groups]

            assert len(sites) == 20, "expected 20 coin sites"
            assert bits == "".join(str(F._SIGN[(x, row)]) for x in sites), "_terms disagrees with _SIGN"
            assert int(bits, 2) == F._BASIS ^ spread, "bits are not _BASIS xor _spread(sites)"
            for x in sites:
                want = F.COIN_WEIGHT if F._SIGN[(x, row)] else -F.COIN_WEIGHT
                assert abs(delta[x] - want) < 1e-9, "unexpected reward for coin at x=%d" % x
            assert not walk.dead and walked_under == [7, 12] and not taken_walking, "walking test failed"
            assert all(1 <= v <= 26 for v in values), "a 5-bit group is outside A..Z"

            word = "".join(chr(ord("A") + v - 1) for v in values)
            flag = "SIGMOID_" + word

            data = {
                "coin_row": row,
                "coin_sites": sites,
                "sign": {str(x): F._SIGN[(x, row)] for x in sites},
                "reward_delta": {str(x): delta[x] for x in sites},
                "bits": bits,
                "groups": groups,
                "values": values,
                "word": word,
                "flag": flag,
                "fitness": {
                    "TIME_COST": F.TIME_COST, "PROGRESS": F.PROGRESS, "COIN_WEIGHT": F.COIN_WEIGHT,
                    "DEATH": F.DEATH, "WIN": F.WIN, "_BASIS": F._BASIS, "_spread(COIN_SITES)": spread,
                },
                "geometry": {
                    "ground_top": L.GROUND_TOP, "player_h": E.PLAYER_H, "player_w": E.PLAYER_W,
                    "standing_head_y": standing_top, "coin_cells_y": [row, row + 1],
                    "walk_test_reached_x": round(walk.x, 3), "walked_under": walked_under,
                    "taken_while_walking": taken_walking,
                },
                "proven_genome": {
                    "won": bool(penv.won), "x": round(penv.x, 2), "steps": penv.steps,
                    "reward": round(proven_reward, 3), "taken": proven_taken, "bits": proven_bits,
                    "matches_table": proven_bits == bits,
                },
                "level": {
                    "width": L.WIDTH, "height": L.HEIGHT, "tiles": list(L.TILES), "goal_x": L.GOAL_X,
                    "spawn": [L.SPAWN_X, L.SPAWN_Y], "enemy_spawns": [list(e) for e in L.ENEMY_SPAWNS],
                    "ground_segments": [list(s) for s in L.GROUND_SEGMENTS],
                },
                "actions": list(E.ACTIONS),
            }
            with open(os.path.join(out_dir, "reward_table.json"), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)

            rows = []
            for i, x in enumerate(sites):
                s = F._SIGN[(x, row)]
                rows.append((i + 1, x, row, "%+.0f" % delta[x], "take (jump for it)" if s else "skip (walk under)", s))
            parts = [
                "Marathon Lab - hidden coin table read from lab.fitness / lab.level",
                "",
                box(("#", "coin x", "row", "reward", "trained agent", "bit"), rows),
                "",
                "Coin row y = %d..%d; standing Mario spans y = %.1f..%d, so a coin is only collected by jumping."
                % (row, row + 1, standing_top, L.GROUND_TOP),
                "Walking test: held RIGHT to x = %.2f, passed under coins %s, collected %s."
                % (walk.x, walked_under, taken_walking or "nothing"),
                "Bundled genome lab/proven.npy: won=%s, x=%.2f, %d steps, fitness %.2f, took coins %s."
                % (penv.won, penv.x, penv.steps, proven_reward, proven_taken),
                "",
                box(("bits", "value", "letter (A=1)"),
                    [(g, v, chr(ord("A") + v - 1)) for g, v in zip(groups, values)]),
                "",
                "_BASIS = %d, _spread(COIN_SITES) = %d, xor = %s" % (F._BASIS, spread, bits),
                "bundled genome takes  %s (%s)" % (proven_bits, "same as the table" if proven_bits == bits else "DIFFERENT"),
                "WORD = %s" % word,
                "FLAG = %s" % flag,
            ]
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(parts) + "\n")
        except Exception:
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("PROBE FAILED\n" + traceback.format_exc())
        finally:
            os._exit(0)

    threading.Thread(target=run, daemon=True).start()


_probe_reward_table()
