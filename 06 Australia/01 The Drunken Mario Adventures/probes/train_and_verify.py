"""Probe run INSIDE Marathon Lab (see tools/inject.c).

Behavioural check of the flag: trains a fresh population with the game's own
evolution code (lab.evolution.Lab), headless and deterministic for a given
SEED, and after every generation replays the champion (Lab.best_ever) to see
which coins it takes. Stops once the champion has reached the flagpole and its
coin choices have not changed for STABLE_GENS generations (or at MAX_GENS).
Reaching the flagpole is not enough: the first finishers still take or skip
some coins by accident and keep improving for a while, so the window is long
(200 generations, i.e. several stagnation "rescues" of lab.evolution).

Important: the game's GUI training is paused first. The brain's feature
extractor writes into module-level numpy buffers, so replaying an agent from a
second thread while the GUI keeps training corrupts both and gives noisy
results.

Environment: SEED (default 11), POP (120), STABLE_GENS (200), MAX_GENS (3000),
PROBE_OUT (output folder).

Writes: training_log.csv, champion_genome.npy, champion_path.csv,
        verify_summary.json, verify_summary.txt
"""


def _train_and_verify():
    import gc
    import json
    import os
    import sys
    import threading
    import time
    import traceback

    out_dir = os.environ.get("PROBE_OUT", ".")
    os.makedirs(out_dir, exist_ok=True)
    seed = int(os.environ.get("SEED", "11"))
    pop = int(os.environ.get("POP", "120"))
    stable_gens = int(os.environ.get("STABLE_GENS", "200"))
    max_gens = int(os.environ.get("MAX_GENS", "3000"))

    def run():
        summary_txt = os.path.join(out_dir, "verify_summary.txt")
        try:
            import numpy as np
            from lab import engine as E
            from lab import evolution as V
            from lab import fitness as F
            from lab import level as L

            app_cls = getattr(sys.modules.get("__main__"), "App", None)
            paused = 0
            for obj in gc.get_objects():
                try:
                    if app_cls is not None and isinstance(obj, app_cls):
                        obj.playing = False
                        paused += 1
                except Exception:
                    pass

            sites = list(L.COIN_SITES)
            row = L.COIN_ROW
            expected = F._terms(sites)

            def word_of(bits):
                vals = [int(bits[i:i + 5], 2) for i in range(0, len(bits), 5)]
                return "".join(chr(ord("A") + v - 1) if 1 <= v <= 26 else "?" for v in vals)

            def rollout(brain, record=False):
                env = E.MarathonEnv(F.reward, E.DEFAULT_MAX_STEPS)
                env.reset()
                path = []
                total = 0.0
                while True:
                    action = brain.act(env)
                    _, r, done, _ = env.step(action)
                    total += r
                    if record:
                        path.append((env.steps, round(env.x, 4), round(env.y, 4), int(action), len(env.taken)))
                    if done:
                        break
                taken = sorted(t[0] for t in env.taken)
                bits = "".join("1" if x in taken else "0" for x in sites)
                return {"bits": bits, "word": word_of(bits), "won": bool(env.won), "dead": bool(env.dead),
                        "x": round(env.x, 2), "steps": env.steps, "reward": round(total, 3),
                        "taken": taken, "path": path}

            lab = V.Lab(pop, seed)
            lab.begin()
            log_rows = []
            last_score = None
            champ = None
            first_win = None
            stable = 0
            t0 = time.time()
            log_path = os.path.join(out_dir, "training_log.csv")
            with open(log_path, "w", encoding="utf-8") as log:
                log.write("generation,seconds,best_score,champion_won,champion_x,champion_steps,coin_bits,word,rescues\n")
                while lab.generation < max_gens:
                    while not lab.finished:
                        lab.step(500)
                    lab.finish()
                    if lab.finished:
                        lab.begin()
                    score = lab.best_ever_score
                    if score != last_score:
                        new = rollout(lab.best_ever)
                        changed = champ is None or new["bits"] != champ["bits"] or new["won"] != champ["won"]
                        champ, last_score = new, score
                        stable = 0 if changed else stable + 1
                    else:
                        stable += 1
                    if champ["won"] and first_win is None:
                        first_win = lab.generation
                    log.write("%d,%.1f,%.3f,%d,%.2f,%d,%s,%s,%d\n" % (
                        lab.generation, time.time() - t0, score, champ["won"], champ["x"], champ["steps"],
                        champ["bits"], champ["word"], lab.rescues))
                    log.flush()
                    if first_win is not None and champ["won"] and stable >= stable_gens:
                        break

            final = rollout(lab.best_ever, record=True)
            again = rollout(lab.best_ever)
            genome = np.asarray(lab.best_ever.genome(), dtype=np.float32)
            bundled = np.asarray(V._proven().genome(), dtype=np.float32)
            same_as_bundled = bool(genome.shape == bundled.shape and np.array_equal(genome, bundled))
            np.save(os.path.join(out_dir, "champion_genome.npy"), genome)
            with open(os.path.join(out_dir, "champion_path.csv"), "w", encoding="utf-8") as fh:
                fh.write("step,x,y,action,coins_taken\n")
                for rec in final["path"]:
                    fh.write("%d,%.4f,%.4f,%d,%d\n" % rec)

            summary = {
                "seed": seed, "population": pop, "generations": lab.generation,
                "first_win_generation": first_win, "stable_generations": stable,
                "seconds": round(time.time() - t0, 1), "gui_paused": paused,
                "champion": {k: final[k] for k in ("bits", "word", "won", "dead", "x", "steps", "reward", "taken")},
                "replay_is_deterministic": again["bits"] == final["bits"] and again["reward"] == final["reward"],
                "reward_table_bits": expected, "matches_reward_table": final["bits"] == expected,
                "champion_equals_bundled_genome": same_as_bundled,
            }
            with open(os.path.join(out_dir, "verify_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=1)
            with open(summary_txt, "w", encoding="utf-8") as fh:
                fh.write("seed %d, population %d: first reached the flagpole at generation %s, stopped at %d "
                         "(%d generations without a change in the champion's coin choices, %.0f s)\n"
                         % (seed, pop, first_win, lab.generation, stable, summary["seconds"]))
                fh.write("champion: won=%s x=%.2f steps=%d reward=%.2f\n"
                         % (final["won"], final["x"], final["steps"], final["reward"]))
                fh.write("coins taken at x = %s\n" % final["taken"])
                fh.write("champion bits     %s -> %s\n" % (final["bits"], final["word"]))
                fh.write("reward table bits %s -> %s\n" % (expected, word_of(expected)))
                fh.write("match: %s, deterministic replay: %s\n"
                         % (summary["matches_reward_table"], summary["replay_is_deterministic"]))
                fh.write("champion is the bundled genome lab/proven.npy: %s\n" % same_as_bundled)
        except Exception:
            with open(summary_txt, "w", encoding="utf-8") as fh:
                fh.write("VERIFY FAILED\n" + traceback.format_exc())
        finally:
            os._exit(0)

    threading.Thread(target=run, daemon=True).start()


_train_and_verify()
