#!/usr/bin/env python3
"""Draw the level, the 20 coins with their hidden reward, the champion's run
and the decoded word.

Usage:  python3 plot_coin_message.py RESULTS_DIR
Reads   reward_table.json (required), champion_path.csv, training_log.csv,
        verify_summary.json (optional, from probes/train_and_verify.py)
Writes  RESULTS_DIR/coin_message.png
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BG = "#1a1a2e"
PRIMARY = "#00d4aa"
SECONDARY = "#ff6b6b"
ACCENT = "#ffd93d"
BLUE = "#4da6ff"
PURPLE = "#b388ff"
TEXT = "#e6e6f0"
MUTED = "#8a8aa8"

TILE_COLOUR = {
    "X": "#5c4033", "B": "#8d5a3b", "?": ACCENT, "U": "#6b5a4b", "S": "#7d6b5d",
    "T": "#2e8b57", "P": "#246b45", "F": "#cfcfcf", "f": PRIMARY, "c": "#5c4033",
    "C": "#2a2a50", "H": "#22223f", "V": "#22223f",
}


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def draw_strip(ax, data, x0, x1, path):
    tiles = data["level"]["tiles"]
    height = data["level"]["height"]
    row = data["coin_row"]
    ax.set_facecolor(BG)
    for y, line in enumerate(tiles):
        for x in range(x0, min(x1, len(line))):
            ch = line[x]
            if ch in TILE_COLOUR:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=TILE_COLOUR[ch], edgecolor=BG, linewidth=0.3))
    for x in data["coin_sites"]:
        if not x0 <= x < x1:
            continue
        good = data["sign"][str(x)] == 1
        colour = PRIMARY if good else SECONDARY
        ax.add_patch(plt.Circle((x + 0.5, row + 0.5), 0.42, facecolor=colour if good else "none",
                                edgecolor=colour, linewidth=1.6, zorder=4))
        ax.text(x + 0.5, row - 0.9, "%+.0f" % data["reward_delta"][str(x)], color=colour,
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
    if path:
        pw, ph = data["geometry"]["player_w"], data["geometry"]["player_h"]
        xs = [float(r["x"]) + pw / 2 for r in path]
        ys = [float(r["y"]) + ph / 2 for r in path]
        ax.plot(xs, ys, color=BLUE, linewidth=1.2, alpha=0.9, zorder=3)
        prev = 0
        for r, cx, cy in zip(path, xs, ys):
            n = int(r["coins_taken"])
            if n > prev and x0 <= cx < x1:
                ax.plot(cx, cy, marker="*", color=ACCENT, markersize=11, zorder=6)
            prev = n
    ax.set_xlim(x0, x1)
    ax.set_ylim(height, 2)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#33334d")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results"
    data = json.load(open(os.path.join(out, "reward_table.json")))
    path = read_csv(os.path.join(out, "champion_path.csv"))
    log = read_csv(os.path.join(out, "training_log.csv"))
    summary_path = os.path.join(out, "verify_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else None

    bits, groups, word = data["bits"], data["groups"], data["word"]
    assert len(bits) == 20 and len(groups) == 4
    assert word == "".join(chr(ord("A") + int(g, 2) - 1) for g in groups)

    fig = plt.figure(figsize=(16, 9.2), facecolor=BG)
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.95], width_ratios=[1.15, 1], hspace=0.32, wspace=0.12,
                  left=0.03, right=0.985, top=0.9, bottom=0.07)
    width = data["level"]["width"]
    half = (width + 1) // 2
    for i, (x0, x1) in enumerate(((0, half), (half, width))):
        ax = fig.add_subplot(gs[i, :])
        draw_strip(ax, data, x0, x1, path)

    title = "Marathon Lab  -  the coins a trained agent jumps for (green, +25) and walks under (red, -25)"
    fig.text(0.03, 0.955, title, color=TEXT, fontsize=15, fontweight="bold", ha="left")
    sub = "Coin row y = %d sits above a walking Mario (y = %.1f to %d), so every taken coin is a deliberate jump." % (
        data["coin_row"], data["geometry"]["standing_head_y"], data["geometry"]["ground_top"])
    if path:
        sub += "  Blue: champion's run, stars: coins it took."
    fig.text(0.03, 0.925, sub, color=MUTED, fontsize=10, ha="left")

    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    for i, g in enumerate(groups):
        ax.text(i + 0.5, 0.78, " ".join(g), color=TEXT, fontsize=17, ha="center", family="monospace")
        ax.text(i + 0.5, 0.56, "= %d" % int(g, 2), color=MUTED, fontsize=12, ha="center")
        ax.text(i + 0.5, 0.25, chr(ord("A") + int(g, 2) - 1), color=ACCENT, fontsize=38, ha="center",
                fontweight="bold")
    ax.text(0.0, 0.98, "coin order, 1 = taken, groups of 5 bits, A = 1", color=MUTED, fontsize=10, va="top")
    ax.text(0.0, -0.08, "FLAG:  " + data["flag"], color=PRIMARY, fontsize=16, fontweight="bold",
            family="monospace")

    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor(BG)
    if log:
        gens = [int(r["generation"]) for r in log]
        best = [float(r["best_score"]) for r in log]
        match = [r["coin_bits"] == bits and r["champion_won"] == "1" for r in log]
        ax.plot(gens, best, color=BLUE, linewidth=1.6)
        hits = [(g, b) for g, b, m in zip(gens, best, match) if m]
        if hits:
            ax.scatter([h[0] for h in hits], [h[1] for h in hits], color=PRIMARY, s=10, zorder=3,
                       label="champion finishes and takes exactly the green coins")
            ax.legend(loc="lower right", fontsize=8, facecolor=BG, edgecolor="#33334d", labelcolor=TEXT)
        label = "best fitness per generation"
        if summary:
            label += " (seed %d, population %d)" % (summary["seed"], summary["population"])
        ax.set_title(label, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("generation", color=MUTED, fontsize=9)
    else:
        ax.text(0.5, 0.5, "run probes/train_and_verify.py for the training curve", color=MUTED,
                ha="center", va="center", transform=ax.transAxes)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#33334d")

    target = os.path.join(out, "coin_message.png")
    fig.savefig(target, dpi=110, facecolor=BG)
    print("wrote", target)


if __name__ == "__main__":
    main()
