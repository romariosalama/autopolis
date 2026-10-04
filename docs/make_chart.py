"""Draws docs/results.png from the saved benchmark JSON files."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAY = "#858b91"
GREEN = "#0b7a4b"
INK = "#2b2f33"
MUTED = "#6b7075"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
for ax, name, title in [(axes[0], "3x3", "3×3 grid (trained here)"),
                        (axes[1], "6x6", "6×6 grid (never seen in training)")]:
    r = json.load(open(f"docs/benchmark_{name}.json"))
    labels = [f"Fixed {g}s" for g in r["fixed"]] + ["AI (PPO)"]
    values = [v["avg_wait"] for v in r["fixed"].values()] + [r["ai"]["avg_wait"]]
    colors = [GRAY] * len(r["fixed"]) + [GREEN]
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, f"{v:.1f}s", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0d3d6")
    ax.set_xlabel("Average wait per trip (seconds)", fontsize=8.5, color=MUTED)

fig.tight_layout(w_pad=3)
fig.savefig("docs/results.png", facecolor="white")
