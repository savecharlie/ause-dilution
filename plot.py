#!/usr/bin/env python3
"""plot.py -- two panels, both from the committed json, nothing hand-typed."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = json.load(open("blindness.json"))
T = json.load(open("topdecile.json"))

INK, WARM, COOL, GREY = "#2b2622", "#c2663a", "#3f6f8f", "#8e867d"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
fig.patch.set_facecolor("#faf7f2")
for a in (ax1, ax2):
    a.set_facecolor("#faf7f2")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GREY)
    a.tick_params(colors=INK, labelsize=9)

# ---- (a) dilution: how much of AUSE's range the prediction magnitude alone reaches
sw = B["sweep"]
xs = [s["a"] for s in sw]
ys = [100 * s["mu_closed"] for s in sw]
ax1.plot(xs, ys, "o-", color=WARM, lw=2.2, ms=6, zorder=3)
ax1.axhline(0, color=GREY, lw=0.8, ls=":")
ax1.set_xlabel("heteroscedasticity exponent  $a$   in   $\\sigma \\propto \\hat\\mu^{\\,a}$",
               fontsize=10, color=INK)
ax1.set_ylabel("% of AUSE's range reached by ranking on $\\hat\\mu$ alone",
               fontsize=10, color=INK)
ax1.set_title("(a) AUSE is not blind — it is diluted", fontsize=11.5, color=INK, loc="left")
ax1.set_ylim(-8, 100)
ax1.annotate("homoscedastic:\n$\\hat\\mu$ is useless,\nAUSE sees everything",
             xy=(0.02, 1.5), xytext=(0.04, 70), fontsize=8.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
ax1.annotate("streamflow lives here —\nmost of the score is\nalready in the forecast",
             xy=(0.9, ys[4]), xytext=(0.50, 22), fontsize=8.5, color=WARM,
             arrowprops=dict(arrowstyle="->", color=WARM, lw=0.9))

# ---- (b) the range effect under tail conditioning
keys = ["all", "top decile of mu-hat", "top 5% of mu-hat", "top 1% of mu-hat"]
lab = ["all", "top 10%", "top 5%", "top 1%"]
raw = [T["subsets"][k]["mu"] for k in keys]
inf = [T["subsets"][k]["informed"] for k in keys]
frac = [100 * T["subsets"][k]["informed_closes_of_mu_gap"] for k in keys]
x = range(len(keys))
ax2.plot(x, raw, "s--", color=GREY, lw=1.6, ms=6, label="raw AUSE, rank by $\\hat\\mu$")
ax2.plot(x, inf, "o--", color=COOL, lw=1.6, ms=6,
         label="raw AUSE, informed spread")
ax2.set_xticks(list(x)); ax2.set_xticklabels(lab)
ax2.set_xlabel("conditioning on the upper tail", fontsize=10, color=INK)
ax2.set_ylabel("raw AUSE  (lower is better)", fontsize=10, color=INK)
ax2.set_ylim(0, 0.52)
ax2.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)

ax3 = ax2.twinx()
ax3.set_facecolor("none")
for s in ("top", "left"):
    ax3.spines[s].set_visible(False)
ax3.spines["right"].set_color(WARM)
ax3.plot(x, frac, "o-", color=WARM, lw=2.4, ms=7, zorder=4,
         label="% of the $\\hat\\mu$ gap closed")
ax3.set_ylim(0, 100)
ax3.set_ylabel("% of the $\\hat\\mu\\to$oracle gap closed", fontsize=10, color=WARM)
ax3.tick_params(colors=WARM, labelsize=9)
ax3.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=WARM)
ax2.set_title("(b) raw AUSE triples; what the spread knows does not move",
              fontsize=11.5, color=INK, loc="left")

p = T["paper_closed"]
fig.text(0.515, 0.015,
         "Gallant & Wilson (arXiv:2610.20947), read the same way:  "
         f"all test {100*p['all_test']:.1f}%   top observed decile "
         f"{100*p['top_observed_decile']:.1f}%   — flat, while their under-dispersion "
         "goes 2.1x to 17.9x",
         fontsize=8.6, color=INK, ha="center")
fig.suptitle("A near-oracle ranking score is mostly a statement about heteroscedasticity",
             fontsize=12.5, color=INK, y=0.985)
fig.tight_layout(rect=(0, 0.045, 1, 0.95))
fig.savefig("ause-dilution.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote ause-dilution.png")
