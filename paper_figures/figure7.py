"""Figure 7 -- effective exponent of the local error in alpha.

Both panels use the *local* error of a single application of each formula,

    eps(t) = || exp(-i H t) - U_step(t) ||_F ,

which is the quantity the theorems and Remark (i) of the paper refer to.  The
accumulated error of an N-step run has a different scaling in alpha, so mixing
the two would not test the remark.

    python3 figure7.py              # replot from data/figure7_local_errors.json
    python3 figure7.py --compute    # recompute (about 20 min) and replot

Outputs ../figures/alpha_exponent_summary_L3_t1.0.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from norm_aware.constants import METHODS
from norm_aware.pauli import Stepper, spin_model, square
from norm_aware.style import (FIG_W_2COL, LABELS, STYLES, method_kw,
                              use_pra_style)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "figure7_local_errors.json")
OUT = os.path.join(HERE, os.pardir, "figures")

# 3x3 open-boundary XY lattice, Eq. (29) of the paper, with alpha_x = -alpha_y.
LATTICE = (9, square(3, 3), 1.0, -1.0)
ALPHAS = np.geomspace(0.02, 1.0, 20)
TIMES = [1, 2, 3, 4, 5, 10, 20, 40, 80, 160, 320, 640]      # plotted as t = 1/n

WIN = (0.02, 0.10)        # window where the power law holds
WIN_WIDE = (0.02, 1.0)    # whole range, shown pale: it biases the exponent
PANEL_B_T = 1000          # t = 1e-3; see note in compute()


def compute():
    """Local error of every formula, for each t and alpha.

    t = 1e-3 is the smallest value used in panel (b).  Going to 1e-4 is not
    possible in double precision: the genuine local error of the second-order
    alpha^2-suppressed formulas is then ~2e-14, at the round-off floor, and
    they measure 35-43 % high.
    """
    n, edges, rx, ry = LATTICE
    h0, terms = spin_model(n, edges, rx, ry)
    st = Stepper(h0, terms)
    out = {m: {} for m in METHODS}
    t0 = time.time()
    for n_steps in TIMES + [PANEL_B_T]:
        t = 1.0 / n_steps
        for m in METHODS:
            out[m][str(n_steps)] = [st.local_error(m, a, t) for a in ALPHAS]
        print(f"  t={t:.4g}  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    json.dump({"alphas": ALPHAS.tolist(),
               "n_steps": TIMES + [PANEL_B_T],
               "errors": out}, open(DATA, "w"))
    print("[saved]", DATA)


def _fit(alphas, err, lo, hi):
    m = (alphas >= lo) & (alphas <= hi) & (err > 0) & np.isfinite(err)
    return np.polyfit(np.log(alphas[m]), np.log(err[m]), 1)[0] if m.sum() >= 3 else np.nan


def _panel_a(ax, alphas, err, times):
    ts = 1.0 / np.array(times, float)
    for m in METHODS:
        narrow = [_fit(alphas, err[m][str(n)], *WIN) for n in times]
        wide = [_fit(alphas, err[m][str(n)], *WIN_WIDE) for n in times]
        ax.semilogx(ts, wide, color=STYLES[m]["color"], ls=STYLES[m]["ls"],
                    lw=0.8, alpha=0.25)
        ax.semilogx(ts, narrow, **method_kw(m, label=LABELS[m]))
    for y in (1, 2):
        ax.axhline(y, color="0.6", ls=":", lw=0.8)
    ax.annotate(r"$U^{[1]}_{\mathrm{PS}}$: $t\!\sim\!\alpha$ crossover",
                xy=(0.1, _fit(alphas, err["ps"]["10"], *WIN)), xytext=(0.0055, 1.22),
                fontsize=6.5, color=STYLES["ps"]["color"],
                arrowprops=dict(arrowstyle="->", color=STYLES["ps"]["color"], lw=0.8))
    ax.text(0.97, 0.03, r"pale: $[0.02,\,1.0]$ window", fontsize=6, color="0.45",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S$   ($\|U-\tilde{U}\|_F \propto \alpha^{S}$)")
    ax.set_ylim(0.70, 2.30)
    ax.text(0.045, 0.93, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_title(rf"local error, fit over $\alpha\in[{WIN[0]:g},\,{WIN[1]:g}]$",
                 fontsize=8, pad=3)


def _panel_b(ax, alphas, err):
    mid = np.sqrt(alphas[:-1] * alphas[1:])
    for m in METHODS:
        e = np.array(err[m][str(PANEL_B_T)])
        ax.semilogx(mid, np.diff(np.log(e)) / np.diff(np.log(alphas)),
                    **method_kw(m, label=LABELS[m]))
    for y in (1, 2):
        ax.axhline(y, color="0.6", ls=":", lw=0.8)
    ax.axvspan(*WIN, color="0.9", zorder=0, lw=0)
    ax.text(np.sqrt(WIN[0] * WIN[1]), 2.71, "fit\nwindow", fontsize=6,
            ha="center", va="top", color="0.5")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{d}\log\|U-\tilde{U}\|_F\,/\,\mathrm{d}\log\alpha$")
    ax.set_xlim(left=WIN[0])          # the band starts flush with the axis
    ax.set_ylim(0.4, 3.0)
    ax.text(0.045, 0.93, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    mant, expo = f"{1 / PANEL_B_T:.1e}".split("e")
    ax.set_title(rf"local error, fixed $t = {mant}\times10^{{{int(expo)}}}$",
                 fontsize=8, pad=3)


def plot():
    d = json.load(open(DATA))
    alphas = np.array(d["alphas"])
    err = {m: {k: np.array(v) for k, v in d["errors"][m].items()} for m in METHODS}
    times = [n for n in d["n_steps"] if n in TIMES]
    os.makedirs(OUT, exist_ok=True)
    use_pra_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W_2COL, 3.6))
    _panel_a(ax1, alphas, err, times)
    _panel_b(ax2, alphas, err)
    h, l = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.30, wspace=0.30)
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=True,
               framealpha=0.9, edgecolor="0.6", handlelength=2.0,
               columnspacing=1.3, handletextpad=0.5, borderpad=0.5,
               bbox_to_anchor=(0.5, 0.015))
    path = os.path.join(OUT, "alpha_exponent_summary_L3_t1.0.pdf")
    fig.savefig(path)
    print("[saved]", os.path.normpath(path))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compute", action="store_true",
                   help="recompute the data instead of reusing data/")
    if p.parse_args().compute:
        compute()
    plot()
