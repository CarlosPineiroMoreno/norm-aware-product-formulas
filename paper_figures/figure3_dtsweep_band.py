"""Figure 3 -- error against the time step, banded over a range of couplings.

Each formula is swept in dt = 1/N at seven couplings spanning
alpha in [0.10, 0.20].  The line is the sweep at the middle coupling and the
shaded band is the envelope over all seven, so the figure shows the
convergence order without committing to one value of alpha.

    python3 paper_figures/figure3_dtsweep_band.py            # replot
    python3 paper_figures/figure3_dtsweep_band.py --compute  # rerun the sweeps

Outputs figures/dtsweep_band_L3_alpha0.10-0.20_t1.pdf.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from norm_aware.constants import METHODS
from norm_aware.style import FIG_W_1COL, LABELS, method_kw, use_pra_style

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

L = 3
T = 1.0
ALPHAS = [0.1, 0.1167, 0.1333, 0.15, 0.1667, 0.1833, 0.2]
NSTEPS = [1, 2, 3, 4, 5, 7, 10, 13, 17, 23, 31, 42, 56, 74, 100, 133, 177, 237, 316]


def run_dir(alpha):
    return os.path.join(RESULTS, f"dtsweep_XXMYY_L{L}_J{alpha:g}_t{T}")


def compute():
    """Sweep dt at each of the seven couplings.  Hours of work."""
    from norm_aware.evolver import frobenius_error
    from norm_aware.spin import SpinLattice2D
    from norm_aware.sweeps import _jsonable, method_table

    lambda_values = {'Uλ̃(λ*=0.211325)': 0.211325}
    names, keys, lams, _, _ = method_table(lambda_values, 0.211325, 0.211325)

    for alpha in ALPHAS:
        d = run_dir(alpha)
        os.makedirs(d, exist_ok=True)
        u_exact = SpinLattice2D(L, alpha, -alpha, 0.0).exact_propagator(T)
        errors = {}
        for name, key, lam in zip(names, keys, lams):
            errors[name] = [frobenius_error(L, alpha, -alpha, T, n, key,
                                            lam=lam, u_exact=u_exact)
                            for n in NSTEPS]
            print(f"  alpha={alpha:g}  {name:24s} done", flush=True)
        json.dump(_jsonable(dict(
            hamiltonian='sum_k Zk + J*sum_ij(XiXj - YiYj)',
            L=L, J=alpha, Jx=alpha, Jy=-alpha, t=T,
            nsteps=NSTEPS, all_methods=names, method_keys=keys,
            method_lams=lams, lam_lambda=0.211325, lam_ps2=0.211325,
            lam_omelyan=0.211325, errors=errors)),
            open(os.path.join(d, "data.json"), "w"), indent=2)
        print(f"[saved] {d}/data.json", flush=True)


def load():
    """Every sweep found on disk, keyed by coupling."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, f"dtsweep_XXMYY_L{L}_J*_t{T}"))):
        alpha = float(os.path.basename(path).split("_J")[1].split("_t")[0])
        with open(os.path.join(path, "data.json")) as f:
            runs[alpha] = json.load(f)
    if not runs:
        raise FileNotFoundError(f"no dt sweeps under {RESULTS}; run with --compute")
    return runs


def plot(runs, out_dir=OUT, save=True):
    alphas = sorted(runs)
    ref = runs[alphas[len(alphas) // 2]]              # the middle coupling
    dt = 1.0 / np.array(ref["nsteps"], float)
    # the display names carry the lambda used, so map them back by position
    name_of = dict(zip(ref["method_keys"], ref["all_methods"]))

    use_pra_style()
    fig, ax = plt.subplots(figsize=(FIG_W_1COL, 2.9))

    for m in METHODS:
        name = name_of[m]
        stack = np.array([runs[a]["errors"][name] for a in alphas])
        ax.fill_between(dt, stack.min(axis=0), stack.max(axis=0),
                        color=method_kw(m)["color"], alpha=0.18, lw=0, zorder=1)
        ax.loglog(dt, np.array(ref["errors"][name]),
                  **method_kw(m, label=LABELS[m], zorder=3))

    ax.set_xlabel(r"$\Delta t = N^{-1}$")
    ax.set_ylabel(r"$\|U-\tilde{U}\|_F$")
    ax.text(0.04, 0.96, rf"$\alpha\in[{alphas[0]:.2f},\,{alphas[-1]:.2f}]$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.6))
    ax.legend(loc="lower right", ncol=2, fontsize=5.6, handlelength=1.6,
              columnspacing=0.9, handletextpad=0.4, labelspacing=0.25,
              borderpad=0.4, framealpha=0.95, edgecolor="0.6")
    fig.tight_layout(pad=0.3)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"dtsweep_band_L{L}_alpha{alphas[0]:.2f}-{alphas[-1]:.2f}_t{T:g}.pdf")
        fig.savefig(path)
        print("[saved]", os.path.normpath(path))
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compute", action="store_true",
                   help="rerun the seven dt sweeps instead of reusing results/")
    if p.parse_args().compute:
        compute()
    plot(load())
