"""Figure 4 -- state error per molecule, for both orders.

    mean over 10 Haar-rotated Hartree-Fock states of || (U - U_method)|psi> ||

at t = 1 a.u. in 100 steps.  Error bars are the standard deviation over the
ensemble; they are imperceptible at this scale, which is the point -- the
ranking does not depend on the initial state.

    python3 paper_figures/figure4_molecules.py            # replot from the .npz
    python3 paper_figures/figure4_molecules.py --compute  # rerun (needs ffsim/pyscf)

Outputs figures/plot_all_orders_PRL_errorbars.pdf.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from norm_aware.benchmarks import MOLECULE_LABELS
from norm_aware.style import use_molecule_style

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, "simulation_results_multi.npz")
OUT = os.path.join(ROOT, "figures")

STYLES_O0 = {
    "trotter": dict(color='black',   ls='-', marker='o', lw=2.0, ms=6, mfc='black'),
    "strang":  dict(color='#d62728', ls='-', marker='s', lw=2.0, ms=6, mfc='#d62728'),
    "lam1":    dict(color='#2ca02c', ls='-', marker='D', lw=1.8, ms=5, mfc='#2ca02c'),
}
STYLES_O1 = {
    "trotter": dict(color='black',   ls='--', marker='o', lw=1.7, ms=5,  mfc='black'),
    "strang":  dict(color='#d62728', ls='--', marker='s', lw=1.7, ms=5,  mfc='white'),
    "lam1":    dict(color='#ff7f0e', ls='--', marker='v', lw=1.7, ms=7,  mfc='white'),
    "lam_opt": dict(color='#17becf', ls='--', marker='v', lw=1.7, ms=10, mfc='#17becf'),
}
LABELS_O0 = {"trotter": r"LTS$^{[1]}$", "strang": r"PS$^{[1]}$",
             "lam1": r"$U_{\tilde{\lambda}}^{[1]}$"}
LABELS_O1 = {"trotter": r"LTS$^{[2]}$", "strang": r"PS$^{[2]}$",
             "lam1": r"$U_{\tilde{\lambda}}^{[2]}$", "lam_opt": r"OG$^{[2]}$"}


def _curves(ax, x, means, stds, styles, labels, draw_order):
    for z, key in enumerate(draw_order, start=2):
        st = styles[key]
        ax.errorbar(x, means[key], yerr=stds[key], fmt="none", ecolor=st["color"],
                    elinewidth=0.8, capsize=2.5, capthick=0.8, alpha=0.55, zorder=z)
        ax.plot(x, means[key], color=st["color"], linewidth=st["lw"],
                linestyle=st["ls"], marker=st["marker"], markersize=st["ms"],
                markerfacecolor=st["mfc"], markeredgecolor=st["color"],
                markeredgewidth=0.8, label=labels[key], zorder=z + 0.5)


def _handle(st):
    return Line2D([], [], color=st["color"], linewidth=st["lw"], linestyle=st["ls"],
                  marker=st["marker"], markersize=st["ms"],
                  markerfacecolor=st["mfc"], markeredgecolor=st["color"])


def plot(data_path=DATA, out_dir=OUT, save=True):
    blob = np.load(data_path, allow_pickle=True)
    results, meta = blob["results"].item(), blob["meta"].item()

    names = meta["mol_names"]
    # NH3 and H2O last, so the horizontal axis runs by system size
    order = [m for m in names if m not in ("NH3", "H2O")] + ["NH3", "H2O"]
    names = [m for m in order if m in names]

    def mean_std(key):
        vals = np.array([results[m][key] for m in names])
        return vals.mean(axis=1), vals.std(axis=1)

    o0_keys = {"trotter": "err_o0_trotter", "strang": "err_o0_strang",
               "lam1": "err_o0_lam1"}
    o1_keys = {"trotter": "err_o1_trotter", "strang": "err_o1_strang",
               "lam1": "err_o1_lam1", "lam_opt": "err_o1_lam_opt"}
    m0, s0 = {}, {}
    for k, key in o0_keys.items():
        m0[k], s0[k] = mean_std(key)
    m1, s1 = {}, {}
    for k, key in o1_keys.items():
        m1[k], s1[k] = mean_std(key)

    x = np.arange(len(names))

    use_molecule_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    _curves(ax, x, m0, s0, STYLES_O0, LABELS_O0, ["lam1", "strang", "trotter"])
    _curves(ax, x, m1, s1, STYLES_O1, LABELS_O1,
            ["lam_opt", "lam1", "strang", "trotter"])

    ax.set_yscale("log")
    ax.set_xlabel("Molecule")
    ax.set_ylabel(r'$\|U - \tilde{U}\|$')
    ax.set_xticks(x)
    ax.set_xticklabels([MOLECULE_LABELS.get(m, m) for m in names])
    ax.tick_params(axis='x', which='both', top=False, bottom=True, length=4)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(which='major', length=4, width=0.7)
    ax.tick_params(which='minor', length=2, width=0.5)
    ax.grid(False)

    # tick numbers on the right, axis label on the left
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.text(0.05, 0.97, rf'$\Delta t={meta["DT"]}$', transform=ax.transAxes,
            fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='black', lw=0.6))

    keys0 = ["trotter", "strang", "lam1"]
    keys1 = ["trotter", "strang", "lam1", "lam_opt"]
    leg = ax.legend([_handle(STYLES_O0[k]) for k in keys0]
                    + [_handle(STYLES_O1[k]) for k in keys1],
                    [LABELS_O0[k] for k in keys0] + [LABELS_O1[k] for k in keys1],
                    frameon=True, framealpha=1.0, edgecolor='black', fancybox=False,
                    fontsize=11, ncol=2, loc='lower right', handlelength=2.2,
                    handletextpad=0.4, labelspacing=0.35, borderpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.8)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "plot_all_orders_PRL_errorbars.pdf")
        fig.savefig(path, dpi=300)
        print("[saved]", os.path.normpath(path))
    return fig


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compute", action="store_true",
                   help="rerun the simulation (needs ffsim and pyscf)")
    if p.parse_args().compute:
        from norm_aware.benchmarks import run_benchmark
        run_benchmark(out_file=DATA)
    plot()
