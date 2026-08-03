"""Figure 2(b) -- error ratios between formulas, against the coupling alpha.

Each curve is one formula's error divided by another's, so a value below one
means the numerator wins.  The ratios come from the same sweep as panel (a)
and are read from the same cache.

THRIFT2 ratios are excluded from the figure: THRIFT is not efficiently
implementable in the settings this comparison is meant to inform, so its
second-order ratio would not be actionable.

    python3 paper_figures/figure2b_ratios.py            # replot from results/
    python3 paper_figures/figure2b_ratios.py --compute  # rerun the sweep first

Outputs figures/Ratios_PRL_1col_L3_t1.pdf.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from norm_aware.style import RATIO_STYLES, use_ratio_style
from norm_aware.sweeps import load_alpha_sweep, run_alpha_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
RUN_DIR = os.path.join(ROOT, "results", "Jsweep_XXMYY_L3_t1.0_n100")
OUT = os.path.join(ROOT, "figures")


def plot(data, width=7, height=5, two_column=False, out_dir=OUT, save=True):
    alphas = np.array(data['J_values'])
    ratios = data.get('ratios_fixed', {})
    L, t, n_fixed = data['L'], data['t'], data['n_fixed']
    dt_ref = t / n_fixed

    if not ratios:
        raise ValueError("no ratios_fixed in the cache; rerun with --compute")

    ratios = {k: v for k, v in ratios.items() if 'THRIFT2' not in k}
    n_done = len(next(iter(ratios.values())))
    alphas = alphas[:n_done]

    use_ratio_style()
    fig, ax = plt.subplots(figsize=(width, height))

    for i, (name, values) in enumerate(ratios.items()):
        st = RATIO_STYLES[i % len(RATIO_STYLES)]
        ax.plot(alphas, np.array(values[:n_done]),
                linestyle=st['ls'], marker=st['marker'], color=st['color'],
                linewidth=st['lw'], markersize=st['ms'],
                markerfacecolor='white', markeredgewidth=0.8,
                markeredgecolor=st['color'], label=name, zorder=i + 2)

    ax.axhline(1.0, color='black', ls='--', lw=0.9, alpha=0.5,
               label='ratio $= 1$', zorder=1)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Vert U_\mathrm{num} \Vert_F \,/\, \Vert U_\mathrm{den} \Vert_F$')
    ax.set_title(rf'$\Delta t={dt_ref:.3f}$,  $L={L}$,  $t={t}$', fontsize=10, pad=4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='major', length=4, width=0.7)
    ax.tick_params(which='minor', length=2, width=0.5)
    ax.grid(False)

    leg = ax.legend(frameon=True, framealpha=1.0, edgecolor='black',
                    fancybox=False, fontsize=8 if two_column else 7,
                    ncol=1, loc='best', handlelength=2.2, handletextpad=0.4,
                    labelspacing=0.35, borderpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.5)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        tag = '2col' if two_column else '1col'
        path = os.path.join(out_dir, f'Ratios_PRL_{tag}_L{L}_t{t:g}.pdf')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print("[saved]", os.path.normpath(path))
    return fig


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compute", action="store_true",
                   help="rerun the alpha sweep instead of reusing results/")
    if p.parse_args().compute:
        run_alpha_sweep(L=3, t=1.0, n_fixed=100,
                        results_dir=os.path.join(ROOT, "results"))
    plot(load_alpha_sweep(RUN_DIR))
