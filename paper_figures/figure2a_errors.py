"""Figure 2(a) -- error of every formula against the coupling alpha.

    || U - U_method ||_F   at fixed dt = t / n_steps,

for the 3x3 open XY lattice of Eq. (29), together with the fitted power law
err = a alpha^n whose exponent is quoted in the legend.

    python3 paper_figures/figure2a_errors.py            # replot from results/
    python3 paper_figures/figure2a_errors.py --compute  # rerun the sweep first

Outputs figures/Errors_screen_L3_t1.pdf.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from norm_aware.style import build_sweep_styles, use_error_style
from norm_aware.sweeps import fit_power_law, load_alpha_sweep, run_alpha_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
RUN_DIR = os.path.join(ROOT, "results", "Jsweep_XXMYY_L3_t1.0_n100")
OUT = os.path.join(ROOT, "figures")


def plot(data, out_dir=OUT, save=True):
    alphas = np.array(data['J_values'])
    methods = data['all_methods']
    errors = data['errors_fixed']
    L, t = data['L'], data['t']

    # a sweep may have been interrupted: keep only the alphas actually done
    n_done = len(errors[methods[0]])
    alphas = alphas[:n_done]

    styles, labels = build_sweep_styles(methods)
    draw_order = sorted(methods, key=lambda m: styles.get(m, {}).get('order', 99))

    use_error_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for z, m in enumerate(draw_order, start=2):
        err = np.array(errors[m])[:n_done]
        st = styles.get(m, dict(color='gray', ls='-', marker='o', lw=1.8, ms=5))

        fit = fit_power_law(alphas, err)
        fit_label = ''
        if fit:
            a_fit, n_fit, grid, curve = fit
            fit_label = rf'  $({a_fit:.2e} \cdot \alpha^{{{round(n_fit, 4):.4f}}})$'

        ax.plot(alphas, err, linestyle=st['ls'], marker=st['marker'],
                color=st['color'], linewidth=st['lw'], markersize=st['ms'],
                label=labels.get(m, m) + fit_label, zorder=z)
        if fit:
            ax.plot(grid, curve, ':', color=st['color'], linewidth=1.4,
                    alpha=0.55, zorder=z)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$', fontsize=16)
    ax.set_ylabel(r'$\|U - \tilde{U}\|_F$', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, length=5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='lightgray',
              fontsize=11, ncol=2, loc='best')
    fig.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'Errors_screen_L{L}_t{t:g}.pdf')
        fig.savefig(path, bbox_inches='tight', dpi=150)
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
