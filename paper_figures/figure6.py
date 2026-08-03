"""Figure 6 -- universality of the optimal parameter lambda-tilde.

For every benchmark system, the optimal parameter is obtained from

    lambda-tilde(alpha) = argmin_lambda || U - U_OG^[2](lambda) ||_F ,

with U_OG^[2] the generalisation of Ostmeyer's scheme, Eq. (27), applied over a
total time t = 1 in N = 100 steps.

    python3 figure6.py              # replot from data/figure6_lambda_N100.npz
    python3 figure6.py --compute    # recompute (about 30 min) and replot

Outputs ../figures/og_universality_a.pdf and _b.pdf.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.optimize import minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (LAMBDA_TILDE, Stepper, all_to_all, chain, hubbard_model,
                    random_klocal_model, spin_model, square, use_pra_style,
                    FIG_W_2COL)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "figure6_lambda_N100.npz")
OUT = os.path.join(HERE, os.pardir, "figures")

NSTEPS = 100
ALPHAS = np.geomspace(1e-3, 1.0, 16)
C_ALPHA2 = (2 - np.sqrt(3)) / 8          # coefficient of Eq. (A12)

# The first eight systems all have H^(1) built from two-site terms.  The last
# two break that pattern on purpose, so the collapse is not tested on a single
# algebraic family.
MODELS = [
    (r"chain $XY$ ($N{=}8$)",           lambda: spin_model(8, chain(8), 1, -1)),
    (r"chain Ising ($N{=}8$)",          lambda: spin_model(8, chain(8), 1, 0)),
    (r"chain $XXZ$ ($N{=}8$)",          lambda: spin_model(8, chain(8), 1, -1, 0.5)),
    (r"ring $XY$ ($N{=}8$)",            lambda: spin_model(8, chain(8, True), 1, -1)),
    (r"square $3{\times}3$ $XY$",       lambda: spin_model(9, square(3, 3), 1, -1)),
    (r"ladder $2{\times}4$ $XY$",       lambda: spin_model(8, square(2, 4), 1, -1)),
    (r"all-to-all ($N{=}7$)",           lambda: spin_model(7, all_to_all(7), 1, -1)),
    (r"Hubbard 1D ($\alpha{=}U$)",      lambda: hubbard_model(5, (2, 2))),
    (r"$XY$ + transv. field ($N{=}8$)", lambda: spin_model(8, chain(8), 1, -1, 0, 0.7)),
    (r"random $k$-local ($N{=}8$)",     lambda: _random_matched()),
]


def _random_matched():
    """Random k-local model, rescaled to the norm of the XY chain so that the
    two are compared at the same perturbation strength."""
    ref = np.linalg.norm(sum(spin_model(8, chain(8), 1, -1)[1]), 2)
    return random_klocal_model(8, 12, target_norm=ref)

COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00",
          "#0072B2", "#CC79A7", "#8C7B1F", "#7F7F7F", "#8C564B"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<"]


# ---------------------------------------------------------------- optimiser
def lambda_star(st, alpha, n_steps=NSTEPS, refine=True):
    """argmin over lambda of the error after ``n_steps`` applications.

    The minimum is quadratic but very flat (curvature ~1e-3), so Brent locates
    the bracket and a local parabola through +-2e-3 extracts the vertex with
    far more signal than tightening Brent's tolerance would give.
    """
    t = 1.0 / n_steps
    u_exact = st.exact(alpha, 1.0)

    def err(lam):
        step = st.step("og", alpha, t, lam)
        return np.linalg.norm(np.linalg.matrix_power(step, n_steps) - u_exact, "fro")

    lam = minimize_scalar(err, bounds=(0.10, 0.30), method="bounded",
                          options={"xatol": 1e-7}).x
    if refine:
        h = 2e-3
        ym, y0, yp = err(lam - h), err(lam), err(lam + h)
        a = (ym - 2 * y0 + yp) / (2 * h * h)
        if a > 0:
            lam -= (yp - ym) / (2 * h) / (2 * a)
    return lam


def omelyan_lambda(alpha):
    """Analytic prediction: argmin of gamma_1^2 + gamma_2^2, Eq. (A10)."""
    def f(lam):
        g1 = alpha * (1 - 6 * lam) / 24
        g2 = -1 / 12 + lam / 2 - lam ** 2 / 2
        return g1 ** 2 + g2 ** 2
    return minimize_scalar(f, bounds=(0.05, 0.30), method="bounded",
                           options={"xatol": 1e-10}).x


def compute():
    out = {"alphas": ALPHAS}
    t0 = time.time()
    for name, build in MODELS:
        h0, terms = build()
        st = Stepper(h0, terms)
        out[name] = np.array([lambda_star(st, a) for a in ALPHAS])
        print(f"  {name:34s} lam*(a->0)={out[name][0]:.8f}  "
              f"L={len(terms):2d}  d={st.d}  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez(DATA, **out)
    print("[saved]", DATA)


# ------------------------------------------------------------------ figure
def _panel_a(ax, d, legend=False):
    fine = np.geomspace(ALPHAS[0], ALPHAS[-1], 300)
    handles = [ax.semilogx(fine, [omelyan_lambda(a) for a in fine], "k--", lw=1.6,
                           label=r"$\lambda_1(\alpha)$ (Omelyan)")[0]]
    for (name, _), c, m in zip(MODELS, COLORS, MARKERS):
        handles.append(ax.semilogx(d["alphas"], d[name], marker=m, color=c,
                                   ms=3.0, lw=1.0, mec=c, mfc=c, label=name)[0])
    ax.axhline(LAMBDA_TILDE, ls=":", c="0.55", lw=1.0)
    ax.set_ylim(0.2085, 0.2175)
    ax.text(2.2e-3, LAMBDA_TILDE - 2.2e-4,
            r"$\widetilde{\lambda}=\frac{1}{2}-\frac{\sqrt{3}}{6}$",
            va="top", ha="left", fontsize=7.5, color="0.45")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\widetilde{\lambda}(\alpha)$")
    ax.text(0.05, 0.93, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    if legend:
        ax.legend(handles=handles, fontsize=5.3, ncol=2, loc="upper left",
                  frameon=True, framealpha=0.92, edgecolor="0.75",
                  handlelength=1.5, labelspacing=0.26, columnspacing=0.9,
                  handletextpad=0.4, borderpad=0.4, bbox_to_anchor=(0.03, 0.90))
    return handles


def _panel_b(ax, d):
    for (name, _), c, m in zip(MODELS, COLORS, MARKERS):
        ax.loglog(d["alphas"], np.abs(d[name] - LAMBDA_TILDE), marker=m, color=c,
                  ms=3.0, lw=1.0, mec=c, mfc=c)
    # The floor is not a discretisation effect: it barely moves with N (5.1e-6
    # at N=100 vs 4.0e-6 at N=300, while dt^2 changes by nine).  It is set by
    # how precisely the minimiser can be resolved, so it is read off the data.
    floor = max(np.abs(d[name][:4] - LAMBDA_TILDE).max() for name, _ in MODELS)
    guide = np.geomspace(1e-3, 1.0, 2)
    ax.loglog(guide, C_ALPHA2 * guide ** 2, "k--", lw=1.6, zorder=20)
    ax.text(5.5e-2, 7e-4, r"$\propto\alpha^{2}$", fontsize=8.5, zorder=21)
    ax.axhspan(1e-8, floor, color="0.9", zorder=0)
    ax.text(1.1e-3, floor * 1.2, r"finite-$N$ floor", fontsize=6, color="0.5",
            va="bottom")
    ax.set_ylim(1e-8, 5e-2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$|\widetilde{\lambda}(\alpha)-\widetilde{\lambda}|$")
    ax.text(0.05, 0.93, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")


def plot():
    d = np.load(DATA)
    os.makedirs(OUT, exist_ok=True)
    use_pra_style()
    for tag, fn in (("a", lambda ax: _panel_a(ax, d, legend=True)),
                    ("b", lambda ax: _panel_b(ax, d))):
        fig, ax = plt.subplots(figsize=(FIG_W_2COL / 2, 3.1))
        fn(ax)
        fig.tight_layout()
        path = os.path.join(OUT, f"og_universality_{tag}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print("[saved]", os.path.normpath(path))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compute", action="store_true",
                   help="recompute the data instead of reusing data/")
    if p.parse_args().compute:
        compute()
    plot()
