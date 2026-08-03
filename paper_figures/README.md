# Appendix figures

Self-contained code for the two appendix figures of the paper. Nothing here
depends on the notebooks or on `trotter_methods.py`; the only requirements are
NumPy, SciPy, Matplotlib, and — for the Fermi–Hubbard benchmark of Figure 6 —
`ffsim`.

| file | what it does |
| --- | --- |
| `common.py` | benchmark Hamiltonians, the nine product formulas, plotting style |
| `figure6.py` | universality of the optimal $\tilde\lambda$ |
| `figure7.py` | effective exponent of the local error in $\alpha$ |
| `data/` | precomputed results, so the figures can be replotted in seconds |

## Usage

```bash
python3 figure6.py              # replot from data/  (seconds)
python3 figure6.py --compute    # recompute from scratch (~30 min), then replot
```

Same interface for `figure7.py` (~20 min to recompute). Figures are written to
`../figures/`.

## What each figure measures

**Figure 6** — for every benchmark system, the optimal parameter is obtained as

$$\tilde\lambda(\alpha)=\arg\min_\lambda\|U-U^{[2]}_{\mathrm{OG}}(\lambda)\|_F,$$

with $U^{[2]}_{\mathrm{OG}}$ the generalisation of Ostmeyer's scheme, Eq. (27),
applied over a total time $t=1$ in $N=100$ steps. All systems collapse onto
$\tilde\lambda=\tfrac12-\tfrac{\sqrt3}{6}$ as $\alpha\to0$.

The first eight systems have $H^{(1)}$ built entirely from two-site terms. The
last two — an $XY$ chain with a transverse field inside $H^{(1)}$, and a random
$k$-local Hamiltonian — break that pattern deliberately, so the collapse is not
demonstrated on a single algebraic family.

**Figure 7** — both panels use the *local* error of a **single** application,

$$\varepsilon(t)=\|e^{-iHt}-U_{\mathrm{step}}(t)\|_F,$$

which is the quantity the theorems and Remark (i) refer to. The accumulated
error of an $N$-step run scales differently in $\alpha$, so using it would not
test the remark. Panel (a) fits $\varepsilon\propto\alpha^{S}$ over
$\alpha\in[0.02,0.1]$ for each $t$; panel (b) shows the local logarithmic slope
at fixed $t=10^{-3}$.

## Two numerical limits worth knowing

**Figure 6, the floor in panel (b).** It is *not* a discretisation effect. Going
from $N=100$ to $N=300$ moves it only from $5.1\times10^{-6}$ to
$4.0\times10^{-6}$, while $\Delta t^{2}$ changes by a factor of nine. Near its
optimum the error is quadratic with a small curvature $a\sim10^{-3}$, so a
perturbation $\delta$ of the evaluated norm displaces the minimiser by
$\sqrt{\delta/a}$; with $\delta$ at the level of accumulated round-off this
gives $\sim10^{-6}$. The shaded band is therefore read off the data rather than
computed from a formula.

**Figure 7, the smallest usable $t$.** $t=10^{-3}$ is close to the limit of
double precision. At $t=10^{-4}$ the genuine local error of the second-order
$\alpha^{2}$-suppressed formulas is $\sim2\times10^{-14}$, at the round-off
floor, and they measure 35–43 % high — checked against the $t^{\,\mathrm{order}+1}$
extrapolation from a known-clean point.

## Ordering of the terms

The bonds are swept in the order given by `common.square`: all horizontal bonds
first, then all vertical ones. The terms do not commute, so a different sweep
order changes the result at the order of the splitting error itself. The effect
on the reported errors is below 2.2 % at $\alpha=0.2$ and below 0.15 % in the
perturbative regime, but the order is fixed here for reproducibility.

`common.bond_terms` groups the Pauli strings of one bond into a single
$H_k^{(1)}$. For eight of the nine formulas the grouping is irrelevant, since
the strings of a bond commute; for THRIFT it is not, because its block
$e^{+iH_0t}e^{-i(H_0+\alpha H_k)t}$ does not factorise.
