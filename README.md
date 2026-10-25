# Norm-Aware Product Formulas for Hamiltonian Simulations

Companion code for the paper *"Norm-Aware Product Formulas for Hamiltonian
Simulations"* (Piñeiro Moreno, Krause, Bande).

A set of Trotter–Suzuki product formulas (LTS, PS, THRIFT, $U_{\tilde\lambda}$,
OG) that exploit a norm hierarchy between the dominant and the remaining parts
of a Hamiltonian to reduce simulation error, benchmarked on a 2D anisotropic
spin lattice and on molecular electronic-structure Hamiltonians.

## Layout

The reusable code is the `norm_aware` package; the scripts under
`paper_figures/` are thin drivers, one per figure, and the two notebooks drive
the same functions interactively.

```
norm_aware/          the library
  constants.py       lambda-tilde and the list of the nine formulas
  spin.py            the 2D lattice, its gate cache, the exact propagator
  evolver.py         the nine formulas on that lattice, by tensor contraction
  pauli.py           Pauli-string machinery and the benchmark models of Fig. 6
  df.py              the formulas for double-factorized molecular Hamiltonians
  molecules.py       double-factorization quality, alpha-bar, random states
  benchmarks.py      the molecular geometries, the state-error run, Table I
  sweeps.py          the alpha sweep and its on-disk cache
  style.py           the plotting conventions shared by every figure

paper_figures/       one script per figure, named after the paper's numbering
  figure2a_errors.py       error against alpha
  figure2b_ratios.py       ratios between formulas
  figure3_dtsweep_band.py  error against the time step, banded over alpha
  figure4_molecules.py     state error per molecule
  figure6.py               universality of the optimal lambda-tilde
  figure7.py               effective exponent of the local error in alpha
  data/                    precomputed results for Figs. 6 and 7
```

The formulas are implemented **twice**, against two different
representations: `evolver` builds full propagators for the spin lattice, and
`pauli` works from Pauli strings for the benchmark models of Fig. 6. The
duplication is deliberate — the collapse onto $\tilde\lambda$ is then not an
artefact of one implementation. `constants.py` holds the values both share so
they cannot drift apart.

## Reproducing a figure

Every script replots from cached data in seconds, and recomputes with
`--compute`:

```bash
python3 paper_figures/figure2a_errors.py             # replot
python3 paper_figures/figure6.py --compute           # recompute, then replot
```

Figures are written to `figures/`. Recomputation costs roughly: Fig. 2 several
hours (both panels share one sweep), Fig. 3 several hours more (seven sweeps,
one per coupling), Fig. 4 about an hour, Fig. 6 about 30 minutes, Fig. 7 about
20 minutes.

Caches live under `results/`: the alpha sweep of Fig. 2 checkpoints after
every alpha, so an interrupted run resumes, and Fig. 3 reads one directory per
coupling. The molecular errors are in `simulation_results_multi.npz`, and
Figs. 6 and 7 read `paper_figures/data/`.

## Installation

```bash
pip install -r requirements.txt
```

`numpy`, `scipy` and `matplotlib` are enough for the spin-lattice figures
(2, 3, 6 and 7). The molecular parts — `norm_aware.df`, `norm_aware.benchmarks`
and Fig. 4's `--compute` path — also need `ffsim` and `pyscf`; those are the
two dependencies most likely to need a specific version.

## What the appendix figures measure

**Figure 6** — for every benchmark system the optimal parameter is

$$\tilde\lambda(\alpha)=\arg\min_\lambda\|U-U^{[2]}_{\mathrm{OG}}(\lambda)\|_F,$$

with $U^{[2]}_{\mathrm{OG}}$ the generalisation of Ostmeyer's scheme, Eq. (27),
over a total time $t=1$ in $N=100$ steps. All systems collapse onto
$\tilde\lambda=\tfrac12-\tfrac{\sqrt3}{6}$ as $\alpha\to0$.

The first eight systems have $H^{(1)}$ built entirely from two-site terms. The
last two — an $XY$ chain with a transverse field inside $H^{(1)}$, and a random
$k$-local Hamiltonian — break that pattern deliberately, so the collapse is not
demonstrated on a single algebraic family.

**Figure 7** — both panels use the *local* error of a **single** application,

$$\varepsilon(t)=\|e^{-iHt}-U_{\mathrm{step}}(t)\|_F,$$

which is the quantity the theorems and Remark (i) refer to. The accumulated
error of an $N$-step run scales differently in $\alpha$, so using it would not
test the remark.

## Three numerical limits worth knowing

**The floor in Fig. 6(b)** is *not* a discretisation effect. Going from
$N=100$ to $N=300$ moves it only from $5.1\times10^{-6}$ to
$4.0\times10^{-6}$, while $\Delta t^{2}$ changes by a factor of nine. Near its
optimum the error is quadratic with a small curvature $a\sim10^{-3}$, so a
perturbation $\delta$ of the evaluated norm displaces the minimiser by
$\sqrt{\delta/a}$; with $\delta$ at the level of accumulated round-off this
gives $\sim10^{-6}$. The shaded band is therefore read off the data rather than
computed from a formula.

**The smallest usable $t$ in Fig. 7** is $10^{-3}$. At $t=10^{-4}$ the genuine
local error of the second-order $\alpha^{2}$-suppressed formulas is
$\sim2\times10^{-14}$, at the round-off floor, and they measure 35–43 % high.

**The order of the bonds matters.** `norm_aware.spin.SpinLattice2D` sweeps all
horizontal bonds first, then all vertical ones; `norm_aware.pauli.square` does
the same. The terms do not commute, so a different sweep order changes the
result at the order of the splitting error itself — below 2.2 % at
$\alpha=0.2$ and below 0.15 % in the perturbative regime, but fixed here for
reproducibility. Relatedly, `bond_terms` groups the Pauli strings of one bond
into a single $H_k^{(1)}$: for eight of the nine formulas the grouping is
irrelevant, since the strings of a bond commute, but for THRIFT it is not,
because its block $e^{+iH_0t}e^{-i(H_0+\alpha H_k)t}$ does not factorise.

## A note on Figure 3

The original plotting code for this figure was lost; only its inputs survived,
as the seven `results/dtsweep_XXMYY_L3_J*` sweeps. `figure3_dtsweep_band.py`
was rebuilt from them. It reads the very files the published figure was drawn
from, so the curves are the same numbers, but it is not byte-identical to the
published PDF: the axes box lands within 2 px and the legend is laid out
slightly differently.

## Citation

```bibtex
@article{,
  title   = {Norm-Aware Product Formulas for Hamiltonian Simulations},
  author  = {Pi{\~n}eiro Moreno, Carlos A. and Krause, Pascal and Bande, Annika},
  journal = {arXiv:2610.31070},
  year    = {2026},
  url     = {http://arxiv.org/abs/2610.31070}
}
```
