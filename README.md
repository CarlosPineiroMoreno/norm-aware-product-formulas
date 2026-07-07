# Norm-Aware Product Formulas for Hamiltonian Simulations

Companion code for the paper *"Norm-Aware Product Formulas for Hamiltonian
Simulations"* (Piñeiro Moreno, Krause, Bande).

This repository reproduces the numerical results and figures: a set of
Trotter-Suzuki product formulas (LTS, PS, $U_{\tilde\lambda}$, OG) that
exploit a norm hierarchy between the one-body and two-body parts of a
Hamiltonian to reduce simulation error, benchmarked on a 2D Heisenberg/XYZ
spin model and on molecular electronic-structure Hamiltonians.

## Structure

- **`trotter_methods.py`** -- the product formulas themselves. Pure `ffsim` +
  `numpy`, no other dependencies:
  - `simulate_trotter_double_factorized_LTS` -- Lie-Trotter-Suzuki
  - `simulate_trotter_double_factorized_PS` -- Partial-Strang
  - `simulate_trotter_double_factorized_PS_RANDOM` -- Partial-Strang Random
    (exploratory variant, not one of the 5 compared methods in the paper)
  - `simulate_trotter_double_factorized_U_lambda` -- optimized-$\lambda$ decomposition
  - `simulate_trotter_double_factorized_OG` -- Omelyan-General (fixed 2nd order)
  - `TrotterHamiltonian` -- a small adapter around
    `ffsim.DoubleFactorizedHamiltonian` exposing the precomputed
    one-body eigendecomposition the step functions need.

  All of LTS/PS/PS_RANDOM/U_lambda accept an `order` parameter (0/1 = first
  order, 2 = second order, 4 = fourth order, ...). OG has no `order`
  parameter -- it's inherently the fixed second-order construction described
  in the paper.

- **`ISING_Model_Paper.ipynb`** -- 2D Heisenberg/XYZ spin model ($L\times L$
  lattice). Self-contained (no dependency on `trotter_methods.py` -- it
  re-implements the same methods directly against the spin Hamiltonian via
  exact tensor contraction rather than double factorization). Produces:
  - `figures/Errors_screen_L3_t1.0.pdf`
  - `figures/Ratios_PRL_1col_L3_t1.0.pdf`

  Caches expensive sweeps under `results/<run-name>/` so re-running with the
  same parameters reuses cached data instead of recomputing.

- **`molecular_hamiltonian_simulation.ipynb`** -- molecular
  double-factorized Hamiltonians (H2, LiH, BeH2, H2O, NH3; STO-3G). Imports
  the product formulas from `trotter_methods.py`. Produces:
  - the double-factorization benchmark table (Table I in the paper)
  - `figures/plot_all_orders_PRL_errorbars.pdf`
  - `figures/nmin_vs_molecule.pdf` (appendix; uses pre-tabulated values from
    the Pauli-grouping/partitioning analysis, not recomputed by this notebook)

  Raw per-sample errors are cached in `simulation_results_multi.npz`.

## Installation

```bash
pip install -r requirements.txt
```

Tested with the package versions pinned in `requirements.txt`; `ffsim` and
`pyscf` are the two dependencies most likely to need a specific version.

## Reproducing a figure

1. Open the relevant notebook and run it top to bottom.
2. Figures are written to `figures/`; intermediate data to `results/` or
   `simulation_results_multi.npz`. Delete the cache file/directory for a
   given run if you want to force a clean recomputation.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{pineiromoreno2025normaware,
  title   = {Norm-Aware Product Formulas for Hamiltonian Simulations},
  author  = {Pi{\~n}eiro Moreno, Carlos A. and Krause, Pascal and Bande, Annika},
  journal = {in preparation},
  year    = {2025}
}
```

## License

-- see `LICENSE`.
