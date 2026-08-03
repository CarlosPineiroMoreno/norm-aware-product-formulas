"""The molecular benchmark: geometries, the state-error run, and Table I.

Five closed-shell molecules in STO-3G, evolved to t = 1 a.u. in 100 steps.
Unlike the spin lattice, ``ffsim`` evolves states rather than operators, so
the error reported here is the state distance

    || (U - U_method) |psi> ||

averaged over ``n_samples`` Haar-rotated Hartree--Fock determinants.  Seeds
are derived from a single master seed, so the ensemble is reproducible:

    seed = master_seed * 10_000 + molecule_index * 1_000 + sample_index

Requires ``ffsim`` and ``pyscf``.
"""
from __future__ import annotations

import numpy as np

from .constants import LAMBDA_TILDE
from .molecules import make_random_state

__all__ = ["MOLECULES", "MOLECULE_LABELS", "ERROR_KEYS", "simulate_molecule",
           "run_benchmark", "df_benchmark_table",
           "TIME_FINAL", "DT", "N_STEPS", "LAMBDA_1", "LAMBDA_2", "LAMBDA_OPT"]

TIME_FINAL = 1.0
DT = 1e-2
N_STEPS = int(round(TIME_FINAL / DT))        # 100
N_SAMPLES = 10
MASTER_SEED = 42

LAMBDA_1 = LAMBDA_TILDE                       # 0.211325..., the paper's value
LAMBDA_2 = 0.193183                           # Omelyan's alpha = 1 value
LAMBDA_OPT = LAMBDA_TILDE

#: Geometries in Angstrom, all STO-3G.
MOLECULES = {
    "H2": {"basis": "sto-3g", "atom": """
H 0.0000 0.0000 -0.3700
H 0.0000 0.0000  0.3700
"""},
    "LiH": {"basis": "sto-3g", "atom": """
Li 0.0000 0.0000 0.0000
H  0.0000 0.0000 3.0100
"""},
    "BeH2": {"basis": "sto-3g", "atom": """
Be 0.0000 0.0000  0.0000
H  0.0000 0.0000  1.3264
H  0.0000 0.0000 -1.3264
"""},
    "H2O": {"basis": "sto-3g", "atom": """
O 0.0000  0.0000  0.0000
H 0.0000  0.7572  0.5865
H 0.0000 -0.7572  0.5865
"""},
    "NH3": {"basis": "sto-3g", "atom": """
N  0.0000  0.0000  0.1173
H  0.0000  0.9391 -0.2730
H  0.8136 -0.4696 -0.2730
H -0.8136 -0.4696 -0.2730
"""},
}

#: How each molecule is labelled in the figure.
MOLECULE_LABELS = {"H2": r"$H_2$", "LiH": r"$LiH$", "BeH2": r"$BeH_2$",
                   "NH3": r"$NH_3$", "H2O": r"$H_2O$"}

#: Error keys, in the order the figure draws them.
ERROR_KEYS = ["err_o0_trotter", "err_o0_strang", "err_o0_lam1", "err_o0_lam2",
              "err_o1_trotter", "err_o1_strang", "err_o1_lam1", "err_o1_lam_opt"]


def simulate_molecule(atom, basis, mol_index, n_samples=N_SAMPLES,
                      time_final=TIME_FINAL, n_steps=N_STEPS,
                      master_seed=MASTER_SEED, verbose=True):
    """Every formula's state error for one molecule, over ``n_samples`` states."""
    import ffsim
    import scipy.sparse.linalg
    from pyscf import gto, scf

    from .df import (TrotterHamiltonian,
                     simulate_trotter_double_factorized_OG,
                     simulate_trotter_double_factorized_PS,
                     simulate_trotter_double_factorized_U_lambda)

    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()

    mol_data = ffsim.MolecularData.from_scf(mf)
    norb, nelec = mol_data.norb, mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    df_ham = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(mol_hamiltonian)
    ham = TrotterHamiltonian.from_ffsim(df_ham)

    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    trace_h = ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec)

    # With optimize=False the factorization is exact, so these sit at machine
    # precision; they are the check that nothing was truncated.
    recon = df_ham.to_molecular_hamiltonian()
    err_one = float(np.linalg.norm(recon.one_body_tensor - mol_hamiltonian.one_body_tensor))
    err_two = float(np.linalg.norm(recon.two_body_tensor - mol_hamiltonian.two_body_tensor))

    raw = {k: [] for k in ERROR_KEYS}

    for sample in range(n_samples):
        rng = np.random.default_rng(master_seed * 10_000 + mol_index * 1_000 + sample)
        psi = make_random_state(norb, nelec, rng)

        exact = scipy.sparse.linalg.expm_multiply(
            -1j * time_final * linop, psi, traceA=-1j * time_final * trace_h)

        def err(state):
            return float(np.linalg.norm(state - exact))

        common = dict(norb=norb, nelec=nelec, n_steps=n_steps)

        raw["err_o0_trotter"].append(err(ffsim.simulate_trotter_double_factorized(
            psi, df_ham, time_final, order=0, **common)))
        raw["err_o0_strang"].append(err(simulate_trotter_double_factorized_PS(
            psi, ham, time_final, order=0, **common)))
        raw["err_o0_lam1"].append(err(simulate_trotter_double_factorized_U_lambda(
            psi, ham, time_final, Lambda=LAMBDA_1, order=0, **common)))
        raw["err_o0_lam2"].append(err(simulate_trotter_double_factorized_U_lambda(
            psi, ham, time_final, Lambda=LAMBDA_2, order=0, **common)))

        raw["err_o1_trotter"].append(err(ffsim.simulate_trotter_double_factorized(
            psi, df_ham, time_final, order=1, **common)))
        raw["err_o1_strang"].append(err(simulate_trotter_double_factorized_PS(
            psi, ham, time_final, order=1, **common)))
        raw["err_o1_lam1"].append(err(simulate_trotter_double_factorized_U_lambda(
            psi, ham, time_final, Lambda=LAMBDA_1, order=1, **common)))
        raw["err_o1_lam_opt"].append(err(simulate_trotter_double_factorized_OG(
            psi, ham, time_final, lambda_opt=LAMBDA_OPT, **common)))

        if verbose:
            print(f"    sample {sample+1:>3}/{n_samples}  "
                  f"o0_trotter={raw['err_o0_trotter'][-1]:.3e}  "
                  f"o1_trotter={raw['err_o1_trotter'][-1]:.3e}", flush=True)

    return {**{k: np.array(v) for k, v in raw.items()},
            "err_one_body": err_one, "err_two_body": err_two,
            "norb": norb, "nelec": nelec, "basis": basis, "N_SAMPLES": n_samples}


def run_benchmark(out_file="simulation_results_multi.npz", n_samples=N_SAMPLES,
                  molecules=None, verbose=True):
    """Run every molecule and cache the raw per-sample errors.

    No averaging is done here -- the figure decides how to aggregate.
    """
    molecules = molecules if molecules is not None else MOLECULES
    results = {}
    for idx, (name, cfg) in enumerate(molecules.items()):
        if verbose:
            print(f"\n-> {name}  (basis={cfg['basis']})", flush=True)
        results[name] = simulate_molecule(cfg["atom"], cfg["basis"], idx,
                                          n_samples=n_samples, verbose=verbose)

    meta = {"TIME_FINAL": TIME_FINAL, "DT": DT, "N_STEPS_SIM": N_STEPS,
            "N_SAMPLES": n_samples, "reference": "expm_multiply (exact)",
            "LAMBDA_1": LAMBDA_1, "LAMBDA_2": LAMBDA_2, "LAMBDA_OPT": LAMBDA_OPT,
            "MASTER_SEED": MASTER_SEED, "mol_names": list(results.keys())}

    np.savez(out_file, results=results, meta=meta)
    if verbose:
        print(f"\n[saved] {out_file}")
    return results, meta


def df_benchmark_table(molecules=None, verbose=True):
    """Table I: orbitals, DF terms, reconstruction error and alpha-bar.

    ``alpha_bar`` is the quantity that decides whether the norm-aware
    formulas help: the smaller it is, the larger the gain over plain LTS.
    """
    import ffsim
    from pyscf import gto, scf

    from .molecules import compute_alpha_bar, two_body_frobenius_error

    molecules = molecules if molecules is not None else MOLECULES
    rows = {}
    if verbose:
        print(f"{'molecule':<10} {'N_orb':>6} {'L':>5} "
              f"{'||V-Vtilde||_F':>16} {'alpha_bar':>12}")
        print("-" * 54)

    for name, cfg in molecules.items():
        mol = gto.Mole()
        mol.build(atom=cfg["atom"], basis=cfg["basis"], unit="Angstrom", verbose=0)
        mol_data = ffsim.MolecularData.from_scf(scf.RHF(mol).run())
        mol_hamiltonian = mol_data.hamiltonian
        df_ham = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
            mol_hamiltonian)

        rows[name] = {
            "norb": mol_data.norb,
            "n_df_terms": len(df_ham.diag_coulomb_mats),
            "frobenius_error": two_body_frobenius_error(mol_hamiltonian, df_ham),
            "alpha_bar": compute_alpha_bar(df_ham),
        }
        if verbose:
            r = rows[name]
            print(f"{name:<10} {r['norb']:>6} {r['n_df_terms']:>5} "
                  f"{r['frobenius_error']:>16.3e} {r['alpha_bar']:>12.4f}")

    return rows
