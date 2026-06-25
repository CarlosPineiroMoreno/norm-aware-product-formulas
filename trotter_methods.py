"""Norm-hierarchy Trotter-Suzuki product formulas for double-factorized Hamiltonians.

Companion code for the paper. Depends only on `ffsim`, `numpy`, and the
standard library -- no private/internal packages.

Methods (matching the paper's notation):
    LTS        Lie-Trotter-Suzuki (the textbook product formula)
    PS         Partial-Strang
    PS_RANDOM  Partial-Strang Random (PS with the sandwiching term chosen at random)
    U_lambda   optimized-lambda decomposition, U_{lambda~}
    OG         Omelyan-General (Theorem 1 in the paper)

All of LTS/PS/PS_RANDOM/U_lambda accept an `order` parameter via ffsim's own
`simulate_trotter_step_iterator`:
    order=0  ->  1st order (LTS^[1], PS^[1], U_lambda^[1], ...)
    order=1  ->  2nd order (LTS^[2], PS^[2], U_lambda^[2], ...) -- a single
                 Strang/forward-backward symmetrization of the order=0 seed
    order=2, 4, ...  ->  4th, 8th, ... order (recursive Suzuki composition;
                 each increment beyond 1 doubles the order)
OG is inherently a fixed second-order (4-group forward/backward)
construction, so it has no `order` parameter.

Usage:
    import ffsim
    from trotter_methods import TrotterHamiltonian, simulate_trotter_double_factorized_PS

    df_ham = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian, optimize=False
    )
    ham = TrotterHamiltonian.from_ffsim(df_ham)
    vec = simulate_trotter_double_factorized_PS(
        initial_state, ham, time=1.0, norb=norb, nelec=nelec, n_steps=100, order=1
    )
"""

from __future__ import annotations

import cmath
import dataclasses
from typing import Optional, Union

import numpy as np
from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
)
from ffsim.trotter._util import simulate_trotter_step_iterator

__all__ = [
    "TrotterHamiltonian",
    "simulate_trotter_double_factorized_LTS",
    "simulate_trotter_double_factorized_PS",
    "simulate_trotter_double_factorized_PS_RANDOM",
    "simulate_trotter_double_factorized_U_lambda",
    "simulate_trotter_double_factorized_OG",
]


@dataclasses.dataclass(frozen=True)
class TrotterHamiltonian:
    """Double-factorized Hamiltonian data needed by the Trotter steps below.

    A thin adapter so these functions don't depend on any custom Hamiltonian
    class: build one directly from ffsim's own `DoubleFactorizedHamiltonian`.
    """

    one_body_energies: np.ndarray
    one_body_basis_change: np.ndarray
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    nuc_constant: float

    @staticmethod
    def from_ffsim(df_hamiltonian) -> "TrotterHamiltonian":
        """Build from `ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(...)`."""
        one_body_energies, one_body_basis_change = np.linalg.eigh(
            df_hamiltonian.one_body_tensor
        )
        return TrotterHamiltonian(
            one_body_energies=one_body_energies,
            one_body_basis_change=one_body_basis_change,
            diag_coulomb_mats=df_hamiltonian.diag_coulomb_mats,
            orbital_rotations=df_hamiltonian.orbital_rotations,
            nuc_constant=df_hamiltonian.constant,
        )


def _apply_one_body_term(
    vec: np.ndarray,
    current_basis: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exp[-itH0] (time can be positive or negative)."""
    vec = apply_orbital_rotation(
        vec,
        one_body_basis_change.T.conj() @ current_basis,
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    vec = apply_num_op_sum_evolution(
        vec, one_body_energies, time, norb=norb, nelec=nelec, copy=False
    )
    return vec, one_body_basis_change


def _apply_two_body_term(
    vec: np.ndarray,
    current_basis: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exp[-itH1^gamma] (time can be positive or negative)."""
    vec = apply_orbital_rotation(
        vec,
        orbital_rotation.T.conj() @ current_basis,
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    vec = apply_diag_coulomb_evolution(
        vec, diag_coulomb_mat, time, norb=norb, nelec=nelec, copy=False
    )
    return vec, orbital_rotation


def _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn):
    """Shared driver: loop over Trotter steps and apply ``step_fn`` each time."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}.")
    if copy:
        vec = vec.copy()
    if n_steps == 0:
        return vec

    step_time = time / n_steps
    current_basis = np.eye(norb, dtype=complex)
    vecs = [] if save_intermediate_states else None
    accumulated_phase = 0.0

    for _ in range(n_steps):
        vec, current_basis = step_fn(vec, current_basis, step_time)
        accumulated_phase += step_time * hamiltonian.nuc_constant

        if save_intermediate_states:
            vec_rotated = apply_orbital_rotation(
                vec, current_basis, norb=norb, nelec=nelec, copy=True
            )
            vec_rotated *= cmath.exp(-1j * accumulated_phase)
            vecs.append(vec_rotated)

    vec = apply_orbital_rotation(vec, current_basis, norb=norb, nelec=nelec, copy=False)
    vec *= cmath.exp(-1j * accumulated_phase)
    return vecs if save_intermediate_states else vec


# ---------------------------------------------------------------------------
# LTS -- Lie-Trotter-Suzuki
# ---------------------------------------------------------------------------

def simulate_trotter_double_factorized_LTS(
    vec: np.ndarray,
    hamiltonian: TrotterHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    order: int = 0,
    copy: bool = True,
    save_intermediate_states: bool = False,
) -> Union[np.ndarray, list]:
    """Lie-Trotter-Suzuki (LTS): H0, H1^1, ..., H1^Gamma (symmetrized for order>0)."""
    n_terms = 1 + len(hamiltonian.diag_coulomb_mats)

    def step_fn(vec, current_basis, step_time):
        for term_index, term_time in simulate_trotter_step_iterator(n_terms, step_time, order):
            if term_index == 0:
                vec, current_basis = _apply_one_body_term(
                    vec, current_basis,
                    hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
                    term_time, norb, nelec,
                )
            else:
                gamma = term_index - 1
                vec, current_basis = _apply_two_body_term(
                    vec, current_basis,
                    hamiltonian.diag_coulomb_mats[gamma], hamiltonian.orbital_rotations[gamma],
                    term_time, norb, nelec,
                )
        return vec, current_basis

    return _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn)


# ---------------------------------------------------------------------------
# PS -- Partial-Strang
# ---------------------------------------------------------------------------

def simulate_trotter_double_factorized_PS(
    vec: np.ndarray,
    hamiltonian: TrotterHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    order: int = 0,
    copy: bool = True,
    save_intermediate_states: bool = False,
) -> Union[np.ndarray, list]:
    """Partial-Strang (PS): H0/2, H1^1, ..., H1^Gamma, H0/2 (symmetrized for order>0).

    U_PS(t) = e^{-iH0 t/2} . prod_k e^{-i alpha H_k t} . e^{-iH0 t/2}
    """
    n_df_terms = len(hamiltonian.diag_coulomb_mats)
    n_terms = 2 + n_df_terms
    last_index = n_terms - 1

    def step_fn(vec, current_basis, step_time):
        for term_index, term_time in simulate_trotter_step_iterator(n_terms, step_time, order):
            if term_index in (0, last_index):
                vec, current_basis = _apply_one_body_term(
                    vec, current_basis,
                    hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
                    term_time / 2, norb, nelec,
                )
            else:
                gamma = term_index - 1
                vec, current_basis = _apply_two_body_term(
                    vec, current_basis,
                    hamiltonian.diag_coulomb_mats[gamma], hamiltonian.orbital_rotations[gamma],
                    term_time, norb, nelec,
                )
        return vec, current_basis

    return _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn)


# ---------------------------------------------------------------------------
# PS_RANDOM -- Partial-Strang Random
# ---------------------------------------------------------------------------

def simulate_trotter_double_factorized_PS_RANDOM(
    vec: np.ndarray,
    hamiltonian: TrotterHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    order: int = 0,
    copy: bool = True,
    save_intermediate_states: bool = False,
    random_seed: Optional[int] = None,
) -> Union[np.ndarray, list]:
    """Partial-Strang Random (PS_RANDOM): like PS, but the sandwiching term
    is a randomly chosen two-body term H1^gamma instead of H0.

    Seed: H_rand, H0, [H1^gamma for gamma != rand] (symmetrized for order>0).
    The random term is chosen once (seeded by ``random_seed``) and reused for
    every step.
    """
    n_df_terms = len(hamiltonian.diag_coulomb_mats)
    if n_df_terms == 0:
        raise ValueError("No two-body terms available for random selection.")

    rng = np.random.default_rng(random_seed)
    random_idx = int(rng.integers(n_df_terms))
    remaining_indices = [i for i in range(n_df_terms) if i != random_idx]
    H_random_mat = hamiltonian.diag_coulomb_mats[random_idx]
    H_random_rotation = hamiltonian.orbital_rotations[random_idx]
    n_terms = 2 + len(remaining_indices)

    def step_fn(vec, current_basis, step_time):
        for term_index, term_time in simulate_trotter_step_iterator(n_terms, step_time, order):
            if term_index == 0:
                vec, current_basis = _apply_two_body_term(
                    vec, current_basis, H_random_mat, H_random_rotation,
                    term_time, norb, nelec,
                )
            elif term_index == 1:
                vec, current_basis = _apply_one_body_term(
                    vec, current_basis,
                    hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
                    term_time, norb, nelec,
                )
            else:
                gamma = remaining_indices[term_index - 2]
                vec, current_basis = _apply_two_body_term(
                    vec, current_basis,
                    hamiltonian.diag_coulomb_mats[gamma], hamiltonian.orbital_rotations[gamma],
                    term_time, norb, nelec,
                )
        return vec, current_basis

    return _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn)


# ---------------------------------------------------------------------------
# U_lambda -- optimized-lambda decomposition
# ---------------------------------------------------------------------------

def simulate_trotter_double_factorized_U_lambda(
    vec: np.ndarray,
    hamiltonian: TrotterHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    Lambda: float = 0.211325,
    order: int = 0,
    copy: bool = True,
    save_intermediate_states: bool = False,
) -> Union[np.ndarray, list]:
    """Optimized-lambda decomposition U_lambda (symmetrized for order>0).

    U_lambda(t) = e^{-i lambda H0 t} . [prod_k e^{-iH_k t/2}]
                  . e^{-i(1-2 lambda) H0 t} . [prod_k e^{-iH_k t/2}] . e^{-i lambda H0 t}

    Each [prod_k] is a symmetric (forward-backward) product. With the default
    Lambda=0.211325 = 1/2 - sqrt(3)/6 this is U_{lambda~}.
    """
    n_df_terms = len(hamiltonian.diag_coulomb_mats)

    def step_fn(vec, current_basis, step_time):
        vec, current_basis = _apply_one_body_term(
            vec, current_basis,
            hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
            Lambda * step_time, norb, nelec,
        )
        for term_index, term_time in simulate_trotter_step_iterator(n_df_terms, step_time / 2, order):
            vec, current_basis = _apply_two_body_term(
                vec, current_basis,
                hamiltonian.diag_coulomb_mats[term_index], hamiltonian.orbital_rotations[term_index],
                term_time, norb, nelec,
            )
        vec, current_basis = _apply_one_body_term(
            vec, current_basis,
            hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
            (1 - 2 * Lambda) * step_time, norb, nelec,
        )
        for term_index, term_time in simulate_trotter_step_iterator(n_df_terms, step_time / 2, order):
            vec, current_basis = _apply_two_body_term(
                vec, current_basis,
                hamiltonian.diag_coulomb_mats[term_index], hamiltonian.orbital_rotations[term_index],
                term_time, norb, nelec,
            )
        vec, current_basis = _apply_one_body_term(
            vec, current_basis,
            hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
            Lambda * step_time, norb, nelec,
        )
        return vec, current_basis

    return _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn)


# ---------------------------------------------------------------------------
# OG -- Omelyan-General (Theorem 1: fixed 4-group forward/backward)
# ---------------------------------------------------------------------------

def simulate_trotter_double_factorized_OG(
    vec: np.ndarray,
    hamiltonian: TrotterHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    lambda_opt: float = 0.211325,
    copy: bool = True,
    save_intermediate_states: bool = False,
) -> Union[np.ndarray, list]:
    """Omelyan-General (OG): fixed second-order, 4-group forward/backward sweep.

    H = A_1 + A_2 + ... + A_L, with A_1 = H0 and A_2..A_L = H1^1..H1^m.
    Group 1 (forward,  coeff c1): A_1, A_2, ..., A_L
    Group 2 (backward, coeff d1): A_L, ..., A_2, A_1
    Group 3 (forward,  coeff c2): A_1, A_2, ..., A_L
    Group 4 (backward, coeff d2): A_L, ..., A_2, A_1
    with c1 = d2 = lambda_opt, d1 = c2 = 1/2 - lambda_opt.
    OG has no `order` parameter -- it is inherently this fixed second-order
    construction (there is no first-order OG in the paper).
    """
    c1 = lambda_opt
    d1 = 0.5 - lambda_opt
    c2 = 0.5 - lambda_opt
    d2 = lambda_opt
    n_df_terms = len(hamiltonian.diag_coulomb_mats)

    def _group(vec, current_basis, coeff, step_time, forward):
        order_range = range(n_df_terms) if forward else range(n_df_terms - 1, -1, -1)
        if forward:
            vec, current_basis = _apply_one_body_term(
                vec, current_basis,
                hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
                coeff * step_time, norb, nelec,
            )
        for gamma in order_range:
            vec, current_basis = _apply_two_body_term(
                vec, current_basis,
                hamiltonian.diag_coulomb_mats[gamma], hamiltonian.orbital_rotations[gamma],
                coeff * step_time, norb, nelec,
            )
        if not forward:
            vec, current_basis = _apply_one_body_term(
                vec, current_basis,
                hamiltonian.one_body_energies, hamiltonian.one_body_basis_change,
                coeff * step_time, norb, nelec,
            )
        return vec, current_basis

    def step_fn(vec, current_basis, step_time):
        vec, current_basis = _group(vec, current_basis, c1, step_time, forward=True)
        vec, current_basis = _group(vec, current_basis, d1, step_time, forward=False)
        vec, current_basis = _group(vec, current_basis, c2, step_time, forward=True)
        vec, current_basis = _group(vec, current_basis, d2, step_time, forward=False)
        return vec, current_basis

    return _run_steps(vec, hamiltonian, time, norb, nelec, n_steps, copy, save_intermediate_states, step_fn)
