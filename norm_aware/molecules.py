"""Double-factorization benchmark quantities for the molecular Hamiltonians.

These are the numbers in Table I and the ingredients of Fig. 4: the quality of
the double factorization, the norm hierarchy alpha-bar that the formulas of
this paper exploit, and the ensemble of initial states the state error is
averaged over.

Requires ``ffsim`` (and ``pyscf`` upstream, to produce the Hamiltonians).
"""
from __future__ import annotations

import numpy as np

__all__ = ["reconstruct_two_body_tensor", "two_body_frobenius_error",
           "compute_alpha_bar", "make_random_state"]


def reconstruct_two_body_tensor(df_ham) -> np.ndarray:
    """Rebuild V from its double-factorized form,

        V[p,q,r,s] = sum_k Z^k_ij U^k_pi U^k*_qi U^k_rj U^k*_sj.

    Uses the number representation, as ``ffsim`` does internally, so the
    result is directly comparable with the original tensor.
    """
    df_nr = df_ham.to_number_representation()
    return np.einsum(
        "kij,kpi,kqi,krj,ksj->pqrs",
        df_nr.diag_coulomb_mats,
        df_nr.orbital_rotations,
        df_nr.orbital_rotations.conj(),
        df_nr.orbital_rotations,
        df_nr.orbital_rotations.conj(),
    )


def two_body_frobenius_error(mol_hamiltonian, df_ham) -> float:
    """``||V - V_tilde||_F``, the reconstruction error of the factorization.

    For an exact (untruncated) DF this sits at round-off, ~1e-15; it grows
    once eigenvalues are discarded or ``optimize=True`` is used.
    """
    v_orig = mol_hamiltonian.two_body_tensor
    v_tilde = reconstruct_two_body_tensor(df_ham).real
    return float(np.linalg.norm(v_orig - v_tilde))


def compute_alpha_bar(df_ham) -> float:
    """Mean of ``alpha_k = ||Z_k||_F / ||Z_0||_F`` over k >= 1.

    This is the norm hierarchy the formulas exploit: the smaller it is, the
    more they gain over plain Lie--Trotter--Suzuki.  The number
    representation is used because there the ``Z_k`` carry the physical scale.
    """
    df_nr = df_ham.to_number_representation()
    zs = df_nr.diag_coulomb_mats                       # (L, norb, norb)
    norms = np.linalg.norm(zs.reshape(len(zs), -1), axis=1)
    if norms[0] == 0.0:
        raise ValueError("||Z_0||_F = 0, cannot normalise")
    return float((norms[1:] / norms[0]).mean())


def make_random_state(norb: int, nelec: tuple, rng: np.random.Generator) -> np.ndarray:
    """A random normalized state in the ``(n_alpha, n_beta)`` sector.

    A Haar-random orbital rotation is applied to the Hartree--Fock
    determinant, which keeps the state inside the correct particle-number
    sector without any projection.
    """
    import ffsim

    hf_state = ffsim.hartree_fock_state(norb, nelec)

    z = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    haar_unitary = q * phase[np.newaxis, :]

    state = ffsim.apply_orbital_rotation(hf_state, haar_unitary, norb=norb, nelec=nelec)
    return state / np.linalg.norm(state)
