"""The anisotropic spin lattice of Sec. V and its gate cache.

    H = sum_i Z_i  +  sum_<ij> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j)

on an open (non-periodic) rows x cols grid.  The partition the product
formulas act on is

    H_0     = sum_i Z_i                     one-qubit field, cheap to exponentiate
    H_k^(1) = the three terms of one bond    a 4x4 block

The paper's model, Eq. (29), is the case (Jx, Jy, Jz) = (alpha, -alpha, 0).

Everything is done by tensor contraction on a state of shape (2,)*n rather
than by building 2^n x 2^n matrices, so an L=3 lattice (d=512) stays cheap.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .constants import LAMBDA_TILDE  # noqa: F401  (re-exported for convenience)

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

XX4 = np.kron(X2, X2)
YY4 = np.kron(Y2, Y2)
ZZ4 = np.kron(Z2, Z2)


def apply_1q_gate(state, gate, qubit, n):
    """Contract a 2x2 gate into one leg of a rank-n state."""
    result = np.tensordot(gate, state, axes=([1], [qubit]))
    return np.moveaxis(result, 0, qubit)


def apply_2q_gate(state, gate, qubit1, qubit2, n):
    """Contract a 4x4 (or 2x2x2x2) gate into two legs of a rank-n state."""
    if gate.shape == (4, 4):
        gate = gate.reshape(2, 2, 2, 2)
    result = np.tensordot(gate, state, axes=([2, 3], [qubit1, qubit2]))
    remaining = [i for i in range(n) if i != qubit1 and i != qubit2]
    axes = [None] * n
    axes[qubit1] = 0
    axes[qubit2] = 1
    for k, orig in enumerate(remaining):
        axes[orig] = 2 + k
    return np.transpose(result, axes)


class GateCache:
    """Memoizes the small exponentials each formula reuses every step.

    Within a run the time step is fixed, so a handful of 2x2 and 4x4
    exponentials serve every one of the ``n_steps`` applications.
    """

    def __init__(self):
        self._cache = {}

    def get_1q(self, label, coeff):
        key = (label, coeff)
        if key not in self._cache:
            if label != 'Z':
                raise ValueError(f"unknown one-qubit generator {label!r}")
            self._cache[key] = expm(coeff * Z2)
        return self._cache[key]

    def get_int(self, Jx, Jy, Jz, dt):
        """exp(-i dt (Jx XX + Jy YY + Jz ZZ)), the bond gate."""
        key = ('int', Jx, Jy, Jz, dt)
        if key not in self._cache:
            h_int = Jx * XX4 + Jy * YY4 + Jz * ZZ4
            self._cache[key] = expm(-1j * dt * h_int)
        return self._cache[key]

    def get_thrift_pair(self, dt, Jx, Jy, Jz):
        """The THRIFT block of one bond,

            exp(+i (Z_i + Z_j) dt) exp(-i (Z_i + Z_j + H_k) dt),

        which is why THRIFT cares how the bond terms are grouped: unlike the
        other formulas it does not factorise over the strings of a bond.
        """
        key = ('thrift', dt, Jx, Jy, Jz)
        if key not in self._cache:
            zz2 = np.kron(Z2, I2) + np.kron(I2, Z2)
            h_int = Jx * XX4 + Jy * YY4 + Jz * ZZ4
            self._cache[key] = expm(+1j * dt * zz2) @ expm(-1j * dt * (zz2 + h_int))
        return self._cache[key]

    def get_thrift_pair_dag(self, dt, Jx, Jy, Jz):
        """Adjoint THRIFT block: the same two factors in the opposite order."""
        key = ('thrift_dag', dt, Jx, Jy, Jz)
        if key not in self._cache:
            zz2 = np.kron(Z2, I2) + np.kron(I2, Z2)
            h_int = Jx * XX4 + Jy * YY4 + Jz * ZZ4
            self._cache[key] = expm(+1j * dt * (zz2 + h_int)) @ expm(-1j * dt * zz2)
        return self._cache[key]

    def clear(self):
        self._cache.clear()


class SpinLattice2D:
    """Open rows x cols lattice; ``pairs`` fixes the order the bonds are swept.

    All horizontal bonds come first, then all vertical ones.  The bond terms
    do not commute, so a different sweep order changes the result at the order
    of the splitting error itself -- the order is part of the definition of
    the formula, not an implementation detail.
    """

    def __init__(self, L, Jx=0.1, Jy=0.15, Jz=0.05):
        self.L = L
        self.N = L * L
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.dim = 2 ** self.N
        self.pairs = self._build_pairs()
        self.m = len(self.pairs)

    def _build_pairs(self):
        L = self.L
        pairs = []
        for row in range(L):
            for col in range(L - 1):
                pairs.append((row * L + col, row * L + col + 1, 'H'))
        for row in range(L - 1):
            for col in range(L):
                pairs.append((row * L + col, (row + 1) * L + col, 'V'))
        return pairs

    def build_hamiltonian(self):
        """The full H as a dense matrix, for the exact reference propagator."""
        from scipy.sparse import csr_matrix, kron as skron
        I_s, X_s = csr_matrix(I2), csr_matrix(X2)
        Y_s, Z_s = csr_matrix(Y2), csr_matrix(Z2)
        H = csr_matrix((self.dim, self.dim), dtype=complex)

        for k in range(self.N):
            op = csr_matrix([[1.0 + 0j]])
            for q in range(self.N):
                op = skron(op, Z_s if q == k else I_s, format='csr')
            H = H + op

        for (i, j, _) in self.pairs:
            for J_val, P_s in [(self.Jx, X_s), (self.Jy, Y_s), (self.Jz, Z_s)]:
                op = csr_matrix([[1.0 + 0j]])
                for q in range(self.N):
                    op = skron(op, P_s if q in (i, j) else I_s, format='csr')
                H = H + J_val * op

        return H.toarray()

    def exact_propagator(self, t):
        """exp(-i H t), the reference every error is measured against."""
        return expm(-1j * self.build_hamiltonian() * t)
