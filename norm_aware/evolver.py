"""The nine product formulas of the paper, on the spin lattice of `spin.py`.

First order
    lts        e^{-iH_0 dt} prod_k e^{-iH_k dt}
    ps         e^{-iH_0 dt/2} prod_k e^{-iH_k dt} e^{-iH_0 dt/2}      Eq. (7)
    thrift     prod_k [e^{+i(Z_i+Z_j)dt} e^{-i(Z_i+Z_j+H_k)dt}] e^{-iH_0 dt}
    u_lambda   e^{-i lam H_0 dt} prod_k e^{-iH_k dt/2} e^{-i(1-2lam)H_0 dt}
               prod_k e^{-iH_k dt/2} e^{-i lam H_0 dt}

Second order -- palindromic U(dt/2) U*(dt/2)
    lts2, ps2, thrift2   symmetrised counterparts of the above
    u_lambda2            the lambda-decomposition at second order
    og                   Ostmeyer's scheme generalised, Eq. (27)

The whole basis is evolved at once ("batch" methods below): the columns of
the identity are carried as one extra leading index, so a single pass yields
the full propagator and the Frobenius error can be taken directly.
"""
from __future__ import annotations

import numpy as np

from .constants import LAMBDA_TILDE, METHODS
from .spin import GateCache, SpinLattice2D


class Evolver:
    """Builds the approximate propagator of ``system`` over total time ``t``."""

    def __init__(self, system: SpinLattice2D, t: float, cache: GateCache | None = None):
        self.sys = system
        self.t = t
        self.cache = cache if cache is not None else GateCache()
        self.N = system.N
        self.dim = system.dim

    # -- batch helpers: leg 0 indexes the basis vector, legs 1..N the qubits --

    def _batch_apply_1q(self, states, gate, qubit):
        result = np.tensordot(gate, states, axes=([1], [qubit + 1]))
        return np.moveaxis(result, 0, qubit + 1)

    def _batch_apply_2q(self, states, gate, q1, q2):
        if gate.shape == (4, 4):
            gate = gate.reshape(2, 2, 2, 2)
        result = np.tensordot(gate, states, axes=([2, 3], [q1 + 1, q2 + 1]))
        remaining = [0] + [i + 1 for i in range(self.N) if i != q1 and i != q2]
        axes = [None] * (self.N + 1)
        axes[0] = 2
        axes[q1 + 1] = 0
        axes[q2 + 1] = 1
        ri = 3
        for orig in remaining[1:]:
            axes[orig] = ri
            ri += 1
        return np.transpose(result, axes)

    def _batch_Z_all(self, states, gate_Z):
        for q in range(self.N):
            states = self._batch_apply_1q(states, gate_Z, q)
        return states

    def _batch_int_all(self, states, gate_int):
        for (i, j, _) in self.sys.pairs:
            states = self._batch_apply_2q(states, gate_int, i, j)
        return states

    def _batch_int_all_reverse(self, states, gate_int):
        for (i, j, _) in reversed(self.sys.pairs):
            states = self._batch_apply_2q(states, gate_int, i, j)
        return states

    def compute_unitary(self, nsteps, method, lam=None):
        """The propagator of ``method`` after ``nsteps`` steps of dt = t/nsteps."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

        dt = self.t / nsteps
        states = np.eye(self.dim, dtype=complex).reshape([self.dim] + [2] * self.N)
        self.cache.clear()

        Jx, Jy, Jz = self.sys.Jx, self.sys.Jy, self.sys.Jz

        # Every gate below is fixed for the whole run, so it is built once.
        if method == 'lts':
            gZ = self.cache.get_1q('Z', -1j * dt)
            gInt = self.cache.get_int(Jx, Jy, Jz, dt)
        elif method == 'ps':
            gZh = self.cache.get_1q('Z', -1j * dt / 2)
            gInt = self.cache.get_int(Jx, Jy, Jz, dt)
        elif method == 'thrift':
            gZ = self.cache.get_1q('Z', -1j * dt)
            gTH = self.cache.get_thrift_pair(dt, Jx, Jy, Jz)
        elif method == 'u_lambda':
            gZl = self.cache.get_1q('Z', -1j * lam * dt)
            gZm = self.cache.get_1q('Z', -1j * (1 - 2 * lam) * dt)
            gInt = self.cache.get_int(Jx, Jy, Jz, dt / 2)
        elif method == 'lts2':
            h = dt / 2
            gZh = self.cache.get_1q('Z', -1j * h)
            gInth = self.cache.get_int(Jx, Jy, Jz, h)
        elif method == 'ps2':
            h = dt / 2
            gZq = self.cache.get_1q('Z', -1j * h / 2)     # dt/4
            gZfus = self.cache.get_1q('Z', -1j * h)       # dt/2, the fused middle H_0
            gInth = self.cache.get_int(Jx, Jy, Jz, h)
        elif method == 'thrift2':
            h = dt / 2
            gZh = self.cache.get_1q('Z', -1j * h)
            gTHh = self.cache.get_thrift_pair(h, Jx, Jy, Jz)
            gTHhd = self.cache.get_thrift_pair_dag(-h, Jx, Jy, Jz)
        elif method == 'u_lambda2':
            _lam = lam if lam is not None else LAMBDA_TILDE
            gZl = self.cache.get_1q('Z', -1j * _lam * dt)
            gZm = self.cache.get_1q('Z', -1j * (1 - 2 * _lam) * dt)
            gInt4 = self.cache.get_int(Jx, Jy, Jz, dt / 4)
        elif method == 'og':
            _lam = lam if lam is not None else LAMBDA_TILDE
            gZl = self.cache.get_1q('Z', -1j * _lam * dt)
            gZm = self.cache.get_1q('Z', -1j * (1 - 2 * _lam) * dt)
            gIntL = self.cache.get_int(Jx, Jy, Jz, _lam * dt)
            gIntM = self.cache.get_int(Jx, Jy, Jz, (1 - 2 * _lam) * dt / 2)

        for _ in range(nsteps):

            if method == 'lts':
                states = self._batch_int_all(states, gInt)
                states = self._batch_Z_all(states, gZ)

            elif method == 'ps':
                states = self._batch_Z_all(states, gZh)
                states = self._batch_int_all(states, gInt)
                states = self._batch_Z_all(states, gZh)

            elif method == 'thrift':
                states = self._batch_int_all(states, gTH)
                states = self._batch_Z_all(states, gZ)

            elif method == 'u_lambda':
                states = self._batch_Z_all(states, gZl)
                states = self._batch_int_all(states, gInt)
                states = self._batch_Z_all(states, gZm)
                states = self._batch_int_all(states, gInt)
                states = self._batch_Z_all(states, gZl)

            elif method == 'lts2':
                # U_LTS(dt/2) then U_trotter*(dt/2)
                states = self._batch_Z_all(states, gZh)
                states = self._batch_int_all_reverse(states, gInth)
                states = self._batch_int_all(states, gInth)
                states = self._batch_Z_all(states, gZh)

            elif method == 'ps2':
                # the two middle H_0/2 factors fuse into one H_0
                states = self._batch_Z_all(states, gZq)
                states = self._batch_int_all_reverse(states, gInth)
                states = self._batch_Z_all(states, gZfus)
                states = self._batch_int_all(states, gInth)
                states = self._batch_Z_all(states, gZq)

            elif method == 'thrift2':
                states = self._batch_Z_all(states, gZh)
                states = self._batch_int_all_reverse(states, gTHhd)
                states = self._batch_int_all(states, gTHh)
                states = self._batch_Z_all(states, gZh)

            elif method == 'u_lambda2':
                states = self._batch_Z_all(states, gZl)
                states = self._batch_int_all(states, gInt4)
                states = self._batch_int_all_reverse(states, gInt4)
                states = self._batch_Z_all(states, gZm)
                states = self._batch_int_all(states, gInt4)
                states = self._batch_int_all_reverse(states, gInt4)
                states = self._batch_Z_all(states, gZl)

            elif method == 'og':
                states = self._batch_Z_all(states, gZl)
                states = self._batch_int_all(states, gIntL)
                states = self._batch_int_all_reverse(states, gIntM)
                states = self._batch_Z_all(states, gZm)
                states = self._batch_int_all(states, gIntM)
                states = self._batch_int_all_reverse(states, gIntL)
                states = self._batch_Z_all(states, gZl)

        return states.reshape(self.dim, self.dim).T


def frobenius_error(L, Jx, Jy, t, n_steps, method, lam=None, u_exact=None, Jz=0.0):
    """|| U - U_method ||_F after ``n_steps`` steps over total time ``t``."""
    system = SpinLattice2D(L, Jx, Jy, Jz)
    if u_exact is None:
        u_exact = system.exact_propagator(t)
    u = Evolver(system, t, GateCache()).compute_unitary(n_steps, method, lam=lam)
    return float(np.linalg.norm(u - u_exact, 'fro'))
