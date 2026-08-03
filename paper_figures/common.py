"""Shared machinery for the appendix figures of

    "Norm-Aware Product Formulas for Hamiltonian Simulations".

Everything needed by ``figure6.py`` and ``figure7.py`` lives here: the benchmark
Hamiltonians, the nine product formulas, and the plotting style.  The module has
no dependencies beyond NumPy, SciPy, Matplotlib and -- only for the Fermi-Hubbard
benchmark -- ffsim.

Conventions
-----------
``H = H_0 + alpha * sum_k H_k``, with ``H_0 = sum_m Z_m`` diagonal in the
computational basis.  Qubit 0 is the most significant bit.

``t`` is the time spanned by a *single* application of a product formula; a
scheme that covers a total time ``T`` in ``N`` steps uses ``t = T/N`` per step.
The error bounds of the paper are local, i.e. they refer to one application.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "PauliString", "chain", "square", "all_to_all", "bond_terms",
    "spin_model", "random_klocal_model", "hubbard_model",
    "Stepper", "METHODS", "LAMBDA_TILDE", "use_pra_style", "FIG_W_2COL",
]

# Optimal lambda: the root of gamma_2 that is stable under the alpha^2
# correction (Appendix A).
LAMBDA_TILDE = 0.5 - np.sqrt(3) / 6          # 0.2113248654...

_P = {"X": np.array([[0, 1], [1, 0]], complex),
      "Y": np.array([[0, -1j], [1j, 0]], complex),
      "Z": np.array([[1, 0], [0, -1]], complex),
      "I": np.eye(2, dtype=complex)}


# --------------------------------------------------------------------------
# Pauli strings
# --------------------------------------------------------------------------
class PauliString:
    """A term ``c * P`` with ``P`` a tensor product of Pauli matrices.

    Because ``P**2 = I`` the exponential is closed-form,
    ``exp(-i t c P) = cos(tc) I - i sin(tc) P``, and ``P`` acts as a signed
    permutation, so applying it costs O(d^2) rather than O(d^3).
    """

    def __init__(self, n, sites, paulis, coeff=1.0):
        self.n, self.d, self.c = n, 2 ** n, float(coeff)
        self.sites, self.paulis = list(sites), list(paulis)
        flip = zmask = n_y = 0
        for s, p in zip(sites, paulis):
            bit = 1 << (n - 1 - s)               # qubit 0 -> most significant
            if p in ("X", "Y"):
                flip |= bit
            if p in ("Y", "Z"):
                zmask |= bit
            if p == "Y":
                n_y += 1
        self.idx = np.arange(self.d) ^ flip
        # phi(x) = i^{n_Y} (-1)^{popcount(x & zmask)}.  np.bitwise_count returns
        # uint8, so it must be cast before the subtraction or 1-2*1 wraps to 255.
        pc = np.array([bin(v & zmask).count("1") for v in self.idx], dtype=np.int64)
        self.phase = ((1j ** n_y) * (1 - 2 * (pc & 1))).astype(complex)

    def dense(self):
        mats = [_P["I"]] * self.n
        for s, p in zip(self.sites, self.paulis):
            mats[s] = _P[p]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return self.c * out

    def apply_exp(self, M, t):
        """Return ``exp(-i t c P) @ M``."""
        ang = t * self.c
        return np.cos(ang) * M - 1j * np.sin(ang) * (self.phase[:, None] * M[self.idx, :])


# --------------------------------------------------------------------------
# geometries and benchmark models
# --------------------------------------------------------------------------
def chain(n, periodic=False):
    e = [(i, i + 1) for i in range(n - 1)]
    return e + ([(n - 1, 0)] if periodic and n > 2 else [])


def square(rows, cols):
    """Bonds of a rows x cols lattice, horizontal ones first.

    The ordering matters: the terms do not commute, so a different sweep order
    changes the result at the order of the splitting error itself.
    """
    ix = lambda r, c: r * cols + c
    e = [(ix(r, c), ix(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    e += [(ix(r, c), ix(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return e


def all_to_all(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def bond_terms(n, edges, rx=1.0, ry=-1.0, rz=0.0, field=0.0):
    """One term per bond, ``rx XX + ry YY + rz ZZ``, as a list of dense matrices.

    Grouping per bond (rather than per Pauli string) is what THRIFT assumes: its
    block ``exp(+i H_0 t) exp(-i (H_0 + alpha H_k) t)`` does not factorise, so
    splitting a bond into its strings changes the formula.  For the other eight
    methods the two groupings coincide, because the strings of one bond commute.
    """
    out = []
    for (i, j) in edges:
        blk = np.zeros((2 ** n, 2 ** n), complex)
        if rx:
            blk += PauliString(n, [i, j], ["X", "X"], rx).dense()
        if ry:
            blk += PauliString(n, [i, j], ["Y", "Y"], ry).dense()
        if rz:
            blk += PauliString(n, [i, j], ["Z", "Z"], rz).dense()
        out.append(blk)
    if field:
        out += [PauliString(n, [i], ["X"], field).dense() for i in range(n)]
    return out


def _h0_diag(n):
    """Diagonal of ``H_0 = sum_m Z_m``."""
    x = np.arange(2 ** n)
    bits = (x[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (1 - 2 * bits).sum(axis=1).astype(float)


def spin_model(n, edges, rx=1.0, ry=-1.0, rz=0.0, field=0.0):
    return _h0_diag(n), bond_terms(n, edges, rx, ry, rz, field)


def random_klocal_model(n=8, n_terms=12, seed=20260730, target_norm=None):
    """``H^(1)`` built from random Pauli strings of weight 2 or 3.

    Purely-Z strings are rejected: they would commute with ``H_0`` and the
    leading commutators would vanish.
    """
    rng = np.random.default_rng(seed)
    terms = []
    while len(terms) < n_terms:
        w = int(rng.integers(2, 4))
        sites = sorted(rng.choice(n, size=w, replace=False).tolist())
        paulis = [str(rng.choice(["X", "Y", "Z"])) for _ in sites]
        if all(p == "Z" for p in paulis):
            continue
        terms.append(PauliString(n, sites, paulis, float(rng.normal())))
    dense = [t.dense() for t in terms]
    if target_norm is not None:
        scale = target_norm / np.linalg.norm(sum(dense), 2)
        dense = [scale * t for t in dense]
    return _h0_diag(n), dense


def hubbard_model(norb=5, nelec=(2, 2)):
    """1D Fermi-Hubbard: ``H_0`` is the hopping, ``H^(1)`` the on-site term.

    Returned as dense matrices in the fixed particle-number sector, so ``alpha``
    is the interaction strength ``U``.  ``H_0`` is not diagonal here, which is
    why :class:`Stepper` accepts a dense ``H_0`` as well.
    """
    import ffsim
    d = ffsim.dim(norb, nelec)
    eye = np.eye(d, dtype=complex)
    dens = lambda op: np.column_stack(
        [ffsim.linear_operator(op, norb=norb, nelec=nelec) @ eye[:, j]
         for j in range(d)])
    h0 = dens(ffsim.fermi_hubbard_1d(norb, tunneling=1.0, interaction=0.0))
    h1 = dens(ffsim.fermi_hubbard_1d(norb, tunneling=0.0, interaction=1.0))
    return h0, [h1]


# --------------------------------------------------------------------------
# the nine product formulas
# --------------------------------------------------------------------------
METHODS = ["lts", "ps", "thrift", "u_lambda",
           "lts2", "ps2", "thrift2", "u_lambda2", "og"]

LABELS = {
    "lts":       r"$U^{[1]}_{\mathrm{LTS}}$",
    "ps":        r"$U^{[1]}_{\mathrm{PS}}$",
    "thrift":    r"$U^{[1]}_{\mathrm{THRIFT}}$",
    "u_lambda":  r"$U^{[1]}_{\tilde\lambda}$",
    "lts2":      r"$U^{[2]}_{\mathrm{LTS}}$",
    "ps2":       r"$U^{[2]}_{\mathrm{PS}}$",
    "thrift2":   r"$U^{[2]}_{\mathrm{THRIFT}}$",
    "u_lambda2": r"$U^{[2]}_{\tilde\lambda}$",
    "og":        r"$U^{[2]}_{\mathrm{OG}}$",
}

# colour / marker per method, identical across every figure of the paper
_C = {"LTS": "black", "PS": "#d62728", "THRIFT": "#1f77b4",
      "ULAM": "#009e73", "OG": "#b8860b"}
STYLES = {
    "lts":       dict(color=_C["LTS"],    marker="o", ls="-",  filled=True),
    "ps":        dict(color=_C["PS"],     marker="s", ls="-",  filled=True),
    "thrift":    dict(color=_C["THRIFT"], marker="^", ls="-",  filled=False),
    "u_lambda":  dict(color=_C["ULAM"],   marker="D", ls="-",  filled=True),
    "lts2":      dict(color=_C["LTS"],    marker="o", ls="--", filled=True),
    "ps2":       dict(color=_C["PS"],     marker="s", ls="--", filled=False),
    "thrift2":   dict(color=_C["THRIFT"], marker="^", ls="--", filled=True),
    "u_lambda2": dict(color=_C["ULAM"],   marker="D", ls="--", filled=False),
    "og":        dict(color=_C["OG"],     marker="v", ls="--", filled=False),
}


class Stepper:
    """One application of a product formula for ``H = H_0 + alpha * sum_k H_k``.

    Gates are applied left-to-right onto the identity, so the factor applied
    first is the one acting first (rightmost in the operator product).
    """

    def __init__(self, h0, terms, pauli=None):
        h0 = np.asarray(h0)
        self.diagonal = h0.ndim == 1
        if self.diagonal:
            self.h0d, self.d = h0.real, len(h0)
            self.H0 = np.diag(self.h0d).astype(complex)
        else:
            self.H0, self.d = h0, h0.shape[0]
            w, v = np.linalg.eigh(h0)
            self._w, self._v = w, v
        self.terms = [np.asarray(T) for T in terms]
        self.pauli = pauli
        self._c = {}

    # -- building blocks ---------------------------------------------------
    def _exp_h0(self, M, th):
        if self.diagonal:
            return np.exp(-1j * th * self.h0d)[:, None] * M
        return (self._v * np.exp(-1j * th * self._w)) @ (self._v.conj().T @ M)

    def _exp_term(self, M, k, th):
        if self.pauli is not None:
            return self.pauli[k].apply_exp(M, th)
        key = ("t", k, th)
        if key not in self._c:
            self._c[key] = expm(-1j * th * self.terms[k])
        return self._c[key] @ M

    def _thrift_block(self, k, th, a, dag=False):
        """``exp(+i H_0 th) exp(-i (H_0 + a H_k) th)``; ``dag`` swaps the order.

        Only the part of ``H_0`` supported on ``H_k`` matters -- the rest
        commutes and cancels between the two factors -- but using the full
        ``H_0`` is equivalent and simpler.
        """
        key = ("th", k, th, a, dag)
        if key not in self._c:
            A = expm(+1j * th * self.H0)
            B = expm(-1j * th * (self.H0 + a * self.terms[k]))
            self._c[key] = (B.conj().T @ A.conj().T) if dag else (A @ B)
        return self._c[key]

    def _sweep(self, M, th, rev=False, kind="t", a=None):
        idx = range(len(self.terms) - 1, -1, -1) if rev else range(len(self.terms))
        for k in idx:
            if kind == "t":
                M = self._exp_term(M, k, th)
            else:
                M = self._thrift_block(k, th, a, dag=(kind == "thd")) @ M
        return M

    # -- the formulas ------------------------------------------------------
    def step(self, method, alpha, t, lam=LAMBDA_TILDE):
        s = alpha * t
        M = np.eye(self.d, dtype=complex)
        if method == "lts":                                    # Eq. (24)
            M = self._sweep(M, s)
            M = self._exp_h0(M, t)
        elif method == "ps":                                   # Eq. (18)
            M = self._exp_h0(M, t / 2)
            M = self._sweep(M, s)
            M = self._exp_h0(M, t / 2)
        elif method == "thrift":                               # Eq. (4)
            M = self._sweep(M, t, kind="th", a=alpha)
            M = self._exp_h0(M, t)
        elif method == "u_lambda":                             # Eq. (21)
            M = self._exp_h0(M, lam * t)
            M = self._sweep(M, s / 2)
            M = self._exp_h0(M, (1 - 2 * lam) * t)
            M = self._sweep(M, s / 2)
            M = self._exp_h0(M, lam * t)
        elif method == "lts2":                                 # Eq. (28)
            M = self._exp_h0(M, t / 2)
            M = self._sweep(M, s / 2, rev=True)
            M = self._sweep(M, s / 2)
            M = self._exp_h0(M, t / 2)
        elif method == "ps2":                                  # Eq. (23)
            M = self._exp_h0(M, t / 4)
            M = self._sweep(M, s / 2, rev=True)
            M = self._exp_h0(M, t / 2)
            M = self._sweep(M, s / 2)
            M = self._exp_h0(M, t / 4)
        elif method == "thrift2":
            M = self._exp_h0(M, t / 2)
            M = self._sweep(M, -t / 2, rev=True, kind="thd", a=alpha)
            M = self._sweep(M, t / 2, kind="th", a=alpha)
            M = self._exp_h0(M, t / 2)
        elif method == "u_lambda2":                            # Eq. (26)
            M = self._exp_h0(M, lam * t)
            M = self._sweep(M, s / 4)
            M = self._sweep(M, s / 4, rev=True)
            M = self._exp_h0(M, (1 - 2 * lam) * t)
            M = self._sweep(M, s / 4)
            M = self._sweep(M, s / 4, rev=True)
            M = self._exp_h0(M, lam * t)
        elif method == "og":                                   # Eq. (27)
            m = (1 - 2 * lam) / 2
            M = self._exp_h0(M, lam * t)
            M = self._sweep(M, lam * s)
            M = self._sweep(M, m * s, rev=True)
            M = self._exp_h0(M, (1 - 2 * lam) * t)
            M = self._sweep(M, m * s)
            M = self._sweep(M, lam * s, rev=True)
            M = self._exp_h0(M, lam * t)
        else:
            raise ValueError(f"unknown method: {method}")
        return M

    def exact(self, alpha, t):
        return expm(-1j * (self.H0 + alpha * sum(self.terms)) * t)

    def local_error(self, method, alpha, t, lam=LAMBDA_TILDE):
        """``|| exp(-i H t) - U_step(t) ||_F`` for one application."""
        return float(np.linalg.norm(self.exact(alpha, t) -
                                    self.step(method, alpha, t, lam), "fro"))


# --------------------------------------------------------------------------
# plotting style (PRA two-column)
# --------------------------------------------------------------------------
FIG_W_1COL, FIG_W_2COL = 3.4, 7.0


def use_pra_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "legend.framealpha": 0.9, "legend.edgecolor": "0.6",
        "lines.linewidth": 1.1, "lines.markersize": 3.4,
        "figure.constrained_layout.use": False,
    })


def method_kw(method, **extra):
    st = STYLES[method]
    kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"],
              lw=1.1, ms=3.5, markeredgewidth=0.9,
              markerfacecolor=st["color"] if st["filled"] else "none",
              markeredgecolor=st["color"])
    kw.update(extra)
    return kw
