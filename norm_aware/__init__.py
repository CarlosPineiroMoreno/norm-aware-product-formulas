"""Norm-aware product formulas for Hamiltonian simulation.

Companion code for *"Norm-Aware Product Formulas for Hamiltonian Simulations"*
(Piñeiro Moreno, Krause, Bande).

The nine formulas of the paper are implemented twice, against two different
representations, and the modules are named after the representation rather
than after the formulas:

``spin`` / ``evolver``
    The lattice of Sec. V, by tensor contraction.  Builds the full propagator,
    so errors are operator norms.  This is what the alpha sweep uses.
``pauli``
    Arbitrary Hamiltonians given as Pauli strings, plus the benchmark models
    of Fig. 6 (chains, ladders, all-to-all, Hubbard, random k-local).
``df``
    Molecular Hamiltonians in double-factorized form, on top of ``ffsim``.
    Evolves states rather than operators, so errors are state distances.

The duplication is deliberate: the appendix figures are a check that the
collapse onto lambda-tilde is not an artefact of one implementation.

``sweeps`` drives the alpha sweep and its cache, ``benchmarks`` the molecular
run and Table I, and ``style`` holds the plotting conventions shared by every
figure.
"""
from __future__ import annotations

from .constants import LAMBDA_TILDE, METHODS

__all__ = ["LAMBDA_TILDE", "METHODS", "__version__"]

__version__ = "1.0.0"
