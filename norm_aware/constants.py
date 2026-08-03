"""Values shared by every module, so the two independent implementations of
the formulas (`pauli.Stepper` and `evolver.Evolver`) cannot drift apart."""
from __future__ import annotations

import numpy as np

# The root of gamma_2 that stays a minimum under the alpha^2 correction
# (Appendix A).  Omelyan's alpha = 1 value, 0.193183, is a different root.
LAMBDA_TILDE = 0.5 - np.sqrt(3) / 6          # 0.2113248654...

#: The nine formulas, in the order they are drawn.
METHODS = ["lts", "ps", "thrift", "u_lambda",
           "lts2", "ps2", "thrift2", "u_lambda2", "og"]
