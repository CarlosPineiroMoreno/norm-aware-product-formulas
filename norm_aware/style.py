"""One place for how every figure of the paper looks.

Four styles live here because the paper uses four sizes:

* :func:`use_pra_style` -- the PRA style of Figs. 3, 6 and 7;
* :func:`use_error_style` and :func:`use_ratio_style` -- the two panels of
  Fig. 2, which the notebook set separately;
* :func:`use_molecule_style` -- the larger type of Fig. 4.

Keeping them together with :data:`STYLES` is what makes a method the same
colour in every figure it appears in.
"""
from __future__ import annotations

from .constants import LAMBDA_TILDE, METHODS

__all__ = [
    "LABELS", "STYLES", "method_kw", "use_pra_style",
    "use_error_style", "use_ratio_style", "use_molecule_style",
    "build_sweep_styles", "RATIO_STYLES", "FIG_W_1COL", "FIG_W_2COL",
]

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

# Colour per method, identical across every figure of the paper.  These are
# the values the published figures were drawn with; a formula and its
# second-order counterpart share a colour and are told apart by the line
# style and whether the marker is filled.
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

FIG_W_1COL, FIG_W_2COL = 3.4, 7.0


def use_pra_style():
    """PRA style: Figs. 3, 6 and 7."""
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


def use_error_style():
    """Style of Fig. 2(a).

    Slightly larger than :func:`use_ratio_style` and without the tick
    settings, which that figure applies itself through ``tick_params``.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "font.family": "serif",
    })


def use_ratio_style():
    """Style of Fig. 2(b)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.family": "serif", "text.usetex": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })


def use_molecule_style():
    """Style of Fig. 4: much larger type, since it is reduced in the paper."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 25, "axes.labelsize": 20, "legend.fontsize": 25,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "font.family": "serif", "text.usetex": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })


def method_kw(method, **extra):
    """Plot kwargs for one of the nine :data:`METHODS`."""
    st = STYLES[method]
    kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"],
              lw=1.1, ms=3.5, markeredgewidth=0.9,
              markerfacecolor=st["color"] if st["filled"] else "none",
              markeredgecolor=st["color"])
    kw.update(extra)
    return kw


# --------------------------------------------------------------------------
# styles for the alpha sweep, whose curves are named at run time
# --------------------------------------------------------------------------
_BASE_STYLES = {
    'LTS':     dict(color='black',   ls='-',  marker='o', lw=2.0, ms=6, order=1),
    'PS':      dict(color='red',     ls='-',  marker='s', lw=2.0, ms=6, order=2),
    'THRIFT':  dict(color='#1f77b4', ls='-',  marker='^', lw=2.0, ms=6, order=3),
    'LTS2':    dict(color='black',   ls='--', marker='o', lw=1.7, ms=5, order=5),
    'PS2':     dict(color='red',     ls='--', marker='s', lw=1.7, ms=5, order=6),
    'THRIFT2': dict(color='#1f77b4', ls='--', marker='^', lw=1.7, ms=5, order=7),
}

_SWEEP_LABELS = {
    'LTS':     'LTS (1st order)',
    'PS':      'Partial–PS (1st)',
    'THRIFT':  'THRIFT (1st order)',
    'LTS2':    'LTS (2nd order)',
    'PS2':     'Partial–PS (2nd)',
    'THRIFT2': 'THRIFT (2nd order)',
}

_LAMBDA_COLORS = ['#2ca02c', '#9467bd', '#e377c2', '#8c564b']
_PS2_OML_COLORS = ['#ff7f0e', '#17becf', '#d62728']

RATIO_STYLES = [
    dict(color='#000000', ls='-', marker='o', lw=1.8, ms=5.5),
    dict(color='#444444', ls='-', marker='s', lw=1.8, ms=5.5),
    dict(color='#888888', ls='-', marker='^', lw=1.8, ms=5.5),
    dict(color='#bbbbbb', ls='-', marker='D', lw=1.8, ms=5.5),
]


def build_sweep_styles(methods):
    """Assign a style to every curve of a sweep.

    The fixed methods take their colour from :data:`_BASE_STYLES`; the
    lambda-dependent ones are named after the lambda they were run with
    (``'λ=0.2113'``, ``'OG(λ=0.211325)'``, ...), so they can only be coloured
    once the run is known.
    """
    styles = dict(_BASE_STYLES)
    labels = dict(_SWEEP_LABELS)

    lam_1st = [m for m in methods
               if m not in styles
               and not any(m.startswith(p) for p in ('ULam2', 'OG'))
               and not m.endswith('2')]
    for i, m in enumerate(lam_1st):
        c = _LAMBDA_COLORS[i % len(_LAMBDA_COLORS)]
        styles[m] = dict(color=c, ls='-', marker='D', lw=1.8, ms=5, order=4)
        labels[m] = m

    ps2_oml = [m for m in methods if any(m.startswith(p) for p in ('ULam2', 'OG'))]
    for i, m in enumerate(ps2_oml):
        c = _PS2_OML_COLORS[i % len(_PS2_OML_COLORS)]
        styles[m] = dict(color=c, ls='--', marker='v', lw=1.7, ms=5, order=8)
        labels[m] = m

    return styles, labels
