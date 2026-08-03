"""The alpha sweep behind Fig. 2, with its on-disk cache.

For each coupling alpha the exact propagator is built once and every formula
is measured against it, at a fixed number of steps.  Results are checkpointed
to ``results/<run>/data.json`` after each alpha, so an interrupted sweep
resumes where it stopped instead of starting over.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .evolver import frobenius_error
from .spin import SpinLattice2D


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def method_table(lambda_values, lam_ps2, lam_omelyan):
    """Display names, formula keys and lambda values, in plotting order."""
    lam_names = list(lambda_values.keys())
    name_ps2 = f'ULam2(λ={lam_ps2})'
    name_oml = f'OG(λ={lam_omelyan})'

    names = (['LTS', 'PS', 'THRIFT'] + lam_names
             + ['LTS2', 'PS2', 'THRIFT2'] + [name_ps2, name_oml])
    keys = (['lts', 'ps', 'thrift'] + ['u_lambda'] * len(lambda_values)
            + ['lts2', 'ps2', 'thrift2'] + ['u_lambda2', 'og'])
    lams = ([None, None, None] + list(lambda_values.values())
            + [None, None, None] + [lam_ps2, lam_omelyan])
    return names, keys, lams, name_ps2, name_oml


def run_alpha_sweep(
    L=3,
    alphas=None,
    t=1.0,
    n_fixed=100,
    results_dir='results',
    lambda_values=None,
    lam_ps2=0.211325,
    lam_omelyan=0.211325,
    n_workers=None,
):
    """Sweep alpha for H = sum_k Z_k + alpha sum_<ij> (X_iX_j - Y_iY_j).

    Returns ``(data, run_dir)``.  ``data`` is exactly what lands in the JSON
    and what the figure scripts consume.
    """
    if alphas is None:
        alphas = np.geomspace(0.02, 1.0, 10)
    alphas = np.array(alphas)

    if lambda_values is None:
        lambda_values = {'λ=0.1932': 0.193183, 'λ=0.2113': 0.211325}
    if n_workers is None:
        n_workers = os.cpu_count()

    all_methods, method_keys, method_lams, name_ps2, name_oml = method_table(
        lambda_values, lam_ps2, lam_omelyan)
    lam_names = list(lambda_values.keys())

    ratio_defs = []
    if len(lam_names) >= 2:
        ratio_defs.append((f'{lam_names[0]} ÷ {lam_names[1]}', lam_names[0], lam_names[1]))
    ratio_defs += [
        ('THRIFT ÷ PS', 'THRIFT', 'PS'),
        (f'THRIFT2 ÷ {name_ps2}', 'THRIFT2', name_ps2),
        (f'{name_oml} ÷ {name_ps2}', name_oml, name_ps2),
    ]
    ratio_names = [r[0] for r in ratio_defs]

    run_dir = os.path.join(results_dir, f"Jsweep_XXMYY_L{L}_t{t}_n{n_fixed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'data.json')

    errors_fixed = {m: [] for m in all_methods}
    ratios_fixed = {r: [] for r in ratio_names}
    start_idx = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            saved = json.load(f)
        if saved.get('all_methods') == all_methods:
            done = len(saved['errors_fixed'][all_methods[0]])
            if done > 0:
                print(f"[checkpoint] resuming at alpha[{done}] "
                      f"({done}/{len(alphas)} done)")
                errors_fixed = saved['errors_fixed']
                ratios_fixed = saved.get('ratios_fixed', {r: [] for r in ratio_names})
                start_idx = done
        else:
            print("[checkpoint] method list changed -- starting over")

    print("H = sum_k Z_k + alpha sum_<ij> (X_iX_j - Y_iY_j)")
    print(f"L={L}  t={t}  n_fixed={n_fixed}  workers={n_workers}")
    print(f"alpha in [{alphas[0]:.4f}, {alphas[-1]:.4f}]  ({len(alphas)} points)")
    print(f"methods ({len(all_methods)}): {all_methods}\n")

    t_total = time.time()

    for idx in range(start_idx, len(alphas)):
        alpha = alphas[idx]
        Jx, Jy = alpha, -alpha
        print(f"[{idx+1}/{len(alphas)}]  alpha={alpha:.5f}", end='  ', flush=True)

        u_exact = SpinLattice2D(L, Jx, Jy, 0.0).exact_propagator(t)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(frobenius_error, L, Jx, Jy, t, n_fixed,
                            mkey, mlam, u_exact): mname
                for mkey, mname, mlam in zip(method_keys, all_methods, method_lams)
            }
            for fut in as_completed(futs):
                errors_fixed[futs[fut]].append(fut.result())

        for rname, num, den in ratio_defs:
            e_num, e_den = errors_fixed[num][-1], errors_fixed[den][-1]
            ratios_fixed[rname].append(
                e_num / e_den
                if e_den > 0 and np.isfinite(e_num) and np.isfinite(e_den)
                else float('nan'))

        print("  ".join(f"{m}={errors_fixed[m][-1]:.3e}" for m in all_methods))
        print(f"      [{time.time()-t_total:.1f}s]\n", flush=True)

        ckpt = dict(
            hamiltonian='sum_k Zk + J*sum_ij(XiXj - YiYj)',
            L=L, t=t, n_fixed=n_fixed,
            J_values=alphas.tolist(),
            all_methods=all_methods,
            lambda_values=lambda_values,
            lam_ps2=lam_ps2,
            lam_omelyan=lam_omelyan,
            errors_fixed=errors_fixed,
            ratio_defs=[list(r) for r in ratio_defs],
            ratios_fixed=ratios_fixed,
        )
        with open(ckpt_path, 'w') as f:
            json.dump(_jsonable(ckpt), f, indent=2)

    print(f"total {time.time()-t_total:.1f}s  ->  {ckpt_path}")
    with open(ckpt_path) as f:
        return json.load(f), run_dir


def load_alpha_sweep(run_dir, quiet=False):
    """Read back a sweep written by :func:`run_alpha_sweep`."""
    with open(os.path.join(run_dir, 'data.json')) as f:
        data = json.load(f)
    if not quiet:
        print(f"[load] L={data['L']}  t={data['t']}  n_fixed={data['n_fixed']}  "
              f"alpha in [{min(data['J_values']):.4f}, {max(data['J_values']):.4f}]  "
              f"({len(data['J_values'])} points)")
    return data


def fit_power_law(alphas, errors):
    """Least-squares fit of err = a * alpha^n in log-log.

    Returns ``(a, n, alpha_grid, fitted_curve)``, or None if fewer than two
    usable points remain.
    """
    a_vals, err = np.array(alphas, float), np.array(errors, float)
    valid = (a_vals > 0) & (err > 0) & np.isfinite(err)
    if valid.sum() < 2:
        return None
    n, log_a = np.polyfit(np.log(a_vals[valid]), np.log(err[valid]), 1)
    a = np.exp(log_a)
    grid = np.linspace(a_vals[valid].min(), a_vals[valid].max(), 200)
    return a, n, grid, a * grid ** n
