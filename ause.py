#!/usr/bin/env python3
"""ause.py -- Area Under the Sparsification Error, and its validation.

Definition used (Ilg et al. 2018, ref [20] of arXiv:2610.20947): sort the predictions in
DESCENDING order of a ranking variable, progressively remove them, and at each removal
fraction f record the RMSE of what is retained, normalised by the RMSE of the full set.
That is the sparsification curve. AUSE is the area between a method's curve and the oracle
curve, where oracle orders by the realised |error|.

NOTHING in this module is believed until validate() prints 3/3. The three known answers:
  (1) oracle ordering -> AUSE exactly 0
  (2) scale invariance -- multiply every error by c, AUSE unchanged
  (3) random ordering -> the analytic value integral(1 - oracle(f)) df, because a random
      curve is flat at 1 in expectation. The paper's own Fig. 2a shows that flat line at
      1.00, so this checks the implementation against their figure as well as the algebra.
"""
import numpy as np


def sparsification(err, rank, fmax=1.0, npts=201):
    """RMSE of retained predictions, normalised by full-set RMSE, as a function of the
    fraction removed. Removal is in DESCENDING order of `rank` (most-uncertain first)."""
    err = np.asarray(err, float)
    order = np.argsort(-np.asarray(rank, float), kind="stable")
    e2 = err[order] ** 2
    n = len(e2)
    # suffix mean of squared error after removing the first k
    csum = np.concatenate([[0.0], np.cumsum(e2)])
    total = csum[-1]
    full_rmse = np.sqrt(total / n)
    fracs = np.linspace(0.0, fmax, npts)
    ks = np.clip((fracs * n).astype(int), 0, n - 1)
    retained = n - ks
    rmse = np.sqrt((total - csum[ks]) / retained)
    return fracs, rmse / full_rmse


def ause(err, rank, fmax=1.0, npts=201):
    """Area between the method's sparsification curve and the oracle's."""
    f, c = sparsification(err, rank, fmax, npts)
    _, o = sparsification(err, np.abs(err), fmax, npts)
    return float(np.trapezoid(c - o, f))


def oracle_area(err, fmax=1.0, npts=201):
    """integral (1 - oracle(f)) df -- the analytic AUSE of a random ordering, and the
    total range the metric has available on this error distribution."""
    f, o = sparsification(err, np.abs(err), fmax, npts)
    return float(np.trapezoid(1.0 - o, f))


def validate(verbose=True):
    rng = np.random.default_rng(11)
    n = 20000
    mu = np.exp(rng.normal(0, 1.2, n))          # discharge-like, heavy right tail
    err = rng.normal(0, 1, n) * mu ** 0.9       # multiplicative error
    checks = []

    a_oracle = ause(err, np.abs(err))
    checks.append(("oracle ordering gives AUSE 0", abs(a_oracle) < 1e-12, f"{a_oracle:.3e}"))

    r = rng.normal(size=n)
    a1 = ause(err, r)
    a2 = ause(err * 1000.0, r)
    rel = abs(a1 - a2) / max(abs(a1), 1e-30)
    checks.append(("scale invariance (x1000)", rel < 1e-10,
                   f"{a1:.12f} vs {a2:.12f}  rel {rel:.2e}"))

    # random ordering, averaged over draws, against the analytic area
    rs = [ause(err, rng.normal(size=n)) for _ in range(40)]
    want = oracle_area(err)
    got, sd = float(np.mean(rs)), float(np.std(rs))
    checks.append(("random ordering = integral(1-oracle)",
                   abs(got - want) < 4 * sd / np.sqrt(len(rs)) + 0.002,
                   f"mean {got:.4f} +- {sd/np.sqrt(len(rs)):.4f} vs analytic {want:.4f}"))

    ok = sum(c[1] for c in checks)
    if verbose:
        for name, passed, detail in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        print(f"  {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    print("validating ause.py against known answers:")
    raise SystemExit(0 if validate() else 1)
