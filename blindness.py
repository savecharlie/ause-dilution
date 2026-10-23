#!/usr/bin/env python3
"""blindness.py -- how much of AUSE's range is reachable with ZERO information about
error beyond the magnitude of the prediction?

Setup matched to arXiv:2610.20947's situation, not to its data (their runs are not public):
discharge-like predictions, multiplicative error, and an ensemble spread that necessarily
scales with the magnitude of what it predicts.

  mu          ~ lognormal                      the ensemble-mean prediction
  kappa_i     ~ lognormal, mean 1              the LOCAL error-scale factor an uncertainty
                                               estimate could know and mu cannot
  sigma_true  = mu**a * kappa                  the realised local error scale
  r           = sigma_true * N(0,1)            the realised residual

Three ranking variables:
  mu        -- the trivial baseline: rank by how big the prediction is.
  blind     -- spread = mu**a * L, L lognormal INDEPENDENT of kappa. Conditional on mu it
               carries exactly zero information about the error. This is the null.
  informed  -- spread = sigma_true * (estimation noise). Knows kappa.

Reported for each: AUSE, and the FRACTION OF THE AVAILABLE RANGE CLOSED,
   closed = 1 - AUSE / AUSE(random),  AUSE(random) = integral(1 - oracle(f)) df,
because the absolute AUSE depends on how heavy the error tail is and on where the
sparsification axis is truncated, and the fraction does not.
"""
import json
import numpy as np
from scipy import stats
from ause import ause, oracle_area, validate

N = 200_000
A = 0.9           # heteroscedasticity exponent, sigma ~ mu**a
KAPPA_SD = 0.6    # how much local error scale varies beyond the magnitude (log sd)
EST_SD = 0.3      # noise in the informed estimate of sigma (log sd)
BLIND_SD = 0.6    # spread noise for the blind arm, matched to KAPPA_SD so the two arms
                  # have the SAME marginal distribution and differ only in what they know
FMAX = (1.0, 0.8)


def cond_spearman(x, y, strat, nbins=10):
    """Spearman(x, y) WITHIN deciles of strat, pooled by Fisher-z average."""
    q = np.quantile(strat, np.linspace(0, 1, nbins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    b = np.digitize(strat, q[1:-1])
    zs, ws = [], []
    for k in range(nbins):
        m = b == k
        if m.sum() < 50:
            continue
        rho = stats.spearmanr(x[m], y[m]).statistic
        rho = np.clip(rho, -0.999999, 0.999999)
        zs.append(np.arctanh(rho)); ws.append(m.sum() - 3)
    z = np.average(zs, weights=ws)
    return float(np.tanh(z))


def world(a=A, n=N, seed=3):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0.0, 1.2, n))
    kappa = np.exp(rng.normal(0.0, KAPPA_SD, n))
    sigma = mu ** a * kappa
    r = sigma * rng.normal(0.0, 1.0, n)
    blind = mu ** a * np.exp(rng.normal(0.0, BLIND_SD, n))
    informed = sigma * np.exp(rng.normal(0.0, EST_SD, n))
    return mu, kappa, sigma, r, blind, informed


def main():
    print("instrument first:")
    if not validate():
        return 1
    print()

    mu, kappa, sigma, r, blind, informed = world()
    out = {"n": N, "a": A, "kappa_sd": KAPPA_SD, "est_sd": EST_SD, "blind_sd": BLIND_SD,
           "paper": {"ause_spread": 0.071, "ause_mu": 0.089,
                     "ause_spread_topdecile": 0.222, "ause_mu_topdecile": 0.283},
           "arms": {}, "sweep": []}

    for fmax in FMAX:
        rand = oracle_area(r, fmax=fmax)
        row = {}
        for name, rank in (("mu", mu), ("blind", blind), ("informed", informed),
                           ("sigma_true", sigma)):
            a_ = ause(r, rank, fmax=fmax)
            row[name] = {"ause": a_, "closed": 1.0 - a_ / rand}
        row["random"] = {"ause": rand, "closed": 0.0}
        out["arms"][f"fmax={fmax}"] = row
        print(f"  sparsification truncated at f = {fmax}:   AUSE(random) = {rand:.4f}")
        for name in ("mu", "blind", "informed", "sigma_true"):
            d = row[name]
            print(f"    {name:11s} AUSE {d['ause']:.4f}   closes {100*d['closed']:5.2f}% "
                  f"of the range")
        marg_blind = row["mu"]["ause"] - row["blind"]["ause"]
        marg_info = row["mu"]["ause"] - row["informed"]["ause"]
        print(f"    marginal gain over ranking by mu:  blind {marg_blind:+.4f}   "
              f"informed {marg_info:+.4f}")
        row["marginal_over_mu"] = {"blind": marg_blind, "informed": marg_info}
        print()

    # P2 / P3: what each arm actually knows, conditional on the prediction
    print("  conditional on deciles of mu, Spearman with |residual|:")
    for name, rank in (("mu", mu), ("blind", blind), ("informed", informed),
                       ("sigma_true", sigma)):
        cs = cond_spearman(rank, np.abs(r), mu)
        mg = stats.spearmanr(rank, np.abs(r)).statistic
        out["arms"].setdefault("spearman", {})[name] = {"marginal": float(mg),
                                                        "conditional": cs}
        print(f"    {name:11s} marginal {mg:+.4f}   conditional on mu {cs:+.4f}")
    print()

    # P4: it is all about the heteroscedasticity exponent
    print("  P4 -- AUSE as a function of the heteroscedasticity exponent a:")
    print(f"    {'a':>5} {'random':>8} {'mu':>8} {'blind':>8} {'informed':>9}  "
          f"{'mu closes':>10}")
    for a in (0.0, 0.2, 0.4, 0.6, 0.9, 1.2):
        m2, k2, s2, r2, b2, i2 = world(a=a, n=60_000, seed=5)
        rand = oracle_area(r2)
        am, ab, ai = ause(r2, m2), ause(r2, b2), ause(r2, i2)
        print(f"    {a:5.1f} {rand:8.4f} {am:8.4f} {ab:8.4f} {ai:9.4f}  "
              f"{100*(1-am/rand):9.2f}%")
        out["sweep"].append({"a": a, "random": rand, "mu": am, "blind": ab,
                             "informed": ai, "mu_closed": 1 - am / rand})

    json.dump(out, open("blindness.json", "w"), indent=1)
    print("\nwrote blindness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
