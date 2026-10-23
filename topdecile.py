#!/usr/bin/env python3
"""topdecile.py -- P5. Is the paper's 'ranking degrades at extremes' an information loss or
a range effect? Restrict to the top decile of the conditioning variable and see what moves.

Reported: AUSE for each arm, the available range AUSE(random), and the scale-free
    closed = (AUSE(mu) - AUSE(arm)) / AUSE(mu)
which is the fraction of the mu->oracle gap the arm closes.
"""
import json
import numpy as np
from ause import ause, oracle_area, validate
from blindness import world, A

if not validate(verbose=False):
    raise SystemExit("instrument failed")

mu, kappa, sigma, r, blind, informed = world(a=A, n=400_000, seed=3)
out = {"a": A, "n": len(mu), "subsets": {}}

def report(tag, m):
    rand = oracle_area(r[m])
    am = ause(r[m], mu[m]); ai = ause(r[m], informed[m]); ab = ause(r[m], blind[m])
    row = {"n": int(m.sum()), "random": rand, "mu": am, "informed": ai, "blind": ab,
           "mu_closes_of_range": 1 - am / rand,
           "informed_closes_of_mu_gap": (am - ai) / am,
           "blind_closes_of_mu_gap": (am - ab) / am}
    out["subsets"][tag] = row
    print(f"  {tag:26s} n={row['n']:7d}  random {rand:.4f}  mu {am:.4f}  informed {ai:.4f}"
          f"   informed closes {100*row['informed_closes_of_mu_gap']:6.2f}% of the mu gap")
    return row

print("P5 -- conditioning on high flow:")
report("all", np.ones(len(mu), bool))
for q, tag in ((0.90, "top decile of mu-hat"), (0.95, "top 5% of mu-hat"),
               (0.99, "top 1% of mu-hat")):
    report(tag, mu >= np.quantile(mu, q))
# and conditioned on OBSERVED flow, which is what their Fig 2a top-decile panel uses
y = mu + r
for q, tag in ((0.90, "top decile of observed y"), (0.99, "top 1% of observed y")):
    report(tag, y >= np.quantile(y, q))

print()
print("the paper's own numbers, read with the same statistic:")
for tag, sp, m in (("all test", 0.071, 0.089), ("top observed decile", 0.222, 0.283)):
    print(f"  {tag:22s} spread {sp:.3f}  mu-hat {m:.3f}   "
          f"spread closes {100*(m-sp)/m:5.1f}% of the mu-hat gap")
out["paper_closed"] = {"all_test": (0.089 - 0.071) / 0.089,
                       "top_observed_decile": (0.283 - 0.222) / 0.283}
json.dump(out, open("topdecile.json", "w"), indent=1)
print("\nwrote topdecile.json")
