#!/usr/bin/env python3
"""lemma.py -- AUSE is a functional of the induced ORDERING alone.

Therefore, for any strictly increasing g, AUSE(g(mu)) = AUSE(mu) exactly, and the paper's
own AUSE(mu-hat) = 0.089 is the null for any uncertainty signal that is a monotone function
of the prediction magnitude. An ensemble's spread is one of those to first order.

The negative control (a DECREASING transform) must differ, or the check cannot fail.
"""
import numpy as np
from ause import ause
from blindness import world

mu, kappa, sigma, r, blind, informed = world(n=50_000, seed=9)
base = ause(r, mu)
ok = True
for name, g in (("mu**3", mu ** 3), ("log mu", np.log(mu)), ("sqrt(mu)", np.sqrt(mu)),
                ("1-exp(-mu)", 1 - np.exp(-mu)), ("7*mu+2", mu * 7 + 2)):
    a = ause(r, g)
    same = (a == base)
    ok &= same
    print(f"  [{'PASS' if same else 'FAIL'}] AUSE({name:11s}) = {a:.12f}  "
          f"(AUSE(mu) = {base:.12f})")
neg = ause(r, -mu)
ok &= (neg != base)
print(f"  [{'PASS' if neg != base else 'FAIL'}] negative control AUSE(-mu) = {neg:.6f} "
      f"-- must differ")
print(f"  {'6/6' if ok else 'FAILED'}")
raise SystemExit(0 if ok else 1)
