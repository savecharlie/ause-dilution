# Registered before writing a line of the simulation

23 Oct 2026, 4:12 PM MST. Paper: Gallant & Wilson, **arXiv:2610.20947v1**, *Ranked Right,
Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes* (stat.ML, 14 Oct 2026;
NeurIPS 2026 Climate Change AI workshop). 269 unregulated catchments in Western Canada,
regional LSTM, M = 10 random-initialisation ensemble, test 2013–2022.

## What they measured (read off the paper, not remembered)

| quantity | value |
|---|---|
| under-dispersion factor, central 90% of flows | 2.1× |
| UDF, top observed-flow decile | 5.8× |
| UDF, top ensemble-mean decile | 4.3× |
| log-log slope of residual SD on spread | 1.30, 95% CI [1.20, 1.41] |
| UDF above the training-period max | 17.93× (n = 603 catchment-days) |
| **AUSE, ensemble spread, all test** | **0.071** |
| **AUSE, ensemble mean µ̂, all test** | **0.089** |
| AUSE, spread, top observed decile | 0.222 |
| AUSE, µ̂, top observed decile | 0.283 |
| per-catchment corr(AUSE, UDF) | ρ = −0.23 |

Their own sentence: the spread is *"only modestly outperforming the ensemble mean."*

## The question I am asking, which is not theirs

They report that a near-oracle **ranking** score coexists with 2–18× **under-dispersion**, and
note in passing that ranking by predicted discharge alone nearly matches ranking by spread.
I want to know whether AUSE, as used, **can fail at all** in this setting — i.e. how much of
its range is reachable by a ranking variable carrying no information about error beyond the
magnitude of the prediction. Streamflow error is strongly multiplicative, so any monotone
function of µ̂ already ranks errors well, and an ensemble's spread is necessarily one of those
to first order.

If the answer is "AUSE near-oracle is achievable with zero conditional information", then
0.071 is a measurement of the heteroscedasticity of streamflow error and not of the ensemble.

## Predictions, registered now

- **P1.** A synthetic spread built as `pure noise × µ̂` — independent of the realised error
  given µ̂, so carrying exactly zero conditional information — will score **AUSE < 0.15**,
  inside or near the band the paper reads as near-oracle (0.071–0.089).
- **P2.** That same construction will show Spearman(spread, |r|) ≈ 0 **within** deciles of µ̂
  (|ρ| < 0.05). This is true by construction and is a check on my own code, not a finding.
- **P3.** *The claim that could fail.* A spread that genuinely knows the local error scale
  (oracle σ, plus realistic estimation noise) will close **more than 50%** of the gap between
  µ̂-ranking and perfect ordering. I.e. the metric is NOT saturated — it can see real
  information when it is there, and the paper's spread really is weak. I am predicting the
  direction that is *inconvenient* for the headline I would prefer to write.
- **P4.** AUSE will depend strongly on the heteroscedasticity exponent `a` in `σ ∝ µ̂^a`:
  at `a = 0` (homoscedastic) µ̂-ranking will be near-useless (AUSE within 20% of random), and
  the whole effect is a property of `a`, not of ensembles.

A miss on any of these is a dead hypothesis, not a thing to reinterpret.

## Instrument validation required BEFORE any of the above is believed

My AUSE must reproduce three answers I already know, or it reports nothing:
1. oracle ordering → AUSE exactly 0;
2. multiplying every error by a constant → AUSE unchanged (it is defined on normalised RMSE);
3. random ordering → the analytic value `∫(1 − oracle(f)) df`, since the random curve is flat
   at 1 in expectation. The paper's Fig. 2a shows exactly that flat line, so this is a check
   against *their* figure as well as against algebra.

---

## AMENDMENT, 4:20 PM, after the main run and BEFORE the top-decile run

**P1 MISSED, and the miss killed the framing.** My "blind" arm — `µ̂^a × lognormal noise` —
scored AUSE 0.187 against µ̂'s 0.117, i.e. *worse* than the trivial baseline, not near-oracle.
Obvious in hindsight: adding noise to µ̂ degrades the ordering. So a zero-information spread
does not score near-oracle; it scores *below* µ̂.

**And the exact statement replaces the simulation entirely.** AUSE is a functional of the
induced ORDERING alone. So for any strictly monotone transform `g(µ̂)`, `AUSE(g(µ̂)) = AUSE(µ̂)`
identically — no simulation needed. **The paper's own `AUSE(µ̂) = 0.089` therefore IS the
zero-conditional-information null**, already measured on their data, and the only interpretable
quantity is the margin below it. That reframes the question from "can AUSE fail" to "how big
is the margin, and what scale should it be read against".

**P3 MISSED too:** the informed arm (oracle σ with 0.3 log-sd estimation noise) closed 40.6%
of the µ̂→oracle gap, not >50%.

**P4 HELD, and it is the real result.** The fraction of AUSE's range reachable by µ̂ alone is
a monotone function of the heteroscedasticity exponent `a`: −0.1% at a=0, 52.6% at a=0.4,
86.0% at a=0.9, 93.8% at a=1.2. AUSE is not blind; it is *diluted*, in proportion to how
multiplicative the error is.

### The one prediction left, registered before running it

Their ranking numbers "degrade at extremes": spread 0.071 → 0.222 in the top observed decile.
But µ̂ degrades in lockstep, 0.089 → 0.283. **P5: most of that degradation is the available
range shrinking under conditioning, not information being lost.** Operationally: restricting
my sim to the top decile of µ̂ will raise AUSE for every arm, while the fraction of the
µ̂→oracle gap closed by the informed arm stays within ±10 percentage points of its full-sample
value. If P5 holds, the scale-free statistic
`closed = (AUSE(µ̂) − AUSE(spread)) / AUSE(µ̂)`
is the thing to report, and on their numbers it is 20.2% full-test and 21.6% top-decile —
i.e. flat in flow, while their dispersion failure grows 2.1× → 4.3× → 17.9×.
