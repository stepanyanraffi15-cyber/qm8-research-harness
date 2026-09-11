# Four retractions, and what is left standing

**QM8, Δ-learning, and a referee that took this project's own flagship apart**

This is the long version of `README.md`: the same story with the tables behind it, the
control that killed each claim, and the measurement that replaced it. The README is the
argument; this document is the evidence a reviewer checks it against.

Three conventions, stated first because the first error in this project was a convention
error:

- **Every headline number is on the sealed test set, `n = 2178`.** Where a number comes
  from `validation` (`n = 2179`) it says so in the same sentence. The two partitions
  differ by a single molecule, which is precisely how a validation number passed for a
  sealed one.
- Intervals are 95% nonparametric bootstraps, 5,000 resamples. For any comparison they
  are **paired over molecules** — resampled on the per-molecule differences, never on two
  independent means.
- A claim is **confirmatory** only if `preregister.json` carries it with a hash predating
  the measurement. Everything else is **exploratory**, and coming out favourably does not
  promote it. Two of the retractions below were decided under that rule; so was
  `delta_log`, which survived.

Every section names the artifact it is generated from. No number in this document was
typed by hand.

---

## 1. The result: one failure mode, four times

The headline finding of this project is not about excited states. It is this:

> **Every headline claim this project published failed the same way — report the slice
> where the effect appears, omit the slices where it does not.**

Four instances. Two authors. One week. The second instance was produced while auditing
for exactly that error.

| claim | slice reported | slice omitted | outcome |
|---|---|---|---|
| f1 abstention — *pre-registered, sealed-signed* | validation | sealed | **retracted** — magnitude artifact |
| E1 abstention — *the audit's proposed rescue* | validation | sealed | **retracted** — a free rule wins |
| Δ wins on screening metrics | f≥0.05, f≥0.1 | **f≥0.01, where it loses** | **qualified** |
| Δ survives shift, direct does not | scaffold/E1 | **the other 7 cells** | **1 of 8 survives** |

Note what these four have in common and what they do not. None is a fabrication. Each
claim is **true of the slice that was reported**: the classifier really does beat random
rejection at every coverage on f1; it really does win on validation for E1; Δ really does
win on AUC at f≥0.1; Δ really does beat a feature-matched direct model under scaffold
shift on E1. The error is never in the number. It is in the quantifier around it, and the
quantifier is where nobody looks, because the number checks out.

The second row is the one that makes this a finding rather than an anecdote. An external
audit caught the first error — a validation number sitting four lines above the words
*"sealed-set verified"* — and proposed E1 abstention as the claim that survived in its
place. **The rescue was itself a validation number.** On the sealed set the ordering
reverses:

```
E1 on validation  : classifier 0.05167   free gap rule 0.05728   -> classifier wins
E1 on sealed_test : classifier 0.05953   free gap rule 0.05686   -> classifier LOSES
```

(The audit quoted 0.05182 against 0.05749 for these two rules, and those figures
reproduce **exactly** — on `validation`, which is how they were identified as validation
figures. `results/abstention_battery.json` records 0.05182 / 0.057493 for its own
validation pass, beside the 0.05167 / 0.05728 of the independent re-run above: same
partition, same ordering, differing in the fourth decimal by fit path. The file now
computes both partitions in one pass and reports `ordering_reverses_val_to_sealed` as a
field, so this is a measurement rather than an argument.)

Two parties hunting for the val→test gap, in the same week, both shipped one. That is the
argument for a **mechanical** check rather than a procedural one: the error survives
people looking for it, including people who have just been burned by it. `critic.py`
re-scores on the sealed partition itself and will not take a number it was handed.

### The fifth instance, which never became a headline

Before the audit, the referee caught the author committing the same error at smaller
scale. Δ-learning appeared to beat the cheap baseline for f1 on the TDDFT-gap split
— 0.0206 against 0.0216, the one split where the 2015 negative result seemed to reverse.
It had a mechanism ready (that split isolates near-degenerate states) and it was repeated
in internal notes several times before anyone measured it on held-out data:

```
cheap 0.03077    delta 0.03032
paired difference +0.000442    CI [-0.000463, +0.001328]
excludes zero: NO
```

Noise, from a single fast-mode run scored on validation — best-of-N on a noisy metric.
Withdrawn before publication, by the machinery, against its own author. It is the same
failure as the four above, and the only difference is that something caught it earlier.

### What the four retractions imply about what to check

Each retraction maps to a control that is cheap, mechanical, and was not run:

| failure | the control that would have caught it | cost |
|---|---|---|
| a selective rule scored on one partition | score both partitions in the same function | one extra call |
| a selective rule with trained features | compare against rules that cost nothing | two lines |
| a selective rule on a skewed target | re-score with \|target\| held fixed within strata | one null |
| a metric that exists above a cut | sweep the cut | a loop |
| a comparison from one cell | fill the grid before writing the sentence | 8 cells, not 1 |
| a comparison against a handicapped loser | give the loser the same inputs | `direct_aug` |

All six are now either referee checks or standing analysis scripts. That is the whole
response to the audit: the claims that failed are gone, and the controls that would have
failed them earlier are wired in where they cannot be skipped.

---

## 2. Retraction 1 — abstention on f1

**`h_misorder_signature` — pre-registered, signed on the sealed set, control clean —
is RETRACTED.** Verdict file `results/audit_abstention.json`; battery
`results/abstention_battery.json`; regenerated by
`python analysis/abstention_battery.py`.

### What was claimed

TDDFT and CC2 disagree about which excited state is which for a sixth of QM8. A swap
corrupts the oscillator strength of both states, and that is the mechanism
Ramakrishnan et al. (2015) proposed for the failure of Δ-learning on oscillator
strengths. Knowing a molecule is misordered requires CC2, which is the cost the whole
exercise exists to avoid — so the claim was that misordering is predictable from
**structure alone**, giving an abstention rule that needs no CC2 at all:

```
structure-only classifier       AUC 0.6933   CI [0.6636, 0.7231]
registered control (coin flip)  AUC 0.4991   CI [0.4658, 0.5310]
```

Δ-model error then rises monotonically across the four risk quartiles — 0.005539,
0.011564, 0.013180, 0.019485 a.u., a ratio of **3.52× CI [2.85, 4.32]**. (Those figures
are on `validation`, from `results/misorder_signature.json`, where the registered primary
outcome was measured; the sealed battery refits the same classifier and reports AUC
0.6988 with a coin-flip control at 0.4862 CI [0.4532, 0.518]. That file still records
`"verdict": "SUPPORTED"`, which refers only to its own narrow question — is the swapped
population separable from structure at all — and is superseded by the referee's rejection
in `results/audit_abstention.json`. Where the two disagree, the referee is the
authority.) Scored on the sealed set,
selective error then falls below the random-rejection interval at every coverage, and at
50% coverage 0.013946 → **0.011080** — a 20.6% reduction against a random-rejection null
of 0.013954 CI [0.012426, 0.015482].

Pre-registered before measurement. Re-scored at full precision on a partition nothing
had touched. Registered control flat. It cleared every gate this project had.

### What killed it

Two checks, added after the fact, each fatal on its own, and they are different
diagnoses with different fixes. Sealed f1 at 50% coverage, every rival risk rule:

| risk rule | selective MAE | cost |
|---|---:|---|
| **the pre-registered classifier** | **0.011080** | 1036 features, trained |
| cheap E2−E1 gap threshold | 0.008034 | **free** |
| \|residual\| regressor | 0.004892 | same features, trained |
| \|predicted correction\| from the fitted Δ model | **0.004321** | **free — already computed** |
| oracle \|error\| | 0.001894 | ceiling |
| random rejection | 0.013954 | CI [0.012426, 0.015482] |

The classifier is the **worst of the five**. Two rules that cost nothing beat it, and one
of them is a column `models.py` computes on the way to making the prediction being
abstained on. `free_baseline_dominates`: the 1036 features bought nothing.

And with magnitude held fixed — keep the lowest-risk half *within* each \|f1\| quintile,
so dropping large values cannot be the mechanism:

```
magnitude-stratified selective MAE   0.012587
stratified null                      0.013938   CI [0.012460, 0.015424]
inside the null: YES
Spearman(risk, f1_CC2) = +0.383
```

`magnitude_artifact`: the rule was never ranking error. It was ranking brightness. MAE on
a non-negative, heavily right-skewed target falls whenever the large values are dropped,
so **any** score correlated with \|target\| beats random rejection without predicting a
single error.

### What the rule does in the application it was written for

This is the number that makes the retraction concrete rather than statistical. f1 is an
oscillator strength; the molecules worth finding are the bright ones. At 50% coverage:

```
genuinely bright molecules (f1_CC2 >= 0.05)   266
kept by the abstention rule                    89   (33.5%)
kept by random rejection                      133   [117, 148]  (50.0%)
```

For a photophysics screen, the rule abstains on the hits — and it does so *because* it
works as advertised, since brightness is what it learned to call risky. A rule that
beats random rejection on MAE and keeps two thirds as many hits as a coin flip is worse
than useless; it is useless in a way that looks like a result.

### The control that should have been run first

Both of the checks that killed this claim are two lines of code, and neither is a
statistical refinement of anything that was already there:

1. **Score the free rules.** Before training a risk model, rank by the cheap E2−E1 gap
   and by \|predicted correction\|. Both were available for free; one was already
   computed. If a trained rule does not beat them, the features are the claim and the
   features are worth nothing.
2. **Hold magnitude fixed.** Stratify by \|target\| quintile and randomise within strata.
   On a skewed non-negative target this is not optional, because the naive null cannot
   distinguish "ranks error" from "ranks size".

"Beats random rejection at matched coverage" is the standard selective-prediction
baseline (El-Yaniv & Wiener 2010) and it is the right baseline for a *bounded, symmetric*
loss. On MAE over a right-skewed non-negative target it is a bar almost anything clears.
Both checks are now in `critic.py` (check 9, `free_baselines` and `magnitude_control`) and
in the published battery, which is generated from the same function the referee calls —
one implementation, so the table and the verdict cannot disagree. A table that can
disagree with the referee is how the first version of this claim survived as long as it
did.

### What remains true underneath

**The state-ordering effect is real.** `results/state_ordering.json`, the full dataset,
`n = 21786`:

| cheap level | swap rate | mean CC2 gap when swapped | otherwise | f MAE as given | brightness-ordered | gain |
|---|---:|---:|---:|---:|---:|---:|
| PBE0/def2SVP | **16.4%** | 0.4999 eV | 0.8405 eV | 0.019224 | 0.015434 | 19.7% |
| PBE0/def2TZVP | 12.1% | 0.5032 eV | 0.8237 eV | 0.015692 | 0.013370 | 14.8% |
| CAM-B3LYP | 11.8% | 0.4453 eV | 0.8303 eV | 0.015832 | 0.012545 | 20.8% |

And the control is clean in the strongest available sense: the identical test applied to
the **energies** — which are sorted by construction, so no swap is possible — returns a
swap rate of **exactly 0.0%** and a gain of 0.0% at all three levels. That rules out the
obvious artifact, which is that reshuffling two correlated columns improves any
correlated metric.

The swapped population sits at half the state separation of the rest (0.50 eV against
0.84 eV), which is the predicted mechanism and not a fitted one.

These are properties of the dataset rather than held-out measurements — no model is
fitted anywhere in that table, which is why it is computed over all 21,786 molecules.

So: the physics is real, the structural signal is real (AUC 0.693 against a flat
coin-flip control), and the error really does rise 3.5× across risk quartiles. **The
model built on top of all that does not work**, because the quantity it ranks is
magnitude and a free column ranks error better. Those are compatible statements, and
keeping them separate is the difference between a retraction and a rout.

---

## 3. Retraction 2 — abstention on E1, which was the audit's own rescue

**`h_misorder_signature_E1` — exploratory — is RETRACTED.** Verdict file
`results/audit_abstention_E1.json`.

### What was claimed

The risk rule was built on f1 and transfers to E1, the project's primary energy target:
a 17.4% error reduction at half coverage on the sealed set, 0.072082 → 0.059528, against
a random-rejection null of 0.072074 CI [0.067712, 0.076914]. It fails on both second
states — on the validation screen that chose the two survivors, E2's risk-quartile ratio
is 0.94×, marginally inverted, which is a clean null and a useful check that the
classifier is not merely flagging hard molecules. The honest narrow form was "abstention
works for state-1 quantities", and when f1 fell, E1 was offered as the claim that
survived.

(The four-target screen ran on `validation` deliberately: screening four targets to find
the two that work is exactly the search a sealed partition exists to be protected from.
Only the two survivors were then scored on it. That part of the procedure was right.)

### What killed it

The full battery on sealed E1 at 50% coverage:

| risk rule | selective MAE | cost |
|---|---:|---|
| the classifier | 0.059528 | 1036 features, trained |
| **cheap E2−E1 gap threshold** | **0.056862** | **free** |
| \|predicted correction\| | 0.060385 | free |
| \|residual\| regressor | 0.055814 | same features, trained |
| oracle \|error\| | 0.020268 | ceiling |
| random rejection | 0.072074 | CI [0.067712, 0.076914] |

`free_baseline_dominates`, and nothing else: a one-column threshold that costs nothing
beats the trained rule. Unlike f1, E1 **passes** the magnitude control decisively —
0.058677 against a stratified null of [0.067532, 0.076706] — so this is not the f1
failure repeated. On E1 the rule genuinely ranks error. It is simply not worth its
features.

And the reason it was offered as a rescue at all is the val→test inversion at the top of
this document: on validation the classifier (0.05167) beats the free gap rule (0.05728);
on sealed it loses (0.05953 against 0.05686). The rescue for a validation-number error
was a validation number.

### The control that should have been run first

Compute both partitions in the same call. The `classifier_vs_free_gap` block in
`results/abstention_battery.json` now does exactly that and reports
`ordering_reverses_val_to_sealed` as a field, because the fact that an ordering reverses
is itself the result. No selective number in this project is reportable from one
partition any more — not because of a policy, but because the function that produces it
returns both.

### What remains true underneath

E1 is the one rule in the battery that clears the magnitude-stratified null. The
misordering mechanism does drive a real, recoverable share of E1 error. The deployable
form of that is the **free gap threshold**, not the classifier: abstain on molecules whose
cheap TDDFT E2−E1 gap is small, which requires no training, no fingerprints, and no
additional calculation. That rule reduces sealed E1 error at half coverage from 0.072082
to 0.056862. It is not a claim this project registered and it is not being promoted to
one; it is where the evidence points.

---

## 4. Qualification 1 — the screening-metric result holds above a threshold, and not below

The audit's second proposal was a replacement flagship: Δ-learning loses to the cheap
TDDFT number on MAE, but MAE is not what a screening campaign optimises, and on the
metrics a campaign does use — AUC, average precision, enrichment — Δ wins. That is worth
more than the retracted claim if it is true.

It was adopted only after being put through the same controls that killed the last one,
which in this case meant one thing: **sweep the threshold.** `results/screening_metrics.json`,
sealed f1, random split, PBE0/def2SVP as the cheap level, paired bootstraps,
`delta − cheap`:

| brightness cut | n positive | ΔAUC | CI | ΔAP | CI |
|---|---:|---:|---|---:|---|
| **f ≥ 0.01** | 677 | **−0.028** | [−0.045, −0.011] | **−0.034** | [−0.057, −0.014] |
| f ≥ 0.05 | 266 | +0.052 | [+0.026, +0.080] | +0.095 | [+0.060, +0.129] |
| f ≥ 0.10 | 156 | +0.073 | [+0.035, +0.111] | +0.123 | [+0.077, +0.164] |

f2 shows the same inversion — ΔAUC −0.038 [−0.054, −0.019] at f≥0.01, +0.068
[+0.046, +0.092] at f≥0.05, +0.108 [+0.079, +0.139] at f≥0.10.

The audit cited AUC at f≥0.1 and AP at f≥0.05 — the two cuts where Δ wins — and did not
report f≥0.01, where it is significantly **worse** on both metrics and both targets. Same
failure mode as the two retractions, in the claim proposed to replace them.

A result that exists only above a cut is a result about the cut, so the analysis script
sweeps three and reports all of them. The qualified claim is: **Δ-learning improves
screening metrics for the bright tail and damages them for the broad population**, and
which of those you see is set by where you put the threshold. Stated that way it is
still useful — a campaign chasing strong absorbers at f≥0.1 should use it — and it is no
longer a headline.

---

## 5. The constructive result: `delta_log`, and why Δ damages the dark bulk

The threshold inversion has a mechanism, and the mechanism is the most useful thing in
this document. Sealed f1, Spearman against CC2, split bright/dark at f = 0.01:

| Spearman | cheap | delta | **delta_log** |
|---|---:|---:|---:|
| bright (n = 677) | 0.595 | **0.773** | 0.648 |
| dark (n = 1501) | 0.820 | 0.403 | **0.847** |
| **all (n = 2178)** | 0.846 | 0.681 | **0.888** |

Δ-learning **improves** the ranking among bright transitions — 0.595 → 0.773, which is
real and is what the high-threshold metrics see — and **wrecks** it among the dark bulk,
0.820 → 0.403. The gradient-boosted model regresses a near-zero population toward its
mean and destroys an ordering the cheap TDDFT calculation already had for free. Above
f≥0.05 you see only the bright gain; at f≥0.01 the dark damage dominates; globally Δ is
0.165 worse than doing nothing, CI [−0.196, −0.134].

That diagnosis suggests its own fix. Fit the correction in log space —
`log10(f + 1e-4)` — so the model's error budget is not dominated by the bright tail:

| sealed, global Spearman vs cheap | Δ | CI |
|---|---:|---|
| f1 | **+0.0418** | [0.031, 0.054] |
| f2 | **+0.0497** | [0.034, 0.067] |

`delta_log` is **the only model in this project that beats raw TDDFT on global ranking**,
on both oscillator-strength targets, sealed and significant. It keeps most of the bright
gain (0.648 against cheap's 0.595) and repairs the dark ranking (0.847 against 0.820).
It also wins where plain Δ loses — at f≥0.01 its AUC is 0.932 against cheap's 0.906 and
Δ's 0.877, and its enrichment at 5% is the best of the three (3.011 against 2.863 and
2.951).

Four lines of code, in `models.py`.

Its MAE is much worse — 0.025588 against cheap's 0.011956, both at PBE0/def2SVP, which is
the cheap level throughout §4 and §5 — for the obvious reason: a
model fitted on log-transformed targets is not optimising absolute error, and
back-transforming inflates it. So it is a **ranking** model and it is reported as one.
Anyone reading the MAE column alone would reject it, which is the same category error as
reading the f≥0.1 AUC column alone.

**`delta_log` is exploratory.** It was not pre-registered, it was found by investigating
why a claim inverted, and it cannot become confirmatory now. That rule decided the two
retractions above; it applies identically to a result that comes out well.

---

## 6. Qualification 2 — Δ-learning's advantage is one cell in eight

**The published claim was:** under distribution shift, a structure-only model is worse
than doing nothing, while Δ-learning holds — so Δ's real value is robustness rather than
accuracy.

Both halves are true of the table that was run, and that table was not a fair test.
`cheap` and `delta` both receive the TDDFT calculation; `direct` was the only method
denied it. The comparison measured **input access** and was reported as a result about
the Δ construction.

`direct_aug` closes that: plain direct learning with the four cheap TDDFT numbers
(E1/E2/f1/f2 at the chosen level) appended to the feature matrix. All four, because one
TDDFT calculation returns all four and withholding three would be a handicap rather than
a saving. It costs exactly what `cheap` and `delta` cost — and note it sees *more* cheap
information than `delta` does, four columns against one.

Five methods × two splits × four targets, sealed test set, `speed=full`, 5,000-resample
paired bootstraps. `results/method_comparison.json`:

| split | target | cheap | direct | direct_aug | delta | `delta − direct_aug` |
|---|---|---:|---:|---:|---:|---|
| random | E1 | 0.26928 | 0.19251 | 0.06221 | 0.06298 | tie [−0.0017, +0.0034] |
| random | E2 | 0.40582 | 0.20913 | **0.10610** | 0.11364 | direct_aug [+0.0040, +0.0111] |
| random | f1 | 0.00868 | 0.01613 | 0.00889 | 0.01067 | direct_aug [+0.0012, +0.0024] |
| random | f2 | 0.02351 | 0.03226 | **0.02152** | 0.02483 | direct_aug [+0.0023, +0.0043] |
| scaffold | E1 | 0.20518 | 0.35590 | 0.08488 | **0.07629** | **delta [−0.0117, −0.0056]** |
| scaffold | E2 | 0.28950 | 0.34451 | 0.13627 | 0.13772 | tie [−0.0032, +0.0060] |
| scaffold | f1 | 0.00921 | 0.02381 | 0.01123 | 0.01291 | direct_aug [+0.0010, +0.0023] |
| scaffold | f2 | 0.02365 | 0.04259 | 0.02524 | 0.02870 | direct_aug [+0.0025, +0.0045] |

```
delta significantly beats direct_aug : 1/8   (scaffold/E1)
direct_aug significantly beats delta : 5/8
ties                                 : 2/8

direct     worse than cheap : 6/8
direct_aug worse than cheap : 3/8
```

The published claim generalised from **scaffold/E1 — the single cell where it holds** —
without testing the other seven. Most of what looked like a multi-fidelity advantage was
input access: `direct` is worse than doing nothing in six cells of eight, `direct_aug` in
three, and the gap between those two numbers is the whole of the apparent effect.

**What survives is worth keeping and is stated narrowly: under scaffold shift on E1,
Δ-learning beats a model given the same cheap numbers, CI [−0.0117, −0.0056].** Since
`direct_aug` sees more cheap columns than `delta` does, that cell is a claim about the
*construction* — fitting a correction to a cheap surface — rather than about access to
cheap data. It is one cell, it is significant, and it is the only one.

One deliberate omission: `direct_aug` is **not** in the null arms of `evaluate.py`. The
sampling space and the fixed grid were frozen before any arm ran, and widening them now
would silently change the published agent-versus-baseline comparison. `direct_aug` is a
first-class method in `models.py`, in the agent's tool enum, and in `run_experiment`'s
reported baselines, so the agent sees the fair baseline without spending budget on it.

---

## 7. What else survives

### The parse reproduces the 2015 paper

Raw PBE0/def2SVP against CC2, over the whole dataset: **E1 0.2712 eV** (paper: 0.27) and
**E2 0.3704 eV** (paper: 0.37). `world/build.py` refuses to build an environment whose
parse does not reproduce those, so this is a precondition rather than a result — which is
the right place for it. Everything downstream is conditioned on the parse being right,
and a parse error would have been invisible in every number above.

### MoleculeNet's QM8 distribution is damaged

`qm8.csv`, the file most QM8 papers train on, advertises 16 tasks under 16 column
headers. Its two PBE0 blocks are **byte-identical across all 21,786 molecules** —
max absolute difference 0.0 — so PBE0/def2TZVP is a verbatim copy of PBE0/def2SVP and
the file holds **12 distinct tasks**. That is why most published work reports 12. The
genuine fourth level of theory exists only in the 2015 supplementary release, which is
what this project parses.

This was [reported to DeepChem in 2021](https://github.com/deepchem/deepchem/issues/2747)
and is still present. A consequence worth stating: the cheap-level comparison below is
**unrunnable** for anyone working from the standard distribution.

A second correction in the same area: DeepChem's loader returns 21,747 molecules rather
than 21,786, and earlier planning for this project assumed the gap was RDKit
sanitization failure. It is not — RDKit parses all 21,786 with **zero** failures.
`world/build.py` keeps all 21,786 rows and carries a mask, so the count cannot change
underneath a result.

### Label efficiency, and the censoring in the 174×

E1, PBE0/def2TZVP, three seeds per point, validation. `results/label_budget.json`;
figure at `docs/figures/label-budget-{light,dark}.svg`.

| CC2 labels | direct (random) | Δ (random) | direct (scaffold) | Δ (scaffold) |
|---:|---:|---:|---:|---:|
| 100 | 0.566 | **0.120** | 0.649 | **0.111** |
| 500 | 0.436 | 0.091 | 0.523 | 0.087 |
| 2,500 | 0.298 | 0.075 | 0.437 | 0.074 |
| 10,000 | 0.242 | 0.067 | 0.398 | 0.070 |
| all 17,429 | 0.230 | 0.064 | 0.383 | 0.068 |

Δ-learning on 100 CC2 labels (0.120 eV) beats plain `direct` trained on all 17,429
(0.230 eV). **The 174× is left-censored**: the sweep's first point is 100 labels, so the
reported factor is literally `17429/100 = 174.3` and the true crossing lies somewhere
below the grid's first row. The number is the grid, not the measurement.

Against `direct_aug` rather than `direct` the gap closes almost entirely (see §6), so the
honest physical content is narrower and more interesting than the headline: **the cheap
level carries a large, learnable constant offset**, and correcting it is worth more than
thousands of fingerprint labels. Returns flatten early — 2,500 to 17,429 labels, roughly
seven times the CC2 compute, buys 0.011 eV.

**`h_budget_plateau` — pre-registered — is FALSIFIED.** The registered prediction was
that scaffold shift would need *more* labels to plateau, on the reasoning that a shifted
test distribution needs more support for a local correction. Measured: the scaffold curve
reaches within 10% of its fully-trained error at 2,500 labels against the random split's
10,000 — **earlier, not later**. The correction is more global than the mechanism assumed,
which is a statement about the chemistry rather than about the model.

Recorded in full because the first version of the analysis script printed `MOVES`, which
is true and useless — direction is part of the claim. That is exactly the bug the
referee's `direction` check exists to catch, committed again in this project's own
analysis code a few hours after being fixed in the referee.

### A better cheap baseline gives a better Δ-model

Sealed test set, full precision, E1, random split, Δ-model on two cheap levels:

```
delta on PBE0/def2TZVP   0.062985    (results/method_comparison.json)
delta on PBE0/def2SVP    0.072082    (results/abstention_battery.json, full-coverage MAE)
difference               0.0091 eV in favour of the better cheap level
```

Both figures come from committed sealed artifacts produced by different scripts, and they
agree on the ordering: **a better cheap baseline gives a better Δ-model**, which is only
measurable at all because the environment parses the raw 2015 release rather than
`qm8.csv`.

It is reported as a **measurement, not as a signed claim**, for two reasons worth stating
rather than hiding. The claim that carried it, `c_svp_beats_tzvp` in
`results/claims_test.json`, is worded "SVP baseline beats TZVP" — backwards relative to
what was measured — and the verdict file that signed it, `results/audit.json`, predates
the `direction` check. That file also can no longer be regenerated: re-running
`python critic.py results/claims_test.json` today rejects all five of its claims for
`unsupported_claim`, because the fixture cites evidence ids (`e_001`, `e_002`) that
predate the globally-unique-id fix and no longer resolve in the log. The paired interval
the earlier write-up quoted for this comparison exists only inside that stale file, so it
is not restated here.

The number is right; the sentence around it was not, and the artifact behind it no longer
reproduces. That is this document's theme applied to its own back pages.

### The project's own earlier explanation, refuted by a baseline

An early hypothesis held that "Δ-learning fails for oscillator strengths" was a metric
artifact: f is non-negative and heavily right-skewed, so MAE should reward predicting
≈ 0. Sealed f1, random split, PBE0/def2TZVP:

```
predict-zero  0.023983      cheap  0.008676      delta  0.010666
```

Predict-zero is nearly three times worse than the cheap baseline. The explanation is
wrong; the 2015 negative result replicates and this project's reason for it does not.

Worth being precise about what that refutes, because it is narrower than it looks:
predict-zero refutes a **degeneracy-of-the-metric** artifact. It says nothing about an
**ordering** artifact, which is a different mechanism and the one §2 finds to be real.
The strawman was refuted; the live version stands.

### The leakage control

Permuting the training labels drives E1 MAE from **0.0750 to 0.9665** — 12.9× worse,
i.e. all skill destroyed. There is no path by which test information reaches the model.
This runs in `selftest.py` on every invocation, not as a one-off.

---

## 8. The agent

Eight seeds, budget 12 experiments per run, local Qwen3 8B at temperature 0.7, 657 s wall
clock, mean 33,885 tokens per run. Hash-bound traces in `results/arms/`, report in
`results/arms_agent8_clean.json`, verdicts in `results/audit_arms_agent8_clean.json`.

### Search: the agent loses to a schedule written before it ran

Best validation MAE found within budget — this is a *search* metric, scored on
`validation` because it is the loop's own objective; nothing here touches the sealed
partition. The grid is a 12-line fixed schedule written before any arm ran; `random`
samples the same space uniformly. Both nulls are from `results/arms_null.json`.

| target | agent | 95% CI | grid | random |
|---|---:|---|---:|---:|
| E1 | 0.1926 | [0.1152, 0.2730] | **0.0645** | 0.2048 |
| E2 | 0.2011 | [0.1585, 0.2296] | **0.1204** | 0.1428 |
| f1 | 0.0154 | [0.0132, 0.0179] | **0.0090** | 0.0121 |
| f2 | 0.0333 | [0.0302, 0.0354] | — | 0.0310 |

The grid sits below the agent's interval on every target it covers. The agent also loses
to uniform random sampling on E2 and f1, and is indistinguishable from it on E1 and f2.

Two caveats on that table, both in the nulls' favour and neither fixable by rewording:
the agent's E1 and f1 rows are 8 seeds while E2 and f2 are 7 (one seed never reached those
families), and the null arms are 3 seeds each — 2 for random/E1 — with the grid
deterministic by construction, so its "interval" is a point. The comparison is
agent-versus-plan, and the plan wins by a margin no amount of seeding will close.

### Why: it does not vary the axis that matters

| seed | experiments | methods tried |
|---:|---:|---|
| 2 | 8 | `direct` × 4, `delta` × 4 |
| 5 | 8 | `direct` × 4, `delta` × 4 |
| 3 | 7 | `direct` × 4, `delta` × 3 |
| 0, 1, 4, 6, 7 | 4 each | `direct` only |

```
3 of 8 seeds tried delta-learning at all
mean 5.4 of 12 experiments used (45% of budget); 5 of 8 seeds stopped at exactly 4
7 of 8 seeds produced a claim; one stopped without a tool call
```

Impatience is real — mean 5.4 of 12 — and it is ResearchGym's named failure mode. It is
**not the main problem**. Five of eight seeds never tried `delta`, the single most
important method in the space and the one every other result in this document depends on;
all five are stuck on `direct`. The tool schema lists every method with a description and
the agent fixates on one anyway, which makes this a harness and prompt finding before it
is a model finding.

### What the referee did to the eight claims

| seed | verdict | why |
|---:|---|---|
| 0 | rejected | **wrong_direction** — cheap beats direct on f1/CAM by 0.00540 |
| 1 | narrowed | overscoped — claims `target=E2`, ran `f1` |
| 2 | **signed** | delta beats direct for E2, random split |
| 3 | rejected | gave_up — no claim (stopped with no tool call) |
| 4 | rejected | unsupported_claim — comparison claim with no `config_a`/`config_b` |
| 5 | **signed** | delta beats direct for E1, random split |
| 6 | rejected | **wrong_direction** — cheap beats direct on f1 by 0.00417 |
| 7 | rejected | **wrong_direction** — cheap beats direct on f1 by 0.00417 |

```
VERDICTS           signed 2 · narrowed 1 · rejected 5
FAILURE HISTOGRAM  wrong_direction 3 · overscoped 1 · gave_up 1 · unsupported_claim 1
```

**Three of the seven claims produced are exactly backwards** — the agent asserted a
direction the sealed data contradicts, and in every case it is the same error: `direct`
does not beat `cheap` on oscillator strengths, which is the 2015 negative result the agent
kept rediscovering with the sign flipped. They are caught only by the `direction` check,
which exists because an earlier rollout produced such a claim and an earlier version of
the referee **signed it**. One late-added check accounts for three of the five rejections.

The referee reports a histogram rather than a pass rate deliberately: a pass rate tells
you nothing about what to fix, and a histogram of named failure modes tells you exactly
what to fix.

### The previously published verdict count is not reproducible, and that is reported

An earlier version of this document published `1 signed · 2 narrowed · 5 rejected` for
eight seeds. **That line cannot be re-derived from anything in the repository**: it was
computed by an ad-hoc command against a report whose eight trace files are all missing
from disk. Rather than re-state it, the rollouts were re-run cleanly under the fixed
harness, with traces bound to their report rows by content hash, and audited by a
committed script (`analysis/audit_agent_claims.py`). The table above is from those
traces.

On the clean traces, the referee change moved two verdicts:

```
old referee   signed 0 · narrowed 2 · rejected 6
new referee   signed 2 · narrowed 1 · rejected 5
```

Both movements are **referee bugs, not agent merit**, and both are described in §10: seed
2 was rejected for omitting `split` exactly as the prompt instructed it to, and seed 5
was narrowed because citing three extra supporting runs was scored as over-claiming.
Reporting the old and new counts side by side is the point — the change is a finding about
referee design, and absorbing it silently would hide the more interesting half.

One anecdote from the earlier write-up is also gone: a seed said to have spent eleven of
twelve experiments running `direct` eleven times. No trace in the repository shows that,
so it is not repeated.

---

## 9. The referee

`critic.py` is deterministic and is never an LLM. An LLM judge is refused for a measured
reason rather than on taste: ChemCrow found GPT-4 could not separate confidently-wrong
chemistry from correct chemistry. A held-out number is not persuadable.

### The invariant

> No number reaches the write-up except through a scorer that **neither the agent nor the
> author** can query adaptively.

Both halves matter, and this project has now demonstrated both. A sealed test set defeats
an agent that would otherwise optimise against the metric it is judged on, one scalar at
a time, over thirty tool calls. It does **not** defeat an author who submits eleven claims
and reports the three that landed — so a claim is confirmatory only if `preregister.json`
carries it with a hash predating the measurement.

Enforcement is mechanical, not procedural: `models.run(eval_on="sealed_test")` raises
without an audit token that nothing importable from `tools.py` holds. `selftest.py`
asserts both halves — that the sealed partition is unreachable without the token, and
that no sealed number appears anywhere in an agent observation.

### The checks

`critic.py` numbers its checks 0 through 10. Checks 0 and 10 decide whether a submission
is adjudicable at all; 1 through 9 score its content. The first three catch fabrication
and leakage and were in the original design. The rest catch how ML research actually goes
wrong, and were not.

| # | check | catches |
|---:|---|---|
| 0 | well-formedness | a submission that is not a claim; **added after a malformed claim crashed an audit** |
| 1 | split hash | a claim measured on a different partition than it names — or, where the claim omits one, the partition its cited evidence agrees on |
| 2 | evidence resolves | a number that was never computed |
| 3 | sealed re-run | precision mismatch; produces the val→test gap as a by-product |
| 4 | noise floor | paired bootstrap over molecules, not a `k × sd` threshold |
| 5 | direction | a claim that is exactly backwards |
| 6 | multiplicity | Holm over the confirmatory family — best-of-N overestimates by construction |
| 7 | scope | a quantifier wider than the evidence covers |
| 8 | control | an artifact presented as a mechanism |
| 9 | **rival baselines** ← *new* | **a selective rule that bought nothing, or never ranked error** |
| 10 | **verifiable kind** ← *new* | **a claim of a kind nothing can measure on held-out data** |

Three verdicts, not two. **Narrowed** is the one worth having: most bad claims in ML
research are not fabrications, they are true statements with an oversized quantifier, and
a binary referee must either accept them or destroy them. Demonstrated end to end,
offline: the mock agent claims *"Delta-learning beats direct learning for excited-state
prediction"* from two runs on E1, and the referee narrows it to *"— for target=E1,
split=random, cheap_level=PBE0-SVP only."*

### Check 9, in two halves, and why the old bar was wrong

Check 9 asks two questions. They are not refinements of each other, they have different
fixes, and they are labelled separately in the failure histogram:

```
free_baselines     is the rule better than one that costs nothing?
                   -> failure label: free_baseline_dominates  (the features bought nothing)
magnitude_control  does it survive with |target| held fixed?
                   -> failure label: magnitude_artifact        (it never ranked error)
```

The bar it replaces was "beats random rejection at matched coverage". That is the
standard selective-prediction baseline, the project's flagship cleared it comfortably,
and it is **the wrong bar for this loss on this target**: MAE over a non-negative,
right-skewed quantity falls whenever the large values are dropped, so any score
correlated with \|target\| clears it without predicting a single error. Random rejection
is retained — check 4 still runs it — but it is no longer sufficient.

The measurement lives in `analysis/misorder_signature.selective_battery`, which
`critic.py` calls and which the published tables in §2 and §3 are generated from. One
implementation, deliberately: the alternative is two copies of the sealed-scoring
construction, and the copy that drifts is the one nobody is adjudicating against.

### The guard that shared the claim's blind spot

`selftest.py` had a check named *"abstention beats random rejection at matched
coverage"*. It passed. It ran on validation. It certified the retracted claim, and it was
written by the same person, on the same day, as the claim — so it inherited exactly the
blind spot it was supposed to cover. It now asserts the **retraction**, against the
sealed battery, and fails if the classifier ever beats every free baseline again.

A test written alongside the claim it guards is not an independent check. That
generalises past this repository.

---

## 10. The harness bugs are the same bugs

The audit surfaced five defects in the harness and referee. They are worth a section
because they are the **same class** as the science errors above, and the pattern is a
single sentence:

> **The code that handles the terminal case or the error case is the code the happy path
> never exercises.**

All five were found by running, not by reading. None would have been caught by a test
written against the path the harness takes when everything works.

### 1. The terminal tool was the only tool that could kill a run

`agent.py` wraps every tool call in `try/except` so a bad call returns an `ERROR`
observation the model can read and recover from. `claim` is intercepted *before* dispatch
so it can double as the stopping rule — and that path had no guard at all. A clean
eight-seed re-run died at seed 2:

```
TypeError: claim() got an unexpected keyword argument 'budget'
```

The one tool that ends a run was the one tool that could crash it, which inverts the
property the entire loop is built on. Now guarded, with the accepted argument list
returned in the error message so the model can correct itself.

### 2. The referee crashed on malformed claims, which its own comments promised it would not

Roughly nine of nineteen malformed claim shapes raised instead of returning a verdict,
and `Critic().audit([good, None])` died before the good claim beside it was ever reported
— so one bad claim cost an eight-seed sweep its entire audit. `adjudicate` and `audit`
are now **total**: a shape check runs first and outside the try (a malformed claim has no
content to measure, only something to report), and anything escaping a real check becomes
a `referee_error` rejection rather than a lost audit. `selftest.py` now feeds 20 malformed
shapes — `None`, `{}`, a non-dict scope, evidence as a string, a non-string id, a
coverage of 5 — and asserts each returns a verdict with a reason, plus that
`audit([good, None, {}])` reports three verdicts.

### 3. The prompt told the agent to omit axes the referee then punished

`tools.py` instructed the agent to omit an axis it was not claiming about. `critic.py`
then rejected any claim omitting `split` as `unsupported_claim`. The agent did as
instructed and was penalised for it — and this cost seed 2 a signature.

An omitted `split` is now **inferred** from the cited evidence: every logged run carries
its `config.split`, so the partition is recoverable. The claim fails only where the
evidence genuinely cannot answer — no cited run carries a split, or the cited runs
disagree, in which case there is no single partition to hash and no honest way to pick
one. The claim tool's `scope` description now says so, so the prompt and the referee
agree.

### 4. Extra evidence made a claim score worse

`_check_scope` compared the set of values in the cited runs against the claim's scope with
`==`. Consequence: a claim scoped `target="E1"` passed on one E1 run and flipped to
`overscoped` the moment a **second** supporting run was cited beside it. A referee that
punishes extra evidence trains precisely the wrong behaviour.

It now tests **coverage**: `target="E1"` is satisfied when the cited runs include E1 runs;
the claim is overscoped only where it quantifies beyond its evidence — `"all"` with one
value present, or `target="E2"` with no E2 run behind it. The narrowed verdict still fires
on the latter, which is the case it was built for.

### 5. Two claim kinds reached SIGNED without touching data

`kind="mechanism"` and `kind="value"` were both in the agent's tool enum and both could
reach SIGNED **without a single sealed-set number being computed**: only the `comparison`
and `selective` branches of `adjudicate` touch data, so every other kind fell through to
the default verdict. A claim that cannot be falsified must not be signable.

Both now reject as `unverifiable_kind` and are removed from the tool enum. The
alternative — inventing a sealed measurement for them — was refused because neither kind
carries the structure to support one: a mechanism claim names an intervention
(`reindex_states`, `ablate_features`) that exists in `tools.py` as a mutation of the
target table and has no representation in `models.run`, so the referee cannot reproduce it
on the sealed partition at all, which is exactly why it was signing for free. A `value`
claim carries no asserted number for a re-measurement to contradict. Refusing to sign is
the honest verdict.

### A scaffold finding, for the same reason

"Reasoning effort off" is **not portable**, and the scaffold is part of the score. Through
an OpenAI-compatible endpoint, `think: false` and `reasoning_effort: low` are both
silently ignored — the model reasons unboundedly and never reaches an answer. The native
endpoint accepts the flag and answers in 4 tokens (`llm.py`):

| route | tokens | answer |
|---|---:|---|
| OpenAI-compatible, `think=False` in `extra_body` | 300 | none |
| OpenAI-compatible, `reasoning_effort=low` | 300 | none |
| OpenAI-compatible, 2000-token budget | 2000 | none |
| native endpoint, `"think": false` | **4** | correct |

Fixing it took a harness check from 647 completion tokens to 18, and 28.1 s per tool call
to 5.7 s. An interface that commits to nothing guarantees nothing: a score is a property
of the model **and** the scaffold serving it, which is the Holistic Agent Leaderboard's
finding landing on this project's own setup.

---

## 11. Honest limits

- **The agent is an 8B local model.** The constraint was that everything run locally, and
  nothing available could serve a larger one. Its poor search is therefore **confounded
  with capacity** and must not be read as "LLM agents cannot do this". The registered
  hypothesis that would separate the two — `h_frontier_explores`, which predicts a
  stronger model explores the method axis in ≥ 6 of 8 seeds and reaches within 20% of the
  grid's 0.06446 eV — remains **unrun**. It is registered with a hash, and a null there
  would be the more useful outcome, because it would move the diagnosis from capacity to
  harness.
- **Gradient boosting on fingerprints is a weak predictor**, chosen for throughput
  (~10 s/fit). It discards the 3D geometry QM8 ships. The Δ-model barely notices; the
  direct model is crippled by it, so any Δ-vs-direct gap is partly a statement about the
  featurization rather than about multi-fidelity learning. Published QM8 leaders are GNNs.
- **`delta_log` is exploratory.** Not pre-registered, found while diagnosing an
  inversion, and this project now carries four demonstrations of why that distinction is
  not bookkeeping.
- **The traces are tool results, not transcripts.** They record what each tool returned,
  bound by content hash to its report row; they do not contain the model's own tokens. Any
  claim about *why* the agent fixated on `direct` would need the transcripts, so no such
  claim is made here.
- **Two pre-registered hypotheses were falsified.** `h_budget_plateau` runs backwards from
  its prediction (§7) and `h_misorder_signature` is retracted (§2). Of three registered
  confirmatory hypotheses, one is falsified, one is retracted, and one is unrun. A
  project that only reports the registrations that worked has not demonstrated that
  registration does anything.
- **The reindexing affordances are supplied.** `slice_error(by="state_gap")` and
  `intervene("reindex_states")` are handed to the agent. An agent that recovers the
  state-ordering result through them has done **tool-assisted hypothesis generation**, not
  independent discovery.
- **Seed variance on the models is exactly zero.** LightGBM on fixed data is
  deterministic; seeds are the wrong instrument and the paired bootstrap over molecules is
  the right one. Seeds vary the *agent*, not the fits.
- **Six design decisions have no support in the cited literature** and are labelled
  reasoning rather than evidence: pre-registration against author-side selection,
  exploratory/confirmatory typing, multiplicity correction, the random-search and
  fixed-grid baselines, OOD-based abstention, and the LLM-versus-gradient-boosting
  substitution.

---

## 12. Numbers the previous version of this document carried, and why they are gone

A reviewer comparing against the earlier write-up should find each of these accounted for
rather than quietly deleted.

| was published | status now |
|---|---|
| abstention on f1: 31% error reduction at half coverage | **validation number.** The sealed figure is 20.6%, and the claim is retracted anyway (§2) |
| abstention on E1: 24% reduction | **validation number.** Sealed is 17.4%, and retracted (§3) |
| "beats random rejection at every coverage" as a result | true, and the **wrong bar** (§2, §9) |
| Δ-learning's value is robustness; direct is worse than doing nothing | **qualified to 1 of 8 cells** (§6) |
| Δ wins on screening metrics | **qualified**: it holds at f ≥ 0.05 and above and inverts at f ≥ 0.01 (§4) |
| agent verdicts `1 signed · 2 narrowed · 5 rejected` | **not reproducible** — traces gone. Clean re-run gives `2 · 1 · 5` (§8) |
| a seed that ran `direct` eleven times | **no trace in the repository shows it.** Dropped (§8) |
| Δ beats cheap for f1 on the gap split | **withdrawn as noise** before publication, by the referee (§1) |
| sealed E1 error by cheap-gap quartile, ratio 2.26× CI [1.97, 2.59] | measured, and **has no standalone artifact** in `results/`. Not repeated here; the surviving form of the mechanism is the gap-threshold rule in §3 |
| the referee's selective null is calibrated at 2.29% empirical FPR over 20,000 noise draws | asserted in the audit-response plan with **no artifact behind it**. Not repeated |
| "seven checks" in one place and "8 checks" in another | `critic.py` numbers them 0–10; the **nine** that score a claim's content are 1–9, and 0 and 10 decide whether it is adjudicable at all (§9) |

`results/audit.json` is a **stale artifact** from the pre-`direction` referee, and it no
longer regenerates: one of its two signed claims is worded backwards, and every claim in
the fixture behind it now fails `evidence_resolves` because its ids predate the
globally-unique-id fix (§7). It is kept because deleting it would remove the evidence that
the `direction` check was needed — but nothing in this document rests on it.

---

## 13. References

**Dataset and the original result**

- Ramakrishnan, Hartmann, Tapavicza & von Lilienfeld, *Electronic spectra from TDDFT and
  machine learning in chemical space*, J. Chem. Phys. **143**, 084111 (2015).
  [doi:10.1063/1.4928757](https://doi.org/10.1063/1.4928757) — QM8, the Δ-learning
  result, and the oscillator-strength failure this work pushes against. Data from the
  first author's [ExcitedStatesQM8](https://github.com/raghurama123/ExcitedStatesQM8).
- Ramakrishnan, Dral, Rupp & von Lilienfeld, *Quantum chemistry structures and properties
  of 134 kilo molecules*, Sci. Data **1**, 140022 (2014).
  [doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)
- Wu et al., *MoleculeNet*, [arXiv:1703.00564](https://arxiv.org/abs/1703.00564);
  duplication [reported 2021](https://github.com/deepchem/deepchem/issues/2747).

**Method**

- Anthropic, *Building Effective Agents* (Dec 2024) — the policy / tools / observation /
  memory / stopping-rule decomposition.
- Efron, *Bootstrap Methods: Another Look at the Jackknife*, Ann. Statist. **7**(1), 1979.
  [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) — every interval
  here, including the paired variant over molecules.
- El-Yaniv & Wiener, *On the Foundations of Noise-free Selective Classification*,
  JMLR **11** (2010) — risk–coverage curves, and why random rejection at matched coverage
  is the only honest baseline. §2 is a note on where that stops being sufficient.

**Evidence for design decisions** *(verified against the papers)*

- *Holistic Agent Leaderboard*, [arXiv:2510.11977](https://arxiv.org/abs/2510.11977) —
  21,730 rollouts; a score is not a property of the model alone, and higher reasoning
  effort reduced accuracy in the majority of runs.
- *ResearchGym*, [arXiv:2602.15112](https://arxiv.org/abs/2602.15112) — a GPT-5 agent
  improved over baselines in 1 of 15 evaluations; named failure modes include impatience
  and poor resource management.
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*,
  [arXiv:2503.13657](https://arxiv.org/abs/2503.13657) — why this ships three roles, not
  seven.
- *Search-Time Contamination in Deep Research Agents*,
  [arXiv:2606.05241](https://arxiv.org/abs/2606.05241) — why there is no literature-search
  agent.
- Bran et al., *ChemCrow*, [arXiv:2304.05376](https://arxiv.org/abs/2304.05376) — GPT-4
  could not separate confidently-wrong chemistry from correct. Why the referee is a
  script.
- Yao et al., *τ-bench*, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) — pass^k,
  and why it is uninformative at temperature 0.

---

## 14. Provenance

Planned and written with LLM assistance, including the design review that produced this
architecture and the audit that retracted its flagship. Stated plainly because the
alternative is both false and less interesting. What the LLM did *not* do: run the
chemistry, decide what counts as a signed claim, or produce any number in this document.
Those go through `critic.py`, which is a script.

Reproduce the claims in this document:

```sh
uv venv && uv pip install -e .
python world/build.py --seed 0           # the environment, byte-for-byte from a seed
python selftest.py                       # 18 checks; re-derives the headline numbers
python analysis/abstention_battery.py    # both retractions, with the full battery
python analysis/screening_metrics.py     # thresholds, decomposition, delta_log
python analysis/method_comparison.py     # 8 cells, sealed, paired
python analysis/audit_agent_claims.py    # the eight agent verdicts
QM8_PROFILE=mock python agent.py         # a full rollout, no GPU, no network
```

`selftest.py` **re-derives** the headline numbers rather than reading them from a results
file, and one of its checks now asserts a **retraction** — it fails if the abstention
classifier ever beats every free baseline again. `--quick` runs the harness checks only,
skipping the model fits.

Two companion documents, with their standing stated rather than implied. `HUMAN.md` is the
hand-written running log, and it runs up to the point the audit arrived: the retractions in
this document are **not** in it, and the record of them is this file plus the commit
history. `docs/AUDIT_RESPONSE.md` is the response *plan*, written before the verification
it schedules; where it predicted an outcome the measurement then contradicted — it expected
E1 abstention to survive — **this document is the authority**.
