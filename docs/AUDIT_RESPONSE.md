# Audit response — plan

An external review found the flagship claim does not survive, and several numbers in the repo do
not reconcile with each other. **I re-verified the load-bearing findings myself before accepting
them.** This file records what I checked, what I concluded, and what happens next.

Branch: `audit-response`. Deadline: 14 Sep.

---

## Verification status

Not taken on faith. Each of these was re-run against the repo's own artifacts.

| # | Finding | Status | Evidence I reproduced |
|---|---|---|---|
| 1 | The headline 31% is a **validation** number | **CONFIRMED** | `risk_coverage.json` coverage 1.0 → `n_kept=2179`; validation is 2179, sealed is 2178. Sealed reductions are **20.6%** (f1) and **17.4%** (E1), not 31%/24%. |
| 2 | Shipped traces contradict the shipped headline | **CONFIRMED** | Seed 0's trace: `['direct','direct','delta','delta']` + control + slice + claim. Its report row: all-`direct`. Seed 1's trace has no claim row. `arms_qwen3` runs 0–2 byte-identical to `arms_agent8`. |
| 3 | `uv pip install -e .` fails | **CONFIRMED** | No `[build-system]`; setuptools flat-layout discovery aborts on five top-level dirs. |
| 4 | Experiment ids collide across rollouts | **CONFIRMED** | `experiment_log.jsonl`: `e_0000` appears **12×**, `e_0001` 12×. The referee resolves against whichever it hits first. |
| 5 | The one signed confirmatory verdict has no regeneration path | **CONFIRMED** | No `claims_abstention.json`; `audit_abstention.json` referenced by no code and no document. |
| 6 | The f1 abstention claim is a **magnitude artifact** | **CONFIRMED** | See table below. Classifier is the worst of five risk scores; magnitude-stratified null **contains** it. |
| 8 | Claim C is false as stated | **CONFIRMED** | `direct` + 4 cheap columns: random 0.0649 vs Δ 0.0645; scaffold 0.0809 vs Δ 0.0676. |
| 9 | 174× is left-censored | **CONFIRMED** | `SIZES` starts at 100; `17429/100 = 174.29`. The reported factor *is* the first grid row. |
| 7, 10–14 | E1 survival, brightness discordance, DeepChem 2021, referee gaps, scoring artifact, thin traces | **accepted pending own check** | Scheduled Day 1–4 below; each is checkable in the repo. |

### The measurement that ends the flagship claim

Sealed f1, 50% coverage, every rival risk score:

| risk rule | selective MAE | cost |
|---|---:|---|
| **the pre-registered classifier** | **0.011080** | 1036 features, trained |
| cheap E2−E1 gap threshold | 0.008034 | **free** |
| \|residual\| regressor | 0.004892 | same features |
| \|predicted correction\| from the fitted Δ model | **0.004321** | **free — already computed** |
| oracle \|error\| | 0.001894 | — |
| random rejection | 0.013954 | CI [0.012426, 0.015482] |

Magnitude-stratified control (keep the lowest-risk half *within* each \|f1_CC2\| quintile):
**0.012597**, against a stratified null of **[0.012540, 0.015460]**. Inside the null.
Spearman(risk, f1_CC2) = **+0.383** — the classifier learned *"bright molecules are risky."*

For a photophysics screen, that is abstaining on the hits.

---

## The decision

**Retract the flagship, lead with the retraction, and promote what survived.**

The repository's thesis is that claims require adversarial verification and that the verifier must
outlive the claim. It already demonstrates that three times: the referee killed `h_f_on_gap_split`;
a real rollout exposed the missing `direction` check; eight seeds changed the diagnosis rather than
the error bars.

**This is the fourth instance and the largest** — the pre-registered, sealed-signed, control-clean
flagship did not survive a control nobody had run. That is the thesis being true about itself at
maximum stakes, and it is a stronger submission than the clean win would have been.

Two conditions for it to land, and they are not negotiable:

1. **The numbers must reconcile.** Day 1 is integrity only. A reviewer who finds the repo
   contradicting itself stops trusting everything else, and every Tier-1 item is findable in under
   ten minutes.
2. **The tone is confident, not apologetic.** Lead with what survived.

### One addition to the reviewer's plan, and it is the lesson

The review offers a replacement headline — Δ-learning wins on every screening metric (AUC, AP,
enrichment) even though it loses on MAE. **It does not become the headline until I have run the
same controls that killed the last one.** Adopting an unverified replacement flagship would repeat,
in the same week, the exact error being corrected.

So Day 3 begins with an adversarial pass on the *new* claim: magnitude stratification, free
baselines, a bright/dark decomposition, and the oracle ceiling. It ships as the lead only if it
survives all of them; otherwise it ships as exploratory with its failures stated.

---

## Plan

### Day 1 — integrity. Nothing else.

Every item here is a place the repo contradicts itself.

- [x] Replace 31% → **20.6%** and 24% → **17.4%** in `README.md`, `writeup/RESULTS.md`, and
      **the SVG figure**, which currently plots the validation series.
      *Done. `analysis/misorder_signature.py::risk_coverage_sealed` regenerates the curve on
      `sealed_test` with the classifier fit on `train` alone — same construction as
      `critic._sealed_selective`, so its 50% row reproduces the referee's 0.011080 exactly.
      Written to `results/risk_coverage_sealed.json`; `analysis/figures.py` now reads that file
      and refuses to plot anything whose `eval_on` is not `sealed_test`.*
- [x] Move "sealed-set verified" so it sits beside sealed numbers, never validation ones.
      *Done. The validation curve is still reported in `RESULTS.md`, explicitly labelled, as the
      val→test gap — never as the headline.*
- [ ] `pyproject.toml`: add `[build-system]` + explicit `py-modules`, so step 1 of *Reproduce it*
      actually runs. Verify in a clean venv.
- [ ] **Globally-unique experiment ids** (`tools.py:83`) — run-scoped prefix, not `len(entries)`.
- [ ] Fix `evaluate.py` resume/unlink: never reuse a report row whose trace file was rewritten.
      Bind each trace to its report row by content hash.
- [ ] **Re-run seeds 0–7 cleanly under the fixed harness**, or delete the contradicted rows and say
      why. Do not ship a trace that refutes its own report row.
- [ ] Add `results/claims_abstention.json` + an `audit_arms.py` driver so every verdict regenerates.
- [ ] `results/README.md` indexing what each artifact is and which command produced it.

### Day 2 — retract and replace the flagship.

- [ ] Add four baselines to `critic._sealed_selective` **and** the figure: cheap gap,
      \|predicted correction\|, residual-magnitude regressor, magnitude-stratified random.
- [ ] Publish the stratified control and the Spearman diagnostic.
- [ ] Demote f1 abstention to **"does not survive"**, with the table above.
- [ ] Verify **E1** against the same battery. Promote it only if it passes (review says it does —
      free gap rule 0.05749 vs 0.05182, stratified null clears decisively; I check this myself).
- [ ] Add the **bright-retention** row: 89/266 truly-bright kept vs 133 [119, 148] random. This is
      the number a drug-discovery reviewer asks for.
- [ ] Add the oracle ceiling to every risk-coverage plot.

### Day 3 — the replacement result, adversarially first.

- [ ] **Verify the screening-metric claim myself** before it is promoted: AUC/AP/enrichment for Δ vs
      cheap on f1 and f2, paired CIs, magnitude stratification, bright/dark Spearman decomposition.
- [ ] If it survives: new section, and it becomes the lead result.
- [ ] `delta_log` (train Δ on `log10(f + 1e-4)`) as a fifth method — reported as exploratory.
- [ ] State precisely why this does **not** collide with §5: predict-zero refutes a *degeneracy*
      artifact; it says nothing about an *ordering* artifact. The strawman was refuted; the live
      version stands.

### Day 4 — harness, then rewrite.

- [ ] Fix the prompt/referee contradiction: `tools.py:593` tells the agent to omit un-claimed axes;
      `critic.py:212` rejects claims omitting `split`. Re-audit; **report both the old and new
      "1 of 8"**, since the change is itself a finding about referee design.
- [ ] `_check_scope`: membership, not set equality — citing an extra supporting experiment should
      not make a claim worse.
- [ ] Make `kind="mechanism"` / `kind="value"` touch data or refuse to sign.
- [ ] Make malformed claims produce verdicts (`audit([good, None])` currently dies).
- [ ] Add `direct_aug` as a fourth method; restate claims B and C against it.
- [ ] Rewrite `README.md` and `writeup/RESULTS.md` around the corrected story.

### Cut entirely

The frontier-model arm, any multi-agent team arm, 3D geometry features. Each is *"would add
value"*; none survives contact with a submission whose current numbers do not reconcile.

---

## What survives, and is stronger for the audit

Stated first in the rewrite, not last.

1. **The referee's selective null is properly calibrated** — 2.29% empirical false-positive rate
   over 20,000 noise draws against 2.5% nominal; the coin-flip control clears 0 times in 40 fits.
   The machinery works. It was aimed at the wrong null, which is a different and fixable problem.
2. ~~**E1 abstention is real and mechanism-driven**, and beats every rival including the
   magnitude-stratified control.~~ **SUPERSEDED by this document's own Day 2 verification.**
   E1 is the one rule that clears the magnitude-stratified null, so the mechanism is real —
   but on the sealed set a *free* cheap-gap threshold beats it (0.056862 vs 0.059528), so
   the 1036 features buy nothing. The audit's survival figures (0.05182 / 0.057493)
   reproduce on **validation**, which is the error this whole document exists to correct.
   Retracted alongside f1.
3. **Δ under scaffold shift beats even a feature-augmented direct model by ~16%**
   (0.0676 vs 0.0809). This is the claim that maps to screening, and it holds in a fair fight.
4. **The environment, parse, split hashing and selftest are solid** — 14/14, numbers re-derive.
5. **The Ramakrishnan reproduction is genuine** — 0.2712 / 0.3704 against 0.27 / 0.37.
