# What was done by hand

A running log, written as it happens rather than reconstructed afterwards. The point of
writing it live is that a reconstructed version is always flattering.

The short version: **the human built the environment, the referee and the search space.
The agent ran experiments inside it.** Where that boundary sits on any particular result
is recorded below.

---

## Tooling disclosure

This project was planned and is being written with LLM assistance (Claude), including the
design review that produced the current architecture. That is stated plainly because the
role is an agentic-AI one and the alternative — implying the design arrived unaided — is
both false and less interesting than the truth.

What the LLM did *not* do: run the chemistry, decide what counts as a signed claim, or
produce any number in the write-up. Those go through `critic.py`, which is a script.

---

## Log

### 2026-09-07 · design

- Two earlier design documents (`QM8 Agent Team Brief`, `QM8 Agent Pipeline`) were written
  first and are superseded. They are kept for provenance, not as specifications.
- A design review found the architecture was well defended against an agent that *lies*
  and undefended against an agent that *cannot do research*: the action space was an
  enumerated config grid, so the moves that produce a finding (conditioning, controls,
  interventions) had no expression in it.
- A second review found the remaining holes. Four changed the design materially:
  1. **The sealed set leaked through the author.** "Score once per signed claim" still
     lets a human submit many claims and report the ones that landed. Pre-registration
     was added.
  2. **`effect > k × seed sd` is not statistics.** Replaced with paired-seed comparison,
     bootstrap CIs, and a multiplicity correction over the confirmatory family.
  3. **The CC2 E2−E1 gap cannot define a deployment-realistic split** — it requires
     paying the cost the project is trying to avoid. Switched to the TDDFT gap. CC2-gap
     analysis is retained as post-hoc diagnosis only.
  4. **The state-reindexing result is not independent rediscovery.** The affordances
     (`state_gap`, `brightness`, `reindex_states`) are supplied by the human. It is
     **tool-assisted hypothesis generation**, and will be described that way.

### 2026-09-07 · what was found on disk

- **The earlier QM8 analysis was never written as files.** The figures quoted in both
  design documents — Δ at 0.066 eV, the 0.271/0.370 parse check, the 16.4% swap rate,
  the 0.30-vs-0.81 eV gap separation, the label-budget curve — have no artifact behind
  them and are treated as **unverified prior claims**, not as results.
  Nothing from them will be reported unless this repo regenerates it.
- Consequence: `world/build.py` exists so the environment resets from a seed, byte for
  byte. That is the fix, and it is also the thing the earlier version was missing.

### 2026-09-07 · repo

- `git init`, scaffold, first commit.
- Harness written for this project: the loop, the OpenAI-compatible client with token
  accounting, the bootstrap CIs, the failure histogram, the chemistry, the three-way sealed
  split, pre-registration and the claim verifiers. The agent decomposition follows
  Anthropic's *Building Effective Agents*; the interval method is the nonparametric
  bootstrap (Efron 1979).

### 2026-09-07/08 · build

Order was **world → referee → evaluator → agent**, inverting both earlier design
documents, which put the agent first. Everything below was found by running code, not
by planning.

**Written by hand (human + LLM pair):** every file in the repo. **Produced by the agent:**
three rollouts on a local Qwen3 8B, three claims, of which the referee signed two and
rejected one. No agent wrote any code.

Findings that came out of building, in the order they appeared:

1. **MoleculeNet's `qm8.csv` is damaged.** Its two PBE0 blocks are byte-identical across
   all 21,786 molecules, so it ships **12 distinct tasks under 16 column headers**.
   PBE0/def2TZVP is a verbatim copy of PBE0/def2SVP. This is why most papers report 12
   tasks. The genuine fourth level exists only in the 2015 supplementary release.
   Consequence: the `cheap_level` axis has three real values only because the environment
   parses the raw file. Anyone working from the CSV cannot run that comparison at all.
2. **DeepChem's 21,747 is not a SMILES-sanitization artifact.** RDKit parses all 21,786
   without a single failure. The earlier documents assumed otherwise.
3. **The prior claims that had no artifact behind them mostly hold.** Independently
   regenerated: |r| 0.903 within energies (claimed 0.90), 0.367 within oscillator
   strengths (0.37), 0.083 across blocks (0.08), 16.7% acyclic, 47.2% top-20 scaffold
   coverage, and a greedy scaffold test set of 2,178 molecules across 2,178 singleton
   scaffolds (claimed 2,180/2,180). So that analysis was real; it was just never written
   to disk.
4. **A hypothesis of mine died.** I proposed that "Δ-learning fails for oscillator
   strengths" was a metric artifact — that MAE on a right-skewed non-negative target
   rewards predicting ≈0. Predict-zero scores **0.0220** on f1, worse than cheap (0.0107)
   and worse than Δ (0.0120). The 2015 negative result replicates; my explanation for it
   does not. Recorded rather than quietly dropped.
5. ~~**Nobody predicted this one.** On the TDDFT-gap split, Δ beats cheap for f1
   (0.0206 vs 0.0216).~~ **WITHDRAWN 2026-09-08** — noise. See the entry below.
6. **The gap mechanism is visible in the energies.** E1/Δ error by TDDFT state gap, on the
   sealed set at full precision: 0.0379 eV on the widest-gap quartile against 0.0859 on the
   narrowest — a 2.26× ratio, CI [1.97, 2.59]. Not a monotone trend, which the first
   write-up of it got wrong.
7. **PBE0/def2TZVP is a better cheap baseline than def2SVP** for Δ-learning: 0.0630 vs
   0.0721 eV on the sealed test set, paired bootstrap CI [-0.011, -0.007]. Only
   measurable because of finding 1.
8. **Seed variance is exactly zero here.** LightGBM on fixed data is deterministic, so
   three seeds return the same number. Reporting "sd = 0.0" as a noise floor would invite
   an agent to treat any difference as real, so the observation now says which instrument
   to use instead — the paired bootstrap over molecules.

Bugs found by running things, each of which would have produced a wrong result:

- The greedy scaffold split put **zero** molecules in the sealed test set.
- The referee treated an **absent** scope key as a universal quantifier, failing claims
  for over-claiming about axes they never mentioned.
- The narrowed statement echoed the **claimed** scope back (`target=all`) instead of what
  was actually run, which defeats the entire purpose of narrowing.
- The comparison count matched on the claimed scope too, so a claim asserting "all"
  matched nothing and reported a family size of zero.

---

### 2026-09-08 · backend

Constraint set by the author: **everything runs locally.** No YSU GPU, no remote host.

That ruled out any remote GPU. Nothing on this machine could serve a model — no API keys in the environment,
and port 8000 turned out to be an unrelated Docker container, not vLLM. So Ollama was
installed and Qwen3 8B pulled: the closest local analogue, on an Apple Silicon laptop with
16 GB.

Consequence to state in the write-up rather than bury: the agent arm runs an **8B** model,
not the 35B/3B-active one the design reasoned about. Any null result from the agent arm is
therefore confounded with model capacity, and cannot be read as "LLM agents cannot do
this."

---

### 2026-09-08 · the referee turned on the author

Two of my own claims were tested rather than asserted, and one died.

- **Withdrawn.** "Δ-learning beats cheap for f1 on the TDDFT-gap split" — repeated several
  times as the most interesting open item in the project. On the sealed set at full
  precision: cheap 0.03077, Δ 0.03032, paired CI [-0.000463, +0.001328]. Includes zero.
  The original figure came from one fast-mode validation run. `h_f_on_gap_split` is marked
  FALSIFIED, which is what its own stated criterion required.
- **Survived.** The gap-stratified error result, re-measured on the sealed set: 2.26×
  ratio between narrowest and widest gap quartile, CI [1.97, 2.59]. But the description
  was wrong — it is not a monotone trend, the second quartile is marginally worse than the
  first. Corrected to what the data supports.
- **Hedged.** "The effect has not previously been reported" became "we found no prior work
  reporting that." QM8 is well studied and absence of a citation is not absence of prior
  art.

Worth naming plainly: the author produced a false finding, promoted it, and repeated it,
and the same machinery built to discipline the agent caught it. That is the argument for
the second half of the invariant.

---

### 2026-09-08 · the flagship claim, finally run

`intervene(reindex_states)` had existed since the tool layer was written and had never
been used. The state-ordering analysis — the most novel thing in the project — was
regenerated properly, with the control the 2015 conjecture needs.

It reproduces. Swap rates match the prior claims **exactly**: 16.4% / 12.1% / 11.8% for
PBE0-SVP / PBE0-TZVP / CAM. The control is clean — 0.0% swap rate and 0.00% gain on
energies, which are sorted by construction, so this is not an artifact of reshuffling two
correlated columns.

Three corrections to how it had been described:

1. **The gap-when-swapped is 0.50 eV, not 0.30.** The prior documents claimed 0.30 against
   0.81 for the rest. We get 0.50 against 0.84. The mechanism holds directionally —
   ambiguity concentrates in near-degenerate states — but that figure does not reproduce.
2. **Brightness-ordering IS the oracle, not a step toward it.** It equals the oracle gain
   exactly in all three rows, which follows from the rearrangement inequality: for two
   elements, sorting both sides by the same key minimises the sum of absolute differences.
   So ~20% is a ceiling. And it is a **task redefinition** — predict the bright transition
   and the dark one rather than the lower-energy and the higher — not a model improvement.
   Reporting "20% error reduction" without that sentence would be dishonest.
3. **Fixing the label does NOT rescue delta-learning.** This is the test that actually
   matters and nobody had run it. In brightness-ordered space, cheap 0.02507 vs delta
   0.02503, paired CI [-0.001239, +0.001391] — indistinguishable. So the 2015 negative
   result is not a labelling artifact; it survives the fix. That is a sharper statement of
   it than the original paper made.

`h_reindex_by_cheap_level` comes out **partially falsified**: swap rate falls with baseline
quality exactly as predicted (16.4 > 12.1 > 11.8), but oracle gain does not (19.7, 14.8,
20.8). The mechanism governs how *often* states are misordered, not how much it costs.

### 2026-09-08 · pre-registration, honestly

`preregister.json` had been a test fixture the whole time, so every real agent claim was
scored exploratory and the confirmatory path was never exercised.

Writing the real one raised the obvious temptation: register everything already measured
and call it confirmatory. That is backdating, and it is precisely what pre-registration
exists to prevent. So only two genuinely untested claims are registered —
`h_budget_plateau` and `h_misorder_signature` — and four others are listed under
`exploratory_by_construction` with the reason each cannot count.

Agent claims are exploratory by definition: the agent chooses what to claim after seeing
validation results, which is the adaptive selection the whole scheme guards against.

---

### 2026-09-08 · citations checked against the papers

The shipped write-up turned out to lean on almost nothing external — Ramakrishnan 2015
(verified by reproducing its own numbers) and two arXiv papers. Both were checked against the actual abstracts rather than against our
own planning documents:

- **HAL, arXiv:2510.11977** — verified. Title, 21,730 rollouts, 9 models × 9 benchmarks,
  2.5B tokens released, and *"higher reasoning effort reducing accuracy in the majority of
  runs"* all match.
- **ResearchGym, arXiv:2602.15112** — verified. Real paper, and the two things we cite are
  exact: GPT-5 improved over provided baselines in *1 of 15 evaluations (6.7%)*, and the
  named failure modes include *impatience, poor time and resource management, overconfidence
  in weak hypotheses* — which is what we attribute the agent's 4-of-12 early stopping to.

**One flagged discrepancy is NOT resolved, and saying it was is its own instance of this
project's failure mode.** The rev-1 planning document claimed HAL's abstract overstated its
own body — that the real finding was "in 21 of 36 runs higher reasoning effort did not
improve accuracy." This log previously concluded the earlier document's correction "was
itself the error."

That conclusion outran its evidence. What was checked is HAL's **abstract**, which says
higher reasoning effort reduced accuracy *in the majority of runs*. The claim being retracted
was about a **body table**. Those are different sentences and both can be accurate; an
abstract summarising 21,730 rollouts and a table reporting 36 configurations are not in
contradiction merely because their phrasings differ.

So the honest status: the abstract's wording is verified, the body table is not, and the
retraction is **withdrawn pending someone opening the PDF and quoting both verbatim**.
Asserting a retraction on evidence that does not cover the claim being retracted is the same
error as asserting a finding on a slice that does not cover the population — which is the
error this entire log is about, pointed the other way. Caught by a reviewer, not by us.

Precision fix that came out of this: `llm.py` had HAL's finding as
"model × scaffold × harness × budget", which is a paraphrase rather than the paper's own
wording ("models, scaffolds, and benchmarks"). Corrected to what HAL actually says.

The citation-dense material lives in `docs/PLAN.md`, which is a planning artifact rather
than part of the submission, and is still labelled *course canon* — printed on a slide,
upstream unverified.

---

### 2026-09-08 · how far the abstention rule reaches

Tested whether the misordering risk score transfers beyond `f1`. Exploratory — run only
after f1 worked, so it is not pre-registered and is labelled that way.

It transfers to **E1** (2.53× quartile ratio, 24% error reduction at half coverage) and
fails on **f2** and **E2**. E2's ratio is 0.94×, marginally inverted, which is a clean null
and a useful check that the classifier is not just flagging generally-hard molecules.

The mechanism holds up: a swap corrupts both states, but the second states are intrinsically
much harder (0.131 eV against 0.068 for E1), so misordering is a smaller fraction of their
error and disappears into it.

Worth having gone looking. The headline was "abstention works"; the true statement is
"abstention works for state-1 quantities", and the second one is the one someone could
deploy without being embarrassed on the second excited state.

---

### 2026-09-08 · E1 abstention through the referee

The E1 transfer had only been measured on validation. Audited on the sealed set: full
0.072082, selective@50% 0.059528, random 0.072074 with CI [0.067712, 0.076914] — selective
falls below the interval, so it holds. 17% error reduction at half coverage on the primary
energy target.

Signed, but typed **exploratory**, and the referee did that on its own because the claim is
not in `preregister.json`. It could not be confirmatory whatever the number said: E1 was
tested only after f1 worked. Registering it now would be backdating, which is the one thing
pre-registration exists to prevent.

---

### 2026-09-10/11 · the audit, and four retractions

An external review found the flagship claim did not survive and that several numbers in the
repo did not reconcile. Eight load-bearing findings were re-run against the repo's own
artifacts before being accepted; all eight confirmed. What follows is what that cost.

**The pattern, which is now the project's actual result.** Every headline claim failed the
same way — report the slice where the effect appears, omit the slices where it does not.

1. **f1 abstention — RETRACTED.** The headline "31% error reduction" was a *validation*
   number, printed four lines above the words "sealed-set verified". Sealed it is 20.6%,
   and that is not the problem: two **free** rules beat the classifier (cheap E2−E1 gap
   0.00803, |predicted correction| 0.00432 against its 0.01108), and under magnitude
   stratification it sits inside the null. Spearman(risk, f1) = +0.383 — it learned that
   bright molecules are risky. At 50% coverage it keeps 89 of 266 bright molecules where
   random keeps 133 [117, 148]. In a photophysics screen it abstains on the hits.
2. **E1 abstention — RETRACTED.** The audit proposed E1 as the claim that survived. Its
   figures reproduce on validation; on sealed the ordering reverses (classifier 0.05953 vs
   free gap 0.05686). **The rescue for a validation-number error was itself a validation
   number.** Two of us, one week, the same mistake, the second while auditing for exactly it.
3. **Screening metrics — QUALIFIED.** Δ wins on AUC/AP above f≥0.05 and loses significantly
   at f≥0.01 (−0.028 CI [−0.045, −0.011]). The proposal quoted only the winning cuts.
4. **Δ survives shift, direct does not — 1 OF 8.** Against `direct_aug` (direct given the
   same cheap columns), Δ wins significantly in one cell of eight — scaffold/E1, the exact
   cell the headline was written from.

**How it hid.** The README had the split sizes swapped — validation 2,178 / sealed 2,179,
when the truth is the reverse. Believing sealed was 2,179 made `n_kept=2179` look sealed.
One transposed pair of numbers concealed the whole thing.

**What I got wrong that was mine, not inherited.** The `--resume` I added for checkpointing
reused report rows while `run_agent` unlinked a fixed trace path, so two of eight shipped
traces refuted their own report rows. The underlying claim (3/8 seeds tried delta, 5.4 of 12
experiments) reproduces exactly on clean traces — the finding was right, the evidence I
shipped for it was not. A reviewer checking my traces would have found them contradicting my
headline and been right to stop trusting everything else.

**Harness bugs, same class as the science bugs.** The terminal `claim` tool bypassed the
try/except every other tool had, so one bad kwarg killed a whole rollout — found when the
clean re-run died at seed 2. The referee crashed on malformed claims despite a comment
promising it would not. The prompt told the agent to omit axes it was not claiming about and
the referee then rejected it for omitting them. Citing *extra* supporting evidence made a
claim score *worse*. Two claim kinds signed without touching data. The pattern: **the code
handling the terminal or error case is the code the happy path never exercises.**

**What survives.** The state-ordering effect is real (16.4% swap rate, energy control at
exactly 0.0%) — only the model built on it failed. `delta_log` is the one model that beats
raw TDDFT on global ranking (f1 +0.042, f2 +0.050), with the mechanism identified: Δ improves
the bright ranking and wrecks the dark bulk. Δ's advantage under scaffold shift on E1 holds
against a same-inputs baseline, CI [−0.0117, −0.0056]. The parse still reproduces the 2015
paper. selftest went 14 → 18.

**Two new referee checks** did the retracting — `free_baselines` and `magnitude_control` —
because the audit showed "beats random rejection" is a bar almost anything clears. And
`selftest`'s own guard had been certifying the flagship on validation: the check shared the
claim's blind spot. It now asserts the retraction and fails if the classifier ever beats
every free baseline again.

**Instance seven, and it is in a commit message about a bug fix.** Commit 4f61581 claimed
that a bug in `_count_comparisons` "silently disabled the Holm correction for exactly the
multi-split claims that need it". That is wrong. `holm()` is fed from `v.checks["_p_value"]`
over confirmatory verdicts in `audit()`; `_count_comparisons` only fills the
`comparisons_in_family` field displayed beside each verdict. The bug hid a number a reader
sees. It never touched the correction.

The fix was right, the diagnosis overstated. Caught by a collaborator reading the code rather
than the message — which is the only way it could have been caught, since a commit message is
not testable. It is recorded here because the commit itself cannot be edited, and because a
project making this argument does not get to leave a false claim in its own history
uncorrected.

**One more instance, caught while writing this.** The README asserted that this log contained
every retraction. It did not — it stopped before the audit entirely. Found by grepping my own
document for a word that should have been in it. That is the fifth occurrence of the same
failure and the reason this entry exists.

---

## Open, and honestly unresolved

- Whether the agent beats random search at equal budget is **not yet known**, and the
  result ships either way.
- Six design decisions have no support in the course literature and are labelled
  *reasoning* rather than *evidence* in the plan: pre-registration, exploratory/
  confirmatory typing, multiplicity correction, the random-search and human-grid
  baselines, OOD-based abstention, and the LLM-vs-GBM substitution.
- Citations inherited from the earlier documents are unverified and are being checked
  against the papers before anything is written up.
