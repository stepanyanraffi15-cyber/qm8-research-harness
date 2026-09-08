# Can the expensive calculation be skipped?

**QM8, delta-learning, and an agent harness built to stop itself lying**

---

## The question

QM8 gives 21,786 small organic molecules the same four excited-state quantities computed
at four levels of quantum theory. One — RI-CC2/def2TZVP — is accurate and costs hours per
molecule. Three are cheap TDDFT approximations already computed for every molecule.

So: **how well can the expensive answer be predicted without paying for it, how few
expensive labels does that take, and where does it break?**

That question is nearly settled in the literature. The more interesting one, and the one
this submission is actually about, is the second-order version: **can an LLM agent do
that research, and how would you know?**

---

## What was built

Build order was **world → referee → evaluator → agent**, which inverts the obvious order
and follows the ArmLLM Day 4 rule: *"Build the referee before the player."* Nothing an
agent produces means anything until something exists that can score a rollout.

```
world/build.py   the environment, regenerable from a seed, byte for byte
critic.py        the referee: 8 checks, 3 verdicts, deterministic
evaluate.py      three arms on one budget, bootstrap CIs, cost
agent.py         the loop: policy + tools + observation + memory + stopping rule
```

### The invariant

> No number reaches the write-up except through a scorer that **neither the agent nor the
> author** can query adaptively.

Both halves matter. A sealed test set defeats an agent that would otherwise optimise
against the metric it is judged on — one scalar at a time, over thirty tool calls. It does
**not** defeat an author who submits eleven claims and reports the three that landed. So a
claim counts as *confirmatory* only if `preregister.json` carries it with a hash predating
the audit; everything else is scored and reported as *exploratory*, with its multiplicity
stated.

Enforcement is mechanical, not procedural: `models.run(eval_on="sealed_test")` raises
without an audit token that nothing importable from `tools.py` holds.

---

## What we found, ordered by what matters

| # | Finding | Status |
|---|---|---|
| A | **Misordering is predictable from structure alone**, giving an abstention rule that needs no CC2. AUC 0.693, error 3.52× higher in the top risk quartile, beats random rejection at every coverage. | **pre-registered · signed on the sealed set · confirmatory** |
| B | **174× label efficiency** — Δ-learning on 100 CC2 labels beats direct learning on all 17,429. | measured, 3 seeds |
| C | **Δ-learning's real value is robustness, not accuracy** — under scaffold shift direct learning is worse than doing nothing; Δ is not. | measured |
| D | **MoleculeNet's `qm8.csv` ships 12 tasks under 16 headers.** One level of theory is a verbatim duplicate. | verified upstream |
| E | **Fixing the state labelling does not rescue Δ-learning for oscillator strengths.** The 2015 negative result survives its own proposed remedy. | measured, CI includes zero |
| F | **The agent loses to a 12-line schedule on search, and only 1 of 8 claims survives the referee** — 2 were exactly backwards. | 8 seeds, 8B model |
| G | `h_budget_plateau` — the plateau moves *backwards* from the registered prediction. | **pre-registered, falsified** |
| H | Δ beats cheap for f1 on the gap split. | **withdrawn — noise** |

Two of these are pre-registered, and one of those is falsified. One is a claim of
ours that the referee killed. Both facts are load-bearing: a project that only
reports what worked has not demonstrated that its referee does anything.

---

## Results

### 1. MoleculeNet's QM8 distribution is damaged

`qm8.csv` — the file most QM8 papers are trained on — advertises 16 tasks under 16 column
headers. Its two PBE0 blocks are **byte-identical across all 21,786 molecules**:

```
PBE0 block A == block B exactly : 21786 / 21786
max abs difference              : 0.0
```

PBE0/def2TZVP is a verbatim copy of PBE0/def2SVP. The file contains **12 distinct tasks**,
which is why most published work reports 12. The genuine fourth level of theory exists
only in the 2015 supplementary release.

This is not a footnote. It means the `cheap_level` comparison below is unrunnable for
anyone working from the standard distribution.

A second correction: DeepChem returns 21,747 molecules rather than 21,786, and both prior
design documents for this project assumed that gap was RDKit sanitization failure. It is
not — RDKit parses all 21,786 with **zero** failures.

### 2. Delta-learning, and where it is actually strong

All numbers from our own parse, validation split, `speed=fast`, seed 0.

| target · split | cheap | direct | Δ | predict-zero |
|---|---:|---:|---:|---:|
| E1 · random | 0.275 | 0.230 | **0.075** | — |
| E1 · scaffold | 0.257 | 0.383 | **0.082** | — |
| E2 · random | 0.371 | 0.244 | **0.144** | — |
| E2 · scaffold | 0.306 | 0.340 | **0.152** | — |
| f1 · random | **0.011** | 0.016 | 0.012 | 0.022 |
| f1 · tddft_gap | 0.022 | 0.025 | 0.021 † | 0.032 |

Energies in eV, oscillator strengths in a.u. † This apparent Δ win is **not**
significant on the sealed set — see §6, where it is withdrawn.

**The strongest result is the scaffold row.** Under structural shift a structure-only
model (0.383) is *worse than doing nothing at all* (0.257), while the multi-fidelity model
holds at 0.082. That is the claim worth making to a company screening novel chemistry: it
is not that Δ-learning is more accurate, it is that Δ-learning is the only one of the two
that survives the distribution shift screening actually involves.

The parse is verified against Ramakrishnan et al. 2015 by reproducing their own numbers:
raw PBE0/def2SVP against CC2 gives **E1 0.2712 eV** (paper: 0.27) and **E2 0.3704 eV**
(paper: 0.37).

### 3. A better cheap baseline gives a better Δ-model

On the sealed test set, at full precision, adjudicated by the referee:

```
PBE0/def2TZVP  0.0630 eV
PBE0/def2SVP   0.0721 eV
paired bootstrap CI [-0.011, -0.007] over 2,178 molecules
val -> test gap  +0.004
```

This comparison is only possible because the environment parses the raw 2015 release.

### 4. The error lives in near-degenerate states

E1 Δ-error, stratified by the *cheap* TDDFT E2−E1 gap. **Sealed test set, full precision**:

| gap quartile (eV) | n | MAE (eV) |
|---|---:|---:|
| 0.001 – 0.282 | 545 | 0.0859 |
| 0.282 – 0.517 | 544 | **0.0887** |
| 0.517 – 0.976 | 544 | 0.0758 |
| 0.976 – 3.422 | 545 | **0.0379** |

```
ratio, narrowest / widest quartile = 2.26x    CI [1.97, 2.59]
```

The interval excludes 1, so the effect is real. But state it precisely: this is **not** a
monotone trend — the second quartile is marginally worse than the first. What the data
supports is that the **widest-gap quartile is roughly half the error of everything else**,
not a smooth decline with separation.

Ramakrishnan et al. attributed the oscillator-strength failure to state ordering being
ambiguous between methods. The same mechanism appears to be measurably present in the
*energies*. We found no prior work reporting that, though we make the weaker claim
deliberately — this is a well-studied dataset and absence of a citation is not absence of
prior art.

Note the split used for distribution shift is on the **TDDFT** gap, not the CC2 gap. The
CC2 gap would be the natural quantity and is the wrong one — it is only available after
paying the cost the whole exercise exists to avoid. The TDDFT gap is computed for every
molecule in a screening set already.

### 5. A hypothesis of ours died, and the baseline killed it

We proposed that "Δ-learning fails for oscillator strengths" was a metric artifact: f is
non-negative and heavily right-skewed (32% of molecules below 10⁻³), so MAE should reward
predicting ≈0.

**Predict-zero scores 0.0220 on f1** — worse than cheap (0.0107) and worse than Δ (0.0120).
The explanation is wrong. The 2015 negative result replicates; our reason for it does not.

Recorded because a project whose thesis is verifiable provenance does not get to quietly
drop its own failed hypotheses.

### 6. A finding of ours that the referee killed — the author's, not the agent's

During the build, Δ-learning appeared to beat the cheap baseline for f1 on the **TDDFT-gap
split only** — 0.0206 against 0.0216, the one split where the 2015 negative result seemed
to reverse. It isolates near-degenerate states, so a mechanism was ready to hand. It was
written up as the most interesting open item in the project and repeated several times
before anyone tested it.

Put through the referee, on the sealed test set, at full precision:

```
cheap 0.03077    delta 0.03032
paired difference +0.000442    CI [-0.000463, +0.001328]
excludes zero: NO
```

**It is noise.** The original number came from a single fast-mode run scored on validation,
which is precisely the "best-of-N on a noisy metric" the referee exists to reject — and it
was the author who produced it, not the agent.

This is the more important half of the invariant doing its job. A sealed set that only
disciplines the agent leaves the human free to promote any suggestive number they happen to
like, and a project whose thesis is verifiable provenance does not get to exempt its own
author. The claim is withdrawn.

### 7. Did the agent do anything?

Three arms, same budget of 12 experiments per run, agent driven by a local Qwen3 8B at
temperature 0.7. Eight seeds for the agent; the null policies are deterministic given the
space and the schedule.

**On search efficiency the agent loses to both null policies.** The grid sits below the
agent's confidence interval on every target, and the agent loses to *random sampling* on
three of four:

| target | agent (8 seeds) | 95% CI | grid | random |
|---|---:|---|---:|---:|
| E1 | 0.190 | [0.111, 0.271] | **0.064** | 0.205 |
| E2 | 0.219 | [0.169, 0.244] | **0.120** | 0.143 |
| f1 | 0.0150 | [0.0126, 0.0177] | **0.0090** | 0.0121 |
| f2 | 0.0336 | [0.0300, 0.0354] | — | **0.0310** |

The obvious explanation is impatience — mean 5.4 of 12 experiments used, 45% of budget,
six of eight seeds stopping at exactly four. That is ResearchGym's named failure mode and
it is real. **It is also not the main problem.**

| seed | experiments | methods tried | best E1 |
|---:|---:|---|---:|
| 5 | 11 | `direct` × 11 | 0.230 |
| 3 | 8 | `direct` × 4, `delta` × 4 | 0.075 |
| 2 | 4 | `direct` × 2, `delta` × 2 | **0.065** |
| 6 | 4 | `direct` × 2, `delta` × 2 | **0.065** |
| 0, 1, 4, 7 | 4 | `direct` only | 0.230 – 0.397 |

**Five of eight seeds never tried delta-learning at all** — the single most important method
in the space, and the one every other result in this write-up depends on. All five are stuck
at 0.230 or worse. All three that did try it landed at 0.065–0.075, essentially matching the
grid's 0.064 in two to four experiments.

Seed 5 settles the question. It spent **eleven of twelve experiments running `direct` eleven
times**: the most budget of any run, and nothing learned. More steps do not help an agent
that is not varying the axis that matters.

So the failure is not impatience, it is **failure to explore the method dimension**, and
impatience is a symptom rather than the cause. That is a harness and prompt finding before
it is a model finding — the tool schema lists all four methods with descriptions, and the
agent fixates on the second one anyway. It is also the most actionable result here: an
agent that reaches `delta` finds the answer almost immediately.

### 8. What the referee does to eight agent claims

Search efficiency is not the only thing worth measuring, and it is not what the brief asked
about. Every seed's claim was put to the referee — re-scored at full precision on a sealed
test set the agent never saw.

```
VERDICTS           signed 1 · narrowed 2 · rejected 5
FAILURE HISTOGRAM  overscoped 3 · unsupported_claim 3 · wrong_direction 2
```

| seed | verdict | why |
|---:|---|---|
| 0 | narrowed | claims `E2`, ran `E1` |
| 1 | rejected | **wrong direction** — cheap beats direct by 0.0042 |
| 2 | **signed** | delta beats direct for f1 |
| 3 | rejected | names no split |
| 4 | rejected | over-scoped, and `config_a`/`config_b` missing entirely |
| 5 | rejected | **wrong direction** — cheap beats direct |
| 6 | narrowed | claims `E1`, ran `E1` and `f1` |
| 7 | rejected | names no split |

**One claim in eight survives intact.** Two are exactly backwards: the agent asserted a
direction the sealed data contradicts. Without the `direction` check — which exists only
because an earlier rollout produced such a claim and an earlier version of the referee
signed it — those two would have passed. A single late-added check catches a quarter of
this agent's output.

Three seeds earlier reported 2 signed and 1 rejected, which was luck. Eight seeds give 1 in
8. That is the same small-sample trap this project keeps catching in its own work, and it
is the reason the referee reports a histogram rather than a pass rate: *"'62%' tells you
nothing. A histogram of failures tells you what to fix."*

The honest answer to "can a team of agents do research on QM8" is therefore narrow and
specific. **It can generate findings; it cannot be trusted to grade them; and the grading
is separable, cheap and mechanical.** Which is the whole argument for building the referee
first.

### 9. The referee's own bug, found by a real agent

An earlier version of `critic.py` **signed seed 1's backwards claim.**

Checking that a paired bootstrap CI excludes zero establishes that a difference is *real*.
It does not establish that the difference runs the way the claim says. `config_b` is by
construction the config a claim asserts is better, so the paired difference must be
positive — and nothing was testing that.

Check 5, `direction`, now does — and across eight seeds it rejects **two of eight** claims,
so a check added late in response to one accident turns out to catch a quarter of this
agent's output.

The mock agent could never have surfaced it: the mock only ever over-claims, never inverts.
It took a real model making a real mistake.

A second robustness bug surfaced the same way. Seed 4 emitted `kind: "comparison"` with a
null `config_b`, and the referee raised a `TypeError` that killed the entire audit run. A
malformed claim must produce a *verdict*, not a traceback — the referee has to be at least
as robust as the loop it grades, where errors are already observations rather than
exceptions. Both bugs were found by running eight rollouts, not by reading the code.

### 10. How few expensive labels? 174× fewer.

The question the project is named after. E1, PBE0/def2TZVP baseline, three seeds
per point with bootstrap intervals — the earlier version of this curve was a
single run per point and its headline was read straight off it.

| CC2 labels | direct (random) | Δ (random) | direct (scaffold) | Δ (scaffold) |
|---:|---:|---:|---:|---:|
| 100 | 0.566 | **0.120** | 0.649 | **0.111** |
| 500 | 0.436 | 0.091 | 0.523 | 0.087 |
| 2,500 | 0.298 | 0.075 | 0.437 | 0.074 |
| 10,000 | 0.242 | 0.067 | 0.398 | 0.070 |
| all 17,429 | 0.230 | 0.064 | 0.383 | 0.068 |

Read the crossing point: **Δ-learning on 100 labels (0.120 eV) beats direct
learning on all 17,429 (0.230 eV)** — 174× fewer expensive calculations, and the
same factor on the scaffold split. Returns flatten early; going from 2,500 to
17,429 labels, roughly seven times the CC2 compute, buys 0.011 eV.

**`h_budget_plateau` — pre-registered — is FALSIFIED.** The registered prediction
was that scaffold shift would need *more* labels to plateau. It plateaus at 2,500
against random's 10,000: **earlier, not later**, the opposite of the prediction.

The first version of the analysis script printed "MOVES", which is true and
useless — direction is part of the claim. That is exactly the bug the referee's
`direction` check exists to catch, committed again in our own analysis code a few
hours after fixing it in the referee. Recorded because it is the second time the
same error appeared, which says something about how easy it is.

### 11. The result that is actually useful: abstention without CC2

**`h_misorder_signature` — pre-registered before measurement — is SUPPORTED.**

Every other result here needs CC2 to know a molecule is misordered, which is
useless: CC2 is the thing being avoided. So: can misordering be predicted from
**structure alone**?

```
structure-only classifier   AUC 0.693   CI [0.664, 0.723]
control (a coin flip)       AUC 0.499   CI [0.466, 0.531]
```

The control is the registered one and it is clean. Δ-model error then follows the
predicted risk monotonically across all four quartiles:

| risk quartile | n | Δ MAE (a.u.) |
|---|---:|---:|
| lowest | 545 | 0.0055 |
| | 544 | 0.0116 |
| | 545 | 0.0132 |
| highest | 545 | **0.0195** |

**3.52× between highest and lowest, CI [2.85, 4.32].**

Which makes the risk–coverage curve computable, against the only baseline that
means anything — random rejection at matched coverage:

| coverage | selective MAE | random MAE | random 95% CI | beats random |
|---:|---:|---:|---|---|
| 100% | 0.01244 | 0.01244 | — | — |
| 90% | 0.01133 | 0.01245 | [0.01199, 0.01281] | yes |
| 80% | 0.01063 | 0.01244 | [0.01181, 0.01301] | yes |
| 70% | 0.00984 | 0.01245 | [0.01161, 0.01323] | yes |
| 60% | 0.00916 | 0.01244 | [0.01144, 0.01338] | yes |
| 50% | 0.00856 | 0.01243 | [0.01120, 0.01364] | yes |

At half coverage the error falls **31%** while random rejection stays flat.

#### Audited on the sealed test set

This is the one claim in the project that clears every gate: registered before
measurement, re-scored at full precision on a partition nothing had touched, with
the registered control.

```
sealed    full 0.013946    selective@50% 0.011080    random 0.013954
          random 95% CI [0.012426, 0.015482]  -> selective falls below it
control   coin-flip classifier AUC 0.4862  CI [0.4532, 0.518]  -> null
verdict   SIGNED · CONFIRMATORY
```

Note the honest val→test gap: full-coverage error is 0.0139 on the sealed set
against 0.0124 on validation. The selective advantage survives it.

This is the part that maps onto screening novel chemistry rather than onto a
benchmark: a model that declines on the compounds it is about to get wrong, from
structure alone, before any expensive calculation is run. It is worth more to a
company screening unfamiliar compounds than a lower average error with no idea
where it fails.

---


## What the referee catches

Seven checks. The first three catch fabrication and leakage. The last four catch how ML
research actually goes wrong, and were absent from the original design:

| check | catches |
|---|---|
| split hash | a claim measured on a different partition than it names |
| evidence resolves | a number that was never computed |
| sealed re-run | precision mismatch; produces the val→test gap |
| **noise floor** | paired bootstrap over molecules, not a `k × sd` threshold |
| **multiplicity** | Holm over the confirmatory family — best-of-N is an overestimate by construction |
| **scope** | a quantifier wider than the configs actually run |
| **control** | a mechanistic claim with no control experiment |

Three verdicts, not two. **Narrowed** is the one worth having: most bad claims in ML
research are not fabrications, they are true statements with an oversized quantifier, and a
binary referee has to either accept or destroy them.

Demonstrated end to end, offline. The mock agent claims *"Delta-learning beats direct
learning for excited-state prediction"* over all targets from two runs on E1. The referee
narrows it to *"— for target=E1, split=random, cheap_level=PBE0-SVP only."*

The leakage control is the one that matters most: permuting the training labels drives MAE
from **0.075 to 0.967**. There is no path by which test information reaches the model.

---

## Honest limits

- **The agent is an 8B model, not the 35B the design reasoned about.** The author's
  constraint was that everything run locally; nothing on the machine could serve a larger
  model. So the agent arm's poor search efficiency is **confounded with model capacity**
  and must not be read as "LLM agents cannot do this."
- **Three seeds is thin.** The agent's E1 interval spans [0.065, 0.230]. The right response
  is more seeds, not a stronger sentence.
- **Citations inherited from earlier planning documents are unverified.** The arXiv ids in
  `docs/PLAN.md` came from LLM-assisted drafts and several sit at or past the assistant's
  knowledge cutoff. They are labelled *course canon* rather than *verified* for that reason,
  and should be checked against the papers before any of them is quoted in a submission.
- **Gradient boosting on fingerprints is a weak predictor**, chosen for throughput
  (~10 s/fit). It discards the 3D geometry QM8 ships. The Δ-model barely notices; the
  direct model is crippled by it, so any Δ-vs-direct gap is partly a statement about the
  featurization. Published QM8 leaders are GNNs.
- **The reindexing affordances are supplied.** `slice_error(by="state_gap")` and
  `intervene("reindex_states")` are handed to the agent. An agent that recovers the
  state-ordering result through them has done **tool-assisted hypothesis generation**, not
  independent discovery.
- **Seed variance is exactly zero.** LightGBM on fixed data is deterministic. Seeds are the
  wrong instrument here; the paired bootstrap over molecules is the right one.
- **`propose_features` is absent, and that is a scope decision, not a safety one.** The
  literature says *sandbox* code execution, not omit it, and the counter-evidence for
  omitting is heavier than the support.

## Where this design runs ahead of its evidence

Six decisions have **no support** in the course literature and are labelled *reasoning*
rather than *evidence*, deliberately: pre-registration against author-side selection,
exploratory/confirmatory typing, multiple-comparison correction, the random-search and
human-grid baselines, OOD-based abstention, and the LLM-vs-gradient-boosting substitution.

Two places an earlier draft over-claimed, now corrected:

- *Extended thinking off by default* is supported for tool-**routing** steps and
  contradicted for hard single-shot reasoning. It scopes per step type, not globally.
- *Deferring code execution* was framed as a safety argument. It is a scope argument.

### One harness finding worth reporting rather than burying

"Reasoning effort off" is **not portable**. Through Ollama's OpenAI-compatible endpoint,
`think: false` and `reasoning_effort: low` are both silently ignored; Qwen3 reasons
unboundedly and never reaches an answer, producing 2,000 tokens of nothing. Its native
`/api/chat` accepts the flag and answers in 4.

| route | tokens | answer |
|---|---:|---|
| OpenAI-compatible, `think=False` in `extra_body` | 300 | none |
| OpenAI-compatible, `reasoning_effort=low` | 300 | none |
| OpenAI-compatible, 2000-token budget | 2000 | none |
| native `/api/chat`, `"think": false` | **4** | correct |

Fixing it took the harness check from 647 to 18 completion tokens and 28.1s to 5.7s per
tool call — the same effect Day 4 measured on vLLM (333→40 tokens, 18.0→3.9s), on a
different stack. An OpenAI-compatible interface commits to nothing, and it turns out to
guarantee nothing about this either. Score is a property of model × scaffold × harness ×
budget, and this is that landing on our own setup.

---

## Provenance

The harness adapts the ArmLLM 2026 Day 4 exercise (`osoblanco/ArmLLM`, `2026/agents`).
Reused: the loop's shape, the OpenAI-compatible client and token accounting, the bootstrap
recipe, the failure-histogram discipline. Written here: all chemistry, the three-way sealed
split, pre-registration, and the claim verifiers.

The science follows Ramakrishnan, Hartmann, Tapavicza & von Lilienfeld, *Electronic spectra
from TDDFT and machine learning in chemical space*, J. Chem. Phys. **143**, 084111 (2015).

What was done by hand, and what was not, is in `HUMAN.md` — written as it happened rather
than reconstructed, including four bugs that would each have produced a wrong result.
