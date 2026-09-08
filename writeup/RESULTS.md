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

E1 Δ-error, stratified by the *cheap* TDDFT E2−E1 gap:

| gap quartile (eV) | n | MAE (eV) |
|---|---:|---:|
| 0.000 – 0.285 | 545 | **0.095** |
| 0.285 – 0.520 | 544 | 0.078 |
| 0.520 – 0.999 | 545 | 0.085 |
| 0.999 – 4.383 | 545 | **0.042** |

A 2.3× spread. Ramakrishnan et al. attributed the oscillator-strength failure to state
ordering being ambiguous between methods; this shows the same mechanism is measurably
present in the *energies*, where the effect has not previously been reported.

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

Three arms, same budget of 12 experiments, 3 seeds, agent driven by a local Qwen3 8B at
temperature 0.7:

| arm | E1 | E2 | f1 | f2 | tokens |
|---|---:|---:|---:|---:|---:|
| random sampling | 0.205 | 0.143 | 0.0121 | 0.0310 | 0 |
| **fixed human grid** | **0.064** | **0.120** | **0.0090** | — | **0** |
| agent (Qwen3 8B) | 0.175 | 0.244 | 0.0144 | 0.0354 | 31,649 |

**On search efficiency the agent loses to both null policies**, and spends 31,649 tokens
doing it. A 12-line schedule written before any results were seen beats it on every target
it covers, and beats uniform sampling 3.2× on E1.

The diagnosis is concrete rather than a shrug: **all three seeds spent 4 of their 12
experiments before claiming.** The agent stops early, which is the *impatience* failure
mode ResearchGym names, and it is a harness-and-prompt problem before it is a model
problem. The E1 confidence interval, [0.065, 0.230], contains the grid's 0.064 — one seed
matched a pre-written plan, two did not.

### 8. But two thirds of its claims survived the referee

Search efficiency is not the only thing worth measuring, and it is not the thing the brief
asked about. Putting each seed's claim to the referee — re-scored at full precision on a
sealed test set the agent never saw:

| seed | claim | verdict | measured |
|---|---|---|---|
| 0 | direct beats cheap for E2 | **signed** | 0.209 vs 0.376 |
| 1 | direct beats cheap for f1 | **rejected** | 0.0161 vs 0.0120 — backwards, and over-scoped |
| 2 | delta beats direct for f1 | **signed** | 0.0123 vs 0.0161 |

**Signed 2, rejected 1.** The agent produces a mixture of true and false findings, and a
deterministic referee separates them. Seed 1 failed on two independent grounds: its claim
pointed the wrong way, and it quantified over a split it had not exclusively run.

That mixture is the honest answer to "can a team of agents do research on QM8": *it can
generate findings, it cannot be trusted to grade them, and the grading is separable and
cheap.* Which is the entire argument for building the referee first.

### 9. The referee's own bug, found by a real agent

An earlier version of `critic.py` **signed seed 1's backwards claim.**

Checking that a paired bootstrap CI excludes zero establishes that a difference is *real*.
It does not establish that the difference runs the way the claim says. `config_b` is by
construction the config a claim asserts is better, so the paired difference must be
positive — and nothing was testing that.

Check 5, `direction`, now does. The mock agent could never have surfaced this: it only ever
over-claims, never inverts. It took a real model making a real mistake.

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
