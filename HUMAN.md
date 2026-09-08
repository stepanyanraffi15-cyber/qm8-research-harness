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
- Harness adapted from ArmLLM 2026 Day 4 (`osoblanco/ArmLLM`, `2026/agents`). Reused
  wholesale: the agent loop's shape, `llm.py`'s client and token accounting, the bootstrap
  CI recipe, the failure-histogram discipline. Written here: everything chemistry, the
  three-way sealed split, pre-registration, and the claim verifiers.

### 2026-09-07/08 · build

Order was **world → referee → evaluator → agent**, inverting both earlier design
documents, which put the agent first. Everything below was found by running code, not
by planning.

**Written by hand (human + LLM pair):** every file in the repo. **Run by the agent:**
nothing yet — the agent arm has only been exercised against the mock backend.

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
5. **Nobody predicted this one.** On the TDDFT-gap split, Δ **beats** cheap for f1
   (0.0206 vs 0.0216) — the only split where it does. That split isolates near-degenerate
   excited states, which is exactly where state-ordering ambiguity should live.
6. **The gap mechanism is visible in the energies.** `slice_error` on E1/Δ by TDDFT state
   gap gives 0.095 eV on the most degenerate quartile against 0.042 eV on the most
   separated — a 2.3× spread.
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

That ruled out the setup the harness was originally written against (Qwen3.6-35B-A3B-FP8
on an L40S). Nothing on this machine could serve a model — no API keys in the environment,
and port 8000 turned out to be an unrelated Docker container, not vLLM. So Ollama was
installed and Qwen3 8B pulled: the closest local analogue, on an Apple Silicon laptop with
16 GB.

Consequence to state in the write-up rather than bury: the agent arm runs an **8B** model,
not the 35B/3B-active one the design reasoned about. Any null result from the agent arm is
therefore confounded with model capacity, and cannot be read as "LLM agents cannot do
this."

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
