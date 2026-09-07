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
