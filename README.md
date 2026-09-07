# QM8 Research Harness

An LLM agent that conducts experiments on QM8, and a deterministic referee that decides
whether any of its claims survive.

> **The invariant the whole system exists to enforce**
>
> No number reaches the write-up except through a scorer that **neither the agent nor the
> author** can query adaptively.

Both halves of that sentence are load-bearing. The agent never observes the sealed test
set, so it cannot optimise against the thing it is judged on. The author pre-registers a
primary outcome before the audit runs, so selection cannot happen through a human either.

---

## The five deliverables

| Asked for | Where it is | State |
|---|---|---|
| Research write-up | `writeup/` | not started |
| Code | `world/`, `splits.py`, `features.py`, `models.py` | in progress |
| Agent harness code | `agent.py`, `tools.py`, `critic.py`, `llm.py`, `evaluate.py` | in progress |
| Run traces | `results/` — full trajectories, not summaries | not started |
| What was done by hand | `HUMAN.md` | running log, written as it happens |

---

## Build order: referee before player

The environment and the referee are built first, and the agent last. This is the order the
ArmLLM 2026 Day 4 session argues for, and the reason is that nothing the agent produces
means anything until something exists that can score a rollout:

> *"The agent is a checkpoint. The referee outlives it. You will replace the model three
> times this year. You will keep the eval."* — Day 4, slide 40
>
> *"The environment is the artifact. The model is a checkpoint that expires."* — slide 51

So: **world → referee → evaluator → agent.**

---

## Provenance

The harness is adapted from the ArmLLM 2026 Day 4 exercise
([`osoblanco/ArmLLM`](https://github.com/osoblanco/ArmLLM), `2026/agents`), taught by
Aram Markosyan. Files carrying an `Adapted from ArmLLM 2026 Day 4` header keep the
original's structure so the diff is legible; the chemistry, the sealed-split protocol,
the pre-registration layer and the claim verifiers are new.

What is reused and what is not is recorded per-file, and in `HUMAN.md`.

The scientific question follows Ramakrishnan et al., *Electronic spectra from TDDFT and
machine learning in chemical space*, J. Chem. Phys. **143**, 084111 (2015) — the paper
that built QM8 for Δ-learning, and reported the oscillator-strength negative result this
project is pushing against.

---

## Layout

```
world/build.py      regenerate the environment from a seed, byte for byte
splits.py           three-way split, SHA-256 per partition
features.py         Morgan r=2 1024-bit + named RDKit descriptors
models.py           cheap / direct / delta
tools.py            TOOL_SCHEMAS + TOOLS — the agent's action space
llm.py              OpenAI-compatible client with token accounting
agent.py            the loop: policy + tools + observation + memory + stopping rule
critic.py           the referee — deterministic, never an LLM judge
evaluate.py         accuracy · CI · pass^k · cost · failure histogram
hypotheses.jsonl    the research questions, each with a verifier
preregister.json    hashed before any audit run
results/            run traces
```

## Running it

```sh
uv run python world/build.py --seed 0     # regenerate the environment
uv run python evaluate.py --arm agent     # score a policy
```

Nothing here needs a GPU except the LLM server; the prediction layer is gradient boosting
on fingerprints and fits in seconds on a laptop.
