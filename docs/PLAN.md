# QM8 Research Harness — build plan

> **The spine.** No number reaches the write-up except through a scorer that **neither the agent nor the author** can query adaptively.

## Context

Take-home for an agentic-AI-researcher role at Deep Origin. Five deliverables: research write-up, code, agent harness code, run traces, and a statement of what was done by hand.

Two findings from exploration reshaped this plan:

1. **The QM8 analysis was never written as files.** Confirmed by the user. Every figure in both design documents — Δ at 0.066 eV, the 0.271/0.370 parse verification, the 16.4% swap rate, the 0.30-vs-0.81 eV gap separation, the whole label-budget curve — has no reproducible artifact behind it. The rev 2 footer, *"all numbers measured on the locked split,"* cannot ship. So the build is not "bolt an agent onto finished analysis"; it is **build the environment, regenerate the findings inside it, then let the agent work there.**
2. **The ARMLLM Day 4 harness is public and directly reusable** — `github.com/osoblanco/ArmLLM` → `2026/agents` (verified public, pushed 2026-08-07). It ships `llm.py`, `tools.py`, `verify.py`, `evaluate.py`, `agent.py`, `world/build.py`, `bench.py`, `serve.sh`, `bootstrap.sh`. Adapting the harness the author was taught is less work *and* better provenance than writing a loop from scratch.

## Build order: referee before player

This inverts rev 1 and rev 2, which both had `agent.py` first. The course states the rule directly, and it is the deck's thesis on slides 1 and 80: **"Build the referee before the player."**

> Slide 40: *"The agent is a checkpoint. The referee outlives it. You will replace the model three times this year. You will keep the eval."*
> Slide 51: *"The environment is the artifact. The model is a checkpoint that expires."*
> Slide 41: *"Three ways to make an agent better: train the model / engineer the harness / redesign the loop. All three need the same thing: an environment that can score a rollout."*

Order: **world → referee → evaluator → agent.**

## Repo layout

Mirrors `2026/agents/` so the provenance is legible and the adapted files diff cleanly against their originals.

```
qm8_research/
  world/build.py      regenerate the whole environment from a seed, byte for byte
  data/               raw 2015 release, downloaded + hashed (not DeepChem — see below)
  splits.py           3-way split, SHA-256 per partition
  features.py         Morgan r=2 1024-bit + named RDKit descriptors
  models.py           cheap / direct / delta
  tools.py            TOOL_SCHEMAS + TOOLS — the action space
  llm.py              adapted from ArmLLM (OpenAI-compatible + Usage accounting)
  agent.py            adapted from ArmLLM (the 172-line loop)
  critic.py           the referee — verify.py's role for chemistry claims
  evaluate.py         adapted from ArmLLM: accuracy · CI · pass^k · cost · failures
  hypotheses.jsonl    the research questions, each with a pre-registered verifier
  preregister.json    hashed before any audit run
  results/
```

### The Day 4 → QM8 mapping

This is what makes the submission legible as *an adaptation of a known harness* rather than an unfalsifiable one-off.

| Day 4 | QM8 equivalent |
|---|---|
| Dvin corpus (178 docs) | QM8 environment: data + splits + fitting pipeline |
| `search` / `read_doc` / `calc` | `run_experiment` / `slice_error` / `run_control` / `intervene` |
| `finish` | `claim` — intercepted before dispatch, so it doubles as the stopping rule |
| `verify.py`, 5 verifier types | `critic.py` — sealed-test numeric verification + `abstain` |
| `tasks.jsonl`, 40 graded questions | `hypotheses.jsonl`, the pre-registered research questions |
| `world/build.py` — reset from a seed | QM8 environment regeneration — **this is what fixes finding #1** |
| failure histogram, 5 labels | QM8 failure labels (below) |

### Failure histogram — the QM8 labels

Slide 36: *"'62%' tells you nothing. A histogram of failures tells you what to fix."*

| Label | Meaning |
|---|---|
| `unsupported_claim` | claimed a result with no matching log entry |
| `noise_as_signal` | claimed an improvement lying inside the CI |
| `overscoped` | claim quantifier exceeds the configs actually run |
| `no_control` | mechanistic claim with no control experiment |
| `gave_up` | budget exhausted without a claim |
| `abstained_on_findable` | declined where a real effect was present |

The last is the direct analogue of `abstained_on_answerable`, and the deck's punchline is the warning: the reference agent scored **87.5% overall and 0% on all 15 unanswerable questions**, with slide 73 predicting *"the obvious fix will crater everything else."*

## Component decisions

**Environment (`world/build.py`).** Slide 46 defines an environment as five things — state, actions, transition, reward, reset — and slide 34: *"No reset, no evaluation."* Rev 2 had **no reset**; that is a real gap this closes. Transition is *"deterministic, code and a database, not a model."* Parse the raw 2015 release, not DeepChem (which returns 21,747, not 21,786). The parse self-check is the 2015 paper's own numbers: PBE0/def2SVP vs CC2 → E1 0.271, E2 0.370 eV.

**Splits.** Carve validation out of train; the existing test partition stays sealed. 17,429 / 2,178 / 2,179, SHA-256 each, hash recorded in every log line.

**Referee (`critic.py`).** Deterministic, not an LLM judge. Checks: split hash · evidence ids resolve · re-run at `speed="full"` on the sealed set · effect exceeds the bootstrap CI · comparison count within the claim family · quantifier vs configs run · control present for mechanistic claims. Verdicts: **signed / narrowed / rejected**.

**Evaluator (`evaluate.py`).** Reuse the bootstrap directly — resample with replacement 5,000×, take the 2.5th/97.5th percentiles; the deck calls it twelve lines and *"there is no excuse for a bare number."* Add pass^k, token cost, and the failure histogram.

**Agent (`agent.py`).** Adapt the 172-line loop. Keep verbatim: errors returned as observations with an `ERROR:` prefix (never raised), `finish`/`claim` intercepted by name before dispatch, four recorded exit reasons (`finish` / `no_tool_call` / `max_steps` / `never`), observation truncation at 4,000 chars for the model and 600 for the log.

**Model.** Qwen3.6-35B-A3B-FP8 on the L40S via vLLM 0.26.0. `--tool-call-parser` is the flag that silently breaks everything: *"Pick the wrong parser and vLLM starts, answers fluently, and never emits a tool call. Nothing errors."*

## Evidence table

Four honest categories. **Verified** = confirmed against the paper itself (none yet — see Verification). **Course canon** = printed on a slide with a citation, upstream unverified. **Reasoning** = argued, not cited. **Conjecture** = untested.

| Decision | Basis | Kind |
|---|---|---|
| Referee before player; environment is the artifact | Day 4 slides 40, 41, 51, 80 | course canon |
| Scorer outside the model's reach | Search-Time Contamination arXiv:2606.05241 — agents found benchmark labels for ~3% of questions on HuggingFace, up to 4% inflation; CaMeL arXiv:2503.18813 — policy enforced outside the model; slide 69 *"something outside the model that can say no"* | course canon |
| Deterministic critic, not LLM judge | `verify.py`'s five types with *"judge — last resort, now you have two things to validate"*; DeepSeek-R1 arXiv:2501.12948 — *"They tried neural RMs. Reward hacking broke the runs"*; BabelJudge arXiv:2606.22329 — trajectory judges show **argument blindness** | course canon |
| Few agents; add a role only on a measurement | Slide 58, **verbatim**: *"You may add structure only when a measurement shows the simpler design cannot do the job."* MAST arXiv:2503.13657 — 41.8% specification, 36.9% misalignment over 1,600+ traces | course canon |
| Observation design is a first-class result | Slide 24: *"None of that is the model. All of it decided whether you saw 20,000 or 26,000."* Slide 13: errors are observations, not exceptions | course canon |
| Bootstrap CI on every number | Slides 37–38: ±15pp at 40 items; 5,000 resamples, 2.5/97.5 percentiles | course canon |
| pass^k and cost reported together | τ-bench arXiv:2406.12045 — GPT-4o τ-retail pass¹ 61% → pass⁸ 25%; AstaBench arXiv:2510.21652 | course canon |
| No Literature Agent / no web search | Search-Time Contamination, as above | course canon |
| Abstention as a graded outcome | `verify.py`'s `abstain` type — *"said it could not tell you, **and** invented no figure"*; reference agent 0% on 15/15 unanswerable | course canon |
| LLM decides, GBM predicts | PoT arXiv:2211.12588 / PAL arXiv:2211.10435 — *"Decide what. Outsource math."* Agent World Model arXiv:2602.10090 — environments *"deliberately not LLM-simulated, so transitions stay consistent"* | course canon (architecture only) |
| Three-way sealed split | Follows from the leak argument — the agent conditions on a metric every call | reasoning |
| Pre-registration against author-side selection | Standard experimental design. **Nothing in any deck covers this** | reasoning |
| Paired seeds + multiplicity correction | Standard statistics. **Nothing in any deck covers either** | reasoning |
| Exploratory vs confirmatory typing | **Nothing in any deck covers this** as a named distinction | reasoning |
| Random-search and human-grid baselines | Equal-budget comparison is well covered (HAL, AstaBench, Snell arXiv:2408.03314). **These two specific baselines are not** | reasoning |
| TDDFT-gap split | The CC2 gap requires paying the cost being avoided; the TDDFT gap is available at screening time | reasoning |
| Gradient boosting over a GNN | Throughput. Published QM8 leaders are GNNs | judgement |
| f-metric artifact hypothesis | Right-skewed non-negative target; not yet measured | conjecture |

## Where this design runs ahead of the evidence — stated, not hidden

Six gaps, in the order they matter. Each is labelled **reasoning** above and will be labelled that way in the write-up.

1. **Pre-registration** — zero coverage. Slide 29's approval of *"people auditing their own benchmarks"* is arguably in mild tension with it.
2. **Exploratory vs confirmatory** — zero coverage as a named distinction. Slide 21 models the behaviour (*"it shows prefill is cheap; it does not by itself prove the agent-loop case"*) without citing anything.
3. **Multiple-comparison correction** — zero coverage. The slide-38 bootstrap is unpaired and single-configuration.
4. **Random-search / human-grid baselines** — the equal-budget *frame* is well evidenced; these two baselines are mine.
5. **OOD-based abstention** — the decks cover abstention as an *outcome* (the answer is absent from the evidence), never as *distributional novelty*. No conformal prediction, no applicability domain, no risk–coverage curve anywhere. My uncertainty-estimator axis is unsupported here.
6. **LLM-vs-GBM for property prediction** — architectural support only; nothing compares them empirically.

### Two places I was over-claiming, now corrected

- **Extended thinking off by default.** Supported for *tool-routing steps inside an agent loop* — HAL's 21,730 rollouts, plus Day 4's own 18.0 s → 3.9 s → 1.5 s and 333 → 40 tokens. Contradicted for hard single-shot reasoning (o1 13→75%, R1 15.6→71.0, s1 50→57 by *forcing more*). **Scope it per step type, not globally.**
- **Deferring code execution.** The decks say *"sandbox — always"* (TTS 151), not defer, and the counter-evidence is heavier than the support: Code as Agent Harness (197 papers), ReTool 40.0→67.0, ToRL 29.3→43.3. Slide 66 also concedes that closing the exfiltration path leaves the dominant harm untouched: *"Corrupting the answer needs one leg."* **So the deferral is a scope decision, not a safety one, and will be written that way.**

### Two measurement traps the decks caught

- **pass^k is uninformative at temperature 0.** Slide 39: *"there is nothing to vary, so the two are equal and you have learned nothing"* — confirmed on slide 72 where pass³ = accuracy = 87.5%. The three-arm comparison must run at nonzero temperature.
- **Paired seeds are weaker than they look for the LLM arm.** Slide 39: *"a batched server is not fully deterministic even at 0: batch composition changes the order of floating-point reductions."* Pairing still works for the GBM fits, which are genuinely seed-deterministic. Scope the pairing claim to the model layer.

## Verification

1. **Environment reset** — run `world/build.py` twice from the same seed; the two trees must be byte-identical. This is the check that finding #1 is actually fixed.
2. **Parse self-check** — raw PBE0/def2SVP vs CC2 must reproduce 0.27 / 0.37 eV against Ramakrishnan et al. 2015.
3. **Split integrity** — mutate a split file; `load_split()` must refuse on hash mismatch.
4. **Leak test** — assert no sealed-test number appears in any observation the agent receives, across a full run's trace.
5. **Pre-registration enforcement** — attempt an audit with no matching `preregister.json` entry; the critic must refuse.
6. **Mock backend** — the whole loop runs with no GPU and no network (the course ships this mode).
7. **Citation verification** — before the write-up, check each arXiv id above against the paper itself. Two known discrepancies to resolve: the rev 1 doc claims HAL says *"in 21 of 36 runs higher reasoning effort did not improve accuracy"* while slide 29 says *"lowered accuracy in the majority of 21,730 rollouts"* — these are different claims and at most one is right; and the rev 1 doc's numbers (47.9% specification-gaming, 1 of 15 ResearchGym) come from papers not present in any deck.
