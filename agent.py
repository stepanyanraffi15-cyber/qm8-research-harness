"""The loop: policy + tools + observation + memory + stopping rule.

An agent is five things: a policy (the model), tools (what it can do), an
observation (what came back), memory (the message list), and a stopping rule
(what ends it). The decomposition follows Anthropic's *Building Effective Agents*
(Dec 2024); everything beyond those five is framework.

Three properties are deliberate, and each is a lesson the literature records:

  * errors are returned as observations with an `ERROR:` prefix, never raised. A
    traceback ends the run; an error string the model can read lets it recover,
    and recovery is what separates an agent from a script.
  * the terminal tool is intercepted by name BEFORE dispatch, so it doubles as
    the stopping rule. Here that tool is `claim` rather than `finish`.
  * four recorded exit reasons. Only the first is success, and the evaluator
    reads this field -- a run that ended because the model answered in prose is a
    different failure from one that exhausted its budget.
  * observations truncated at 4000 chars for the model, 600 for the log.

Changed for this project: the terminal tool submits a research claim rather than
an answer, and the trajectory carries the experiment ids so the referee can tie a
claim to the runs behind it.

What this agent CANNOT do, by construction: reach the sealed test set. Every
number it sees comes from validation. models.run refuses `eval_on="sealed_test"`
without an audit token that nothing importable from here holds.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tools
from llm import LLM, Usage

SYSTEM_PROMPT = """You are a research agent working on QM8, a dataset of 21,786 small
organic molecules with excited-state properties computed at four levels of quantum theory.

One level, RI-CC2/def2TZVP, is accurate and expensive. Three are cheap TDDFT approximations
and are already computed for every molecule. The question is how well the expensive answer
can be predicted without paying for it, and where that prediction breaks.

How to work:
- Look at the dataset first. Then form a hypothesis and design an experiment that could
  show it is WRONG. An experiment that can only confirm you is not evidence.
- A difference is only real if the paired bootstrap in `vs_best` excludes zero. Seed
  variance is 0 here because the fit is deterministic -- do not read it as a noise floor.
- If you claim a mechanism, run a control. A result you cannot distinguish from an
  artifact is not a result, and the referee will reject it.
- `slice_error` tells you WHERE a method fails. That is usually more informative than
  another point estimate of how much it fails on average.

How the run ends:
- Call `claim` with one finding. Nothing else ends the task.
- A referee then re-scores your claim on a sealed test set you have never seen, applies a
  multiplicity correction, and checks that your statement does not quantify over
  configurations you never ran. Claim narrowly and cite every experiment id you used.
- Claiming something broad and unsupported scores worse than claiming something narrow
  and true. If your experiments do not support a finding, say so in the claim."""

MAX_OBSERVATION_CHARS = 4000
LOG_OBSERVATION_CHARS = 600


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    observation: str
    tokens_after: int


@dataclass
class Result:
    question: str
    claim: dict | None = None
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_because: str = "claim"

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "claim": self.claim,
            "stopped_because": self.stopped_because,
            "n_steps": len(self.steps),
            "usage": self.usage.as_dict(),
            "experiment_ids": [
                eid for s in self.steps
                if (eid := _extract_id(s.observation)) is not None
            ],
            "trajectory": [
                {"step": s.n, "tool": s.tool, "args": s.args,
                 "observation": s.observation[:LOG_OBSERVATION_CHARS],
                 "tokens_after": s.tokens_after}
                for s in self.steps
            ],
        }


def _extract_id(observation: str) -> str | None:
    try:
        return json.loads(observation).get("experiment_id")
    except Exception:
        return None


class Agent:
    def __init__(self, llm: LLM | None = None, max_steps: int = 20,
                 budget: int = 30, system_prompt: str = SYSTEM_PROMPT,
                 log_path: Path | None = None, run_prefix: str | None = None) -> None:
        self.llm = llm or LLM()
        self.max_steps = max_steps
        self.budget = budget
        self.system_prompt = system_prompt
        self.log_path = log_path
        # Caller-supplied so the evaluator can name the trace file after the same
        # run the experiment ids carry. Left None, the Session mints its own.
        self.run_prefix = run_prefix

    def run(self, question: str) -> Result:
        session = tools.reset_session(
            budget=self.budget,
            log_path=self.log_path or tools.LOG_PATH,
            run_prefix=self.run_prefix,
        )
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        result = Result(question=question, usage=self.llm.usage)

        for n in range(1, self.max_steps + 1):
            msg = self.llm.chat(messages, tools=tools.TOOL_SCHEMAS)
            messages.append(msg)

            calls = msg.get("tool_calls")
            if not calls:
                # Answered in prose instead of calling a tool. Accept it, but record
                # that the stopping rule did not fire -- this is a failure mode the
                # evaluator classifies, not an error.
                result.stopped_because = "no_tool_call"
                return result

            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "claim":
                    observation = tools.claim(**args) if _claimable(args) else (
                        "ERROR: claim needs at least `statement`, `kind` and `evidence`."
                    )
                    if session.claim is not None:
                        result.claim = session.claim
                        result.steps.append(
                            Step(n, name, args, observation, self.llm.usage.total_tokens)
                        )
                        return result

                else:
                    fn = tools.TOOLS.get(name)
                    if fn is None:
                        observation = (
                            f"ERROR: no tool named {name!r}. "
                            f"Available: {sorted(tools.TOOLS) + ['claim']}."
                        )
                    else:
                        try:
                            observation = fn(**args)
                        except TypeError as exc:
                            observation = f"ERROR: bad arguments for {name}: {exc}"
                        except Exception as exc:  # noqa: BLE001
                            observation = f"ERROR: {name} failed: {exc}"

                if not isinstance(observation, str):
                    observation = json.dumps(observation, ensure_ascii=False)
                if len(observation) > MAX_OBSERVATION_CHARS:
                    observation = observation[:MAX_OBSERVATION_CHARS] + "\n...[truncated]"

                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "name": name, "content": observation})
                result.steps.append(
                    Step(n, name, args, observation, self.llm.usage.total_tokens)
                )

        result.stopped_because = "max_steps"
        return result


def _claimable(args: dict) -> bool:
    return bool(args.get("statement")) and bool(args.get("evidence"))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="run one research agent rollout")
    ap.add_argument("question", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None, help="write the trajectory here")
    args = ap.parse_args()

    question = " ".join(args.question) or (
        "How few expensive CC2 labels are needed to predict excited-state properties, "
        "and where does the prediction break?"
    )

    agent = Agent(llm=LLM(temperature=args.temperature),
                  max_steps=args.max_steps, budget=args.budget)
    res = agent.run(question)

    for s in res.steps:
        preview = s.observation.replace("\n", " ")[:100]
        print(f"  [{s.n}] {s.tool}({json.dumps(s.args)[:60]}) -> {preview}")
    print()
    print(f"stopped : {res.stopped_because}")
    print(f"claim   : {(res.claim or {}).get('statement', '-')}")
    print(f"tokens  : {res.usage.total_tokens} in {res.usage.calls} model calls")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res.as_dict(), indent=2) + "\n")
        print(f"wrote   : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
