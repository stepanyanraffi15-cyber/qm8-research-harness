"""Model endpoint, token accounting, and nothing else.

Adapted from ArmLLM 2026 Day 4 `llm.py` (github.com/osoblanco/ArmLLM, 2026/agents).
The client, the retry-and-adapt logic and the Usage accounting are theirs and are
kept close to the original so the diff stays legible. What is new here: the
profiles, and a QM8 mock backend.

Where the model runs is a config change, never a code change:

    QM8_PROFILE=vllm        http://localhost:8000/v1     (the L40S)
    QM8_PROFILE=openrouter  https://openrouter.ai/api/v1
    QM8_PROFILE=mock        no network at all

    uv run python llm.py check

Token accounting lives here because cost is a first-class result. An agent's
score is a property of model x scaffold x harness x budget (HAL, arXiv:2510.11977),
so a number reported without its cost is half a result.

`enable_thinking` is off by default, and the scope of that decision matters.
It is supported for tool-ROUTING steps inside an agent loop -- the Day 4 measurement
is 333 completion tokens per call falling to 40, and 18.0s per tool call falling
to 3.9s. It is NOT supported for hard single-shot reasoning, where the evidence
runs the other way entirely (o1 13->75% on AIME, R1 15.6->71.0). This harness only
ever asks the model to pick a next experiment, which is the routing case.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROFILES = {
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3.6-35B-A3B-FP8"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.6-35b-a3b"},
    "mock": {"base_url": "", "model": "mock"},
}


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader -- no dependency, no surprises. Existing vars win."""
    env_path = path or (HERE / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return default


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "retries": self.retries,
        }


class LLMError(RuntimeError):
    pass


class LLM:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        profile: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
    ) -> None:
        self.profile = profile or _env("QM8_PROFILE", "ARMLLM_PROFILE", default="mock")
        if self.profile not in PROFILES:
            raise LLMError(f"unknown profile {self.profile!r}; expected one of {list(PROFILES)}")
        defaults = PROFILES[self.profile]

        self.model = model or _env("QM8_MODEL", "ARMLLM_MODEL") or defaults["model"]
        self.base_url = base_url or _env("QM8_BASE_URL", "ARMLLM_BASE_URL") or defaults["base_url"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.usage = Usage()
        self.mock = self.profile == "mock"
        self.thinking = _env("QM8_THINKING", "ARMLLM_THINKING", default="0") == "1"

        # Some endpoints reject temperature != 1 or require max_completion_tokens.
        # Discover that once from the error text, then stop asking.
        self._send_temperature = True
        self._token_param = "max_tokens"

        if self.mock:
            self._client = None
            return

        key = api_key or _env("QM8_API_KEY", "ARMLLM_API_KEY", "OPENROUTER_API_KEY",
                              "OPENAI_API_KEY", default="EMPTY" if self.profile == "vllm" else None)
        if not key:
            raise LLMError(
                f"profile {self.profile!r} needs an API key. Put QM8_API_KEY in "
                f"{HERE / '.env'} (gitignored), or export it. Use QM8_PROFILE=mock to run offline."
            )
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=180.0, max_retries=0)

    def describe(self) -> str:
        return (f"profile={self.profile} model={self.model} "
                f"base_url={self.base_url or '-'} thinking={self.thinking} temp={self.temperature}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the assistant message dict. Records usage as a side effect."""
        if self.mock:
            return _mock_chat(messages, self.usage)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._once(messages, tools)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                text = str(exc)
                if self._adapt(text):
                    continue
                if attempt == self.max_retries or not _is_transient(text):
                    break
                self.usage.retries += 1
                time.sleep(min(2**attempt, 8) + random.random())
        raise LLMError(f"{self.describe()} failed after retries: {last_error}") from last_error

    def _once(self, messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if self._send_temperature:
            kwargs["temperature"] = self.temperature
        kwargs[self._token_param] = self.max_tokens
        if not self.thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        resp = self._client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None,
        }

    def _adapt(self, error_text: str) -> bool:
        low = error_text.lower()
        if "max_completion_tokens" in low and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in low and self._send_temperature and (
            "unsupported" in low or "does not support" in low or "only the default" in low
        ):
            self._send_temperature = False
            return True
        return False


def _is_transient(error_text: str) -> bool:
    low = error_text.lower()
    return any(m in low for m in (
        "rate limit", "429", "timeout", "timed out", "502", "503", "504",
        "overloaded", "connection", "temporarily",
    ))


# --------------------------------------------------------------------------
# Mock backend -- deterministic, offline, and deliberately naive.
#
# It does what an unengineered research agent does: run the obvious experiment,
# see a big number, and claim it as broadly as possible. It never runs a control
# and it quantifies over configurations it did not test.
#
# That is the point. The referee should REJECT or NARROW this agent, on a laptop,
# with no GPU and no network -- which makes the referee testable before any model
# is involved. Build the referee before the player.
# --------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }


def _mock_chat(messages: list[dict], usage: Usage) -> dict:
    prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
    results = [m for m in messages if m.get("role") == "tool"]
    n = len(results)

    if n == 0:
        out = _call("call_1", "describe_dataset", {})
    elif n == 1:
        out = _call("call_2", "run_experiment", {"target": "E1", "method": "delta"})
    elif n == 2:
        out = _call("call_3", "run_experiment", {"target": "E1", "method": "direct"})
    else:
        ids = re.findall(r'"experiment_id":\s*"([^"]+)"', " ".join(
            str(m.get("content") or "") for m in results))
        a, b = (ids + ["e_0000", "e_0001"])[:2]
        out = _call("call_4", "claim", {
            "statement": "Delta-learning beats direct learning for excited-state prediction",
            "kind": "comparison",
            # deliberately over-scoped: claims every target and every split from
            # two runs on E1/random. The referee narrows this.
            "scope": {"target": "all", "split": "random"},
            "evidence": [a, b],
            "config_a": {"target": "E1", "method": "direct", "cheap_level": "PBE0-SVP",
                         "split": "random", "seed": 0},
            "config_b": {"target": "E1", "method": "delta", "cheap_level": "PBE0-SVP",
                         "split": "random", "seed": 0},
        })

    usage.add(_approx_tokens(" " * prompt_chars), _approx_tokens(json.dumps(out)))
    return out


def check() -> int:
    """Verify the endpoint answers, and -- the part that actually breaks -- that it
    returns tool calls in the shape the loop expects.

    Day 4, slide 11: "Pick the wrong parser and vLLM starts, answers fluently, and
    never emits a tool call. Nothing errors." This is that check.
    """
    llm = LLM()
    print(f"config  : {llm.describe()}")

    t0 = time.time()
    plain = llm.chat([{"role": "user", "content": "Reply with exactly: ok"}])
    print(f"chat    : {(plain.get('content') or '').strip()[:40]!r}  ({time.time() - t0:.1f}s)")

    from tools import TOOL_SCHEMAS

    t0 = time.time()
    msg = llm.chat(
        [{"role": "system", "content": "Use the tools. Look at the dataset before experimenting."},
         {"role": "user", "content": "How accurately can CC2 excitation energies be predicted?"}],
        tools=TOOL_SCHEMAS,
    )
    calls = msg.get("tool_calls")
    if calls:
        fn = calls[0]["function"]
        print(f"tools   : {fn['name']}({fn['arguments'][:60]})  ({time.time() - t0:.1f}s)")
    else:
        print("tools   : NO TOOL CALL RETURNED -- the agent loop will not work with this model")
        print("          check --tool-call-parser on the server")
        return 1
    print(f"usage   : {llm.usage.as_dict()}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(check())
