"""Actor agent: takes an observation + task context, queries an LLM, returns a
BrowserGym-compatible action string.

The prompt template is loaded from ``prompts/inference/agent.txt`` (configured
via ``benchmark.actor.prompt_path``). Per-mode obs descriptions are filled
into the template's ``{obs_description}`` placeholder by the runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.llm_client import LLMClient


# --- Action parsing ------------------------------------------------------------


# BrowserGym HighLevelActionSet emits Python-like function calls, e.g.:
#   click('45')
#   fill('51', 'hello world')
#   scroll(0, 400)
#   send_msg_to_user('done')
# We extract the FIRST matching function call from the LLM output (strict=False
# behaviour at the mapper level; BrowserGym will re-validate on execute).
_ACTION_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z_0-9]*)\s*\(", re.MULTILINE)


# Explicit `provider/model` → tokenizer spec for client-side prompt truncation.
# No silent fallback: if a model isn't listed here, the truncator raises
# ValueError so the experimenter chooses deliberately (add an entry, or
# disable truncation with max_prompt_tokens=null in the config).
#
# Spec format: ``"<kind>:<name>"``.
#   * ``tiktoken:<encoding>`` — load via ``tiktoken.get_encoding(<encoding>)``.
#                               OpenAI families: gpt-4o / gpt-5 / o-series share
#                               o200k_base; legacy gpt-4 / gpt-3.5 use cl100k_base.
#   * ``hf:<repo_id>``        — load via ``transformers.AutoTokenizer.from_pretrained``.
#                               Tokenizer-only download (no model weights).
#                               Used for proprietary tokenizers (DeepSeek BBPE, etc.).
_TOKENIZER_BY_MODEL: Dict[str, str] = {
    # -- OpenAI (o200k_base family) --
    "openai/gpt-5.1":             "tiktoken:o200k_base",
    "openai/gpt-5-mini":          "tiktoken:o200k_base",
    "openai/gpt-5-nano":          "tiktoken:o200k_base",
    "openai/gpt-4o":              "tiktoken:o200k_base",
    "openai/gpt-4o-mini":         "tiktoken:o200k_base",
    "openai/o1":                  "tiktoken:o200k_base",
    "openai/o3":                  "tiktoken:o200k_base",
    "openai/o3-mini":             "tiktoken:o200k_base",
    # -- OpenAI legacy (cl100k_base) --
    "openai/gpt-4":               "tiktoken:cl100k_base",
    "openai/gpt-4-turbo":         "tiktoken:cl100k_base",
    "openai/gpt-3.5-turbo":       "tiktoken:cl100k_base",
    # -- DeepSeek (proprietary BBPE shared across V3.x and V3.x-reasoner endpoints) --
    "deepseek/deepseek-chat":     "hf:deepseek-ai/DeepSeek-V3.2",
    "deepseek/deepseek-reasoner": "hf:deepseek-ai/DeepSeek-V3.2",
    # -- Gemini (no public Gemini tokenizer; Gemma-family SentencePiece is the
    #    closest open approximation. Token counts will be off by a few percent
    #    versus the API but well within our truncation safety margin.) --
    "gemini/gemini-3.1-flash-lite-preview": "hf:google/gemma-3-27b-it",
}


def extract_action_string(raw_output: str) -> Optional[str]:
    """Pull the first BrowserGym-style function call out of raw LLM text.

    Strategy: find the earliest `name(` token, then walk the string balancing
    parentheses and string literals. Returns the exact `name(...)` substring,
    or None if no valid call found.
    """
    m = _ACTION_PATTERN.search(raw_output)
    if not m:
        return None
    start = m.start()
    i = m.end() - 1  # at the `(`
    depth = 0
    in_str: Optional[str] = None
    escape = False
    while i < len(raw_output):
        ch = raw_output[i]
        if escape:
            escape = False
        elif in_str is not None:
            if ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return raw_output[start:i + 1]
        i += 1
    return None


def split_think_action(raw_output: str) -> Tuple[str, Optional[str]]:
    """Split LLM output into (think_text, action_string).

    Convention: the action is the last recognised BrowserGym-style call.
    Everything before it is treated as chain-of-thought. When a match is
    accepted, the scan advances past its closing paren so identifier-like
    tokens inside string literals (e.g. inside `fill('x', 'a (b)')`) are
    not mis-selected as a later action.
    """
    last_action: Optional[str] = None
    last_start = -1
    pos = 0
    while pos < len(raw_output):
        m = _ACTION_PATTERN.search(raw_output, pos)
        if not m:
            break
        candidate = extract_action_string(raw_output[m.start():])
        if candidate:
            last_action = candidate
            last_start = m.start()
            pos = m.start() + len(candidate)
        else:
            pos = m.end()
    if last_action is None:
        return raw_output.strip(), None
    think = raw_output[:last_start].strip()
    return think, last_action


# --- Actor ---------------------------------------------------------------------


@dataclass
class ActorCallRecord:
    """One LLM call made by the actor — retained for trajectory logging."""
    prompt: str
    raw_output: str
    think: str
    action: Optional[str]
    parse_ok: bool
    usage: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 0


class Actor:
    """LLM-backed actor. Builds a prompt, calls the LLM, parses an action."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str,
        action_space_description: str,
        max_retries: int = 2,
        max_prompt_tokens: Optional[int] = None,
    ) -> None:
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.action_space_description = action_space_description
        self.max_retries = max_retries
        self.max_prompt_tokens = max_prompt_tokens

    def _truncate_prompt(self, prompt: str) -> str:
        """Bottom-truncate the assembled prompt to fit ``max_prompt_tokens``.

        Dispatches by provider:
          * ``vllm``   — uses the server's ``/tokenize`` + ``/detokenize``
                         endpoints so counts match the serving tokenizer.
          * API path   — looks up an **explicit** ``provider/model`` entry in
                         ``_TOKENIZER_BY_MODEL`` and dispatches to either
                         ``tiktoken.get_encoding`` or
                         ``transformers.AutoTokenizer.from_pretrained`` (HF,
                         tokenizer-only download). Unmapped models raise
                         ``ValueError``; no silent fallback. Add an entry to
                         the map or set ``benchmark.max_prompt_tokens=null``
                         to disable truncation.
        """
        if not self.max_prompt_tokens:
            return prompt
        provider = getattr(self.llm, "provider", None)
        model = getattr(self.llm, "model", None)
        if provider == "vllm":
            return self._truncate_via_vllm(prompt, model)
        return self._truncate_via_api_tokenizer(prompt, provider, model)

    def _truncate_via_vllm(self, prompt: str, model: Optional[str]) -> str:
        base_url = getattr(self.llm, "base_url", None)
        if not base_url or not model:
            return prompt
        import json
        from urllib.request import Request, urlopen

        def _post(path: str, payload: dict) -> dict:
            req = Request(
                f"{base_url.rstrip('/')}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            api_key = getattr(self.llm, "api_key", "") or "EMPTY"
            req.add_header("Authorization", f"Bearer {api_key}")
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())

        try:
            tok = _post("/tokenize", {"model": model, "prompt": prompt})
        except Exception:
            # vLLM unreachable / endpoint absent — skip truncation and let
            # the upstream chat call surface the real error.
            return prompt
        ids = tok.get("tokens", [])
        if len(ids) <= self.max_prompt_tokens:
            return prompt
        try:
            detok = _post(
                "/detokenize",
                {"model": model, "tokens": ids[:self.max_prompt_tokens]},
            )
        except Exception:
            return prompt
        return detok.get("prompt", prompt)

    def _truncate_via_api_tokenizer(
        self, prompt: str, provider: Optional[str], model: Optional[str],
    ) -> str:
        key = f"{provider}/{model}"
        if key not in _TOKENIZER_BY_MODEL:
            raise ValueError(
                f"No tokenizer mapping for {key!r}. "
                f"Add it to `_TOKENIZER_BY_MODEL` in benchmarks/actor.py "
                f"(`tiktoken:<encoding>` or `hf:<repo_id>`), or set "
                f"`benchmark.max_prompt_tokens=null` in the config to skip "
                f"truncation for this run."
            )
        spec = _TOKENIZER_BY_MODEL[key]
        kind, _, name = spec.partition(":")
        if kind == "tiktoken":
            import tiktoken
            enc = tiktoken.get_encoding(name)
            ids = enc.encode(prompt)
            if len(ids) <= self.max_prompt_tokens:
                return prompt
            return enc.decode(ids[:self.max_prompt_tokens])
        if kind == "hf":
            # Use the `tokenizers` (Rust-backed) library directly rather than
            # `transformers.AutoTokenizer`, which in 4.57+ tries to enumerate
            # `additional_chat_templates/` and crashes with a 404 when that
            # directory is absent (most models). We only need encode/decode,
            # so the lower-level Tokenizer is sufficient and faster. It pulls
            # only `tokenizer.json` from the HF hub — no model weights.
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(name)
            ids = tok.encode(prompt, add_special_tokens=False).ids
            if len(ids) <= self.max_prompt_tokens:
                return prompt
            return tok.decode(ids[:self.max_prompt_tokens], skip_special_tokens=True)
        raise ValueError(
            f"Unknown tokenizer kind {kind!r} for {key!r} (expected "
            f"`tiktoken:<encoding>` or `hf:<repo_id>`)."
        )

    def build_prompt(
        self,
        task: str,
        observation: str,
        action_history: List[str] = None,
    ) -> str:
        """Assemble the actor prompt from the template.

        Placeholders supported:
          {task_instruction}, {observation}, {action_space}, {action_history}
        """
        if action_history:
            action_history_text = "\n".join(
                f"{i + 1}. {a}" for i, a in enumerate(action_history)
            )
        else:
            action_history_text = "(none)"

        fill = {
            "task_instruction": task,
            "observation": observation,
            "action_space": self.action_space_description,
            "action_history": action_history_text,
        }
        prompt = self.prompt_template
        for k, v in fill.items():
            prompt = prompt.replace("{" + k + "}", v or "")
        return prompt

    def act(
        self,
        task: str,
        observation: str,
        action_history: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], str, List[ActorCallRecord]]:
        """Produce one action for the given context.

        Returns:
            (action_str_or_None, think_text, llm_call_records)
            — action_str_or_None is the BrowserGym-compatible action call
              (None if all retries failed to parse)
            — think_text is the chain-of-thought from the last attempt
            — llm_call_records logs every LLM attempt (for trajectory.jsonl)
        """
        prompt = self.build_prompt(task, observation, action_history)
        prompt = self._truncate_prompt(prompt)
        records: List[ActorCallRecord] = []
        for attempt in range(self.max_retries + 1):
            raw_output, usage = self.llm.generate(prompt)
            think, action = split_think_action(raw_output)
            records.append(ActorCallRecord(
                prompt=prompt, raw_output=raw_output,
                think=think, action=action,
                parse_ok=action is not None,
                usage=usage, attempt=attempt,
            ))
            if action is not None:
                return action, think, records
            # retry: append a correction hint
            prompt = (
                prompt
                + "\n\nYour previous response did not contain a parsable action."
                + " Output exactly one action call from the action space, "
                + "e.g. `click('45')`."
            )
        return None, records[-1].think, records
