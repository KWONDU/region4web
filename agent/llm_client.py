"""Unified LLM client.

Model string format: "{provider}/{model_name}"
  - openai/o3                           -> OpenAI API (api.openai.com)
  - openai/gpt-5.1                      -> OpenAI API
  - anthropic/claude-sonnet-4-5-20250929 -> Anthropic API (api.anthropic.com)
  - deepseek/deepseek-chat              -> DeepSeek API (api.deepseek.com, OpenAI-compatible)
  - gemini/gemini-3.1-flash             -> Gemini API (OpenAI-compatible layer)
  - Qwen/Qwen3.5-27B                    -> local vLLM server (HuggingFace-style name → vllm)
  - google/gemma-4-31B-it               -> local vLLM (HF org `google/` has no collision here
                                          because Gemini uses the `gemini/` prefix)

Secrets file (secrets.yaml) reads keys under these slots:
    openai_api_key, anthropic_api_key, deepseek_api_key, gemini_api_key

All provider-specific params (temperature, reasoning_effort, top_p, top_k, ...)
are forwarded as kwargs. vLLM-only extensions (top_k, min_p, repetition_penalty)
are sent via ``extra_body`` so the OpenAI SDK doesn't reject them.

Usage:
    client = create_llm_client("openai/o3", params={"reasoning_effort": "medium"})
    client = create_llm_client("deepseek/deepseek-chat")
    client = create_llm_client("Qwen/Qwen3.5-27B", base_url="http://127.0.0.1:8270")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

_API_PROVIDERS: Dict[str, str] = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    # OpenAI-compatible chat completions endpoints — routed through the same
    # _call_openai_compatible path with only base_url + api_key differing.
    "deepseek": "https://api.deepseek.com",
    # NB: use "gemini/" not "google/" because HuggingFace org `google/` hosts
    # e.g. google/gemma-*, google/t5-*, google/flan-*; colliding prefixes
    # would route those HF repos to the Gemini API instead of local vLLM.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

# API key env var + secrets.yaml lookup key per provider.
_API_KEY_LOOKUP: Dict[str, tuple[str, str]] = {
    "openai":    ("OPENAI_API_KEY",    "openai_api_key"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    "deepseek":  ("DEEPSEEK_API_KEY",  "deepseek_api_key"),
    "gemini":    ("GEMINI_API_KEY",    "gemini_api_key"),
}

_VLLM_DEFAULT_URL = "http://localhost:8000"

# Params the OpenAI SDK accepts as first-class kwargs on chat.completions.create.
# Anything else (e.g. vLLM extensions like top_k, min_p, repetition_penalty) is
# forwarded via `extra_body` so vLLM sees it while the OpenAI SDK does not reject it.
_OPENAI_STANDARD_PARAMS = {
    "temperature", "top_p", "max_tokens", "max_completion_tokens",
    "presence_penalty", "frequency_penalty", "stop", "n", "seed",
    "logprobs", "top_logprobs", "response_format", "tools", "tool_choice",
    "logit_bias", "user", "stream", "stream_options",
}

_secrets_cache: Optional[Dict[str, str]] = None


def _load_secrets() -> Dict[str, str]:
    """Load API keys from secrets.yaml (cached)."""
    global _secrets_cache
    if _secrets_cache is not None:
        return _secrets_cache

    import yaml
    from pathlib import Path

    # Try config.yaml for secrets_file path, fallback to secrets.yaml
    secrets_path = "secrets.yaml"
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            secrets_path = cfg.get("secrets_file", secrets_path)
        except Exception:
            pass

    try:
        with open(secrets_path) as f:
            _secrets_cache = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _secrets_cache = {}

    return _secrets_cache


def _parse_model_string(model_string: str) -> tuple[str, str, str]:
    """Parse 'provider/model_name' into (provider, model_name, base_url).

    Returns:
        (provider, model_name, base_url)
        - openai/gpt-5.4-mini -> ("openai", "gpt-5.4-mini", "https://api.openai.com")
        - anthropic/claude-... -> ("anthropic", "claude-...", "https://api.anthropic.com")
        - meta-llama/Llama-3-70B -> ("vllm", "meta-llama/Llama-3-70B", "http://localhost:8000")
    """
    parts = model_string.split("/", 1)
    if len(parts) == 2 and parts[0] in _API_PROVIDERS:
        provider = parts[0]
        model_name = parts[1]
        base_url = _API_PROVIDERS[provider]
    else:
        # HuggingFace model -> local vLLM
        provider = "vllm"
        model_name = model_string
        base_url = _VLLM_DEFAULT_URL
    return provider, model_name, base_url


@dataclass
class LLMClient:
    """Client for OpenAI-compatible or Anthropic API."""

    provider: str = "vllm"
    model: str = "gpt-4o"
    base_url: str = _VLLM_DEFAULT_URL
    api_key: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300

    def __post_init__(self) -> None:
        if not self.api_key and self.provider in _API_KEY_LOOKUP:
            env_var, secrets_key = _API_KEY_LOOKUP[self.provider]
            self.api_key = os.environ.get(env_var, "") or _load_secrets().get(secrets_key, "")

    def generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """Send a prompt. Returns (text, usage_dict)."""
        return self.generate_with_messages([{"role": "user", "content": prompt}])

    def generate_with_messages(
        self, messages: List[Dict[str, str]], **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Chat-style generation. Returns (text, usage_dict)."""
        if self.provider == "anthropic":
            return self._call_anthropic(messages, **kwargs)
        else:
            return self._call_openai_compatible(messages, **kwargs)

    def _call_openai_compatible(
        self, messages: List[Dict[str, str]], **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Call OpenAI-compatible API. Returns (text, usage).

        Works for openai, deepseek, google (all expose the same schema), and
        local vLLM. Provider differences are only in ``base_url`` + ``api_key``.
        """
        try:
            from openai import OpenAI
            client_kwargs: Dict[str, Any] = {}
            if self.provider == "vllm":
                client_kwargs["base_url"] = f"{self.base_url.rstrip('/')}/v1"
                # vLLM accepts any non-empty key; OpenAI SDK refuses to init without one.
                client_kwargs["api_key"] = self.api_key or "EMPTY"
            elif self.provider == "openai":
                # Default OpenAI base_url. SDK handles path.
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
            elif self.provider in ("deepseek", "gemini"):
                # Both expose /v1 chat completions at these base URLs.
                client_kwargs["base_url"] = f"{self.base_url.rstrip('/')}/v1"
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
            elif self.api_key:
                client_kwargs["api_key"] = self.api_key
            all_params = {**self.params, **kwargs}
            standard = {k: v for k, v in all_params.items() if k in _OPENAI_STANDARD_PARAMS}
            extras = {k: v for k, v in all_params.items() if k not in _OPENAI_STANDARD_PARAMS}
            create_kwargs: Dict[str, Any] = {
                "model": self.model, "messages": messages, **standard,
            }
            if extras:
                create_kwargs["extra_body"] = extras
            response = OpenAI(timeout=self.timeout, **client_kwargs).chat.completions.create(**create_kwargs)
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                # Reasoning tokens (OpenAI o-series, gpt-5.* etc.)
                if getattr(response.usage, "completion_tokens_details", None):
                    details = response.usage.completion_tokens_details
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt:
                        usage["reasoning_tokens"] = rt
                # Prompt caching bookkeeping (OpenAI: prompt_tokens_details.cached_tokens;
                # DeepSeek: top-level prompt_cache_hit_tokens / prompt_cache_miss_tokens;
                # Google OpenAI-compatible layer: not exposed consistently as of now).
                ptd = getattr(response.usage, "prompt_tokens_details", None)
                if ptd and getattr(ptd, "cached_tokens", None):
                    usage["cached_tokens"] = ptd.cached_tokens
                for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    v = getattr(response.usage, k, None)
                    if v is not None:
                        usage[k] = v
            return text, usage
        except ImportError:
            return self._call_via_http(messages, **kwargs)

    def _call_anthropic(
        self, messages: List[Dict[str, str]], **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Call Anthropic Messages API. Returns (text, usage)."""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            call_params = {**self.params, **kwargs}
            max_tokens = call_params.pop("max_tokens", 512)
            response = client.messages.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, **call_params,
            )
            text = response.content[0].text
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            return text, usage
        except ImportError:
            raise ImportError("anthropic SDK required for provider='anthropic'. pip install anthropic")

    def _call_via_http(
        self, messages: List[Dict[str, str]], **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Fallback: raw HTTP to OpenAI-compatible endpoint. Returns (text, usage)."""
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            **self.params, **kwargs,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(url, data=data, headers=headers)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
        except URLError as e:
            raise ConnectionError(f"API connection failed ({self.base_url}): {e}") from e
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return text, usage


def create_llm_client(
    model_string: str,
    api_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client from a model string.

    Args:
        model_string: "{provider}/{model}" format.
            - "openai/gpt-5.4-mini" -> OpenAI cloud API
            - "anthropic/claude-sonnet-4-20250514" -> Anthropic cloud API
            - "meta-llama/Llama-3-70B-Instruct" -> local vLLM server
        api_key: API key. Falls back to env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY).
        params: Free kwargs passed to API call (temperature, max_tokens, effort, etc.).
        base_url: Override the endpoint URL (only meaningful for vLLM provider;
            ignored for openai/anthropic which always hit the official cloud).
    """
    provider, model_name, default_base_url = _parse_model_string(model_string)
    if base_url and provider == "vllm":
        default_base_url = base_url
    return LLMClient(
        provider=provider,
        model=model_name,
        base_url=default_base_url,
        api_key=api_key or "",
        params=params or {},
    )
