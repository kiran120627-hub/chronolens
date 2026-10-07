"""Dependency-free LLM client: Anthropic or any OpenAI-compatible API, with image support and a disk cache.

Configure via environment or `.env` in the project root:
    LLM_PROVIDER = anthropic | gemini | groq | openrouter | ollama | openai
    LLM_API_KEY  = ...
    LLM_MODEL    = optional override
    LLM_BASE_URL = optional (OpenAI-compatible providers)
    LLM_CACHE    = 1 (default) — identical prompts replay from .cache/llm (fast, reproducible demos)

Message content may be a string or a list of parts:
    {"type": "text", "text": "..."}   |   {"type": "image", "b64": "<base64 jpeg>", "mime": "image/jpeg"}
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1-mini"}
PRESETS = {
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    "groq": ("openai", "https://api.groq.com/openai/v1", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5"),
    "ollama": ("openai", "http://localhost:11434/v1", "qwen2.5vl:7b"),
}


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and not os.environ.get(k):  # .env fills in missing/empty values (re-read on every call)
                os.environ[k] = v


class LLMError(RuntimeError):
    pass


@dataclass
class LLM:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    cache: bool = True
    cache_dir: Path = ROOT / ".cache" / "llm"
    max_tokens: int = 6000
    fallbacks: list = field(default_factory=list)  # tried in order when the main model is overloaded/unavailable
    exhausted: set = field(default_factory=set)  # models whose daily quota is used up (skipped for this process)
    calls: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "LLM":
        load_env()
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if provider in PRESETS:
            p, url, m = PRESETS[provider]
            provider, base_url, model = p, base_url or url, model or m
        key = os.environ.get("LLM_API_KEY") or (os.environ.get("ANTHROPIC_API_KEY", "") if provider == "anthropic"
                                                else os.environ.get("OPENAI_API_KEY", ""))
        fb = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        if not fb and "generativelanguage" in (base_url or ""):
            # free tier = 20 requests/day *per model*, so spread load over several capable flash models
            fb = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest",
                  "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        return cls(provider=provider, model=model or DEFAULT_MODELS.get(provider, ""), api_key=key,
                   base_url=base_url or ("https://api.openai.com/v1" if provider == "openai" else ""),
                   cache=os.environ.get("LLM_CACHE", "1") != "0", fallbacks=fb)

    @property
    def label(self) -> str:
        host = re.sub(r"^https?://", "", self.base_url).split("/")[0] if self.base_url else self.provider
        return f"{self.model} via {host}"

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base_url

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0, tag: str = "") -> str:
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, messages, temperature, tag],
                                        sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if self.cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # always replay a saved answer (instant, works offline); answers are still cross-checked downstream
            with self._lock:
                self.cache_hits += 1
            return data["text"]
        if not self.configured:
            raise LLMError("No LLM API key configured. Set LLM_API_KEY in .env (see .env.example).")
        primary, last_err = self.model, None
        text, used = None, primary
        for m in [primary] + [f for f in self.fallbacks if f != primary]:
            if m in self.exhausted:
                continue
            self.model = m
            try:
                text = self._with_retries(system, messages, temperature)
                used = m
                break
            except LLMError as e:
                last_err = e
                if "PerDay" in str(e) or "per day" in str(e).lower():
                    self.exhausted.add(m)
                if not any(code in str(e) for code in ("HTTP 503", "HTTP 429", "HTTP 404", "HTTP 500", "network")):
                    break
            finally:
                self.model = primary
        if text is None:
            raise last_err or LLMError("LLM call failed")
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": used, "text": text}), encoding="utf-8")
        return text

    # ------------------------------------------------------------------ transport
    def _with_retries(self, system, messages, temperature) -> str:
        last: Exception | None = None
        for attempt in range(4):
            try:
                with self._lock:
                    self.calls += 1
                fn = self._anthropic if self.provider == "anthropic" else self._openai
                return fn(system, messages, temperature)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                per_day = " [PerDay quota]" if "PerDay" in body else ""
                last = LLMError(f"HTTP {e.code} from {self.label}{per_day}: {body[:400]}")
                if e.code == 400 and "temperature" in body:
                    temperature = None
                    continue
                if e.code in (429, 500, 502, 503, 504, 529):
                    if self.fallbacks and (attempt >= 1 or "PerDay" in body):
                        raise last from e  # let complete() switch to a fallback model quickly
                    time.sleep(2 ** attempt * 2)
                    continue
                raise last from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"network error talking to {self.label}: {e}")
                time.sleep(2 ** attempt)
        raise last or LLMError("LLM call failed")

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"content-type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _parts(content, provider: str):
        if isinstance(content, str):
            return content
        out = []
        for p in content:
            if p["type"] == "text":
                out.append({"type": "text", "text": p["text"]})
            elif provider == "anthropic":
                out.append({"type": "image", "source": {"type": "base64", "media_type": p.get("mime", "image/jpeg"),
                                                        "data": p["b64"]}})
            else:
                out.append({"type": "image_url",
                            "image_url": {"url": f"data:{p.get('mime', 'image/jpeg')};base64,{p['b64']}"}})
        return out

    def _anthropic(self, system, messages, temperature) -> str:
        msgs = [{"role": m["role"], "content": self._parts(m["content"], "anthropic")} for m in messages]
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": msgs}
        if temperature is not None:
            payload["temperature"] = temperature
        url = (self.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        data = self._post(url, payload, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _openai(self, system, messages, temperature) -> str:
        msgs = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": self._parts(m["content"], "openai")} for m in messages]
        payload = {"model": self.model, "messages": msgs, "max_tokens": self.max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(self.base_url.rstrip("/") + "/chat/completions", payload, headers)
        return data["choices"][0]["message"].get("content") or ""


def extract_json(text: str):
    """First JSON object/array in the text (prefers ```json blocks)."""
    decoder = json.JSONDecoder()
    for chunk in re.findall(r"```json\s*(.*?)```", text, re.S) + [text]:
        for opener in ("{", "["):
            pos = chunk.find(opener)
            while pos != -1:
                try:
                    obj, _ = decoder.raw_decode(chunk[pos:])
                    return obj
                except json.JSONDecodeError:
                    pos = chunk.find(opener, pos + 1)
    raise ValueError("no JSON found in model response")


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.S)
    if blocks:
        with_fn = [b for b in blocks if "def answer" in b]
        return (with_fn or blocks)[-1].strip() + "\n"
    raise ValueError("no python code block found in model response")
