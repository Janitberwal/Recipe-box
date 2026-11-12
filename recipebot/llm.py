"""Thin Gemini wrapper. The LLM is used ONLY to (a) parse a user turn into a structured
action and (b) optionally re-phrase an adapted recipe. It never produces counts, IDs,
ingredient lists or evidence - those come from the deterministic tools.

Robustness: per-request timeout, low "thinking" for speed, retries on rate limits, and an
automatic fallback model when the primary is overloaded / retired / times out."""
from __future__ import annotations

import json
import time

from .config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL, GEMINI_THINKING

_client = None
_no_thinking: set[str] = set()     # models that rejected the thinking setting
last_model_used = ""


class LLMError(RuntimeError):
    pass


def available() -> bool:
    return bool(GEMINI_API_KEY)


def client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set. Put it in the .env file (see README).")
        from google import genai
        from google.genai import types
        _client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=60_000))
    return _client


def _config(model: str, **kw):
    from google.genai import types
    if GEMINI_THINKING and model not in _no_thinking:
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=GEMINI_THINKING)
    return types.GenerateContentConfig(**kw)


def _call(contents: str, **cfg) -> str:
    """Try the primary model, then fallbacks. Returns response text."""
    global last_model_used
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    errors = []
    for model in models:
        for attempt in range(3):
            try:
                resp = client().models.generate_content(model=model, contents=contents, config=_config(model, **cfg))
                last_model_used = model
                return resp.text or ""
            except Exception as e:
                msg = str(e)
                if "thinking" in msg.lower() and "INVALID_ARGUMENT" in msg and model not in _no_thinking:
                    _no_thinking.add(model)      # model does not support this thinking level -> retry without
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    errors.append(f"{model}: rate limited")
                    if attempt < 2:
                        time.sleep(6 * (attempt + 1))   # free tier resets per minute: 6s, 12s
                        continue
                    break
                if any(c in msg for c in ("503", "UNAVAILABLE", "504", "DEADLINE", "404", "NOT_FOUND",
                                          "timed out", "Timeout", "500", "INTERNAL")):
                    errors.append(f"{model}: {msg[:90]}")
                    break                        # go to next model
                if "API key" in msg or "PERMISSION_DENIED" in msg or "401" in msg or "403" in msg:
                    raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY in .env.") from e
                errors.append(f"{model}: {msg[:120]}")
                break
    err = LLMError("All Gemini models failed: " + " | ".join(errors))
    err.rate_limited = any("rate limited" in e or "503" in e or "UNAVAILABLE" in e for e in errors)
    raise err


def generate_json(system: str, prompt: str, schema: dict | None = None, temperature: float = 0.0) -> dict:
    cfg = dict(system_instruction=system, temperature=temperature, response_mime_type="application/json")
    if schema:
        cfg["response_json_schema"] = schema
    for _ in range(2):
        text = _call(prompt, **cfg)
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError:
            continue
    raise LLMError("The model did not return valid JSON.")


def generate_text(system: str, prompt: str, temperature: float = 0.2) -> str:
    return _call(prompt, system_instruction=system, temperature=temperature)
