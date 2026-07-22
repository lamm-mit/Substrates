# Copyright 2026.  Apache-2.0.
"""Provider-agnostic LLM interface for the analysis step.

Two backends, chosen by the caller:

    provider="anthropic"  ->  Anthropic Messages API, default model claude-opus-4-8
    provider="openai"     ->  OpenAI Responses API,  default model gpt-5.5

Both accept the same *neutral* content format — a list of blocks that are either
``{"type": "text", "text": ...}`` or ``{"type": "image_png", "data": <b64>}`` —
and each adapter translates to that provider's vision message shape.  This lets
``analyze.py`` send the figure PNGs plus numeric readouts to whichever model the
user picks on the CLI, with one code path.

Auth: standard per-provider env vars — ``ANTHROPIC_API_KEY`` (or an
``ant auth login`` profile) for Anthropic, ``OPENAI_API_KEY`` for OpenAI.
"""

from __future__ import annotations

DEFAULT_MODEL = {"anthropic": "claude-opus-4-8", "openai": "gpt-5.5"}


def default_model(provider: str) -> str:
    if provider not in DEFAULT_MODEL:
        raise ValueError(f"unknown provider {provider!r}; use 'anthropic' or 'openai'")
    return DEFAULT_MODEL[provider]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

def _anthropic_complete(model, system, blocks, max_tokens, effort):
    import anthropic

    client = anthropic.Anthropic()
    content = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image_png":
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b["data"]}})
    # Stream (SDK guidance for long output); adaptive thinking + effort for quality.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    return "".join(p.text for p in msg.content if p.type == "text").strip()


# --------------------------------------------------------------------------- #
# OpenAI  (GPT-5.5) — Responses API
# --------------------------------------------------------------------------- #

def _openai_complete(model, system, blocks, max_tokens, effort, base_url=None):
    """Uses the OpenAI **Responses API** (`client.responses.create`), the
    current interface for GPT-5-family models.

    ``base_url`` points the OpenAI-compatible client at a non-OpenAI endpoint —
    a local server such as mistral.rs, LM Studio, vLLM, or Ollama — instead of
    ``api.openai.com``. Left as ``None``, the SDK uses the default endpoint
    (honoring ``OPENAI_BASE_URL`` if the caller set it in the environment).

    Shape differences from the old chat-completions path this replaces:
    - the system prompt is the top-level ``instructions`` field (not a
      "system" message);
    - user content parts are ``input_text`` / ``input_image`` (an
      ``input_image`` carries the data-URI directly as ``image_url``, a plain
      string — unlike chat-completions' ``{"url": ...}`` object);
    - the output cap is ``max_output_tokens``;
    - reasoning depth is ``reasoning={"effort": ...}`` (minimal/low/medium/high);
    - the aggregated answer is ``response.output_text``.
    """
    import openai
    import os

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
        # Local OpenAI-compatible servers (mistral.rs, LM Studio, vLLM, Ollama)
        # usually need no real key, but the SDK requires one to construct —
        # supply a harmless placeholder when the environment sets none.
        if not os.environ.get("OPENAI_API_KEY"):
            client_kwargs["api_key"] = "not-needed"
    client = openai.OpenAI(**client_kwargs)
    content = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "input_text", "text": b["text"]})
        elif b["type"] == "image_png":
            content.append({"type": "input_image",
                            "image_url": f"data:image/png;base64,{b['data']}"})

    kwargs = {
        "model": model,
        "instructions": system,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": effort},
    }
    try:
        resp = client.responses.create(**kwargs)
    except (openai.BadRequestError, TypeError):
        # A non-reasoning model rejects `reasoning`; drop it and retry.
        kwargs.pop("reasoning", None)
        resp = client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


# --------------------------------------------------------------------------- #

def complete(provider: str, model: str, system: str, blocks: list[dict], *,
             max_tokens: int = 4000, effort: str = "high",
             base_url: str | None = None) -> str:
    """Run one analysis turn on the chosen provider and return the text.

    ``base_url`` applies only to the ``openai`` provider (a local /
    OpenAI-compatible endpoint); it is ignored for ``anthropic``.
    """
    if provider == "anthropic":
        return _anthropic_complete(model, system, blocks, max_tokens, effort)
    if provider == "openai":
        return _openai_complete(model, system, blocks, max_tokens, effort,
                                base_url=base_url)
    raise ValueError(f"unknown provider {provider!r}; use 'anthropic' or 'openai'")
