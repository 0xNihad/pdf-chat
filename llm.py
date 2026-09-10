from __future__ import annotations

import os

from litellm import completion

import config
GATEWAYS = {"opencode": ("https://opencode.ai/zen/go/v1", "OPENCODE_API_KEY")}


def chat(system: str, user: str, model: str | None = None) -> str:
    model = model or config.LLM_MODEL
    prefix, _, rest = model.partition("/")
    extra = {}
    if prefix in GATEWAYS:
        api_base, key_var = GATEWAYS[prefix]
        key = os.getenv(key_var)
        if not key:
            raise SystemExit(f"error: LLM_MODEL={model} needs {key_var} in .env")
        model, extra = f"openai/{rest}", {"api_base": api_base, "api_key": key}

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **extra,
    )
    return response.choices[0].message.content
