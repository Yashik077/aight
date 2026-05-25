"""Aight LLM client - assumes OpenAI-compatible local endpoint.
e.g. http://localhost:11434/v1 (ollama) or http://localhost:8000/v1
Set AIGHT_BASE_URL / AIGHT_MODEL / AIGHT_API_KEY env vars.
"""
import os
import json
import urllib.request

BASE_URL = os.getenv("AIGHT_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("AIGHT_MODEL", "aight-7b")
API_KEY = os.getenv("AIGHT_API_KEY", "not-needed")


def build_prompt(query, contexts, max_chars=3000):
    ctx_text = ""
    for i, (payload, score) in enumerate(contexts):
        snippet = payload["text"][:800]
        ctx_text += f"\n[{i+1}] ({payload['source']} score={score:.2f})\n{snippet}\n"
    ctx_text = ctx_text[:max_chars]
    return f"""You are Aight, a helpful local assistant. Answer ONLY from context.
If answer not in context, say you don't know.

Context:{ctx_text}

Question: {query}
Answer with citations like [1], [2].
"""


def generate(prompt, temperature=0.2, max_tokens=512):
    url = BASE_URL.rstrip("/") + "/chat/completions"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Aight unreachable at {BASE_URL}: {e}] Fallback extract: {prompt[:400]}"
