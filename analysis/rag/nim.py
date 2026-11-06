"""Answer generation through NVIDIA NIM (build.nvidia.com), for comparing a hosted Nemotron model with the
local Qwen generators in analysis/run_rag_case_study.py.

Only the Python standard library is used. The API key is read from NVIDIA_API_KEY and is never written to the
cache, logs or published artifacts. Generation is greedy (temperature 0), like the local generators.

    export NVIDIA_API_KEY=nvapi-...
    python -m analysis.rag.nim --list            # show the Nemotron models your key can use
    export NIM_MODEL=<one of the listed ids>
    python analysis/run_rag_case_study.py        # adds "nemotron-nim" to the compared generators
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
RETRY_STATUS = {429, 500, 502, 503, 504}


def nim_spec(env=os.environ) -> dict | None:
    """Generator spec for run_rag_case_study.GENERATORS, or None when no key/model is configured."""
    if not env.get("NVIDIA_API_KEY") or not env.get("NIM_MODEL"):
        return None
    return {"name": env["NIM_MODEL"], "label": f"{env['NIM_MODEL']} (NVIDIA NIM, hosted)", "provider": "nim"}


def _request(path: str, payload: dict | None, api_key: str, timeout: float) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, method="POST" if data else "GET",
                                 headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class NimGenerator:
    """Same interface as run_rag_case_study.Generator: generate(batch_of_messages, max_new_tokens) -> list[str]."""

    def __init__(self, model: str, cache_path: Path, api_key: str | None = None, retries: int = 5,
                 backoff_s: float = 2.0, timeout_s: float = 120.0, log=print, sleep=time.sleep):
        self.name = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        self.retries, self.backoff_s, self.timeout_s = retries, backoff_s, timeout_s
        self.log, self.sleep = log, sleep

    def close(self) -> None:
        pass

    def _key(self, messages, max_new_tokens):
        return hashlib.sha1(json.dumps([self.name, messages, max_new_tokens]).encode()).hexdigest()

    def _complete(self, messages: list[dict], max_new_tokens: int) -> str:
        payload = {"model": self.name, "messages": messages, "max_tokens": max_new_tokens,
                   "temperature": 0, "top_p": 1, "stream": False}
        for attempt in range(self.retries + 1):
            try:
                out = _request("/chat/completions", payload, self.api_key, self.timeout_s)
                return (out["choices"][0]["message"].get("content") or "").strip()
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt == self.retries:
                    raise RuntimeError(f"NIM request failed with HTTP {exc.code}") from None
                wait = self.backoff_s * 2 ** attempt
                self.log(f"  NIM HTTP {exc.code}; retrying in {wait:.0f}s")
                self.sleep(wait)
        raise AssertionError("unreachable")

    def generate(self, batch: list[list[dict]], max_new_tokens: int = 200, batch_size: int = 8) -> list[str]:
        outputs = []
        for i, messages in enumerate(batch):
            key = self._key(messages, max_new_tokens)
            if key not in self.cache:
                self.cache[key] = self._complete(messages, max_new_tokens)
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
                if (i + 1) % batch_size == 0:
                    self.log(f"  generated {i + 1}/{len(batch)}")
            outputs.append(self.cache[key])
        return outputs


def list_models(api_key: str | None = None, contains: str = "nemotron") -> list[str]:
    out = _request("/models", None, api_key or os.environ["NVIDIA_API_KEY"], 60.0)
    return sorted(m["id"] for m in out.get("data", []) if contains.lower() in m["id"].lower())


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(list_models()))
    else:
        print(__doc__)
