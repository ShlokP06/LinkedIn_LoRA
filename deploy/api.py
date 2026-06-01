"""
FastAPI orchestration backend for FLUX LoRA Demo.

Responsibilities:
  - Cleans user prompts via Groq (llama-3.3-70b) before sending to Flux
  - Fires Groq clean + Modal container warm-up IN PARALLEL on /generate
  - Proxies /loras to Modal (acts as the page-load warm-up trigger)

Local dev:
    uvicorn deploy.api:fastapi_app --reload --port 8000

Deploy to Modal (CPU only, no GPU):
    modal deploy deploy/api.py

Env vars (.env file, loaded automatically):
    GROQ_API_KEY       = ...
    modal_generate_url = https://shlokp06--flux-lora-inference-...-generate-api.modal.run
    modal_loras_url    = https://shlokp06--flux-lora-inference-...-loras-api.modal.run
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import modal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

log = logging.getLogger("flux-api")

from deploy.prompts import _system_prompt


class GenerateRequest(BaseModel):
    prompt: str
    lora_step: Optional[int] = None
    size: int = 768
    num_steps: int = 28
    guidance_scale: float = 4.0
    seed: int = 42

class CleanRequest(BaseModel):
    prompt: str

class CleanResponse(BaseModel):
    cleaned: str
    original: str


def _make_groq_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    return Groq(api_key=key)


async def _groq_clean(raw: str, client) -> str:
    """Clean a casual user prompt into a structured Flux portrait prompt."""
    t0 = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": raw},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        cleaned = (resp.choices[0].message.content or "").strip()
        log.info("[timing] Groq clean: %.2fs", time.perf_counter() - t0)
        # Only hard requirement: LoRA trigger word "Shl0k" must be present
        if "shl0k" in cleaned.lower():
            return cleaned
        log.warning("Groq output missing LoRA trigger 'Shl0k', injecting trigger word")
        return f"LinkedIn Profile picture of Shl0k, clean-shaven, wearing glasses, {cleaned}"
    except Exception as exc:
        log.info("[timing] Groq clean: %.2fs (failed)", time.perf_counter() - t0)
        log.warning("Groq call failed (%s), falling back to raw prompt", exc)
        return raw  # graceful degradation — generation still runs


def _generate_url() -> str:
    url = os.environ.get("modal_generate_url", "")
    if not url:
        raise RuntimeError("modal_generate_url not set in environment")
    return url


def _loras_url() -> str:
    url = os.environ.get("modal_loras_url", "")
    if not url:
        raise RuntimeError("modal_loras_url not set in environment")
    return url


async def _modal_warm(http: httpx.AsyncClient) -> list[int]:
    """Hit /loras-api to wake the inference container. Returns available steps.

    L40S cold-start takes 20-60 s — timeout must exceed that or the ping is useless.
    """
    t0 = time.perf_counter()
    try:
        resp = await http.get(_loras_url(), timeout=90)
        resp.raise_for_status()
        log.info("[timing] Modal warm-up: %.2fs", time.perf_counter() - t0)
        return resp.json().get("steps", [])
    except Exception as exc:
        log.info("[timing] Modal warm-up: %.2fs (failed)", time.perf_counter() - t0)
        log.warning("Modal warm-up ping failed: %s", exc)
        return []


async def _modal_generate(http: httpx.AsyncClient, cleaned: str, req: GenerateRequest) -> bytes:
    """Call Modal /generate-api and return raw PNG bytes."""
    params: dict = {
        "prompt": cleaned,
        "size": (req.size // 8) * 8,  # FLUX requires multiples of 8
        "num_steps": req.num_steps,
        "guidance_scale": req.guidance_scale,
        "seed": req.seed,
    }
    if req.lora_step is not None:
        params["lora_step"] = req.lora_step

    t0 = time.perf_counter()
    resp = await http.post(_generate_url(), params=params, timeout=300)
    log.info("[timing] Modal generate: %.2fs", time.perf_counter() - t0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Modal error {resp.status_code}: {resp.text[:200]}")
    return resp.content


_limiter = Limiter(key_func=get_remote_address)


def build_fastapi_app() -> FastAPI:
    groq_client = _make_groq_client()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _app.state.http = httpx.AsyncClient()
        yield
        await _app.state.http.aclose()

    api = FastAPI(title="FLUX LoRA API", version="1.0", lifespan=lifespan)
    api.state.limiter = _limiter
    api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-cleaned-prompt"],  # lets browser JS read this header
    )

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    @api.get("/loras")
    async def loras(request: Request):
        """Proxy to Modal /loras-api. Call this on page load to warm the container."""
        steps = await _modal_warm(request.app.state.http)
        return {"steps": steps}

    @api.post("/clean", response_model=CleanResponse)
    @_limiter.limit("2/minute")
    async def clean(req: CleanRequest, request: Request):
        """Run just the Groq prompt cleaning step. ~0.5-1 s."""
        cleaned = await _groq_clean(req.prompt, groq_client)
        return CleanResponse(cleaned=cleaned, original=req.prompt)

    @api.post("/generate")
    @_limiter.limit("2/minute")
    async def generate(req: GenerateRequest, request: Request):
        """
        Full pipeline:
          1. Fire Groq (clean prompt) + Modal warm-up IN PARALLEL
          2. Use cleaned prompt to call Modal generate
          3. Return PNG bytes with x-cleaned-prompt header

        Frontend can read the header to show the cleaned prompt without an extra call.
        """
        http = request.app.state.http
        t_total = time.perf_counter()

        # Groq cleans the prompt + Modal container starts warming in parallel
        cleaned, _ = await asyncio.gather(
            _groq_clean(req.prompt, groq_client),
            _modal_warm(http),
        )

        # StreamingResponse sends headers immediately so the client gets the cleaned
        # prompt right away; the async generator streams the PNG body when ready.
        async def image_stream():
            img_bytes = await _modal_generate(http, cleaned, req)
            log.info("[timing] Total /generate: %.2fs", time.perf_counter() - t_total)
            yield img_bytes

        return StreamingResponse(
            image_stream(),
            media_type="image/png",
            headers={"x-cleaned-prompt": cleaned},
        )

    return api


# uvicorn deploy.api:fastapi_app --reload --port 8000
fastapi_app = build_fastapi_app()


_app = modal.App("flux-lora-api")

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "httpx",
        "groq",
        "python-dotenv",
        "slowapi",
    )
)


@_app.function(
    image=_image,
    secrets=[modal.Secret.from_dotenv()],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web() -> object:
    return build_fastapi_app()
