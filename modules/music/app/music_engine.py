# -*- coding: utf-8 -*-
"""
MusicEngine -- ACE-Step 1.5 REST API wrapping class

ACE-Step API server (default localhost:8006) sends HTTP requests to generate music.
Flow: POST /release_task -> poll POST /query_result -> GET /v1/audio

Environment variables (optional):
  ACESTEP_STUB=1              -- dummy wav without ACE-Step (dev/test)
  ACESTEP_API_URL             -- ACE-Step API server URL (default http://localhost:8006)
  ACESTEP_CACHE_DIR           -- downloaded audio storage path
  ACESTEP_POLL_INTERVAL       -- polling interval seconds (default 2.0)
  ACESTEP_POLL_TIMEOUT        -- max wait seconds (default 600)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODULE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _MODULE_ROOT / "output" / "cache"


@dataclass
class TrackMeta:
    track_id: str = ""
    prompt: str = ""
    lyrics: Optional[str] = None
    genre: Optional[str] = None
    bpm: int = 120
    key_scale: str = ""
    duration_sec: float = 90.0
    file_path: Optional[Path] = None
    seed: int = -1
    generation_time_sec: float = 0.0
    requester: Optional[str] = None


@dataclass
class GenerationParams:
    prompt: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    key_scale: str = ""
    duration: float = 90.0
    inference_steps: int = 8
    guidance_scale: float = 7.0
    thinking: bool = True
    audio_format: str = "wav"
    batch_size: int = 1
    seed: int = -1


class MusicEngine:
    """ACE-Step 1.5 REST API music generation engine."""

    def __init__(
        self,
        model_name: str = "acestep-v15-turbo",
        cache_dir: Optional[Path] = None,
    ):
        self._model_name = model_name
        self._api_url = os.environ.get(
            "ACESTEP_API_URL", "http://localhost:8006"
        ).rstrip("/")
        self._cache_dir = Path(
            os.environ.get("ACESTEP_CACHE_DIR", str(cache_dir or _DEFAULT_CACHE))
        )
        self._poll_interval = float(os.environ.get("ACESTEP_POLL_INTERVAL", "2.0"))
        self._poll_timeout = float(os.environ.get("ACESTEP_POLL_TIMEOUT", "600"))

        self._stub_mode = os.environ.get("ACESTEP_STUB", "").lower() in (
            "1", "true", "yes",
        )
        self._is_ready = False
        self._session: Any = None  # aiohttp.ClientSession

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # -- lifecycle -------------------------------------------------

    async def initialize(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self._stub_mode:
            self._is_ready = True
            logger.warning("MusicEngine running in ACESTEP_STUB mode")
            return

        import aiohttp

        self._session = aiohttp.ClientSession()

        try:
            async with self._session.get(
                f"{self._api_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"ACE-Step API health check failed: HTTP {resp.status}"
                    )
                data = await resp.json()
                models_ok = data.get("models_initialized", False)
                llm_ok = data.get("llm_initialized", False)
                if not (models_ok and llm_ok):
                    logger.warning(
                        "ACE-Step API responded but models not fully loaded: %s", data
                    )
            self._is_ready = True
            logger.info(
                "MusicEngine ready (api=%s, cache=%s)", self._api_url, self._cache_dir
            )
        except Exception as exc:
            logger.error(
                "ACE-Step API connection failed (%s). "
                "Ensure the API server is running. "
                "Set ACESTEP_STUB=1 for testing without GPU.",
                exc,
            )
            await self._close_session()
            self._is_ready = False

    async def shutdown(self) -> None:
        await self._close_session()
        self._is_ready = False
        logger.info("MusicEngine shutdown complete")

    async def _close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- generation ------------------------------------------------

    async def generate(self, params: GenerationParams) -> TrackMeta:
        if not self._is_ready:
            raise RuntimeError(
                "MusicEngine is not ready. Start ACE-Step API or set ACESTEP_STUB=1."
            )

        started = time.perf_counter()

        if self._stub_mode:
            path = await asyncio.to_thread(self._generate_stub_wav, params)
            meta_extra: dict = {}
        else:
            path, meta_extra = await self._generate_via_api(params)

        elapsed = time.perf_counter() - started
        track_id = uuid.uuid4().hex[:12]

        return TrackMeta(
            track_id=track_id,
            prompt=params.prompt,
            lyrics=params.lyrics or None,
            bpm=int(meta_extra.get("bpm") or params.bpm or 120),
            key_scale=str(meta_extra.get("keyscale") or params.key_scale or ""),
            duration_sec=params.duration,
            file_path=path,
            seed=int(meta_extra.get("seed", params.seed)),
            generation_time_sec=elapsed,
        )

    async def generate_with_reference(
        self,
        params: GenerationParams,
        reference_audio_path: Path,
        cover_strength: float = 0.5,
    ) -> TrackMeta:
        if not self._is_ready:
            raise RuntimeError("MusicEngine is not ready.")

        ref = Path(reference_audio_path)
        if not ref.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref}")

        started = time.perf_counter()

        if self._stub_mode:
            path = await asyncio.to_thread(self._generate_stub_wav, params)
            meta_extra: dict = {}
        else:
            path, meta_extra = await self._generate_via_api(
                params,
                task_type="cover",
                src_audio_path=str(ref),
                audio_cover_strength=cover_strength,
            )

        elapsed = time.perf_counter() - started
        return TrackMeta(
            track_id=uuid.uuid4().hex[:12],
            prompt=params.prompt,
            lyrics=params.lyrics or None,
            bpm=params.bpm or meta_extra.get("bpm", 120),
            key_scale=meta_extra.get("keyscale", params.key_scale),
            duration_sec=params.duration,
            file_path=path,
            seed=meta_extra.get("seed", params.seed),
            generation_time_sec=elapsed,
        )

    # -- util ------------------------------------------------------

    def estimate_generation_time(self, duration_sec: float) -> float:
        if self._stub_mode:
            return min(2.0, duration_sec * 0.05)
        return max(5.0, duration_sec * 0.4)

    async def get_gpu_status(self) -> dict:
        status: dict = {
            "api_url": self._api_url,
            "stub_mode": self._stub_mode,
            "engine_ready": self._is_ready,
        }

        if self._stub_mode or not self._session or self._session.closed:
            return status

        try:
            import aiohttp

            async with self._session.get(
                f"{self._api_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    health = await resp.json()
                    status.update({
                        "models_initialized": health.get("models_initialized"),
                        "llm_initialized": health.get("llm_initialized"),
                        "gpu_type": health.get("gpu_type"),
                    })
        except Exception as exc:
            status["health_error"] = str(exc)

        return status

    # -- internal: ACE-Step REST API -------------------------------

    async def _generate_via_api(
        self,
        params: GenerationParams,
        task_type: str = "text2music",
        src_audio_path: Optional[str] = None,
        audio_cover_strength: float = 1.0,
    ) -> tuple[Path, dict]:
        """POST /release_task -> poll /query_result -> GET /v1/audio"""
        import aiohttp

        if not self._session or self._session.closed:
            raise RuntimeError("aiohttp session not available")

        # -- 1. submit task (/release_task) --
        payload: dict[str, Any] = {
            "task_type": task_type,
            "prompt": params.prompt,
            "lyrics": params.lyrics or "",
            "audio_duration": params.duration,
            "inference_steps": params.inference_steps,
            "guidance_scale": params.guidance_scale,
            "thinking": params.thinking if task_type == "text2music" else False,
            "batch_size": params.batch_size,
        }
        if params.seed >= 0:
            payload["seed"] = params.seed
        if src_audio_path:
            payload["src_audio_path"] = src_audio_path
            payload["audio_cover_strength"] = audio_cover_strength

        # retry on 429 (queue full) -- ACE-Step processes one at a time
        max_submit_retries = 30
        task_resp: Optional[dict] = None

        for attempt in range(max_submit_retries):
            async with self._session.post(
                f"{self._api_url}/release_task",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    wait = min(10.0, 3.0 + attempt * 2.0)
                    logger.info(
                        "ACE-Step queue full (429), retry %d/%d in %.0fs",
                        attempt + 1, max_submit_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"ACE-Step /release_task failed (HTTP {resp.status}): {body}"
                    )
                task_resp = await resp.json()
                break
        else:
            raise RuntimeError(
                "ACE-Step queue full after all retries -- server overloaded"
            )

        # task_id: top-level or nested in data.task_id
        task_id = (
            task_resp.get("task_id")
            or (task_resp.get("data") or {}).get("task_id")
        )
        if not task_id:
            raise RuntimeError(
                f"No task_id in /release_task response: {task_resp}"
            )

        logger.info("ACE-Step task submitted: %s", task_id)

        # -- 2. poll result (/query_result) --
        # ACE-Step API expects task_id_list (JSON array string),
        # returns {"data": [{"task_id":..., "status": int, "result": str}]}
        # status: 0=processing, 1=succeeded, 2=failed/timeout
        import json as _json

        deadline = time.monotonic() + self._poll_timeout
        result_item: Optional[dict] = None

        while time.monotonic() < deadline:
            await asyncio.sleep(self._poll_interval)

            async with self._session.post(
                f"{self._api_url}/query_result",
                json={"task_id_list": _json.dumps([task_id])},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "ACE-Step /query_result HTTP %d -- retrying", resp.status
                    )
                    continue
                resp_json = await resp.json()

            data_list = resp_json.get("data", [])
            if not data_list:
                logger.debug("ACE-Step task %s: empty data, retrying", task_id)
                continue

            item = data_list[0] if isinstance(data_list, list) else data_list
            poll_status = item.get("status", 0)

            if poll_status == 1:  # succeeded
                result_item = item
                break
            if poll_status == 2:  # failed / timeout
                error_msg = item.get("error") or item.get("progress_text") or "unknown error"
                raise RuntimeError(f"ACE-Step task failed: {error_msg}")

            # status 0 = processing -- keep waiting
            progress = item.get("progress_text", "")
            logger.debug("ACE-Step task %s processing: %s", task_id, progress)
        else:
            raise TimeoutError(
                f"ACE-Step generation timed out after {self._poll_timeout}s"
            )

        # -- 3. extract audio path and download --
        # result_item["result"] is a JSON string: [{"file":"...","metas":{...},...}]
        result_str = result_item.get("result", "[]")
        try:
            result_parsed = _json.loads(result_str) if isinstance(result_str, str) else result_str
        except _json.JSONDecodeError:
            result_parsed = []

        if not result_parsed or not isinstance(result_parsed, list):
            raise RuntimeError(f"ACE-Step returned no audio: {result_item}")

        first = result_parsed[0]
        remote_path = first.get("file", "") if isinstance(first, dict) else str(first)

        if not remote_path:
            raise RuntimeError(f"Empty audio path in result: {result_parsed}")

        local_path = await self._download_audio(remote_path, params.audio_format or "mp3")

        # extract metadata
        meta: dict = {}
        if isinstance(first, dict):
            metas = first.get("metas", {}) or {}
            meta["bpm"] = metas.get("bpm") or params.bpm
            meta["keyscale"] = metas.get("keyscale") or params.key_scale or ""
            meta["seed"] = metas.get("seed", params.seed)
            meta["prompt"] = first.get("prompt", params.prompt)
        else:
            meta["seed"] = params.seed

        return local_path, meta

    async def _download_audio(self, remote_path: str, fmt: str = "wav") -> Path:
        """Download audio file from ACE-Step server to local cache."""
        import aiohttp
        from urllib.parse import urlparse, parse_qs, unquote

        # Handle file paths returned as "/v1/audio?path=..." URLs
        if remote_path.startswith("/v1/audio"):
            parsed = urlparse(remote_path)
            qs = parse_qs(parsed.query)
            actual_path = qs.get("path", [remote_path])[0]
            actual_path = unquote(actual_path)
        else:
            actual_path = remote_path

        url = f"{self._api_url}/v1/audio"
        req_params = {"path": actual_path}

        async with self._session.get(
            url,
            params=req_params,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"ACE-Step /v1/audio failed (HTTP {resp.status}): {body}"
                )

            ext = fmt if fmt in ("wav", "mp3", "flac") else "wav"
            out_path = self._cache_dir / f"ace_{uuid.uuid4().hex[:8]}.{ext}"

            with open(out_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(65536):
                    f.write(chunk)

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded audio file is empty: {out_path}")

        logger.info(
            "Audio downloaded: %s (%.1f KB)",
            out_path.name,
            out_path.stat().st_size / 1024,
        )
        return out_path

    # -- internal: stub --------------------------------------------

    def _generate_stub_wav(self, params: GenerationParams) -> Path:
        import soundfile as sf

        sr = 44100
        duration = max(5.0, min(float(params.duration), 600.0))
        n = int(sr * duration)

        t = np.linspace(0, duration, n, dtype=np.float32)
        bpm = params.bpm or 120
        freq = 220.0 * (bpm / 120.0)
        tone = 0.08 * np.sin(2 * np.pi * freq * t)
        noise = 0.01 * np.random.randn(n).astype(np.float32)
        mono = (tone + noise).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1)

        out_path = self._cache_dir / f"stub_{uuid.uuid4().hex[:8]}.wav"
        sf.write(str(out_path), stereo, sr, subtype="PCM_16")
        logger.info("Stub track written: %s (%.1fs)", out_path.name, duration)
        return out_path

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
