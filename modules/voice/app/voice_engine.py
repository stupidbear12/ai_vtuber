# -*- coding: utf-8 -*-
"""
app/voice_engine.py — GPT-SoVITS TTS 엔진

GPTSoVITSEngine:
  - GPT-SoVITS API 서버(api_v2.py, 기본 http://127.0.0.1:9880)를 HTTP로 호출
  - 시온 전용 파인튜닝 모델(sion_jfla_v3, 13곡 오리지널 학습)을 사용
  - 레퍼런스 오디오(data/reference_audio/)로 음색 고정, 한국어/영어 모두 지원
  - GPT-SoVITS 서버는 별도 프로세스로 실행되어야 함:
      cd GPT-SoVITS
      python api_v2.py -a 127.0.0.1 -p 9880
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── 경로 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
REFERENCE_AUDIO_DIR = DATA_DIR / "reference_audio"
GPT_SOVITS_CONFIG_PATH = DATA_DIR / "gpt_sovits_config.json"

# ── GPT-SoVITS API 서버 설정 ────────────────────────────────────
GPT_SOVITS_API_URL = os.environ.get("GPT_SOVITS_API_URL", "http://127.0.0.1:9880")

# 언어 자동 감지용 코드 매핑 (GPT-SoVITS v2 지원 언어: ko, en, ja, zh, yue, auto)
SUPPORTED_LANGUAGES = {"ko", "en", "ja", "zh", "yue", "auto"}
_HANGUL_RE = re.compile(r"[가-힣]")
_JAPANESE_RE = re.compile(r"[぀-ヿ]")


def detect_language(text: str) -> str:
    """텍스트 내용을 보고 ko/en/ja 언어 코드를 추정한다."""
    if _HANGUL_RE.search(text):
        return "ko"
    if _JAPANESE_RE.search(text):
        return "ja"
    return "en"


class GPTSoVITSEngine:
    """GPT-SoVITS API 서버를 호출하는 얇은 클라이언트.

    실제 추론은 GPT-SoVITS 저장소의 api_v2.py 프로세스가 수행하며,
    이 클래스는 레퍼런스 오디오/모델 경로를 채워 HTTP 요청만 보낸다.
    """

    def __init__(self) -> None:
        self._api_url = GPT_SOVITS_API_URL.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._api_url, timeout=120.0)
        self._weights_ready = False
        self._config = self._load_config()

    # ── 설정 로드 ────────────────────────────────────────────────

    def _load_config(self) -> dict:
        with open(GPT_SOVITS_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

        ref_audio_path = REFERENCE_AUDIO_DIR / config["ref_audio"]
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"레퍼런스 오디오를 찾을 수 없습니다: {ref_audio_path}")
        config["ref_audio_path"] = str(ref_audio_path.resolve())
        return config

    def reload_config(self) -> dict:
        """설정 파일을 다시 읽어 레퍼런스 오디오 등을 갱신한다."""
        old_ref = self._config.get("ref_audio_path")
        self._config = self._load_config()
        new_ref = self._config.get("ref_audio_path")
        changed = old_ref != new_ref
        if changed:
            logger.info(f"레퍼런스 오디오 변경: {Path(old_ref).name} → {Path(new_ref).name}")
        return {"changed": changed, "ref_audio": new_ref}

    # ── 모델 가중치 보장 ─────────────────────────────────────────

    async def _ensure_weights(self) -> None:
        """GPT-SoVITS API 서버에 시온 모델 가중치가 로드되어 있는지 확인한다.

        서버가 다른 모델로 기동됐을 수 있으므로 첫 호출 시 한 번만 전환한다.
        """
        if self._weights_ready:
            return

        gpt_path = self._config["gpt_weights_path"]
        sovits_path = self._config["sovits_weights_path"]

        resp = await self._client.get("/set_gpt_weights", params={"weights_path": gpt_path})
        if resp.status_code != 200:
            raise RuntimeError(f"GPT 가중치 전환 실패: {resp.text}")

        resp = await self._client.get("/set_sovits_weights", params={"weights_path": sovits_path})
        if resp.status_code != 200:
            raise RuntimeError(f"SoVITS 가중치 전환 실패: {resp.text}")

        self._weights_ready = True
        logger.info(f"GPT-SoVITS 시온 모델 로드 완료: {Path(gpt_path).name}, {Path(sovits_path).name}")

    # ── TTS 핵심 메서드 ──────────────────────────────────────────

    async def synthesize_async(
        self,
        text: str,
        language: Optional[str] = None,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> bytes:
        """텍스트를 WAV 바이트로 변환한다.

        Args:
            text:        변환할 텍스트 (한국어/영어)
            language:    "ko"|"en"|"ja"|"auto" (None이면 텍스트로부터 자동 감지)
            top_p:       샘플링 top-p
            temperature: 샘플링 temperature

        Returns:
            WAV 오디오 바이트
        """
        await self._ensure_weights()

        text_lang = language if language in SUPPORTED_LANGUAGES else detect_language(text)

        payload = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self._config["ref_audio_path"],
            "prompt_text": self._config["prompt_text"],
            "prompt_lang": self._config["prompt_lang"],
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": "cut5",
            "media_type": "wav",
        }

        resp = await self._client.post("/tts", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"GPT-SoVITS TTS 실패: {resp.text}")
        return resp.content

    # ── 상태 조회 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "engine": "gpt-sovits",
            "api_url": self._api_url,
            "weights_loaded": self._weights_ready,
            "gpt_weights": self._config["gpt_weights_path"],
            "sovits_weights": self._config["sovits_weights_path"],
            "ref_audio": self._config["ref_audio_path"],
            "languages": sorted(SUPPORTED_LANGUAGES),
        }

    async def close(self) -> None:
        await self._client.aclose()
