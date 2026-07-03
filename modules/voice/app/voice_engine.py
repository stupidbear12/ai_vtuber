# -*- coding: utf-8 -*-
"""
app/voice_engine.py — TTS 엔진 모음

1. EdgeTTSEngine  — Edge TTS (Microsoft, 무료, async)  ← 기본 TTS
2. ChatterboxEngine — Chatterbox TTS (로컬 GPU, 음성 클로닝) ← 데뷔곡 보컬 전용

EdgeTTSEngine:
  - edge-tts 라이브러리 사용 (async, 무료)
  - 한국어 여성 음성 (ko-KR-SunHiNeural 기본)
  - 감정/속도/피치 조절 가능

ChatterboxEngine:
  - Chatterbox TTS 로컬 모델로 텍스트 → 음성 변환 (API 키 불필요)
  - 레퍼런스 오디오 기반 음성 클로닝 (JFla 스타일 보이스 등)
  - 감정 태그 → exaggeration 파라미터 매핑
"""

import asyncio
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ── 경로 설정 ─────────────────────────────────────────────────────
REFERENCE_AUDIO_DIR = Path(__file__).parent.parent / "data" / "reference_audio"
VOICE_CONFIG_PATH = Path(__file__).parent.parent / "data" / "voice_config.json"

# ── Edge TTS 기본 설정 ──────────────────────────────────────────
DEFAULT_EDGE_VOICE = "ko-KR-SunHiNeural"  # 한국어 여성 기본 음성

# 감정 → Edge TTS prosody 매핑
EMOTION_PROSODY: dict[str, dict[str, str]] = {
    "happy":     {"rate": "+10%",  "pitch": "+5%"},
    "excited":   {"rate": "+20%",  "pitch": "+10%"},
    "sad":       {"rate": "-10%",  "pitch": "-5%"},
    "worried":   {"rate": "-5%",   "pitch": "-3%"},
    "angry":     {"rate": "+5%",   "pitch": "+3%"},
    "surprised": {"rate": "+15%",  "pitch": "+8%"},
    "thinking":  {"rate": "-15%",  "pitch": "-2%"},
    "calm":      {"rate": "-10%",  "pitch": "-3%"},
    "love":      {"rate": "-5%",   "pitch": "+3%"},
    "shy":       {"rate": "-8%",   "pitch": "+2%"},
}


class EdgeTTSEngine:
    """Edge TTS 엔진 (Microsoft, 무료, async).

    기본 TTS 엔진으로 사용. GPU 불필요, 빠른 응답.
    """

    def __init__(self, voice: str = DEFAULT_EDGE_VOICE) -> None:
        self._voice = voice
        self._sample_rate = 24000  # WAV 변환 시 리샘플링 타겟

    @property
    def voice(self) -> str:
        return self._voice

    def set_voice(self, voice: str) -> None:
        self._voice = voice
        logger.info(f"Edge TTS 음성 변경: {voice}")

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> bytes:
        """텍스트를 MP3 바이트로 변환 (async).

        Args:
            text:    변환할 텍스트
            voice:   음성 이름 (None이면 기본값)
            emotion: 감정 태그 → rate/pitch 자동 설정
            rate:    말하기 속도 (예: "+10%", "-20%")
            pitch:   음높이 (예: "+5Hz", "-10%")

        Returns:
            MP3 오디오 바이트
        """
        import edge_tts

        use_voice = voice or self._voice

        # 감정 → prosody 매핑 (명시적 rate/pitch가 우선)
        if emotion and emotion in EMOTION_PROSODY:
            prosody = EMOTION_PROSODY[emotion]
            rate = rate or prosody.get("rate")
            pitch = pitch or prosody.get("pitch")

        communicate = edge_tts.Communicate(
            text,
            use_voice,
            rate=rate or "+0%",
            pitch=pitch or "+0Hz",
        )

        # 전체 오디오를 메모리에 수집
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> bytes:
        """텍스트를 MP3 바이트로 변환 (동기 래퍼).

        이미 실행 중인 이벤트 루프가 있으면 새 스레드에서 실행.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.synthesize_async(text, voice, emotion, rate, pitch),
                )
                return future.result()
        else:
            return asyncio.run(
                self.synthesize_async(text, voice, emotion, rate, pitch)
            )

    async def get_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ):
        """텍스트를 오디오 청크 스트림으로 반환 (async generator)."""
        import edge_tts

        use_voice = voice or self._voice

        if emotion and emotion in EMOTION_PROSODY:
            prosody = EMOTION_PROSODY[emotion]
            rate = rate or prosody.get("rate")
            pitch = pitch or prosody.get("pitch")

        communicate = edge_tts.Communicate(
            text,
            use_voice,
            rate=rate or "+0%",
            pitch=pitch or "+0Hz",
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def list_voices(self, locale: str = "ko-KR") -> list[dict]:
        """사용 가능한 음성 목록 반환."""
        import edge_tts
        voices = await edge_tts.list_voices()
        filtered = [
            {
                "short_name": v["ShortName"],
                "local_name": v["LocalName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
                "is_current": v["ShortName"] == self._voice,
            }
            for v in voices
            if v["Locale"] == locale
        ]
        return filtered

    def get_status(self) -> dict:
        return {
            "engine": "edge-tts",
            "voice": self._voice,
            "sample_rate": self._sample_rate,
            "emotion_tags": list(EMOTION_PROSODY.keys()),
        }


# ── 감정 → exaggeration 매핑 (Chatterbox용) ─────────────────────
# exaggeration: 0.0 = 단조롭고 안정적, 1.0 = 매우 감정적·역동적
EMOTION_EXAGGERATION: dict[str, float] = {
    "happy":     0.60,
    "excited":   0.90,
    "sad":       0.30,
    "worried":   0.40,
    "angry":     0.80,
    "surprised": 0.70,
    "thinking":  0.20,
    "calm":      0.10,
    "love":      0.50,
    "shy":       0.35,
}

DEFAULT_EXAGGERATION = 0.50
# cfg_weight: 레퍼런스 오디오가 있을 때 충실도. 높을수록 원본에 가까움.
DEFAULT_CFG_WEIGHT = 0.50


class ChatterboxEngine:
    """Chatterbox TTS 엔진 (로컬 GPU 추론).

    싱글톤으로 사용. 모델은 첫 synthesize 호출 시 GPU에 로드된다(lazy load).
    레퍼런스 오디오를 data/reference_audio/ 에 배치하면 음성 클로닝이 활성화된다.
    """

    def __init__(self) -> None:
        # torch/chatterbox는 호출 시점에 임포트 (서버 시작 오류 방지)
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None          # ChatterboxTTS (lazy load)
        self._sample_rate = 24000   # Chatterbox 기본 샘플레이트 (모델 로드 후 갱신)
        self._default_reference: str = ""
        self._references: dict[str, str] = {}  # {name: absolute_path}
        self._load_config()
        self._scan_references()

    # ── 모델 관리 ────────────────────────────────────────────────

    def _get_model(self):
        """Chatterbox 모델 lazy load (첫 호출 시 GPU에 로드)."""
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS
            logger.info(f"Chatterbox 모델 로딩 중... (device={self._device})")
            self._model = ChatterboxTTS.from_pretrained(device=self._device)
            self._sample_rate = self._model.sr
            logger.info(f"Chatterbox 모델 로드 완료 (sr={self._sample_rate}Hz)")
        return self._model

    # ── 설정 파일 관리 ───────────────────────────────────────────

    def _load_config(self) -> None:
        try:
            if VOICE_CONFIG_PATH.exists():
                with open(VOICE_CONFIG_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                self._default_reference = data.get("default_reference", "")
        except Exception as e:
            logger.warning(f"voice_config.json 로드 실패 (무시): {e}")

    def _save_config(self) -> None:
        VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"default_reference": self._default_reference},
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ── 레퍼런스 오디오 스캔 ────────────────────────────────────

    def _scan_references(self) -> None:
        """data/reference_audio/*.wav 목록을 갱신한다."""
        REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self._references = {
            p.stem: str(p)
            for p in sorted(REFERENCE_AUDIO_DIR.glob("*.wav"))
        }
        if self._references:
            logger.info(
                f"레퍼런스 오디오 {len(self._references)}개 발견: "
                f"{list(self._references.keys())}"
            )

    # ── 속성 ─────────────────────────────────────────────────────

    @property
    def default_reference(self) -> Optional[str]:
        return self._default_reference or None

    def set_default_reference(self, name: str) -> None:
        """기본 레퍼런스 오디오를 설정하고 파일에 저장한다."""
        self._default_reference = name
        self._save_config()
        logger.info(f"기본 레퍼런스 오디오 변경: {name}")

    # ── 오디오 변환 유틸 ─────────────────────────────────────────

    def _tensor_to_wav_bytes(self, wav, sample_rate: int) -> bytes:
        """Torch 텐서를 WAV 바이트로 변환한다."""
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, wav.cpu(), sample_rate, format="wav")
        buf.seek(0)
        return buf.read()

    # ── TTS 핵심 메서드 ──────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        reference_name: Optional[str] = None,
        emotion: Optional[str] = None,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
    ) -> bytes:
        """텍스트를 WAV 바이트로 변환한다.

        Args:
            text:           변환할 텍스트 (한국어 포함)
            reference_name: 레퍼런스 오디오 이름 (None이면 기본값 또는 클로닝 없이 생성)
            emotion:        감정 태그 → exaggeration 자동 설정
            cfg_weight:     음성 충실도 (0.0~1.0, 레퍼런스 있을 때 유효)

        Returns:
            WAV 오디오 바이트 (24kHz, mono)
        """
        model = self._get_model()

        # 레퍼런스 오디오 결정
        ref_name = reference_name or self._default_reference or None
        ref_path: Optional[str] = None
        if ref_name:
            self._scan_references()  # 최신 목록 반영
            ref_path = self._references.get(ref_name)
            if ref_path is None:
                logger.warning(f"레퍼런스 오디오 '{ref_name}' 없음 → 클로닝 없이 생성")

        exaggeration = EMOTION_EXAGGERATION.get(emotion or "", DEFAULT_EXAGGERATION)

        logger.debug(
            f"TTS 생성: len={len(text)}, ref={ref_name}, "
            f"exaggeration={exaggeration}, cfg_weight={cfg_weight}"
        )

        wav = model.generate(
            text=text,
            audio_prompt_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        return self._tensor_to_wav_bytes(wav, self._sample_rate)

    def get_sync_stream(
        self,
        text: str,
        reference_name: Optional[str] = None,
        emotion: Optional[str] = None,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
    ) -> Iterator[bytes]:
        """WAV 바이트를 청크로 분할하는 동기 제너레이터.

        Chatterbox는 일괄 생성 방식이므로 전체 생성 후 4KB 청크로 분할 전송.
        """
        audio_bytes = self.synthesize(text, reference_name, emotion, cfg_weight)
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    # ── 레퍼런스 목록 ────────────────────────────────────────────

    def list_references(self) -> list[dict]:
        """사용 가능한 레퍼런스 오디오 목록을 반환한다."""
        self._scan_references()
        return [
            {
                "name": name,
                "path": path,
                "is_default": name == self._default_reference,
            }
            for name, path in self._references.items()
        ]

    # ── 상태 조회 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """현재 엔진 상태를 반환한다."""
        self._scan_references()
        return {
            "engine": "chatterbox",
            "device": self._device,
            "model_loaded": self._model is not None,
            "sample_rate": self._sample_rate,
            "default_reference": self._default_reference or None,
            "references": list(self._references.keys()),
            "emotion_tags": list(EMOTION_EXAGGERATION.keys()),
            "reference_audio_dir": str(REFERENCE_AUDIO_DIR),
        }
