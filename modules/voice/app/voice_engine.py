# -*- coding: utf-8 -*-
"""
app/voice_engine.py — ElevenLabs TTS 엔진 핵심 로직

역할:
  - ElevenLabs Python SDK를 래핑하여 TTS/Voice Design 기능 제공
  - 감정 태그 → 음성 파라미터 매핑 (stability, style 등)
  - 기본 음성(시온) 설정 및 커스텀 음성 로컬 저장 관리
  - 음성 목록 조회, 음성 디자인(Voice Design) API 연동

ElevenLabs SDK 버전: elevenlabs>=1.0.0
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

logger = logging.getLogger(__name__)

# ── 커스텀 음성 데이터 저장 경로 ──────────────────────────────────
# modules/voice/data/custom_voices.json
VOICES_STORE_PATH = Path(__file__).parent.parent / "data" / "custom_voices.json"

# ── ElevenLabs 모델 설정 ──────────────────────────────────────────
# eleven_multilingual_v2: 한국어 포함 29개 언어 지원 (권장)
# eleven_turbo_v2_5: 저지연 스트리밍용 (한국어 지원)
DEFAULT_MODEL = "eleven_multilingual_v2"

# ── 감정 태그 → 음성 파라미터 매핑 ────────────────────────────────
# stability: 음성 일관성 (0=변동 많음, 1=안정적)
# similarity_boost: 목소리 재현도 (0=유연, 1=원본 충실)
# style: 표현력 강도 (0=중립, 1=과장) — v2 모델 전용
# use_speaker_boost: 명료도 향상 (약간의 품질 향상)
EMOTION_SETTINGS: dict[str, VoiceSettings] = {
    "happy": VoiceSettings(
        stability=0.35, similarity_boost=0.80, style=0.55, use_speaker_boost=True
    ),
    "excited": VoiceSettings(
        stability=0.20, similarity_boost=0.85, style=0.75, use_speaker_boost=True
    ),
    "sad": VoiceSettings(
        stability=0.72, similarity_boost=0.70, style=0.08, use_speaker_boost=False
    ),
    "worried": VoiceSettings(
        stability=0.60, similarity_boost=0.72, style=0.15, use_speaker_boost=False
    ),
    "angry": VoiceSettings(
        stability=0.30, similarity_boost=0.90, style=0.65, use_speaker_boost=True
    ),
    "surprised": VoiceSettings(
        stability=0.22, similarity_boost=0.82, style=0.50, use_speaker_boost=True
    ),
    "thinking": VoiceSettings(
        stability=0.80, similarity_boost=0.65, style=0.05, use_speaker_boost=False
    ),
    "calm": VoiceSettings(
        stability=0.82, similarity_boost=0.70, style=0.00, use_speaker_boost=False
    ),
    "love": VoiceSettings(
        stability=0.50, similarity_boost=0.80, style=0.40, use_speaker_boost=True
    ),
    "shy": VoiceSettings(
        stability=0.62, similarity_boost=0.78, style=0.18, use_speaker_boost=False
    ),
}

# 감정 없음 또는 알 수 없는 감정일 때 사용하는 기본 설정
DEFAULT_VOICE_SETTINGS = VoiceSettings(
    stability=0.50, similarity_boost=0.75, style=0.00, use_speaker_boost=True
)

# ElevenLabs 빌트인 보이스 중 한국어 지원 기본값 (API 키 없을 때 안내용)
FALLBACK_VOICE_ID = "FGY2WhTYpPnrIDTdsKH5"  # Laura


class ElevenLabsEngine:
    """ElevenLabs TTS/Voice Design API 연동 엔진.

    싱글톤으로 사용하며, ELEVENLABS_API_KEY 환경변수로 인증한다.
    커스텀 음성(시온)은 data/custom_voices.json에 로컬 저장된다.
    """

    def __init__(self) -> None:
        self.api_key: str = os.environ.get("ELEVENLABS_API_KEY", "")
        # 기본 음성 ID: 환경변수 → 저장된 설정 → 빌트인 기본값 순으로 결정
        self._default_voice_id: str = os.environ.get("ELEVENLABS_VOICE_ID", "")
        self._client: Optional[ElevenLabs] = None
        self._custom_voices: dict = {}  # { voice_id: {name, description, created_at} }
        self._load_custom_voices()

    # ── 클라이언트 관리 ───────────────────────────────────────────

    def _get_client(self) -> ElevenLabs:
        """ElevenLabs 클라이언트를 반환한다 (Lazy initialization).

        api_key가 없으면 ValueError를 발생시켜 503 응답으로 이어진다.
        """
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY가 설정되지 않았습니다. "
                    ".env 파일에 ELEVENLABS_API_KEY=... 를 추가하세요."
                )
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    # ── 커스텀 음성 파일 관리 ────────────────────────────────────

    def _load_custom_voices(self) -> None:
        """data/custom_voices.json 에서 커스텀 음성 설정을 불러온다."""
        try:
            if VOICES_STORE_PATH.exists():
                with open(VOICES_STORE_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                self._custom_voices = data.get("voices", {})
                saved_default = data.get("default_voice_id", "")
                # 환경변수가 비어 있으면 저장된 기본값을 사용
                if not self._default_voice_id and saved_default:
                    self._default_voice_id = saved_default
        except Exception as e:
            logger.warning(f"커스텀 음성 파일 로드 실패 (무시): {e}")

    def _save_custom_voices(self) -> None:
        """현재 커스텀 음성 설정을 data/custom_voices.json 에 저장한다."""
        VOICES_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VOICES_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "voices": self._custom_voices,
                    "default_voice_id": self._default_voice_id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ── 속성 ─────────────────────────────────────────────────────

    @property
    def default_voice_id(self) -> str:
        """현재 기본 음성 ID (설정 없으면 Rachel 반환)."""
        return self._default_voice_id or FALLBACK_VOICE_ID

    def set_default_voice(self, voice_id: str, voice_name: str = "") -> None:
        """기본 음성 ID를 변경하고 파일에 저장한다."""
        self._default_voice_id = voice_id
        self._save_custom_voices()
        logger.info(f"기본 음성 변경: {voice_name or voice_id} ({voice_id})")

    # ── TTS (Text-to-Speech) ─────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> bytes:
        """텍스트를 MP3 바이트로 변환한다 (동기, 전체 로드).

        Args:
            text: 변환할 텍스트
            voice_id: ElevenLabs 보이스 ID (None이면 기본 음성 사용)
            emotion: 감정 태그 (happy/sad/calm 등 — EMOTION_SETTINGS 참조)

        Returns:
            MP3 오디오 바이트 데이터
        """
        client = self._get_client()
        vid = voice_id or self.default_voice_id
        settings = EMOTION_SETTINGS.get(emotion or "", DEFAULT_VOICE_SETTINGS)

        # convert()는 Iterator[bytes]를 반환하므로 join으로 합산
        chunks: Iterator[bytes] = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=DEFAULT_MODEL,
            voice_settings=settings,
            output_format="mp3_44100_128",  # 44.1kHz 128kbps — 품질/용량 균형
        )
        return b"".join(chunks)

    def get_sync_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> Iterator[bytes]:
        """텍스트를 MP3 청크 스트림으로 변환한다 (동기 제너레이터).

        FastAPI StreamingResponse에 직접 넘겨 사용한다.
        낮은 지연 시간으로 실시간 대화 음성 재생에 적합.

        Args:
            text: 변환할 텍스트
            voice_id: 보이스 ID (None이면 기본 음성)
            emotion: 감정 태그

        Yields:
            MP3 바이트 청크
        """
        client = self._get_client()
        vid = voice_id or self.default_voice_id
        settings = EMOTION_SETTINGS.get(emotion or "", DEFAULT_VOICE_SETTINGS)

        # stream()은 실시간으로 청크를 생성하는 동기 제너레이터 반환
        yield from client.text_to_speech.stream(
            voice_id=vid,
            text=text,
            model_id=DEFAULT_MODEL,
            voice_settings=settings,
            output_format="mp3_44100_128",
        )

    # ── 음성 목록 ────────────────────────────────────────────────

    def list_voices(self) -> list[dict]:
        """ElevenLabs 계정의 음성 목록을 반환한다.

        빌트인 + 클론 음성 모두 포함. 커스텀 음성 여부와 기본 음성 여부를 표시.

        Returns:
            [{"voice_id", "name", "category", "is_custom", "is_default"}, ...]
        """
        client = self._get_client()
        response = client.voices.get_all()

        voices = []
        for v in response.voices:
            voices.append(
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": getattr(v, "category", "unknown"),
                    "is_custom": v.voice_id in self._custom_voices,
                    "is_default": v.voice_id == self.default_voice_id,
                    "labels": getattr(v, "labels", {}),
                }
            )

        # 기본 음성을 목록 맨 앞으로 정렬
        voices.sort(key=lambda x: (not x["is_default"], not x["is_custom"]))
        return voices

    # ── 음성 디자인 (Voice Design) ────────────────────────────────

    def design_voice(self, name: str, description: str) -> dict:
        """텍스트 설명으로 새 음성을 생성하고 계정에 저장한다.

        ElevenLabs Voice Generation API 2단계 흐름:
          1단계: 설명 기반 음성 미리보기(preview) 생성
          2단계: 미리보기에서 실제 음성 생성 및 계정 저장

        Args:
            name: 음성 이름 (예: "시온")
            description: 음성 설명 (예: "밝고 귀여운 20대 여성, 약간 높은 톤, 한국어")

        Returns:
            {"voice_id": str, "name": str, "description": str}
        """
        client = self._get_client()

        # 1단계: 텍스트 설명으로 음성 미리보기 생성 (SDK v2: text_to_voice.create_previews)
        logger.info(f"음성 디자인 시작: {name} — {description}")
        previews = client.text_to_voice.create_previews(
            voice_description=description,
            text=(
                f"안녕! 나 {name}이야~ 만나서 진짜 반가워! "
                "오늘 하루 어떻게 보냈어? 나는 엄청 신나는 일이 있었는데, "
                "같이 얘기하면서 즐거운 시간 보내자! 뭐든지 물어봐도 돼, 항상 여기 있을게~"
            ),
        )

        if not previews.previews:
            raise RuntimeError("ElevenLabs에서 음성 미리보기를 생성하지 못했습니다.")

        # 첫 번째 미리보기를 선택 (추후 여러 개 중 선택 지원 가능)
        preview = previews.previews[0]

        # 2단계: 미리보기를 계정에 음성으로 저장 (SDK v2: text_to_voice.create)
        new_voice = client.text_to_voice.create(
            voice_name=name,
            voice_description=description,
            generated_voice_id=preview.generated_voice_id,
        )

        # 로컬 custom_voices 파일에 메타데이터 저장
        self._custom_voices[new_voice.voice_id] = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        self._save_custom_voices()

        logger.info(f"음성 디자인 완료: {name} → voice_id={new_voice.voice_id}")
        return {
            "voice_id": new_voice.voice_id,
            "name": name,
            "description": description,
        }

    # ── 상태 조회 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """현재 음성 엔진 설정 상태를 반환한다."""
        return {
            "api_key_set": bool(self.api_key),
            "default_voice_id": self.default_voice_id,
            "model": DEFAULT_MODEL,
            "custom_voices_count": len(self._custom_voices),
            "custom_voices": [
                {"voice_id": vid, **meta}
                for vid, meta in self._custom_voices.items()
            ],
            "emotion_tags": list(EMOTION_SETTINGS.keys()),
        }
