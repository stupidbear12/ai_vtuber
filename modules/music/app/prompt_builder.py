# -*- coding: utf-8 -*-
"""
PromptBuilder — 시청자 입력 → ACE-Step 프롬프트 변환

역할:
  - 시청자 한국어 요청을 ACE-Step 영어 프롬프트로 변환
  - 장르/분위기 키워드 매핑
  - 자동 선곡용 프롬프트 템플릿
  - BPM/키/장르 추출 및 제안
"""

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .music_engine import GenerationParams

logger = logging.getLogger(__name__)


# ── 장르 매핑 (한국어 → ACE-Step 프롬프트 키워드) ────────────────

GENRE_MAP = {
    "EDM": "electronic dance music, energetic synth, drop, bass heavy",
    "로파이": "lo-fi hip hop, chill beats, vinyl crackle, mellow piano",
    "lo-fi": "lo-fi hip hop, chill beats, vinyl crackle, mellow piano",
    "팝": "pop, catchy melody, bright production, modern pop",
    "K-pop": "K-pop, catchy hook, polished production, dance beat",
    "재즈": "jazz, smooth saxophone, piano trio, swing rhythm",
    "클래식": "classical, orchestral, strings, elegant composition",
    "록": "rock, electric guitar, drums, powerful riff",
    "힙합": "hip hop, trap beat, 808 bass, rhythmic flow",
    "R&B": "R&B, smooth vocals, groove, soul",
    "앰비언트": "ambient, atmospheric pads, ethereal, spacious soundscape",
    "칠": "chill, relaxing, soft beats, downtempo",
    "하우스": "house music, four-on-the-floor, deep bass, club",
    "테크노": "techno, driving beat, industrial, synthesizer",
    "트랩": "trap, heavy 808, hi-hat rolls, dark atmosphere",
    "어쿠스틱": "acoustic, guitar, warm, intimate",
    "보사노바": "bossa nova, soft guitar, Brazilian rhythm, smooth",
    "시티팝": "city pop, 80s Japanese, funky bass, retro synth",
}

# ── 분위기 매핑 ───────────────────────────────────────────────────

MOOD_MAP = {
    "신나는": "upbeat, energetic, exciting",
    "슬픈": "melancholic, sad, emotional",
    "차분한": "calm, peaceful, serene",
    "몽환적인": "dreamy, ethereal, floating",
    "어두운": "dark, moody, intense",
    "밝은": "bright, happy, cheerful",
    "감성적인": "emotional, heartfelt, sentimental",
    "파워풀한": "powerful, epic, grand",
    "귀여운": "cute, playful, light",
    "섹시한": "sultry, smooth, groovy",
}

# ── 시간대별 자동 선곡 템플릿 ─────────────────────────────────────

AUTO_SELECT_TEMPLATES = {
    "morning_bright": [
        "bright acoustic pop with gentle piano and uplifting melody",
        "cheerful indie folk with strumming guitar, morning vibes",
        "soft jazz piano trio, warm and inviting, easy listening",
    ],
    "afternoon_energetic": [
        "upbeat EDM with catchy synth hook and driving bass",
        "energetic K-pop dance track with powerful beat",
        "funky city pop with retro synth and groovy bassline",
    ],
    "evening_chill": [
        "lo-fi hip hop beats, vinyl crackle, chill piano loop",
        "smooth R&B with soft vocals and relaxing groove",
        "bossa nova guitar with gentle percussion, evening cafe",
    ],
    "night_ambient": [
        "ambient electronic, ethereal pads, deep space atmosphere",
        "downtempo chill, soft beats, dreamy synthesizer layers",
        "minimal techno, hypnotic loop, late night drive",
    ],
}


@dataclass
class ParsedRequest:
    """시청자 요청 파싱 결과."""
    prompt_en: str
    genre: Optional[str] = None
    mood: Optional[str] = None
    bpm: Optional[int] = None
    duration: Optional[float] = None


class PromptBuilder:
    """시청자 요청 → ACE-Step 프롬프트 변환기.

    사용 패턴:
        builder = PromptBuilder()
        params = builder.from_viewer_request("신나는 EDM 틀어줘", duration=90)
        params = builder.auto_select("evening_chill")
    """

    def __init__(self):
        self._recent_genres: list = []  # 최근 사용 장르 (중복 방지용)
        self._max_recent = 5

    # ── 시청자 요청 파싱 ──────────────────────────────────────────

    def parse_viewer_input(self, text: str) -> ParsedRequest:
        """시청자 한국어 입력을 파싱.

        Args:
            text: 시청자 채팅 메시지 (예: "신나는 EDM 120bpm")

        Returns:
            ParsedRequest

        TODO:
          1. 텍스트에서 장르 키워드 매칭 (GENRE_MAP)
          2. 분위기 키워드 매칭 (MOOD_MAP)
          3. BPM 숫자 추출 (정규식: r'\d+\s*bpm')
          4. 길이 추출 (정규식: r'(\d+)\s*(분|초)')
          5. 나머지를 영어 프롬프트로 조합
        """
        genre: Optional[str] = None
        mood: Optional[str] = None
        bpm: Optional[int] = None
        duration: Optional[float] = None
        prompt_parts: list = []

        # 1. 장르 키워드 매칭
        for key, keywords in GENRE_MAP.items():
            if key in text:
                genre = key
                prompt_parts.append(keywords)
                break

        # 2. 분위기 키워드 매칭
        for key, keywords in MOOD_MAP.items():
            if key in text:
                mood = key
                prompt_parts.append(keywords)
                break

        # 3. BPM 추출
        bpm_match = re.search(r'(\d+)\s*bpm', text, re.IGNORECASE)
        if bpm_match:
            bpm = int(bpm_match.group(1))

        # 4. 길이 추출
        dur_match = re.search(r'(\d+)\s*(분|초)', text)
        if dur_match:
            value = int(dur_match.group(1))
            unit = dur_match.group(2)
            duration = float(value * 60 if unit == '분' else value)

        # 5. 프롬프트 조합
        if not prompt_parts:
            prompt_parts.append("music")
        prompt_en = ", ".join(prompt_parts)

        return ParsedRequest(
            prompt_en=prompt_en,
            genre=genre,
            mood=mood,
            bpm=bpm,
            duration=duration,
        )

    def from_viewer_request(
        self,
        text: str,
        duration: float = 90.0,
        **overrides,
    ) -> GenerationParams:
        """시청자 요청 → GenerationParams 변환.

        Args:
            text: 시청자 채팅 입력
            duration: 기본 길이 (초)
            **overrides: 추가 파라미터 오버라이드

        Returns:
            GenerationParams

        TODO:
          1. parse_viewer_input()으로 파싱
          2. ParsedRequest → GenerationParams 매핑
          3. overrides 적용
        """
        parsed = self.parse_viewer_input(text)

        params = GenerationParams(
            prompt=parsed.prompt_en,
            bpm=parsed.bpm,
            duration=parsed.duration if parsed.duration is not None else duration,
        )

        for key, value in overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)

        return params

    # ── 자동 선곡 ─────────────────────────────────────────────────

    def auto_select(self, mood_key: str) -> GenerationParams:
        """시간대/분위기 기반 자동 선곡 파라미터 생성.

        Args:
            mood_key: "morning_bright" | "afternoon_energetic" | "evening_chill" | "night_ambient"

        Returns:
            GenerationParams

        TODO:
          1. AUTO_SELECT_TEMPLATES에서 랜덤 템플릿 선택
          2. 최근 장르와 겹치지 않도록 필터링
          3. BPM 범위 랜덤 (분위기별 적절한 범위)
          4. _recent_genres 업데이트
        """
        templates = AUTO_SELECT_TEMPLATES.get(mood_key, AUTO_SELECT_TEMPLATES["evening_chill"])

        # 최근 사용 장르와 겹치지 않는 템플릿 우선
        available = [
            t for t in templates
            if not any(g.lower() in t.lower() for g in self._recent_genres)
        ]
        if not available:
            available = templates

        prompt = random.choice(available)

        # 분위기별 BPM 범위
        bpm_ranges = {
            "morning_bright":      (90, 120),
            "afternoon_energetic": (120, 140),
            "evening_chill":       (70, 95),
            "night_ambient":       (60, 85),
        }
        bpm_min, bpm_max = bpm_ranges.get(mood_key, (90, 120))
        bpm = random.randint(bpm_min, bpm_max)

        # 최근 장르 업데이트 (프롬프트 텍스트에서 장르 힌트 추출)
        for genre in GENRE_MAP:
            if genre.lower() in prompt.lower():
                self._recent_genres.append(genre)
                if len(self._recent_genres) > self._max_recent:
                    self._recent_genres.pop(0)
                break

        return GenerationParams(prompt=prompt, bpm=bpm)

    # ── 유틸 ──────────────────────────────────────────────────────

    @staticmethod
    def suggest_bpm_range(genre: str) -> Tuple[int, int]:
        """장르별 적절한 BPM 범위 제안.

        Returns:
            (min_bpm, max_bpm)

        TODO:
          EDM: 125~140, lo-fi: 70~90, pop: 100~130, jazz: 80~160,
          hip-hop: 80~115, house: 120~130, techno: 125~150, ...
        """
        BPM_RANGES: dict = {
            "EDM":      (125, 140),
            "로파이":    (70,  90),
            "lo-fi":    (70,  90),
            "팝":        (100, 130),
            "K-pop":    (100, 140),
            "재즈":      (80,  160),
            "클래식":    (60,  140),
            "록":        (100, 160),
            "힙합":      (80,  115),
            "R&B":      (60,  100),
            "앰비언트":  (60,  90),
            "칠":        (70,  100),
            "하우스":    (120, 130),
            "테크노":    (125, 150),
            "트랩":      (130, 170),
            "어쿠스틱":  (70,  130),
            "보사노바":  (100, 130),
            "시티팝":    (100, 130),
        }
        return BPM_RANGES.get(genre, (90, 130))

    @staticmethod
    def enhance_prompt(base_prompt: str, genre: Optional[str] = None, mood: Optional[str] = None) -> str:
        """프롬프트에 장르/분위기 키워드를 추가로 보강.

        TODO:
          1. genre가 GENRE_MAP에 있으면 키워드 추가
          2. mood가 MOOD_MAP에 있으면 키워드 추가
          3. 중복 제거 후 조합
        """
        parts = [base_prompt]
        if genre and genre in GENRE_MAP:
            parts.append(GENRE_MAP[genre])
        if mood and mood in MOOD_MAP:
            parts.append(MOOD_MAP[mood])

        seen: set = set()
        unique_keywords: list = []
        for part in parts:
            for kw in part.split(", "):
                kw = kw.strip()
                if kw and kw not in seen:
                    seen.add(kw)
                    unique_keywords.append(kw)

        return ", ".join(unique_keywords)
