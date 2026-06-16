# -*- coding: utf-8 -*-
"""
DJController — AI DJ 자동화 로직
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from .music_engine import MusicEngine, GenerationParams, TrackMeta
from .track_queue import TrackQueue
from .audio_mixer import AudioMixer
from .prompt_builder import AUTO_SELECT_TEMPLATES, PromptBuilder

logger = logging.getLogger(__name__)

# 분위기별 BPM 범위
_MOOD_BPM = {
    "morning_bright": (90, 120),
    "afternoon_energetic": (120, 140),
    "evening_chill": (70, 95),
    "night_ambient": (60, 85),
}


@dataclass
class NowPlaying:
    """현재 재생 중인 트랙 정보."""
    track: Optional[TrackMeta] = None
    started_at: Optional[datetime] = None
    requester: Optional[str] = None


@dataclass
class DJConfig:
    """DJ 동작 설정."""
    buffer_size: int = 3
    crossfade_sec: float = 3.0
    crossfade_trigger_sec: float = 5.0
    auto_select_enabled: bool = True
    default_duration: float = 90.0
    poll_interval_sec: float = 0.5
    buffer_wait_sec: float = 2.0


class DJController:
    """AI DJ 메인 컨트롤러."""

    def __init__(
        self,
        engine: MusicEngine,
        queue: TrackQueue,
        mixer: AudioMixer,
        config: Optional[DJConfig] = None,
    ):
        self._engine = engine
        self._queue = queue
        self._mixer = mixer
        self._config = config or DJConfig()
        self._prompt_builder = PromptBuilder()

        self._now_playing = NowPlaying()
        self._buffer: List[TrackMeta] = []
        self._buffer_lock = asyncio.Lock()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._buffer_loop_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._crossfade_triggered = False
        self._recent_genres: List[str] = []

    # ── 라이프사이클 ──────────────────────────────────────────────

    async def start(self) -> None:
        """DJ 백그라운드 루프 시작."""
        if self._is_running:
            return

        self._is_running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._buffer_loop_task = asyncio.create_task(self._buffer_loop())
        logger.info("DJController started")

    async def stop(self) -> None:
        """DJ 루프 정지."""
        self._is_running = False

        for task in (self._main_loop_task, self._buffer_loop_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._main_loop_task = None
        self._buffer_loop_task = None
        await self._mixer.stop()
        logger.info("DJController stopped")

    # ── 메인 루프 ─────────────────────────────────────────────────

    async def _main_loop(self) -> None:
        """트랙 재생 흐름 관리."""
        try:
            while self._is_running:
                state = self._mixer.get_playback_state()
                duration = state.get("duration") or 0.0
                position = state.get("position") or 0.0
                is_playing = state.get("is_playing", False)

                if self._now_playing.track is None:
                    await self._play_next()
                elif not is_playing and duration > 0 and position >= max(0.0, duration - 0.2):
                    logger.info("Track finished, playing next")
                    self._crossfade_triggered = False
                    await self._play_next()
                elif (
                    is_playing
                    and duration > 0
                    and not self._crossfade_triggered
                    and (duration - position) <= self._config.crossfade_trigger_sec
                ):
                    next_track = await self._take_next_buffered_track()
                    if next_track and next_track.file_path:
                        logger.info(
                            "Auto crossfade -> %s (%.1fs remaining)",
                            next_track.track_id, duration - position,
                        )
                        self._crossfade_triggered = True
                        await self._mixer.crossfade_to(
                            next_track.file_path,
                            self._config.crossfade_sec,
                        )
                        self._set_now_playing(next_track)

                await asyncio.sleep(self._config.poll_interval_sec)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("DJ main loop error")

    async def _buffer_loop(self) -> None:
        """프리-제너레이션 — 버퍼를 buffer_size까지 유지."""
        try:
            while self._is_running:
                async with self._buffer_lock:
                    need = self._config.buffer_size - len(self._buffer)

                if need <= 0:
                    await asyncio.sleep(self._config.buffer_wait_sec)
                    continue

                if not self._engine.is_ready:
                    await asyncio.sleep(self._config.buffer_wait_sec)
                    continue

                params, requester, item_id = await self._resolve_next_params()
                try:
                    track = await self._engine.generate(params)
                    if requester and track.genre is None:
                        track.genre = params.prompt[:40]

                    async with self._buffer_lock:
                        self._buffer.append(track)

                    if item_id:
                        if track.file_path:
                            await self._queue.mark_ready(item_id, str(track.file_path))
                        else:
                            await self._queue.mark_failed(item_id, "No file_path in TrackMeta")

                    logger.info(
                        "Buffered track %s (%d/%d) requester=%s",
                        track.track_id, len(self._buffer), self._config.buffer_size, requester,
                    )
                except NotImplementedError:
                    logger.warning("MusicEngine.generate() not implemented — buffer loop idle")
                    await asyncio.sleep(5.0)
                except Exception as exc:
                    logger.error("Track generation failed: %s", exc)
                    if item_id:
                        await self._queue.mark_failed(item_id, str(exc))
                    await asyncio.sleep(self._config.buffer_wait_sec)

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("DJ buffer loop error")

    # ── 재생 제어 ─────────────────────────────────────────────────

    async def _play_next(self) -> bool:
        """다음 트랙 재생."""
        track = await self._take_next_buffered_track()
        if track is None:
            track = await self._generate_immediate()
        if track is None or not track.file_path:
            return False

        path = Path(track.file_path)
        if not path.exists():
            logger.error("Track file missing: %s", path)
            return False

        await self._mixer.load_track(path)
        await self._mixer.play()
        self._set_now_playing(track)
        self._crossfade_triggered = False
        logger.info("Now playing: %s — %s", track.track_id, track.prompt[:60])
        return True

    async def skip_current(self, crossfade_sec: Optional[float] = None) -> dict:
        """현재 곡 스킵."""
        fade = crossfade_sec if crossfade_sec is not None else self._config.crossfade_sec
        skipped = self._now_playing.track

        next_track = await self._take_next_buffered_track()
        if next_track is None:
            next_track = await self._generate_immediate()

        if next_track and next_track.file_path and Path(next_track.file_path).exists():
            if skipped and self._mixer.get_playback_state().get("is_playing"):
                await self._mixer.crossfade_to(Path(next_track.file_path), fade)
            else:
                await self._mixer.load_track(Path(next_track.file_path))
                await self._mixer.play()
            self._set_now_playing(next_track)
            self._crossfade_triggered = False
        else:
            await self._mixer.stop()

        return {
            "skipped": skipped,
            "next": next_track,
        }

    async def _take_next_buffered_track(self) -> Optional[TrackMeta]:
        """버퍼에서 다음 트랙 하나를 꺼낸다."""
        async with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer.pop(0)

    async def _generate_immediate(self) -> Optional[TrackMeta]:
        """버퍼가 비었을 때 즉시 1곡 생성."""
        if not self._engine.is_ready:
            return None

        params, requester, item_id = await self._resolve_next_params()
        try:
            track = await self._engine.generate(params)
            if item_id and track.file_path:
                await self._queue.mark_ready(item_id, str(track.file_path))
            if requester:
                track.genre = track.genre or params.prompt[:40]
            return track
        except NotImplementedError:
            logger.warning("MusicEngine.generate() not implemented")
            return None
        except Exception as exc:
            logger.error("Immediate generation failed: %s", exc)
            if item_id:
                await self._queue.mark_failed(item_id, str(exc))
            return None

    async def _resolve_next_params(self) -> Tuple[GenerationParams, Optional[str], Optional[str]]:
        """큐 또는 자동 선곡에서 다음 생성 파라미터를 가져온다."""
        item = await self._queue.dequeue()
        if item is not None:
            return item.params, item.requester, item.item_id

        if self._config.auto_select_enabled:
            return await self._auto_select(), None, None

        return GenerationParams(
            prompt="instrumental lo-fi chill beats",
            duration=self._config.default_duration,
        ), None, None

    def _set_now_playing(self, track: TrackMeta) -> None:
        self._now_playing = NowPlaying(
            track=track,
            started_at=datetime.now(),
            requester=getattr(track, "requester", None),
        )

    # ── 자동 선곡 ─────────────────────────────────────────────────

    async def _auto_select(self) -> GenerationParams:
        """시간대 기반 자동 선곡."""
        mood = self._get_mood_by_time()
        templates = AUTO_SELECT_TEMPLATES.get(mood, AUTO_SELECT_TEMPLATES["evening_chill"])

        # 최근 프롬프트와 겹치지 않게 선택
        candidates = [t for t in templates if t not in self._recent_genres]
        prompt = random.choice(candidates or templates)

        self._recent_genres.append(prompt)
        if len(self._recent_genres) > 5:
            self._recent_genres.pop(0)

        bpm_min, bpm_max = _MOOD_BPM.get(mood, (80, 120))
        genre_hint = mood.split("_")[-1]

        return GenerationParams(
            prompt=prompt,
            bpm=random.randint(bpm_min, bpm_max),
            duration=self._config.default_duration,
        )

    def _get_mood_by_time(self) -> str:
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning_bright"
        if 12 <= hour < 18:
            return "afternoon_energetic"
        if 18 <= hour < 23:
            return "evening_chill"
        return "night_ambient"

    # ── 시청자 요청 처리 ──────────────────────────────────────────

    async def handle_viewer_request(self, prompt: str, requester: str, **kwargs) -> str:
        """시청자 채팅 요청을 큐에 추가."""
        duration = float(kwargs.get("duration", self._config.default_duration))
        genre = kwargs.get("genre")
        bpm = kwargs.get("bpm")

        try:
            params = self._prompt_builder.from_viewer_request(
                prompt,
                duration=duration,
                genre=genre,
                bpm=bpm,
            )
        except NotImplementedError:
            params = GenerationParams(
                prompt=prompt,
                bpm=bpm,
                duration=duration,
            )

        priority = int(kwargs.get("priority", 1))
        item_id = await self._queue.enqueue(params, requester=requester, priority=priority)
        logger.info("Viewer request queued: %s by %s", item_id, requester)
        return item_id

    # ── 상태 조회 ─────────────────────────────────────────────────

    def now_playing_info(self) -> Optional[dict]:
        if not self._now_playing.track:
            return None

        elapsed = 0.0
        if self._now_playing.started_at:
            elapsed = (datetime.now() - self._now_playing.started_at).total_seconds()

        mixer_state = self._mixer.get_playback_state()
        if mixer_state.get("position"):
            elapsed = mixer_state["position"]

        t = self._now_playing.track
        return {
            "track_id": t.track_id,
            "prompt": t.prompt,
            "genre": t.genre,
            "bpm": t.bpm,
            "duration": t.duration_sec or mixer_state.get("duration", 0.0),
            "elapsed": elapsed,
            "requester": self._now_playing.requester,
            "is_playing": mixer_state.get("is_playing", False),
        }

    def buffer_status(self) -> dict:
        return {
            "buffer_size": len(self._buffer),
            "max_buffer": self._config.buffer_size,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "prompt": t.prompt,
                    "bpm": t.bpm,
                    "file_path": str(t.file_path) if t.file_path else None,
                }
                for t in self._buffer
            ],
        }
