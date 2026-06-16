# -*- coding: utf-8 -*-
"""
AudioMixer — 실시간 오디오 믹싱 및 스트림 출력

역할:
  - 트랙 간 크로스페이드 트랜지션
  - BPM 매칭 (타임스트레칭)
  - 실시간 PCM 스트림 생성 (WebSocket 클라이언트용)
  - 오디오 이펙트 (페이드인/아웃, 볼륨)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import numpy as np
import soundfile as sf
from fastapi import WebSocket

logger = logging.getLogger(__name__)

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover
    AudioSegment = None


@dataclass
class AudioChunk:
    """오디오 청크 단위."""
    data: np.ndarray        # shape: (samples, channels), dtype: float32
    sample_rate: int = 44100
    channels: int = 2


@dataclass
class PlaybackState:
    """현재 재생 상태."""
    track_path: Optional[Path] = None
    position_sec: float = 0.0
    duration_sec: float = 0.0
    volume: float = 1.0
    is_playing: bool = False


@dataclass
class _CrossfadeState:
    next_audio: np.ndarray
    next_path: Path
    total_samples: int
    remaining: int
    next_pos: int = 0


@dataclass
class _FadeState:
    kind: str               # "in" | "out"
    total_samples: int
    remaining: int
    start_gain: float
    end_gain: float


class AudioMixer:
    """실시간 오디오 믹싱 엔진.

    WebSocket 구독자에게 int16 PCM(stereo interleaved)을 전송한다.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 2, chunk_size: int = 4096):
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._subscribers: Set[WebSocket] = set()
        self._playback = PlaybackState()
        self._stream_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._lock = asyncio.Lock()

        self._audio: Optional[np.ndarray] = None
        self._sample_pos: int = 0
        self._crossfade: Optional[_CrossfadeState] = None
        self._fade: Optional[_FadeState] = None
        self._chunk_gain: float = 1.0

    # ── 라이프사이클 ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """믹서 초기화 및 스트림 루프 시작."""
        self._shutdown = False
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info(
            "AudioMixer initialized (sr=%s, ch=%s, chunk=%s)",
            self._sample_rate, self._channels, self._chunk_size,
        )

    async def shutdown(self) -> None:
        """믹서 종료."""
        self._shutdown = True
        self._playback.is_playing = False

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self._stream_task = None
        self._subscribers.clear()
        self._audio = None
        self._crossfade = None
        self._fade = None
        logger.info("AudioMixer shutdown complete")

    # ── 트랙 로드/재생 ────────────────────────────────────────────

    async def load_track(self, file_path: Path) -> float:
        """오디오 파일 로드. 트랙 길이(초)를 반환한다."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio = await asyncio.to_thread(self._load_audio_file, path)

        async with self._lock:
            self._audio = audio
            self._sample_pos = 0
            self._crossfade = None
            self._fade = None
            self._chunk_gain = 1.0
            self._playback.track_path = path
            self._playback.duration_sec = len(audio) / self._sample_rate
            self._playback.position_sec = 0.0

        logger.info("Loaded track: %s (%.1fs)", path.name, self._playback.duration_sec)
        return self._playback.duration_sec

    async def play(self) -> None:
        """현재 로드된 트랙 재생 시작."""
        if self._audio is None:
            raise RuntimeError("No track loaded. Call load_track() first.")
        self._playback.is_playing = True

    async def pause(self) -> None:
        """재생 일시정지."""
        self._playback.is_playing = False

    async def stop(self) -> None:
        """재생 정지 및 위치 초기화."""
        async with self._lock:
            self._playback.is_playing = False
            self._sample_pos = 0
            self._playback.position_sec = 0.0
            self._crossfade = None
            self._fade = None
            self._chunk_gain = 1.0

    # ── 트랜지션 ──────────────────────────────────────────────────

    async def crossfade_to(self, next_track_path: Path, duration_sec: float = 3.0) -> None:
        """현재 트랙에서 다음 트랙으로 크로스페이드."""
        next_path = Path(next_track_path)
        next_audio = await asyncio.to_thread(self._load_audio_file, next_path)
        fade_samples = max(1, int(duration_sec * self._sample_rate))

        async with self._lock:
            if self._audio is None:
                self._audio = next_audio
                self._sample_pos = 0
                self._playback.track_path = next_path
                self._playback.duration_sec = len(next_audio) / self._sample_rate
                self._playback.position_sec = 0.0
                self._playback.is_playing = True
                return

            self._crossfade = _CrossfadeState(
                next_audio=next_audio,
                next_path=next_path,
                total_samples=fade_samples,
                remaining=fade_samples,
            )
            self._playback.is_playing = True

    async def fade_out(self, duration_sec: float = 2.0) -> None:
        """현재 트랙 페이드아웃."""
        samples = max(1, int(duration_sec * self._sample_rate))
        async with self._lock:
            self._fade = _FadeState(
                kind="out",
                total_samples=samples,
                remaining=samples,
                start_gain=self._chunk_gain,
                end_gain=0.0,
            )

    async def fade_in(self, duration_sec: float = 2.0) -> None:
        """현재 트랙 페이드인."""
        samples = max(1, int(duration_sec * self._sample_rate))
        async with self._lock:
            self._chunk_gain = 0.0
            self._fade = _FadeState(
                kind="in",
                total_samples=samples,
                remaining=samples,
                start_gain=0.0,
                end_gain=1.0,
            )

    # ── BPM 매칭 ──────────────────────────────────────────────────

    async def match_bpm(self, target_bpm: int) -> None:
        """현재 트랙 BPM을 target_bpm에 맞게 타임스트레칭."""
        if librosa is None:
            raise RuntimeError("librosa is required for match_bpm()")

        async with self._lock:
            if self._audio is None:
                raise RuntimeError("No track loaded")

            mono = self._audio.mean(axis=1)
            tempo, _ = librosa.beat.beat_track(y=mono, sr=self._sample_rate)
            current_bpm = float(tempo)
            if current_bpm <= 0:
                raise RuntimeError("Could not detect BPM")

            rate = target_bpm / current_bpm
            stretched_channels = []
            for ch in range(self._audio.shape[1]):
                stretched = librosa.effects.time_stretch(self._audio[:, ch], rate=rate)
                stretched_channels.append(stretched)

            min_len = min(len(c) for c in stretched_channels)
            self._audio = np.stack([c[:min_len] for c in stretched_channels], axis=1)
            self._sample_pos = min(self._sample_pos, len(self._audio))
            self._playback.duration_sec = len(self._audio) / self._sample_rate
            logger.info("BPM matched: %.1f -> %d", current_bpm, target_bpm)

    # ── 볼륨/이펙트 ───────────────────────────────────────────────

    def set_volume(self, volume: float) -> None:
        """볼륨 설정 (0.0~1.0)."""
        self._playback.volume = max(0.0, min(1.0, volume))

    # ── WebSocket 스트리밍 ────────────────────────────────────────

    async def subscribe(self, ws: WebSocket) -> None:
        """WebSocket 클라이언트를 오디오 스트림 구독자로 등록."""
        self._subscribers.add(ws)
        logger.info("Audio stream subscriber added (total=%d)", len(self._subscribers))

    async def unsubscribe(self, ws: WebSocket) -> None:
        """WebSocket 클라이언트 구독 해제."""
        self._subscribers.discard(ws)

    async def _stream_loop(self) -> None:
        """청크 단위로 구독자에게 오디오 전송."""
        chunk_duration = self._chunk_size / self._sample_rate
        try:
            while not self._shutdown:
                if not self._playback.is_playing or self._audio is None:
                    await asyncio.sleep(0.02)
                    continue

                chunk = await self._next_chunk()
                if chunk is None:
                    self._playback.is_playing = False
                    await asyncio.sleep(0.02)
                    continue

                await self._broadcast_chunk(chunk)
                await asyncio.sleep(chunk_duration)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("AudioMixer stream loop error")

    async def _next_chunk(self) -> Optional[AudioChunk]:
        """재생 위치에서 다음 청크를 생성."""
        async with self._lock:
            return self._build_chunk()

    def _build_chunk(self) -> Optional[AudioChunk]:
        """락 내부에서 호출 — 다음 청크를 계산한다."""
        if self._audio is None:
            return None

        n = self._chunk_size
        pos = self._sample_pos
        end = min(pos + n, len(self._audio))

        if pos >= len(self._audio) and self._crossfade is None:
            return None

        current = self._audio[pos:end] if pos < len(self._audio) else np.zeros((0, self._channels), dtype=np.float32)

        # 크로스페이드 구간
        if self._crossfade is not None:
            cf = self._crossfade
            take = min(n, cf.remaining, len(current) if len(current) else n)
            if take <= 0:
                self._finish_crossfade()
                return self._build_chunk()

            if len(current) < take:
                pad = np.zeros((take - len(current), self._channels), dtype=np.float32)
                current = np.vstack([current, pad]) if len(current) else pad

            next_part = cf.next_audio[cf.next_pos:cf.next_pos + take]
            if len(next_part) < take:
                pad = np.zeros((take - len(next_part), self._channels), dtype=np.float32)
                next_part = np.vstack([next_part, pad]) if len(next_part) else pad

            progress = 1.0 - (cf.remaining / cf.total_samples)
            alpha = np.linspace(progress, min(1.0, progress + take / cf.total_samples), take, dtype=np.float32)
            alpha = alpha[:, np.newaxis]
            mixed = current[:take] * (1.0 - alpha) + next_part[:take] * alpha

            cf.remaining -= take
            cf.next_pos += take
            self._sample_pos += take

            if cf.remaining <= 0:
                self._finish_crossfade()

            data = self._apply_effects(mixed)
        else:
            if len(current) == 0:
                return None
            self._sample_pos = end
            data = self._apply_effects(current)

        self._playback.position_sec = self._sample_pos / self._sample_rate
        return AudioChunk(data=data, sample_rate=self._sample_rate, channels=self._channels)

    def _finish_crossfade(self) -> None:
        """크로스페이드 완료 후 다음 트랙으로 전환."""
        if self._crossfade is None:
            return
        cf = self._crossfade
        self._audio = cf.next_audio
        self._sample_pos = cf.next_pos
        self._playback.track_path = cf.next_path
        self._playback.duration_sec = len(self._audio) / self._sample_rate
        self._crossfade = None
        logger.info("Crossfade complete -> %s", cf.next_path.name)

    def _apply_effects(self, data: np.ndarray) -> np.ndarray:
        """페이드·볼륨·청크 게인 적용."""
        out = data.astype(np.float32, copy=True)
        n = len(out)

        if self._fade is not None and n > 0:
            fade = self._fade
            apply_n = min(n, fade.remaining)
            if apply_n > 0:
                t = np.linspace(
                    1.0 - fade.remaining / fade.total_samples,
                    1.0 - (fade.remaining - apply_n) / fade.total_samples,
                    apply_n,
                    dtype=np.float32,
                )
                gain = fade.start_gain + (fade.end_gain - fade.start_gain) * t
                out[:apply_n] *= gain[:, np.newaxis]
                fade.remaining -= apply_n
                if fade.remaining <= 0:
                    self._chunk_gain = fade.end_gain
                    if fade.kind == "out":
                        self._playback.is_playing = False
                    self._fade = None

        out *= self._chunk_gain * self._playback.volume
        return np.clip(out, -1.0, 1.0)

    async def _broadcast_chunk(self, chunk: AudioChunk) -> None:
        """모든 구독자에게 int16 PCM 전송."""
        if not self._subscribers:
            return

        pcm = (chunk.data * 32767.0).astype(np.int16)
        payload = pcm.tobytes()
        dead: Set[WebSocket] = set()

        for ws in self._subscribers:
            try:
                await ws.send_bytes(payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self._subscribers.discard(ws)

    # ── 파일 로드 (동기) ──────────────────────────────────────────

    def _load_audio_file(self, path: Path) -> np.ndarray:
        """wav/flac/mp3 파일을 float32 (samples, channels)로 로드."""
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            if AudioSegment is None:
                raise RuntimeError("pydub is required for MP3 files")
            return self._load_mp3(path)

        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            if AudioSegment is None:
                raise
            logger.warning("soundfile failed for %s, trying pydub: %s", path, exc)
            return self._load_mp3(path)

        data = self._to_target_channels(data)
        if sr != self._sample_rate:
            data = self._resample(data, sr, self._sample_rate)
        return np.clip(data, -1.0, 1.0).astype(np.float32)

    def _load_mp3(self, path: Path) -> np.ndarray:
        seg = AudioSegment.from_file(str(path))
        seg = seg.set_frame_rate(self._sample_rate).set_channels(self._channels)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        max_val = float(1 << (8 * seg.sample_width - 1))
        samples /= max_val
        return samples.reshape(-1, self._channels).astype(np.float32)

    def _to_target_channels(self, data: np.ndarray) -> np.ndarray:
        """채널 수를 self._channels에 맞춘다."""
        if data.ndim == 1:
            data = data[:, np.newaxis]
        ch = data.shape[1]
        if ch == self._channels:
            return data
        if ch == 1 and self._channels == 2:
            return np.repeat(data, 2, axis=1)
        if ch > self._channels:
            return data[:, : self._channels]
        # 2 -> 1 등: 평균
        return data.mean(axis=1, keepdims=True)

    def _resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return data
        if librosa is not None:
            resampled = [
                librosa.resample(data[:, c], orig_sr=orig_sr, target_sr=target_sr)
                for c in range(data.shape[1])
            ]
            min_len = min(len(c) for c in resampled)
            return np.stack([c[:min_len] for c in resampled], axis=1).astype(np.float32)

        # librosa 없을 때 선형 보간 폴백
        ratio = target_sr / orig_sr
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0, 1, len(data), dtype=np.float32)
        x_new = np.linspace(0, 1, new_len, dtype=np.float32)
        out = np.zeros((new_len, data.shape[1]), dtype=np.float32)
        for c in range(data.shape[1]):
            out[:, c] = np.interp(x_new, x_old, data[:, c])
        return out

    # ── 상태 조회 ─────────────────────────────────────────────────

    def get_playback_state(self) -> dict:
        """현재 재생 상태 반환."""
        return {
            "track": str(self._playback.track_path) if self._playback.track_path else None,
            "position": self._playback.position_sec,
            "duration": self._playback.duration_sec,
            "volume": self._playback.volume,
            "is_playing": self._playback.is_playing,
            "subscribers": len(self._subscribers),
        }
