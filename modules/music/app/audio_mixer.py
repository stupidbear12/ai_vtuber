# -*- coding: utf-8 -*-
"""
AudioMixer — 실시간 오디오 재생 및 스트림 출력

역할:
  - 오디오 파일 로드·재생·일시정지·정지
  - 실시간 PCM 스트림 생성 (WebSocket 클라이언트용)
  - 시스템 오디오 출력 (sounddevice, OBS 데스크탑 오디오 캡처용)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from pathlib import Path
from typing import Optional, Set

import numpy as np
import soundfile as sf
from fastapi import WebSocket

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover
    AudioSegment = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None


class AudioMixer:
    """오디오 재생 엔진.

    WebSocket 구독자에게 int16 PCM(stereo interleaved)을 전송하고,
    시스템 오디오로도 동시 출력한다.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 2, chunk_size: int = 4096):
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._subscribers: Set[WebSocket] = set()
        self._stream_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._lock = asyncio.Lock()

        # 재생 상태
        self._audio: Optional[np.ndarray] = None
        self._sample_pos: int = 0
        self._track_path: Optional[Path] = None
        self._duration_sec: float = 0.0
        self._position_sec: float = 0.0
        self._volume: float = 1.0
        self._is_playing: bool = False

        # 시스템 오디오 출력 (sounddevice callback 방식)
        self._sd_stream: Optional[object] = None
        self._sd_buffer: Optional[queue.Queue] = None
        self._use_system_audio = os.environ.get("MUSIC_SYSTEM_AUDIO", "1") == "1"

    # ── 라이프사이클 ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """믹서 초기화 및 스트림 루프 시작."""
        self._shutdown = False
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_loop())

        if self._use_system_audio and sd is not None:
            try:
                self._start_system_audio()
            except Exception as exc:
                logger.warning("System audio output failed: %s", exc)

        logger.info(
            "AudioMixer initialized (sr=%s, ch=%s, chunk=%s, system_audio=%s)",
            self._sample_rate, self._channels, self._chunk_size,
            self._sd_stream is not None,
        )

    async def shutdown(self) -> None:
        """믹서 종료."""
        self._shutdown = True
        self._is_playing = False
        self._stop_system_audio()

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self._stream_task = None
        self._subscribers.clear()
        self._audio = None
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
            self._track_path = path
            self._duration_sec = len(audio) / self._sample_rate
            self._position_sec = 0.0

        logger.info("Loaded track: %s (%.1fs)", path.name, self._duration_sec)
        return self._duration_sec

    async def play(self) -> None:
        """현재 로드된 트랙 재생 시작."""
        if self._audio is None:
            raise RuntimeError("No track loaded. Call load_track() first.")
        self._is_playing = True

    async def pause(self) -> None:
        """재생 일시정지."""
        self._is_playing = False

    async def stop(self) -> None:
        """재생 정지 및 위치 초기화."""
        async with self._lock:
            self._is_playing = False
            self._sample_pos = 0
            self._position_sec = 0.0

    def set_volume(self, volume: float) -> None:
        """볼륨 설정 (0.0~1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    # ── WebSocket 스트리밍 ────────────────────────────────────────

    async def subscribe(self, ws: WebSocket) -> None:
        """WebSocket 클라이언트를 오디오 스트림 구독자로 등록."""
        self._subscribers.add(ws)
        logger.info("Audio stream subscriber added (total=%d)", len(self._subscribers))

    async def unsubscribe(self, ws: WebSocket) -> None:
        """WebSocket 클라이언트 구독 해제."""
        self._subscribers.discard(ws)

    async def _stream_loop(self) -> None:
        """청크 단위로 오디오 전송."""
        chunk_duration = self._chunk_size / self._sample_rate
        try:
            while not self._shutdown:
                if not self._is_playing or self._audio is None:
                    await asyncio.sleep(0.02)
                    continue

                chunk_data = await self._next_chunk()
                if chunk_data is None:
                    self._is_playing = False
                    await asyncio.sleep(0.02)
                    continue

                await self._broadcast_chunk(chunk_data)
                await asyncio.sleep(chunk_duration)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("AudioMixer stream loop error")

    async def _next_chunk(self) -> Optional[np.ndarray]:
        """재생 위치에서 다음 청크를 생성."""
        async with self._lock:
            if self._audio is None:
                return None

            pos = self._sample_pos
            end = min(pos + self._chunk_size, len(self._audio))

            if pos >= len(self._audio):
                return None

            data = self._audio[pos:end].astype(np.float32, copy=True)
            data *= self._volume
            data = np.clip(data, -1.0, 1.0)

            self._sample_pos = end
            self._position_sec = end / self._sample_rate
            return data

    async def _broadcast_chunk(self, data: np.ndarray) -> None:
        """시스템 오디오 출력 + WebSocket 구독자에게 int16 PCM 전송."""
        # 시스템 오디오 출력 (논블로킹 큐 push)
        if self._sd_buffer is not None:
            try:
                self._sd_buffer.put_nowait(data.copy())
            except queue.Full:
                try:
                    self._sd_buffer.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._sd_buffer.put_nowait(data.copy())
                except queue.Full:
                    pass

        # WebSocket 구독자 전송
        if not self._subscribers:
            return

        pcm = (data * 32767.0).astype(np.int16)
        payload = pcm.tobytes()
        dead: Set[WebSocket] = set()

        for ws in self._subscribers:
            try:
                await ws.send_bytes(payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self._subscribers.discard(ws)

    # ── 시스템 오디오 출력 ─────────────────────────────────────────

    def _sd_callback(self, outdata, frames, time_info, status):
        """sounddevice 콜백 — 별도 스레드에서 실행, 논블로킹."""
        if self._sd_buffer is None:
            outdata.fill(0)
            return
        try:
            data = self._sd_buffer.get_nowait()
            n = min(len(data), frames)
            outdata[:n] = data[:n]
            if n < frames:
                outdata[n:] = 0
        except queue.Empty:
            outdata.fill(0)

    def _start_system_audio(self) -> None:
        """sounddevice OutputStream(callback)으로 시스템 스피커 출력 시작."""
        if sd is None:
            return
        self._sd_buffer = queue.Queue(maxsize=30)
        self._sd_stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=self._sd_callback,
        )
        self._sd_stream.start()
        logger.info("System audio output started (sounddevice callback mode)")

    def _stop_system_audio(self) -> None:
        """시스템 오디오 출력 중지."""
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
            self._sd_buffer = None
            logger.info("System audio output stopped")

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
            return data[:, :self._channels]
        return data.mean(axis=1, keepdims=True)

    def _resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """리샘플링 (선형 보간)."""
        if orig_sr == target_sr:
            return data
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
            "track": str(self._track_path) if self._track_path else None,
            "position": self._position_sec,
            "duration": self._duration_sec,
            "volume": self._volume,
            "is_playing": self._is_playing,
            "subscribers": len(self._subscribers),
        }
