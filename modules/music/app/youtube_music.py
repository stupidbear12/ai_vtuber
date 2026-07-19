# -*- coding: utf-8 -*-
"""
YouTube Music 재생 — ytmusicapi 검색 + yt-dlp 오디오 다운로드 + AudioMixer 출력

OBS 연동:
  ws://localhost:8005/music/stream 에 PCM 스트림 구독
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

from .audio_mixer import AudioMixer

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# 최대 트랙 길이 (초) — 이 값을 초과하는 영상은 재생/큐 거부
MAX_TRACK_DURATION_SEC = float(os.environ.get("YTMUSIC_MAX_DURATION", "900"))  # 기본 15분


@dataclass
class YTMTrack:
    """YouTube Music 트랙 메타데이터."""

    video_id: str
    title: str
    artist: str = ""
    duration_sec: Optional[float] = None
    file_path: Optional[Path] = None
    requester: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "artist": self.artist,
            "duration_sec": self.duration_sec,
            "file_path": str(self.file_path) if self.file_path else None,
            "requester": self.requester,
        }


def _parse_duration(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_ytmusic_client():
    try:
        from ytmusicapi import YTMusic
    except ImportError:
        return None

    headers_path = os.environ.get("YTMUSIC_HEADERS_PATH", "").strip()
    if headers_path and Path(headers_path).exists():
        return YTMusic(headers_path)
    return YTMusic()


class YouTubeMusicPlayer:
    """YouTube Music 검색·다운로드·재생 큐."""

    def __init__(self, mixer: AudioMixer):
        self._mixer = mixer
        root = Path(__file__).resolve().parents[3]
        default_cache = root / "cache" / "youtube_music"
        self._cache_dir = Path(os.environ.get("YTMUSIC_CACHE_DIR", str(default_cache)))
        self._use_music_url = os.environ.get("YTMUSIC_USE_MUSIC_URL", "1") == "1"

        self._queue: Deque[YTMTrack] = deque()
        self._now_playing: Optional[YTMTrack] = None
        self._started_at: Optional[datetime] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._advancing = False

    # ── 라이프사이클 ──────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("YouTubeMusicPlayer started (cache=%s)", self._cache_dir)

    async def stop(self) -> None:
        self._running = False
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        self._watch_task = None
        self._queue.clear()
        logger.info("YouTubeMusicPlayer stopped")

    # ── 검색 ──────────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 10) -> List[dict]:
        query = query.strip()
        if not query:
            return []

        if _VIDEO_ID_RE.match(query):
            track = await self.resolve_video(query)
            return [track.to_dict()]

        client = _get_ytmusic_client()
        if client is not None:
            try:
                raw = await asyncio.to_thread(
                    client.search, query, filter="songs", limit=limit
                )
                results = [self._from_ytm_result(item) for item in raw]
                results = [r for r in results if r.get("video_id")]
                # 최대 길이 초과 트랙 필터링
                results = [
                    r for r in results
                    if not r.get("duration_sec") or r["duration_sec"] <= MAX_TRACK_DURATION_SEC
                ]
                if results:
                    return results
            except Exception as exc:
                logger.warning("ytmusicapi search failed, fallback to yt-dlp: %s", exc)

        raw_results = await asyncio.to_thread(self._search_ytdlp, query, limit)
        # 최대 길이 초과 트랙 필터링
        return [
            r for r in raw_results
            if not r.get("duration_sec") or r["duration_sec"] <= MAX_TRACK_DURATION_SEC
        ]

    async def resolve_video(self, video_id: str) -> YTMTrack:
        video_id = video_id.strip()
        if not _VIDEO_ID_RE.match(video_id):
            raise ValueError(f"잘못된 video_id: {video_id}")

        meta = await asyncio.to_thread(self._fetch_metadata, video_id)
        return YTMTrack(
            video_id=video_id,
            title=meta.get("title") or video_id,
            artist=meta.get("artist") or "",
            duration_sec=_parse_duration(meta.get("duration_sec")),
        )

    # ── 재생 제어 ─────────────────────────────────────────────────

    async def play_query(self, query: str, requester: Optional[str] = None) -> YTMTrack:
        results = await self.search(query, limit=1)
        if not results:
            raise ValueError(f"검색 결과 없음: {query}")
        item = results[0]
        return await self.play_video(
            video_id=item["video_id"],
            title=item.get("title") or query,
            artist=item.get("artist") or "",
            duration_sec=_parse_duration(item.get("duration_sec")),
            requester=requester,
        )

    async def play_video(
        self,
        video_id: str,
        title: str = "",
        artist: str = "",
        duration_sec: Optional[float] = None,
        requester: Optional[str] = None,
    ) -> YTMTrack:
        # 최대 길이 초과 시 거부
        if duration_sec and duration_sec > MAX_TRACK_DURATION_SEC:
            mins = int(MAX_TRACK_DURATION_SEC // 60)
            raise ValueError(
                f"트랙이 너무 길어요 ({int(duration_sec//60)}분). 최대 {mins}분까지 가능합니다: {title}"
            )
        async with self._lock:
            state = self._mixer.get_playback_state()
            if self._now_playing and state.get("is_playing"):
                pending = YTMTrack(
                    video_id=video_id,
                    title=title or video_id,
                    artist=artist,
                    duration_sec=duration_sec,
                    requester=requester,
                )
                self._queue.append(pending)
                logger.info("YTMusic queued: %s (by %s)", pending.title, requester or "-")
                return pending
            return await self._start_playback(
                video_id, title, artist, duration_sec, requester
            )

    async def enqueue(
        self,
        video_id: str,
        title: str = "",
        artist: str = "",
        requester: Optional[str] = None,
    ) -> YTMTrack:
        return await self.play_video(
            video_id=video_id,
            title=title,
            artist=artist,
            requester=requester,
        )

    async def skip(self) -> Optional[YTMTrack]:
        async with self._lock:
            await self._mixer.stop()
            self._now_playing = None
            self._started_at = None
            if not self._queue:
                return None
            next_track = self._queue.popleft()
            return await self._start_playback(
                next_track.video_id,
                next_track.title,
                next_track.artist,
                next_track.duration_sec,
                next_track.requester,
            )

    async def pause(self) -> None:
        await self._mixer.pause()

    async def resume(self) -> None:
        await self._mixer.play()

    async def stop(self) -> None:
        async with self._lock:
            self._queue.clear()
            self._now_playing = None
            self._started_at = None
            await self._mixer.stop()

    def now_playing_info(self) -> Optional[dict]:
        if not self._now_playing:
            return None
        playback = self._mixer.get_playback_state()
        return {
            "source": "youtube_music",
            "track": self._now_playing.to_dict(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "playback": playback,
            "queue_size": len(self._queue),
        }

    def queue_list(self) -> List[dict]:
        return [t.to_dict() for t in self._queue]

    # ── 내부 ──────────────────────────────────────────────────────

    async def _start_playback(
        self,
        video_id: str,
        title: str,
        artist: str,
        duration_sec: Optional[float],
        requester: Optional[str],
    ) -> YTMTrack:
        if not title:
            meta = await self.resolve_video(video_id)
            title = meta.title
            artist = artist or meta.artist
            duration_sec = duration_sec or meta.duration_sec

        file_path = await self.download_audio(video_id)
        track = YTMTrack(
            video_id=video_id,
            title=title,
            artist=artist,
            duration_sec=duration_sec,
            file_path=file_path,
            requester=requester,
        )

        actual_duration = await self._mixer.load_track(file_path)
        if not track.duration_sec:
            track.duration_sec = actual_duration
        await self._mixer.play()

        self._now_playing = track
        self._started_at = datetime.now()
        logger.info(
            "YTMusic playing: %s — %s (%.0fs, by %s)",
            track.artist or "?",
            track.title,
            track.duration_sec or 0,
            requester or "-",
        )
        return track

    async def _play_next_from_queue(self) -> None:
        if self._advancing or not self._running:
            return
        self._advancing = True
        try:
            async with self._lock:
                if not self._queue:
                    self._now_playing = None
                    self._started_at = None
                    return
                next_track = self._queue.popleft()
                await self._start_playback(
                    next_track.video_id,
                    next_track.title,
                    next_track.artist,
                    next_track.duration_sec,
                    next_track.requester,
                )
        finally:
            self._advancing = False

    async def _watch_loop(self) -> None:
        """트랙 종료 감지 후 큐에서 다음 곡 재생."""
        try:
            while self._running:
                if self._now_playing:
                    state = self._mixer.get_playback_state()
                    duration = float(state.get("duration") or 0.0)
                    position = float(state.get("position") or 0.0)
                    is_playing = bool(state.get("is_playing"))

                    if duration > 0 and not is_playing and position >= max(0.0, duration - 0.3):
                        await self._play_next_from_queue()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("YTMusic watch loop error")

    async def download_audio(self, video_id: str) -> Path:
        return await asyncio.to_thread(self._download_audio_sync, video_id)

    def _download_audio_sync(self, video_id: str) -> Path:
        for ext in ("mp3", "m4a", "opus", "webm"):
            existing = self._cache_dir / f"{video_id}.{ext}"
            if existing.exists() and existing.stat().st_size > 0:
                return existing

        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("yt-dlp 설치 필요: pip install yt-dlp") from exc

        if self._use_music_url:
            url = f"https://music.youtube.com/watch?v={video_id}"
        else:
            url = f"https://www.youtube.com/watch?v={video_id}"

        out_template = str(self._cache_dir / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        cookies = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
        if cookies and Path(cookies).exists():
            ydl_opts["cookiefile"] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for ext in ("mp3", "m4a", "opus", "webm"):
            path = self._cache_dir / f"{video_id}.{ext}"
            if path.exists() and path.stat().st_size > 0:
                return path

        raise FileNotFoundError(f"다운로드 실패: {video_id}")

    def _fetch_metadata(self, video_id: str) -> dict:
        try:
            import yt_dlp
        except ImportError:
            return {"title": video_id, "artist": "", "duration_sec": None}

        url = (
            f"https://music.youtube.com/watch?v={video_id}"
            if self._use_music_url
            else f"https://www.youtube.com/watch?v={video_id}"
        )
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        cookies = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
        if cookies and Path(cookies).exists():
            ydl_opts["cookiefile"] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        artist = ""
        if isinstance(info.get("artist"), str):
            artist = info["artist"]
        elif isinstance(info.get("creator"), str):
            artist = info["creator"]
        elif info.get("uploader"):
            artist = str(info["uploader"])

        return {
            "title": info.get("title") or video_id,
            "artist": artist,
            "duration_sec": info.get("duration"),
        }

    def _search_ytdlp(self, query: str, limit: int) -> List[dict]:
        try:
            import yt_dlp
        except ImportError:
            return []

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)

        entries = info.get("entries") or []
        results = []
        for entry in entries:
            if not entry:
                continue
            video_id = entry.get("id") or ""
            if not _VIDEO_ID_RE.match(video_id):
                continue
            results.append(
                {
                    "video_id": video_id,
                    "title": entry.get("title") or video_id,
                    "artist": entry.get("uploader") or "",
                    "duration_sec": _parse_duration(entry.get("duration")),
                }
            )
        return results

    @staticmethod
    def _from_ytm_result(item: dict) -> dict:
        video_id = item.get("videoId") or ""
        artists = item.get("artists") or []
        artist = ", ".join(a.get("name", "") for a in artists if a.get("name"))
        duration_text = item.get("duration") or ""
        duration_sec = None
        if duration_text and ":" in duration_text:
            parts = duration_text.split(":")
            try:
                if len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except ValueError:
                duration_sec = None

        return {
            "video_id": video_id,
            "title": item.get("title") or video_id,
            "artist": artist,
            "duration_sec": duration_sec,
        }
