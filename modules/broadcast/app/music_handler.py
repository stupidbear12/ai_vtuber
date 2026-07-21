# -*- coding: utf-8 -*-
"""
app/music_handler.py -- 음악 명령 처리 믹스인

BroadcastChatManager에서 분리된 YouTube Music 채팅 명령 및 Auto-DJ 관련 메서드.
"""

import asyncio
import base64
import logging
import os
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── YouTube Music 채팅 명령 상수 ──────────────────────────────
MUSIC_PLAY_PREFIXES = ("!play ", "!음악 ", "!노래 ", "!신청 ")
# 자연어 음악 요청 패턴: "곡이름 틀어줘", "곡이름 재생해줘" 등
MUSIC_NATURAL_SUFFIXES = ("틀어줘", "틀어주세요", "틀어", "재생해줘", "재생해주세요", "재생해", "들려줘", "들려주세요", "불러줘", "불러주세요")
MUSIC_SKIP_COMMANDS = frozenset({"!skip", "!스킵", "!다음", "!next"})
MUSIC_STOP_COMMANDS = frozenset({"!stop", "!정지", "!음악정지"})
MUSIC_PAUSE_COMMANDS = frozenset({"!pause", "!일시정지"})
MUSIC_RESUME_COMMANDS = frozenset({"!resume", "!재개"})
MUSIC_NOWPLAYING_COMMANDS = frozenset({"!현재곡", "!nowplaying", "!np", "!뭐틀어"})
MUSIC_QUEUE_COMMANDS = frozenset({"!queue", "!대기열", "!큐"})

# ── Auto-DJ 설정 ─────────────────────────────────────────────
AUTO_DJ_ENABLED = os.environ.get("AUTO_DJ_ENABLED", "1") == "1"
AUTO_DJ_IDLE_SEC = int(os.environ.get("AUTO_DJ_IDLE_SEC", "180"))
AUTO_DJ_CHECK_INTERVAL = int(os.environ.get("AUTO_DJ_CHECK_INTERVAL", "30"))


class MusicHandlerMixin:
    """음악 명령 처리 메서드 믹스인.

    BroadcastChatManager가 이 믹스인을 상속하여 사용한다.
    self._music_commands_enabled, self._music_url, self._tts_lock,
    self._voice_enabled, self._voice_url, self._live2d_url, self._chat_url 등
    BroadcastChatManager.__init__에서 초기화된 속성에 의존한다.
    """

    def _is_music_command(self, msg) -> bool:
        """YouTube Music 채팅 명령 여부.

        접두사 명령(!노래, !play 등)과 자연어 요청(곡이름 틀어줘 등) 모두 감지.
        """
        if not self._music_commands_enabled:
            return False
        text = msg.message.strip()
        text_lower = text.lower()
        for prefix in MUSIC_PLAY_PREFIXES:
            if text_lower.startswith(prefix.lower()):
                return True
        # 자연어 음악 요청: "곡이름 틀어줘" 패턴
        for suffix in MUSIC_NATURAL_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix) + 1:
                return True
        all_cmds = (
            MUSIC_SKIP_COMMANDS
            | MUSIC_STOP_COMMANDS
            | MUSIC_PAUSE_COMMANDS
            | MUSIC_RESUME_COMMANDS
            | MUSIC_NOWPLAYING_COMMANDS
            | MUSIC_QUEUE_COMMANDS
        )
        return text_lower in all_cmds

    async def _announce_music(self, text: str, emotion: str = "happy") -> None:
        """신청곡 관련 시온 반응을 TTS + Live2D로 출력.

        _tts_lock으로 직렬화하여 다른 TTS와 겹치지 않도록 보장.
        """
        async with self._tts_lock:
            audio_base64: Optional[str] = None
            if self._voice_enabled:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self._voice_url}/voice/tts",
                            json={"text": text, "emotion": emotion},
                            timeout=aiohttp.ClientTimeout(total=30.0),
                        ) as resp:
                            if resp.status == 200:
                                audio_bytes = await resp.read()
                                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                except Exception as e:
                    logger.warning("[MusicCmd] TTS 실패 (무시): %s", e)

            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self._live2d_url}/live2d/emotion",
                        json={"emotion": emotion},
                        timeout=aiohttp.ClientTimeout(total=3.0),
                    )
                    broadcast_payload: dict = {
                        "cmd": "speak",
                        "text": text,
                        "emotion": emotion,
                        "author": "시온",
                        "platform": "music",
                        "is_donation": False,
                    }
                    if audio_base64:
                        broadcast_payload["audio_base64"] = audio_base64
                    await session.post(
                        f"{self._live2d_url}/live2d/broadcast",
                        json=broadcast_payload,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    )
            except Exception as e:
                logger.warning("[MusicCmd] Live2D 전송 실패 (무시): %s", e)

    async def _handle_music_command(self, msg) -> None:
        """채팅 명령으로 YouTube Music 제어 + 시온 반응."""
        text = msg.message.strip()
        text_lower = text.lower()

        # ── 자연어 신청곡 ("곡이름 틀어줘" 패턴) ─────────────────
        for suffix in MUSIC_NATURAL_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix) + 1:
                query = text[: -len(suffix)].strip()
                if query:
                    # 접두사 방식과 동일한 재생 로직으로 전달
                    logger.info("[MusicCmd] 자연어 신청: '%s' → query='%s'", text[:40], query)
                    # 아래 prefix 루프의 재생 로직을 재사용하기 위해 text를 변환
                    text = f"!노래 {query}"
                    text_lower = text.lower()
                    break

        # ── 신청곡 재생 ──────────────────────────────────────────
        for prefix in MUSIC_PLAY_PREFIXES:
            if text_lower.startswith(prefix.lower()):
                query = text[len(prefix):].strip()
                if not query:
                    await self._announce_music(
                        f"{msg.author}님, 듣고 싶은 곡 이름을 같이 적어주세요!",
                        "confused",
                    )
                    return
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self._music_url}/ymusic/play",
                            json={"query": query, "requester": msg.author},
                            timeout=aiohttp.ClientTimeout(total=120.0),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                track = data.get("track") or {}
                                title = track.get("title", query)
                                artist = track.get("artist", "")
                                now_playing = data.get("now_playing")
                                # 큐에 추가된 건지 바로 재생인지 구분
                                if now_playing and now_playing.get("track", {}).get("video_id") != track.get("video_id"):
                                    announce = (
                                        f"{msg.author}님이 신청하신 {title}"
                                        f"{' - ' + artist if artist else ''}, "
                                        f"대기열에 추가했어요! 곧 틀어드릴게요~"
                                    )
                                else:
                                    announce = (
                                        f"{msg.author}님 신청곡! "
                                        f"{title}"
                                        f"{' - ' + artist if artist else ''} "
                                        f"바로 틀어드릴게요~"
                                    )
                                await self._announce_music(announce, "excited")
                                logger.info(
                                    "[MusicCmd] 재생: %s — %s (by %s)",
                                    artist or "?", title, msg.author,
                                )
                            else:
                                body = await resp.text()
                                logger.warning(
                                    "[MusicCmd] 재생 실패 HTTP %s: %s",
                                    resp.status, body[:200],
                                )
                                await self._announce_music(
                                    f"앗, {query} 검색에 실패했어요... 다시 한번 시도해주세요!",
                                    "sad",
                                )
                except Exception as exc:
                    logger.warning("[MusicCmd] ai_music 연결 실패: %s", exc)
                    await self._announce_music(
                        "음악 서버에 연결할 수 없어요... 잠시 후 다시 시도해주세요!",
                        "sad",
                    )
                return

        # ── 현재곡 정보 ──────────────────────────────────────────
        if text_lower in MUSIC_NOWPLAYING_COMMANDS:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._music_url}/ymusic/now-playing",
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            np = data.get("now_playing")
                            if np and np.get("track"):
                                t = np["track"]
                                title = t.get("title", "알 수 없는 곡")
                                artist = t.get("artist", "")
                                requester = t.get("requester", "")
                                announce = f"지금 듣고 있는 곡은 {title}"
                                if artist:
                                    announce += f" - {artist}"
                                if requester:
                                    announce += f"! {requester}님이 신청해주셨어요~"
                                else:
                                    announce += "!"
                                await self._announce_music(announce, "happy")
                            else:
                                await self._announce_music(
                                    "지금은 재생 중인 곡이 없어요! !노래 로 신청해주세요~",
                                    "calm",
                                )
            except Exception as exc:
                logger.warning("[MusicCmd] now-playing 조회 실패: %s", exc)
            return

        # ── 대기열 정보 ──────────────────────────────────────────
        if text_lower in MUSIC_QUEUE_COMMANDS:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._music_url}/ymusic/now-playing",
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            queue_list = data.get("queue") or []
                            if queue_list:
                                names = [q.get("title", "?") for q in queue_list[:5]]
                                announce = f"대기열에 {len(queue_list)}곡이 있어요! " + ", ".join(names)
                                if len(queue_list) > 5:
                                    announce += f" 외 {len(queue_list) - 5}곡"
                            else:
                                announce = "대기열이 비어있어요~ !노래 로 신청해주세요!"
                            await self._announce_music(announce, "calm")
            except Exception as exc:
                logger.warning("[MusicCmd] queue 조회 실패: %s", exc)
            return

        # ── 스킵/정지/일시정지/재개 ──────────────────────────────
        endpoint = None
        announce_text = None
        if text_lower in MUSIC_SKIP_COMMANDS:
            endpoint = "/ymusic/skip"
            announce_text = f"{msg.author}님 요청으로 다음 곡으로 넘길게요!"
        elif text_lower in MUSIC_STOP_COMMANDS:
            endpoint = "/ymusic/stop"
            announce_text = "음악 재생을 멈출게요~"
        elif text_lower in MUSIC_PAUSE_COMMANDS:
            endpoint = "/ymusic/pause"
            announce_text = "잠깐 일시정지할게요!"
        elif text_lower in MUSIC_RESUME_COMMANDS:
            endpoint = "/ymusic/resume"
            announce_text = "다시 재생할게요!"

        if not endpoint:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._music_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status == 200:
                        logger.info("[MusicCmd] %s by %s", endpoint, msg.author)
                        if announce_text:
                            await self._announce_music(announce_text, "happy")
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[MusicCmd] %s 실패 HTTP %s: %s",
                            endpoint, resp.status, body[:200],
                        )
        except Exception as exc:
            logger.warning("[MusicCmd] ai_music 연결 실패: %s", exc)

    async def _handle_music_action(self, query: str, requester: str) -> None:
        """LLM이 감지한 자연어 음악 요청([액션:play_music:검색어])을 처리.

        재생 성공 시 신청곡 정보를 TTS로 안내한다.
        Auto-DJ 타이머도 리셋한다.

        Args:
            query: LLM이 생성한 YouTube Music 검색어
            requester: 요청한 시청자 닉네임
        """
        if not self._music_commands_enabled or not query.strip():
            return
        # Auto-DJ 타이머 리셋
        if hasattr(self, '_last_music_time'):
            self._last_music_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._music_url}/ymusic/play",
                    json={"query": query, "requester": requester},
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        track = data.get("track") or {}
                        title = track.get("title", query)
                        artist = track.get("artist", "")
                        logger.info(
                            "[MusicAction] 재생: %s — %s (query=%r, by %s)",
                            artist or "?", title, query, requester,
                        )
                        # 신청곡 TTS 안내
                        announce = f"{requester}님이 신청하신 {title}"
                        if artist:
                            announce += f" - {artist}"
                        await self._announce_music(announce, "excited")
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[MusicAction] 재생 실패 HTTP %s (query=%r): %s",
                            resp.status, query, body[:200],
                        )
        except Exception as exc:
            logger.warning("[MusicAction] ai_music 연결 실패 (query=%r): %s", query, exc)

    # ── Auto-DJ ──────────────────────────────────────────────────

    async def _auto_dj_loop(self) -> None:
        """Auto-DJ 루프 -- 음악이 없으면 시온의 대화 맥락으로 자동 선곡.

        동작 흐름:
          1. AUTO_DJ_CHECK_INTERVAL마다 음악 모듈 상태 체크
          2. 음악이 재생 중이면 _last_music_time 갱신
          3. AUTO_DJ_IDLE_SEC 동안 음악이 없으면 LLM에 선곡 요청
          4. 추천곡을 자동 재생 + TTS 안내
        """
        self._last_music_time = time.time()
        logger.info("[AutoDJ] 루프 시작 (idle=%ds, check=%ds)",
                    AUTO_DJ_IDLE_SEC, AUTO_DJ_CHECK_INTERVAL)

        while self._running:
            try:
                await asyncio.sleep(AUTO_DJ_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            # 음악 상태 체크
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._music_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
            except Exception:
                continue

            now_playing = data.get("now_playing")
            queue_size = data.get("queue_size", 0)

            if now_playing or queue_size > 0:
                self._last_music_time = time.time()
                continue

            # 음악 없는 시간 체크
            idle_sec = time.time() - self._last_music_time
            if idle_sec < AUTO_DJ_IDLE_SEC:
                continue

            # ── LLM에 선곡 요청 ──────────────────────────────
            logger.info("[AutoDJ] 음악 idle %.0fs → 자동 선곡 요청", idle_sec)

            try:
                time_period = self._get_time_period()
                dj_prompt = (
                    f"[Auto-DJ] 지금 방송에서 음악이 안 나오고 있어요. "
                    f"현재 시간대: {time_period}. "
                    f"지금 분위기에 어울리는 한국어 노래 하나만 추천해주세요. "
                    f"'아티스트 - 제목' 형식으로 노래 이름만 답해주세요. "
                    f"다른 말은 하지 마세요."
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._chat_url}/chat",
                        json={
                            "message": dj_prompt,
                            "mode": "broadcast",
                            "context": f"[Auto-DJ] {time_period} 시간대. 분위기에 맞는 노래 추천 요청.",
                            "viewer_name": "시온",
                            "is_donation": False,
                        },
                        timeout=aiohttp.ClientTimeout(total=60.0),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("[AutoDJ] ai_chat 오류: HTTP %s", resp.status)
                            self._last_music_time = time.time()
                            continue
                        chat_data = await resp.json()

                reply = (chat_data.get("reply") or "").strip()
                if not reply:
                    logger.warning("[AutoDJ] LLM 응답 비어있음")
                    self._last_music_time = time.time()
                    continue

                # LLM 응답에서 노래 검색어 추출 (감정 태그 등 제거)
                import re
                search_query = re.sub(r"\[감정:[^\]]*\]", "", reply).strip()
                search_query = search_query.strip('"\'').strip()

                if not search_query or len(search_query) < 2:
                    logger.warning("[AutoDJ] 유효한 검색어 없음: %r", reply)
                    self._last_music_time = time.time()
                    continue

                logger.info("[AutoDJ] 선곡: %s", search_query)

                # 음악 재생
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._music_url}/ymusic/play",
                        json={"query": search_query, "requester": "시온 Auto-DJ"},
                        timeout=aiohttp.ClientTimeout(total=120.0),
                    ) as resp:
                        if resp.status == 200:
                            play_data = await resp.json()
                            track = play_data.get("track") or {}
                            title = track.get("title", search_query)
                            artist = track.get("artist", "")
                            song_info = f"{title} - {artist}" if artist else title
                            announce = f"이 노래 한번 들어볼까요? {song_info}"
                            await self._announce_music(announce, "happy")
                            logger.info("[AutoDJ] 재생 성공: %s", song_info)
                        else:
                            body = await resp.text()
                            logger.warning("[AutoDJ] 재생 실패 HTTP %s: %s",
                                           resp.status, body[:200])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[AutoDJ] 오류: %s", e, exc_info=True)
            finally:
                self._last_music_time = time.time()

    async def _auto_dj_safe(self) -> None:
        """Auto-DJ 래퍼 -- crash 시 자동 재시작."""
        while self._running:
            try:
                await self._auto_dj_loop()
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[AutoDJ] crash 감지 → 5초 후 재시작: %s", e, exc_info=True)
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break
        logger.info("[AutoDJ] 루프 종료")
