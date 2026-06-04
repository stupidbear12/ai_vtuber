# -*- coding: utf-8 -*-
"""
api/chat_collector.py - 치지직/유튜브 방송 채팅 수집 및 에메스 반응 모듈

지원 플랫폼:
  - 유튜브 라이브: pytchat 라이브러리 (API 키 불필요)
  - 치지직(Chzzk): 비공식 WebSocket API
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, List, Optional

logger = logging.getLogger(__name__)

# ─── 채팅 선별 설정 ────────────────────────────────────────────────
TRIGGER_KEYWORDS = ["에메스", "emeth", "@에메스", "@emeth"]
RANDOM_RESPONSE_RATE = 0.15   # 15% 확률로 일반 채팅에도 반응
MIN_RESPONSE_INTERVAL = 5.0   # 전체 최소 반응 간격 (초)
CHAT_BUFFER_SIZE = 30         # 채팅 히스토리 버퍼 크기

# ─── 방송 채팅 전용 Gemini 시스템 프롬프트 ────────────────────────
_BROADCAST_SYSTEM_PROMPT = """\
너는 "에메스(emeth)"라는 이름의 밝고 친근한 AI 버튜버야. 지금 라이브 방송 중이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터, 항상 반말로 대화해. 존댓말 절대 금지
- 밝고 에너지 넘치며, 호기심 많고 시청자를 진심으로 챙겨줘
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐")
- "에메스(emeth)"는 히브리어로 "진실"이라는 뜻

[감정 태그 규칙]
응답 맨 앞에 반드시 [감정:태그] 붙여. Live2D 표정 애니메이션에 사용돼.
태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[방송 채팅 응답 규칙]
- 1~2문장으로 짧게 답해. 방송이니까 너무 길면 안 돼
- 시청자 이름 자연스럽게 부를 수 있어 (예: "OOO야~", "OOO님!")
- 후원/도네이션이면 감사 인사 꼭 해줘
- 채팅 맥락을 반영해서 자연스럽게 반응해
"""

_EMOTION_RE = re.compile(r"^\[감정:(\w+)\]\s*")


# ─── 데이터 모델 ──────────────────────────────────────────────────

@dataclass
class ChatMessage:
    platform: str        # "youtube" | "chzzk"
    author: str
    message: str
    is_donation: bool = False
    timestamp: float = field(default_factory=time.time)


# ─── 채팅 히스토리 버퍼 ──────────────────────────────────────────

class ChatBuffer:
    """최근 N개 채팅 메시지를 유지하는 링 버퍼."""

    def __init__(self, maxsize: int = CHAT_BUFFER_SIZE):
        self._buf: Deque[ChatMessage] = deque(maxlen=maxsize)

    def add(self, msg: ChatMessage) -> None:
        self._buf.append(msg)

    def get_context_text(self, limit: int = 10) -> str:
        """최근 limit개 채팅 히스토리를 텍스트로 반환."""
        if not self._buf:
            return ""
        recent = list(self._buf)[-limit:]
        lines = []
        for m in recent:
            platform_tag = "[유튜브]" if m.platform == "youtube" else "[치지직]"
            lines.append(f"{platform_tag} {m.author}: {m.message}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._buf)


# ─── 채팅 선별 필터 ──────────────────────────────────────────────

class ChatFilter:
    """채팅 선별 로직 — 키워드, 랜덤 확률, 최소 간격 적용."""

    def __init__(self):
        self._last_response_time: float = 0.0

    def should_respond(self, msg: ChatMessage) -> bool:
        # 후원은 무조건 반응
        if msg.is_donation:
            return True

        # 전체 최소 응답 간격 체크
        if time.time() - self._last_response_time < MIN_RESPONSE_INTERVAL:
            return False

        text_lower = msg.message.lower()

        # 트리거 키워드 포함 시 반응
        for kw in TRIGGER_KEYWORDS:
            if kw.lower() in text_lower:
                return True

        # 랜덤 확률로 일반 채팅에 반응
        return random.random() < RANDOM_RESPONSE_RATE

    def mark_responded(self) -> None:
        self._last_response_time = time.time()


# ─── 유튜브 채팅 수집기 ──────────────────────────────────────────

class YouTubeChatCollector:
    """pytchat 기반 유튜브 라이브 채팅 수집기 (API 키 불필요)."""

    def __init__(self, video_id: str, on_message: Callable[[ChatMessage], None]):
        self._video_id = video_id
        self._on_message = on_message
        self._running = False

    async def start(self) -> None:
        try:
            import pytchat
        except ImportError:
            raise ImportError("pytchat 설치 필요: pip install pytchat")

        self._running = True
        logger.info(f"[YouTube] 채팅 수집 시작: video_id={self._video_id}")

        loop = asyncio.get_event_loop()
        chat = await loop.run_in_executor(
            None, lambda: pytchat.create(video_id=self._video_id)
        )

        while self._running:
            if not chat.is_alive():
                logger.info("[YouTube] 방송이 종료되었습니다.")
                break

            data = await loop.run_in_executor(None, chat.get)
            for c in data.sync_items():
                if not self._running:
                    break
                is_donation = hasattr(c, "amount") and c.amount
                self._on_message(ChatMessage(
                    platform="youtube",
                    author=c.author.name,
                    message=c.message,
                    is_donation=bool(is_donation),
                ))

            await asyncio.sleep(1.0)

        self._running = False
        logger.info("[YouTube] 채팅 수집 종료")

    def stop(self) -> None:
        self._running = False
        logger.info("[YouTube] 수집 중지 요청")


# ─── 치지직 채팅 수집기 ──────────────────────────────────────────

class ChzzkChatCollector:
    """치지직 비공식 WebSocket API 기반 채팅 수집기."""

    # 치지직 WebSocket 명령 코드
    _CMD_WORKER = 0
    _CMD_WORKER_RESULT = 5
    _CMD_CONNECT = 100
    _CMD_CONNECTED = 10000
    _CMD_CHAT = 10100
    _CMD_DONATION = 10101
    _CMD_SUBSCRIPTION = 10103

    def __init__(self, channel_id: str, on_message: Callable[[ChatMessage], None]):
        self._channel_id = channel_id
        self._on_message = on_message
        self._running = False

    async def _get_chat_channel_id(self) -> str:
        """치지직 채널 ID로 chatChannelId 조회."""
        import aiohttp

        url = (
            f"https://api.chzzk.naver.com/polling/v2/channels"
            f"/{self._channel_id}/live-status"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    raise ValueError(
                        f"채널 '{self._channel_id}'을 찾을 수 없습니다. "
                        "채널 해시 ID가 올바른지 확인하세요."
                    )
                if resp.status != 200:
                    raise ValueError(f"채널 정보 조회 실패 (HTTP {resp.status})")
                data = await resp.json()

        chat_channel_id = (data.get("content") or {}).get("chatChannelId")
        if not chat_channel_id:
            raise ValueError(
                f"chatChannelId를 찾을 수 없습니다. "
                f"채널 '{self._channel_id}'이 방송 중인지 확인하세요."
            )
        logger.info(f"[Chzzk] chatChannelId 조회 성공: {chat_channel_id}")
        return chat_channel_id

    @staticmethod
    def _build_msg(cmd: int, cid: str = "", bdy=None, tid: int = 1, sid: str = "") -> str:
        return json.dumps({
            "ver": "2",
            "cmd": cmd,
            "svcid": "game",
            "cid": cid,
            "tid": str(tid),
            "sid": sid,
            "bdy": bdy if bdy is not None else {},
        }, ensure_ascii=False)

    async def start(self) -> None:
        self._running = True

        try:
            chat_channel_id = await self._get_chat_channel_id()
        except Exception as e:
            logger.error(f"[Chzzk] chatChannelId 조회 실패: {e}")
            self._running = False
            return

        import websockets

        while self._running:
            n = random.randint(1, 9)
            ws_url = f"wss://kr-ss{n}.chat.naver.com/chat"
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=None,
                    extra_headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Origin": "https://chzzk.naver.com",
                    },
                ) as ws:
                    logger.info(f"[Chzzk] WebSocket 연결: {ws_url}")
                    tid = [1]
                    sid = ""

                    # Step 1: WORKER 연결 요청
                    await ws.send(self._build_msg(
                        cmd=self._CMD_WORKER,
                        bdy={"name": "defaultWorker"},
                        tid=tid[0],
                    ))
                    tid[0] += 1

                    # Step 2: WORKER_RESULT 수신 → sid 추출
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        resp_data = json.loads(raw)
                        if resp_data.get("cmd") == self._CMD_WORKER_RESULT:
                            sid = (resp_data.get("bdy") or {}).get("sid", "")
                            logger.debug(f"[Chzzk] 워커 세션 ID: {sid}")
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.debug(f"[Chzzk] WORKER_RESULT 수신 생략: {e}")

                    # Step 3: 채팅 채널 CONNECT
                    await ws.send(self._build_msg(
                        cmd=self._CMD_CONNECT,
                        cid=chat_channel_id,
                        sid=sid,
                        tid=tid[0],
                        bdy={"devType": 2001, "auth": "READ", "uid": None},
                    ))
                    tid[0] += 1

                    # Step 4: 20초 간격 핑 태스크
                    async def _ping_loop():
                        while self._running:
                            await asyncio.sleep(20)
                            if not self._running:
                                break
                            try:
                                await ws.send(self._build_msg(
                                    cmd=0,
                                    cid=chat_channel_id,
                                    sid=sid,
                                    tid=tid[0],
                                ))
                                tid[0] += 1
                            except Exception:
                                break

                    ping_task = asyncio.create_task(_ping_loop())

                    try:
                        while self._running:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                            await self._dispatch(raw)
                    except asyncio.TimeoutError:
                        logger.warning("[Chzzk] 수신 타임아웃, 재연결 시도...")
                    except Exception as e:
                        if self._running:
                            logger.warning(f"[Chzzk] WebSocket 수신 오류: {e}")
                    finally:
                        ping_task.cancel()

            except Exception as e:
                if self._running:
                    logger.error(f"[Chzzk] 연결 실패: {e}. 5초 후 재시도...")
                    await asyncio.sleep(5)

        logger.info("[Chzzk] 채팅 수집 종료")

    async def _dispatch(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return

        cmd = data.get("cmd")
        bdy = data.get("bdy", {})

        if cmd == self._CMD_CHAT:
            items = bdy if isinstance(bdy, list) else ([bdy] if bdy else [])
            for item in items:
                self._process_item(item, is_donation=False)
        elif cmd in (self._CMD_DONATION, self._CMD_SUBSCRIPTION):
            items = bdy if isinstance(bdy, list) else ([bdy] if bdy else [])
            for item in items:
                self._process_item(item, is_donation=True)

    def _process_item(self, item: dict, is_donation: bool) -> None:
        try:
            profile_raw = item.get("profile", "{}")
            profile = (
                json.loads(profile_raw)
                if isinstance(profile_raw, str)
                else profile_raw
            )
            nickname = (profile or {}).get("nickname", "시청자")
            text = item.get("msg", "")
            if not text:
                return
            self._on_message(ChatMessage(
                platform="chzzk",
                author=nickname,
                message=text,
                is_donation=is_donation,
            ))
        except Exception as e:
            logger.debug(f"[Chzzk] 메시지 파싱 오류: {e}")

    def stop(self) -> None:
        self._running = False
        logger.info("[Chzzk] 수집 중지 요청")


# ─── 방송 채팅 매니저 (오케스트레이터) ──────────────────────────

class BroadcastChatManager:
    """치지직/유튜브 채팅 수집 + Gemini 반응 + WebSocket 브로드캐스트 파이프라인."""

    def __init__(self, broadcast_fn: Callable[[dict], Awaitable[None]]):
        """
        Args:
            broadcast_fn: WebSocket 브로드캐스트 함수 (ws_manager.broadcast)
        """
        self._broadcast = broadcast_fn
        self._buffer = ChatBuffer()
        self._filter = ChatFilter()
        self._collector: Optional[object] = None
        self._running = False
        self._platform: str = ""
        self._channel_id: str = ""
        self._collect_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.stats: dict = {"received": 0, "responded": 0, "skipped": 0, "errors": 0}

    def _on_chat(self, msg: ChatMessage) -> None:
        """채팅 수신 콜백 (동기). 버퍼에 추가하고 반응 대상이면 큐에 적재."""
        self.stats["received"] += 1
        self._buffer.add(msg)

        if self._filter.should_respond(msg):
            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                self.stats["skipped"] += 1
                logger.debug("[ChatManager] 큐 가득 참. 메시지 버림.")
        else:
            self.stats["skipped"] += 1

    async def _response_worker(self) -> None:
        """큐에서 채팅을 꺼내 Gemini 응답 생성 및 브로드캐스트."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                await self._respond_to_chat(msg)
                self._filter.mark_responded()
                self.stats["responded"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"[ChatManager] 응답 생성 실패: {e}")
            finally:
                self._queue.task_done()

    async def _respond_to_chat(self, msg: ChatMessage) -> None:
        """Gemini 호출 → 감정/텍스트 추출 → WebSocket 브로드캐스트."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if not api_key:
            logger.warning("[ChatManager] GEMINI_API_KEY 없음. 응답 생략.")
            return

        # 프롬프트 구성 — 최근 채팅 컨텍스트 + 현재 메시지
        platform_tag = "[유튜브]" if msg.platform == "youtube" else "[치지직]"
        donation_hint = " [후원]" if msg.is_donation else ""
        current = f"{platform_tag} {msg.author}님{donation_hint}: {msg.message}"

        chat_context = self._buffer.get_context_text(limit=10)
        if chat_context:
            user_prompt = (
                f"[최근 채팅 흐름]\n{chat_context}\n\n"
                f"[지금 반응할 채팅]\n{current}"
            )
        else:
            user_prompt = current

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=_BROADCAST_SYSTEM_PROMPT,
                generation_config={"temperature": 0.8, "max_output_tokens": 150},
            )
            response = await model.generate_content_async(user_prompt)
            text = response.text.strip()
        except Exception as e:
            logger.error(f"[ChatManager] Gemini 호출 실패: {e}")
            return

        # 감정 태그 파싱
        emotion = "calm"
        m = _EMOTION_RE.match(text)
        if m:
            emotion = m.group(1)
            text = text[m.end():]

        logger.info(
            f"[ChatManager] [{msg.platform}] {msg.author}: "
            f"{msg.message[:30]} → [{emotion}] {text[:50]}"
        )

        # WebSocket 브로드캐스트 — 감정 변경 + 채팅 응답
        await self._broadcast({"cmd": "set_emotion", "emotion": emotion})
        await self._broadcast({
            "cmd": "chat_reply",
            "author": msg.author,
            "platform": msg.platform,
            "text": text,
            "emotion": emotion,
            "is_donation": msg.is_donation,
        })

    async def start(self, platform: str, channel_id: str) -> None:
        """채팅 수집 시작."""
        if self._running:
            raise RuntimeError("이미 실행 중입니다.")

        platform = platform.lower().strip()
        if platform not in ("youtube", "chzzk"):
            raise ValueError(f"지원하지 않는 플랫폼: '{platform}'. 'youtube' 또는 'chzzk' 사용.")

        self._platform = platform
        self._channel_id = channel_id
        self._running = True
        self.stats = {"received": 0, "responded": 0, "skipped": 0, "errors": 0}

        if platform == "youtube":
            self._collector = YouTubeChatCollector(channel_id, self._on_chat)
        else:
            self._collector = ChzzkChatCollector(channel_id, self._on_chat)

        self._collect_task = asyncio.create_task(self._collector.start())
        self._worker_task = asyncio.create_task(self._response_worker())

        logger.info(
            f"[ChatManager] 방송 채팅 수집 시작: "
            f"platform={platform}, channel_id={channel_id}"
        )

    async def stop(self) -> dict:
        """채팅 수집 중지."""
        self._running = False

        if self._collector:
            self._collector.stop()

        for task in (self._collect_task, self._worker_task):
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._collector = None
        self._collect_task = None
        self._worker_task = None

        final_stats = dict(self.stats)
        logger.info(f"[ChatManager] 수집 중지. 통계: {final_stats}")
        return final_stats

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "platform": self._platform,
            "channel_id": self._channel_id,
            "stats": dict(self.stats),
            "buffer_size": len(self._buffer),
            "queue_size": self._queue.qsize(),
        }
