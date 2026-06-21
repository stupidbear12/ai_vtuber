# -*- coding: utf-8 -*-
"""
app/chat_collector.py — 치지직/유튜브 방송 채팅 수집 및 시온 반응 모듈

지원 플랫폼:
  - 유튜브 라이브: pytchat 라이브러리 (API 키 불필요)
  - 치지직(Chzzk): 비공식 WebSocket API

모듈 연동 방식:
  - ai_chat  (기본 http://localhost:8002) → POST /chat 으로 시온 응답 생성
  - ai_live2d (기본 http://localhost:8001) → POST /live2d/emotion 으로 표정 변경
"""

import asyncio
import base64
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── 채팅 선별 설정 ────────────────────────────────────────────────
TRIGGER_KEYWORDS = ["시온", "sion", "@시온", "@sion"]  # 이 키워드 포함 시 무조건 반응
RANDOM_RESPONSE_RATE = 0.15   # 트리거 키워드 없어도 15% 확률로 반응
MIN_RESPONSE_INTERVAL = 5.0   # 연속 응답 최소 간격 (초) — 스팸 방지
CHAT_BUFFER_SIZE = 30         # 최근 채팅 히스토리 보관 개수


# ── 데이터 모델 ──────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """방송 채팅 단일 메시지."""
    platform: str        # "youtube" 또는 "chzzk"
    author: str          # 닉네임
    message: str         # 채팅 내용
    is_donation: bool = False  # 후원/슈퍼챗 여부
    timestamp: float = field(default_factory=time.time)  # 수신 시각 (Unix timestamp)


# ── 채팅 히스토리 버퍼 ──────────────────────────────────────────

class ChatBuffer:
    """최근 N개 채팅 메시지를 유지하는 링 버퍼.

    ai_chat 컨텍스트 프롬프트 생성에 활용한다.
    deque(maxlen=N)으로 자동 오래된 항목 제거.
    """

    def __init__(self, maxsize: int = CHAT_BUFFER_SIZE):
        self._buf: Deque[ChatMessage] = deque(maxlen=maxsize)

    def add(self, msg: ChatMessage) -> None:
        """새 채팅 메시지를 버퍼에 추가."""
        self._buf.append(msg)

    def get_context_text(self, limit: int = 10) -> str:
        """최근 limit개 채팅 히스토리를 텍스트로 변환.

        ai_chat의 /chat 엔드포인트 context 파라미터로 전달한다.

        Args:
            limit: 반환할 최근 메시지 수 (기본값: 10)

        Returns:
            "[유튜브] 닉네임: 채팅내용\\n..." 형식 문자열
        """
        if not self._buf:
            return ""
        recent = list(self._buf)[-limit:]
        lines = []
        for m in recent:
            tag = "[유튜브]" if m.platform == "youtube" else "[치지직]"
            lines.append(f"{tag} {m.author}: {m.message}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._buf)


# ── 채팅 선별 필터 ──────────────────────────────────────────────

class ChatFilter:
    """채팅 선별 로직 — 키워드 포함 여부, 랜덤 확률, 최소 응답 간격 적용."""

    def __init__(self):
        self._last_response_time: float = 0.0

    def should_respond(self, msg: ChatMessage) -> bool:
        """이 채팅에 시온가 반응해야 하는지 판단.

        우선순위:
          1. 후원/슈퍼챗 → 무조건 반응
          2. 최소 응답 간격 미충족 → 건너뜀
          3. 트리거 키워드 포함 → 반응
          4. RANDOM_RESPONSE_RATE 확률로 랜덤 반응

        Args:
            msg: 판단할 채팅 메시지

        Returns:
            True면 반응, False면 건너뜀
        """
        # 후원은 무조건 반응
        if msg.is_donation:
            return True

        # 최소 응답 간격 체크 (스팸 방지)
        if time.time() - self._last_response_time < MIN_RESPONSE_INTERVAL:
            return False

        # 트리거 키워드 검사 (대소문자 무시)
        text_lower = msg.message.lower()
        for kw in TRIGGER_KEYWORDS:
            if kw.lower() in text_lower:
                return True

        # 랜덤 확률로 일반 채팅에도 반응
        return random.random() < RANDOM_RESPONSE_RATE

    def mark_responded(self) -> None:
        """응답 완료 시 호출 — 다음 응답까지 최소 간격 카운트 시작."""
        self._last_response_time = time.time()


# ── 유튜브 채팅 수집기 ──────────────────────────────────────────

class YouTubeChatCollector:
    """pytchat 기반 유튜브 라이브 채팅 수집기 (API 키 불필요).

    pytchat은 유튜브 영상 ID만으로 라이브 채팅을 폴링 방식으로 수집한다.
    """

    def __init__(self, video_id: str, on_message: Callable[[ChatMessage], None]):
        """
        Args:
            video_id: 유튜브 라이브 영상 ID (URL의 ?v= 파라미터)
            on_message: 채팅 수신 시 호출할 동기 콜백 함수
        """
        self._video_id = video_id
        self._on_message = on_message
        self._running = False

    async def start(self) -> None:
        """채팅 수집 루프 시작 (비동기).

        pytchat.create()로 채팅 세션을 만들고,
        1초마다 새 채팅을 폴링해 on_message 콜백에 전달한다.
        """
        try:
            import pytchat
        except ImportError:
            raise ImportError("pytchat 설치 필요: pip install pytchat")

        self._running = True
        logger.info(f"[YouTube] 채팅 수집 시작: video_id={self._video_id}")

        loop = asyncio.get_event_loop()
        # pytchat은 블로킹 라이브러리 → 스레드 풀에서 실행
        chat = await loop.run_in_executor(
            None, lambda: pytchat.create(video_id=self._video_id)
        )

        while self._running:
            if not chat.is_alive():
                logger.info("[YouTube] 방송이 종료되었습니다.")
                break

            # 새 채팅 배치 가져오기 (블로킹 → executor)
            data = await loop.run_in_executor(None, chat.get)
            for c in data.sync_items():
                if not self._running:
                    break
                # 슈퍼챗 여부 판단 (amount 속성 존재 시 후원으로 간주)
                is_donation = hasattr(c, "amount") and c.amount
                self._on_message(ChatMessage(
                    platform="youtube",
                    author=c.author.name,
                    message=c.message,
                    is_donation=bool(is_donation),
                ))

            await asyncio.sleep(1.0)  # 1초 폴링 간격

        self._running = False
        logger.info("[YouTube] 채팅 수집 종료")

    def stop(self) -> None:
        """채팅 수집 중지 요청 (루프 종료는 다음 틱에서 발생)."""
        self._running = False
        logger.info("[YouTube] 수집 중지 요청")


# ── 치지직 채팅 수집기 ──────────────────────────────────────────

class ChzzkChatCollector:
    """치지직 비공식 WebSocket API 기반 채팅 수집기.

    치지직 공식 API가 없어 비공식 WebSocket 프로토콜을 역공학해 구현했다.
    연결이 끊기면 자동으로 재연결을 시도한다.
    """

    # 치지직 WebSocket 명령 코드 (비공식 역공학)
    # 참고: 치지직은 네이버 NBASE 채팅 WebSocket 프로토콜을 사용함
    _CMD_PING = 0             # 핑/heartbeat — cmd=0은 연결 유지용으로도 사용됨
    _CMD_WORKER = 0           # 워커 연결 요청 (초기 핸드셰이크)
    _CMD_WORKER_RESULT = 5    # 워커 연결 결과 (sid 포함)
    _CMD_CONNECT = 100        # 채팅 채널 구독 요청
    _CMD_CONNECTED = 10000    # 채팅 채널 구독 완료
    _CMD_CHAT = 10100         # 일반 채팅 메시지
    _CMD_DONATION = 10101     # 후원 (치즈)
    _CMD_SUBSCRIPTION = 10103 # 구독 알림

    def __init__(self, channel_id: str, on_message: Callable[[ChatMessage], None]):
        """
        Args:
            channel_id: 치지직 채널 해시 ID (URL의 /live/ 뒤 부분)
            on_message: 채팅 수신 시 호출할 동기 콜백 함수
        """
        self._channel_id = channel_id
        self._on_message = on_message
        self._running = False

    async def _get_chat_channel_id(self) -> str:
        """치지직 채널 ID로 chatChannelId 조회.

        치지직 채널 ID(해시)와 채팅 채널 ID는 다르다.
        라이브 상태 API를 호출해 실제 채팅 채널 ID를 가져온다.

        Returns:
            chatChannelId 문자열

        Raises:
            ValueError: 채널을 찾을 수 없거나 방송 중이 아닐 때
        """
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
        """치지직 WebSocket 프로토콜 메시지 직렬화.

        Args:
            cmd: 명령 코드 (예: _CMD_CONNECT = 100)
            cid: 채팅 채널 ID
            bdy: 메시지 본문 딕셔너리
            tid: 트랜잭션 ID (매 요청마다 증가)
            sid: 세션 ID (워커 결과에서 받은 값)

        Returns:
            JSON 직렬화 문자열
        """
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
        """채팅 수집 루프 시작 (비동기).

        흐름:
          1. chatChannelId 조회
          2. 치지직 WebSocket 서버에 연결 (랜덤 서버 선택)
          3. WORKER → CONNECT 핸드셰이크
          4. 20초마다 핑 전송 (연결 유지)
          5. 채팅/후원 메시지 수신 및 파싱
          연결 끊기면 5초 후 재연결 시도
        """
        self._running = True

        try:
            chat_channel_id = await self._get_chat_channel_id()
        except Exception as e:
            logger.error(f"[Chzzk] chatChannelId 조회 실패: {e}")
            self._running = False
            return

        import websockets

        while self._running:
            # 치지직은 kr-ss1 ~ kr-ss9 중 랜덤 선택
            n = random.randint(1, 9)
            ws_url = f"wss://kr-ss{n}.chat.naver.com/chat"
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=None,   # 수동 핑 관리
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

                    # Step 1: defaultWorker 연결 요청
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

                    # Step 4: 20초 간격 핑 태스크 (연결 유지)
                    async def _ping_loop():
                        while self._running:
                            await asyncio.sleep(20)
                            if not self._running:
                                break
                            try:
                                # _CMD_PING(0)으로 서버에 heartbeat 전송
                                await ws.send(self._build_msg(
                                    cmd=self._CMD_PING,
                                    cid=chat_channel_id,
                                    sid=sid,
                                    tid=tid[0],
                                ))
                                tid[0] += 1
                            except Exception:
                                break

                    ping_task = asyncio.create_task(_ping_loop())

                    try:
                        # Step 5: 메시지 수신 루프
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
        """수신된 WebSocket 메시지를 파싱해 ChatMessage로 변환.

        Args:
            raw: 수신된 JSON 문자열
        """
        try:
            data = json.loads(raw)
        except Exception:
            return

        cmd = data.get("cmd")
        bdy = data.get("bdy", {})

        if cmd == self._CMD_CHAT:
            # 일반 채팅 메시지
            items = bdy if isinstance(bdy, list) else ([bdy] if bdy else [])
            for item in items:
                self._process_item(item, is_donation=False)
        elif cmd in (self._CMD_DONATION, self._CMD_SUBSCRIPTION):
            # 후원 또는 구독 메시지
            items = bdy if isinstance(bdy, list) else ([bdy] if bdy else [])
            for item in items:
                self._process_item(item, is_donation=True)

    def _process_item(self, item: dict, is_donation: bool) -> None:
        """단일 채팅 항목에서 닉네임과 텍스트를 추출해 콜백 호출.

        Args:
            item: 채팅 아이템 딕셔너리
            is_donation: 후원 메시지 여부
        """
        try:
            # profile은 JSON 문자열로 중첩 인코딩된 경우가 있음
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
        """채팅 수집 중지 요청."""
        self._running = False
        logger.info("[Chzzk] 수집 중지 요청")


# ── 방송 채팅 매니저 (오케스트레이터) ──────────────────────────

class BroadcastChatManager:
    """치지직/유튜브 채팅 수집 + ai_chat + ai_voice TTS + ai_live2d 파이프라인.

    처리 흐름:
      1. 채팅 수집 (치지직 공식 Session API 또는 비공식 WebSocket / 유튜브 pytchat)
      2. 채팅 선별 (키워드, 후원, 랜덤 확률)
      3. ai_chat POST /chat → 시온 응답 + 감정 태그
      3.5. Chzzk 채팅창에 시온 응답 전송 (공식 API, 선택)
      4. ai_voice POST /voice/tts → TTS 음성 (MP3)
      5. ai_live2d POST /live2d/emotion → 표정 변경
    """

    def __init__(
        self,
        chat_url: str = "http://localhost:8002",
        live2d_url: str = "http://localhost:8001",
        voice_url: str = "http://localhost:8004",
        chzzk_client=None,
    ):
        """
        Args:
            chat_url: ai_chat 서버 URL (기본: http://localhost:8002)
            live2d_url: ai_live2d 서버 URL (기본: http://localhost:8001)
            voice_url: ai_voice 서버 URL (기본: http://localhost:8004)
            chzzk_client: ChzzkClient 인스턴스 (공식 API 사용 시). None이면 비공식 사용.
        """
        # 환경변수로 URL 오버라이드 가능
        self._chat_url = os.environ.get("AI_CHAT_URL", chat_url).rstrip("/")
        self._live2d_url = os.environ.get("AI_LIVE2D_URL", live2d_url).rstrip("/")
        self._voice_url = os.environ.get("AI_VOICE_URL", voice_url).rstrip("/")
        self._voice_enabled = os.environ.get(
            "BROADCAST_VOICE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

        # 공식 Chzzk API 클라이언트 (Access Token이 있을 때만 유효)
        self._chzzk_client = chzzk_client

        self._buffer = ChatBuffer()
        self._filter = ChatFilter()
        self._collector = None
        self._running = False
        self._platform: str = ""
        self._channel_id: str = ""
        self._collect_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        # start() 호출 시 현재 이벤트 루프를 저장 — 스레드 안전 큐 삽입에 사용
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 통계: 수신/응답/건너뜀/오류 횟수
        self.stats: dict = {
            "received": 0,
            "responded": 0,
            "skipped": 0,
            "errors": 0,
        }

    def _on_chat(self, msg: ChatMessage) -> None:
        """채팅 수신 콜백 — 임의 스레드에서 호출될 수 있음.

        YouTube 수집기(pytchat)는 executor 스레드에서 이 콜백을 호출하므로
        asyncio.Queue를 직접 조작하면 스레드 안전성 문제가 생긴다.
        loop.call_soon_threadsafe로 실제 처리를 이벤트 루프 스레드에 위임한다.

        Args:
            msg: 수신된 채팅 메시지
        """
        if self._loop is not None:
            # 이벤트 루프 스레드에 안전하게 _enqueue 예약
            self._loop.call_soon_threadsafe(self._enqueue, msg)

    def _enqueue(self, msg: ChatMessage) -> None:
        """이벤트 루프 스레드에서 실행 — 버퍼 추가 및 큐 삽입.

        call_soon_threadsafe에 의해 항상 이벤트 루프 스레드에서 호출된다.

        Args:
            msg: 처리할 채팅 메시지
        """
        self.stats["received"] += 1
        self._buffer.add(msg)  # 히스토리 버퍼에 추가

        if self._filter.should_respond(msg):
            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                # 처리 큐가 가득 찬 경우 메시지 버림 (부하 방어)
                self.stats["skipped"] += 1
                logger.debug("[ChatManager] 큐 가득 참. 메시지 버림.")
        else:
            self.stats["skipped"] += 1

    async def _response_worker(self) -> None:
        """큐에서 채팅을 꺼내 ai_chat 호출 후 ai_live2d 표정 변경.

        실행 중인 동안 큐를 지속적으로 감시하며 처리한다.
        1초마다 타임아웃 체크로 _running 상태를 확인한다.
        """
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # _running 재확인

            try:
                await self._respond_to_chat(msg)
                self._filter.mark_responded()  # 최소 응답 간격 카운트 시작
                self.stats["responded"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"[ChatManager] 응답 생성 실패: {e}")
            finally:
                self._queue.task_done()

    async def _respond_to_chat(self, msg: ChatMessage) -> None:
        """단일 채팅 메시지 처리 파이프라인.

        흐름:
          1. 방송용 메시지 포맷 구성 (플랫폼 태그 + 후원 힌트)
          2. ai_chat POST /chat 호출 → {reply, emotion} 수신
          3. ai_voice POST /voice/tts → TTS 음성 생성 (선택)
          4. ai_live2d POST /live2d/emotion → 표정 변경

        Args:
            msg: 처리할 채팅 메시지
        """
        # ── Step 1: 메시지 포맷 구성 ───────────────────────────────
        platform_tag = "[유튜브]" if msg.platform == "youtube" else "[치지직]"
        donation_hint = " [후원]" if msg.is_donation else ""
        current = f"{platform_tag} {msg.author}님{donation_hint}: {msg.message}"

        # 최근 채팅 히스토리를 컨텍스트로 전달 (자연스러운 맥락 반응)
        chat_context = self._buffer.get_context_text(limit=10)

        # ── Step 2: ai_chat 호출 → 시온 응답 생성 ───────────────
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "message": current,
                    "mode": "broadcast",
                    "context": chat_context if chat_context else None,
                    "viewer_name": msg.author,
                    "is_donation": msg.is_donation,
                }
                async with session.post(
                    f"{self._chat_url}/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60.0),  # Ollama 응답 대기
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"ai_chat 응답 오류: HTTP {resp.status}")
                    data = await resp.json()

            reply_text = data.get("reply", "")
            emotion = data.get("emotion", "calm")
        except Exception as e:
            logger.error(f"[ChatManager] ai_chat 호출 실패: {e}")
            return

        logger.info(
            f"[ChatManager] [{msg.platform}] {msg.author}: "
            f"{msg.message[:30]} → [{emotion}] {reply_text[:50]}"
        )

        # ── Step 2.5: Chzzk 채팅창에 시온 응답 전송 (공식 API) ─────
        if (
            self._chzzk_client
            and self._platform == "chzzk"
            and reply_text
            and self._chzzk_client.has_token()
        ):
            try:
                await self._chzzk_client.send_chat(reply_text[:100])
                logger.debug(f"[ChatManager] Chzzk 채팅 전송 완료: {reply_text[:30]!r}")
            except Exception as e:
                logger.warning(f"[ChatManager] Chzzk 채팅 전송 실패 (무시): {e}")

        # ── Step 3: ai_voice TTS 음성 생성 (선택) ─────────────────
        audio_base64: Optional[str] = None
        if self._voice_enabled and reply_text:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._voice_url}/voice/tts",
                        json={"text": reply_text, "emotion": emotion},
                        timeout=aiohttp.ClientTimeout(total=30.0),
                    ) as resp:
                        if resp.status == 200:
                            audio_bytes = await resp.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                            logger.debug(
                                f"[ChatManager] TTS 생성 완료: "
                                f"{len(audio_bytes)} bytes"
                            )
                        else:
                            logger.warning(
                                f"[ChatManager] ai_voice TTS 실패: HTTP {resp.status}"
                            )
            except Exception as e:
                logger.warning(f"[ChatManager] ai_voice 연결 실패 (무시): {e}")

        # ── Step 4: ai_live2d 표정 변경 + 음성 재생 ───────────────
        try:
            async with aiohttp.ClientSession() as session:
                # 표정 변경
                async with session.post(
                    f"{self._live2d_url}/live2d/emotion",
                    json={"emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"[ChatManager] ai_live2d 표정 변경 실패: HTTP {resp.status}")

                # 자막 + 음성 브로드캐스트 (Live2D WebSocket 클라이언트로 전달)
                broadcast_payload: dict = {
                    "cmd": "speak",
                    "text": reply_text,
                    "emotion": emotion,
                    "author": msg.author,
                    "platform": msg.platform,
                    "is_donation": msg.is_donation,
                }
                if audio_base64:
                    broadcast_payload["audio_base64"] = audio_base64

                async with session.post(
                    f"{self._live2d_url}/live2d/broadcast",
                    json=broadcast_payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        logger.debug("[ChatManager] speak 브로드캐스트 완료")
                    elif resp.status == 404:
                        # broadcast 엔드포인트 미구현 — 무시
                        logger.debug("[ChatManager] /live2d/broadcast 미구현 (무시)")
                    else:
                        logger.debug(f"[ChatManager] broadcast 전송: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"[ChatManager] ai_live2d 연결 실패 (무시): {e}")

    async def start(self, platform: str, channel_id: str) -> None:
        """채팅 수집을 시작한다.

        Args:
            platform: "youtube" 또는 "chzzk"
            channel_id: 유튜브 영상 ID 또는 치지직 채널 해시 ID

        Raises:
            RuntimeError: 이미 실행 중인 경우
            ValueError: 지원하지 않는 플랫폼인 경우
        """
        if self._running:
            raise RuntimeError("이미 실행 중입니다.")

        platform = platform.lower().strip()
        if platform not in ("youtube", "chzzk"):
            raise ValueError(f"지원하지 않는 플랫폼: '{platform}'. 'youtube' 또는 'chzzk' 사용.")

        self._platform = platform
        self._channel_id = channel_id
        self._running = True
        # _on_chat이 스레드에서 호출될 때 call_soon_threadsafe에서 사용할 루프 저장
        self._loop = asyncio.get_running_loop()
        self.stats = {"received": 0, "responded": 0, "skipped": 0, "errors": 0}

        # 플랫폼에 맞는 수집기 생성
        if platform == "youtube":
            self._collector = YouTubeChatCollector(channel_id, self._on_chat)
        else:
            # 공식 Access Token이 있으면 Session API 사용, 없으면 비공식 WebSocket 폴백
            if self._chzzk_client and self._chzzk_client.has_token():
                from app.chzzk_session import ChzzkOfficialChatCollector
                self._collector = ChzzkOfficialChatCollector(self._chzzk_client, self._on_chat)
                logger.info("[ChatManager] 치지직 공식 Session API 수집기 사용")
            else:
                self._collector = ChzzkChatCollector(channel_id, self._on_chat)
                logger.info("[ChatManager] 치지직 비공식 WebSocket 수집기 사용 (Access Token 없음)")

        # 수집 태스크와 응답 워커 태스크를 동시에 시작
        self._collect_task = asyncio.create_task(self._collector.start())
        self._worker_task = asyncio.create_task(self._response_worker())

        logger.info(
            f"[ChatManager] 방송 채팅 수집 시작: "
            f"platform={platform}, channel_id={channel_id}"
        )

    async def stop(self) -> dict:
        """채팅 수집을 중지하고 최종 통계를 반환한다.

        Returns:
            수집 통계 딕셔너리 {"received", "responded", "skipped", "errors"}
        """
        self._running = False

        if self._collector:
            self._collector.stop()

        # 수집/워커 태스크 취소 및 정리 (최대 3초 대기)
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
        self._loop = None  # 루프 참조 해제 (추가 콜백 예약 차단)

        final_stats = dict(self.stats)
        logger.info(f"[ChatManager] 수집 중지. 통계: {final_stats}")
        return final_stats

    @property
    def is_running(self) -> bool:
        """현재 채팅 수집이 실행 중인지 반환."""
        return self._running

    def get_status(self) -> dict:
        """현재 상태 딕셔너리 반환."""
        return {
            "running": self._running,
            "platform": self._platform,
            "channel_id": self._channel_id,
            "stats": dict(self.stats),
            "buffer_size": len(self._buffer),
            "queue_size": self._queue.qsize(),
            "chat_url": self._chat_url,
            "live2d_url": self._live2d_url,
            "voice_url": self._voice_url,
            "voice_enabled": self._voice_enabled,
            "chzzk_official_api": bool(self._chzzk_client and self._chzzk_client.has_token()),
        }
