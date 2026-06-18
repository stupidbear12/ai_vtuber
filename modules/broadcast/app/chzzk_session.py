# -*- coding: utf-8 -*-
"""
app/chzzk_session.py — 치지직 공식 Session API + Socket.IO 채팅 수집기

흐름:
  1. GET /open/v1/sessions/auth → Socket.IO 연결 URL 획득
  2. Socket.IO 연결 (transports=websocket)
  3. SYSTEM "connected" 이벤트 수신 시:
     - GET /open/v1/sessions → 자신의 세션 키 조회
     - POST .../subscribe/chat + /subscribe/donation + /subscribe/subscription
  4. CHAT / DONATION / SUBSCRIPTION 이벤트 → ChatMessage 변환 → 콜백 호출
  연결 끊기면 5초 후 자동 재연결

의존: python-socketio (pip install python-socketio)
"""

import asyncio
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ChzzkOfficialChatCollector:
    """치지직 공식 Session API 기반 실시간 채팅 수집기.

    공식 Access Token이 있을 때만 사용한다.
    인증된 사용자 본인의 채널 채팅 이벤트를 수신한다.
    """

    def __init__(self, client, on_message: Callable):
        """
        Args:
            client: ChzzkClient 인스턴스 (circular import 방지를 위해 타입 힌트 생략)
            on_message: ChatMessage를 받는 동기/비동기 콜백
        """
        self._client = client
        self._on_message = on_message
        self._running = False

    async def start(self) -> None:
        """채팅 수집 루프 시작. 연결 실패 시 5초 후 재시도."""
        self._running = True
        logger.info("[ChzzkOfficial] 공식 Session API 채팅 수집 시작")

        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if self._running:
                    logger.error(f"[ChzzkOfficial] 연결 오류: {e}. 5초 후 재시도...")
                    await asyncio.sleep(5)

        logger.info("[ChzzkOfficial] 채팅 수집 종료")

    async def _connect_and_listen(self) -> None:
        """Socket.IO 연결, 이벤트 구독, 수신 루프."""
        try:
            import socketio as sio_module
        except ImportError:
            raise ImportError(
                "python-socketio 설치 필요: pip install python-socketio"
            )

        # 1. 세션 URL 조회
        session_info = await self._client.get_session_url()
        session_url = (session_info or {}).get("url", "")
        if not session_url:
            raise RuntimeError(f"세션 URL이 비어있습니다. 응답: {session_info}")

        logger.info(f"[ChzzkOfficial] Socket.IO 연결 시도: {session_url[:60]}...")

        connected_evt = asyncio.Event()
        disconnected_evt = asyncio.Event()

        sio = sio_module.AsyncClient(
            reconnection=False,
            logger=False,
            engineio_logger=False,
        )

        @sio.event
        async def connect():
            logger.info("[ChzzkOfficial] Socket.IO 연결 성공")
            connected_evt.set()

        @sio.event
        async def disconnect():
            logger.info("[ChzzkOfficial] Socket.IO 연결 해제")
            disconnected_evt.set()

        @sio.on("SYSTEM")
        async def on_system(data):
            # 치지직 Session API는 SYSTEM 이벤트 데이터를 JSON 문자열로 전송
            parsed = data
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}

            event_type = parsed.get("type", "")
            logger.info(f"[ChzzkOfficial] SYSTEM 이벤트: type={event_type}, data={parsed}")

            if event_type == "connected":
                # 세션 키가 이벤트 데이터에 포함되어 있으면 바로 사용
                session_key = (parsed.get("data") or {}).get("sessionKey", "")
                if session_key:
                    await self._subscribe_with_key(session_key)
                else:
                    # 폴백: API로 세션 키 조회
                    await self._subscribe_all_events()

        @sio.on("CHAT")
        async def on_chat(data):
            self._dispatch(data, is_donation=False)

        @sio.on("DONATION")
        async def on_donation(data):
            self._dispatch(data, is_donation=True)

        @sio.on("SUBSCRIPTION")
        async def on_subscription(data):
            self._dispatch(data, is_donation=True)

        # 2. 연결
        await sio.connect(
            session_url,
            transports=["websocket"],
            wait_timeout=15,
        )

        # 3. connected 이벤트 대기 (최대 20초)
        try:
            await asyncio.wait_for(connected_evt.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            await sio.disconnect()
            raise RuntimeError("Socket.IO 연결 타임아웃")

        # 4. 연결 해제 또는 수집 중지 대기
        while self._running and not disconnected_evt.is_set():
            await asyncio.sleep(0.5)

        if sio.connected:
            await sio.disconnect()

    async def _subscribe_with_key(self, session_key: str) -> None:
        """주어진 세션 키로 채팅/후원/구독 이벤트를 모두 구독."""
        try:
            logger.info(f"[ChzzkOfficial] 세션 키 {session_key[:12]}... 로 이벤트 구독 시작")
            await self._client.subscribe_chat(session_key)
            await self._client.subscribe_donation(session_key)
            await self._client.subscribe_subscription(session_key)
            logger.info("[ChzzkOfficial] 채팅/후원/구독 이벤트 구독 완료")
        except Exception as e:
            logger.error(f"[ChzzkOfficial] 이벤트 구독 실패: {e}")

    async def _subscribe_all_events(self) -> None:
        """활성 세션 키를 조회 후 채팅/후원/구독 이벤트를 모두 구독."""
        try:
            session_key = await self._find_active_session_key()
            if not session_key:
                logger.warning("[ChzzkOfficial] 활성 세션 키를 찾지 못했습니다.")
                return
            await self._subscribe_with_key(session_key)
        except Exception as e:
            logger.error(f"[ChzzkOfficial] 이벤트 구독 실패: {e}")

    async def _find_active_session_key(self) -> str:
        """GET /open/v1/sessions에서 disconnectedDate가 없는 최신 세션 키를 반환."""
        sessions = await self._client.get_sessions(page=0, size=10)
        # disconnectedDate가 None인 세션 = 현재 활성 세션
        active = [s for s in sessions if not s.get("disconnectedDate")]
        if not active:
            return ""
        # connectedDate 기준 최신 세션 선택
        latest = max(active, key=lambda s: s.get("connectedDate", ""))
        return latest.get("sessionKey", "")

    def _dispatch(self, data, is_donation: bool) -> None:
        """수신 이벤트를 ChatMessage로 변환해 콜백 호출."""
        try:
            # 치지직 Session API는 데이터를 JSON 문자열로 보낼 수 있음
            parsed = data
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    logger.debug(f"[ChzzkOfficial] JSON 파싱 불가: {data[:100]}")
                    return

            items = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])
            for item in items:
                if isinstance(item, dict):
                    self._dispatch_single(item, is_donation)
        except Exception as e:
            logger.debug(f"[ChzzkOfficial] 이벤트 파싱 오류: {e}")

    def _dispatch_single(self, item: dict, is_donation: bool) -> None:
        """단일 이벤트 항목 파싱.

        공식 Session API 이벤트 필드:
          - CHAT: profile(Object), content(String), senderChannelId, ...
          - DONATION: donatorNickname, donationText, payAmount, ...
          - SUBSCRIPTION: subscriberNickname, ...
        """
        from app.chat_collector import ChatMessage

        if is_donation:
            # 후원 이벤트: donatorNickname, donationText
            nickname = item.get("donatorNickname") or "익명"
            text = item.get("donationText", "")
            if not text and not item.get("payAmount"):
                # 구독 이벤트: subscriberNickname
                nickname = item.get("subscriberNickname") or "시청자"
                text = f"구독 알림 (Tier {item.get('tierNo', '?')})"
        else:
            # 채팅 이벤트: profile은 Object(dict), 메시지는 content 필드
            profile = item.get("profile") or {}
            if isinstance(profile, str):
                try:
                    profile = json.loads(profile)
                except Exception:
                    profile = {}
            nickname = (profile or {}).get("nickname", "시청자")
            # 공식 API는 'content' 필드, 비공식은 'msg' 필드
            text = item.get("content") or item.get("message") or item.get("msg", "")

        if not text:
            return

        self._on_message(
            ChatMessage(
                platform="chzzk",
                author=nickname,
                message=text,
                is_donation=is_donation,
            )
        )

    def stop(self) -> None:
        """수집 중지 요청."""
        self._running = False
        logger.info("[ChzzkOfficial] 수집 중지 요청")
