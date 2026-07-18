# -*- coding: utf-8 -*-
"""
app/chzzk_session.py — 치지직 공식 Session API + WebSocket 채팅 수집기

흐름:
  1. GET /open/v1/sessions/auth → Socket.IO 연결 URL 획득
  2. Engine.IO handshake (HTTP) → sid, pingInterval, pingTimeout
  3. WebSocket 연결 + Engine.IO/Socket.IO 핸드셰이크
  4. SYSTEM "connected" 이벤트 수신 → 이벤트 구독
  5. CHAT / DONATION / SUBSCRIPTION 이벤트 → ChatMessage 변환 → 콜백 호출
  연결 끊기면 자동 재연결 (exponential backoff)

websockets 라이브러리 직접 사용 — python-socketio/engineio의
'packet queue is empty' 버그를 우회한다.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Callable, Optional
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

# ── Engine.IO / Socket.IO 패킷 타입 ────────────────────────────
# Engine.IO: 0=open, 1=close, 2=ping, 3=pong, 4=message, 5=upgrade, 6=noop
# Socket.IO (message 안): 0=connect, 1=disconnect, 2=event, 3=ack, 4=error
EIO_OPEN = "0"
EIO_CLOSE = "1"
EIO_PING = "2"
EIO_PONG = "3"
EIO_MESSAGE = "4"

SIO_CONNECT = "0"
SIO_DISCONNECT = "1"
SIO_EVENT = "2"


def _parse_sio_event(payload: str):
    """Socket.IO EVENT 패킷 파싱 → (event_name, data)."""
    # payload = '2["EVENT_NAME", {...}]' or '2["EVENT_NAME", "json_string"]'
    if not payload.startswith(SIO_EVENT):
        return None, None
    body = payload[1:]  # "2" 제거
    # namespace prefix 제거 (e.g. "/chat,")
    if body and body[0] == "/":
        idx = body.find(",")
        if idx >= 0:
            body = body[idx + 1:]
    try:
        arr = json.loads(body)
    except json.JSONDecodeError:
        # Extra data 대응
        try:
            decoder = json.JSONDecoder()
            arr, _ = decoder.raw_decode(body)
        except json.JSONDecodeError:
            return None, None
    if isinstance(arr, list) and len(arr) >= 2:
        return arr[0], arr[1]
    if isinstance(arr, list) and len(arr) == 1:
        return arr[0], None
    return None, None


class ChzzkOfficialChatCollector:
    """치지직 공식 Session API 기반 실시간 채팅 수집기.

    websockets 라이브러리를 직접 사용하여 Engine.IO/Socket.IO 프로토콜을
    수동 처리한다. python-socketio의 packet queue 버그를 우회.
    """

    def __init__(self, client, on_message: Callable):
        self._client = client
        self._on_message = on_message
        self._running = False
        self._ws = None
        # 중복 메시지 필터
        self._seen_messages: OrderedDict = OrderedDict()
        self._dedup_ttl = 60  # seconds
        # 디버그 카운터
        self._debug_ws_messages = 0      # WebSocket 수신 총 메시지 수
        self._debug_chat_events = 0      # CHAT 이벤트 수
        self._debug_self_filtered = 0    # 자기 메시지 필터링 수
        self._debug_dedup_filtered = 0   # 중복 필터링 수
        self._debug_dispatched = 0       # 콜백 호출 수
        # 구독 헬스체크용
        self._session_key: Optional[str] = None
        self._health_check_interval = 60  # seconds

    async def start(self) -> None:
        """채팅 수집 루프 시작."""
        self._running = True
        logger.info("[ChzzkOfficial] 공식 Session API 채팅 수집 시작 (websockets)")
        retry_delay = 5

        while self._running:
            try:
                await self._connect_and_listen()
                retry_delay = 5
            except Exception as e:
                if self._running:
                    err_str = str(e)
                    if "429" in err_str:
                        retry_delay = min(retry_delay * 2, 120)
                        logger.error(
                            f"[ChzzkOfficial] 세션 제한 초과. {retry_delay}초 후 재시도..."
                        )
                    else:
                        retry_delay = 10
                        logger.error(
                            f"[ChzzkOfficial] 연결 오류: {e}. {retry_delay}초 후 재시도..."
                        )
                    await asyncio.sleep(retry_delay)

        logger.info("[ChzzkOfficial] 채팅 수집 종료")

    async def _connect_and_listen(self) -> None:
        """WebSocket 직접 연결 → Engine.IO/Socket.IO 핸드셰이크 → 이벤트 루프."""
        import websockets

        async with self._client.session_slot():
            # 기존 연결 정리
            await self._close_ws()
            await self._client.wait_for_session_slot()

            # 1. 세션 URL 조회
            session_info = await self._client.get_session_url()
            session_url = (session_info or {}).get("url", "")
            if not session_url:
                raise RuntimeError(f"세션 URL이 비어있습니다. 응답: {session_info}")

            logger.info(f"[ChzzkOfficial] 연결 시도: {session_url[:60]}...")

            # 2. WebSocket 직접 연결 (polling 없이)
            parsed = urlparse(session_url)
            ws_path = parsed.path.rstrip("/") + "/socket.io/"
            ws_params = urlencode({"EIO": "4", "transport": "websocket"})
            # auth 파라미터 유지
            orig_query = parsed.query
            if orig_query:
                ws_params = f"{orig_query}&{ws_params}"
            ws_url = f"wss://{parsed.hostname}{':' + str(parsed.port) if parsed.port else ''}{ws_path}?{ws_params}"

            ws = await websockets.connect(
                ws_url,
                ping_interval=None,  # 우리가 직접 ping/pong 관리
                ping_timeout=None,
                close_timeout=5,
                max_size=2**20,
            )
            self._ws = ws

            # 3. Engine.IO open 패킷 수신
            first_msg = await asyncio.wait_for(ws.recv(), timeout=15)
            if isinstance(first_msg, bytes):
                first_msg = first_msg.decode("utf-8", errors="replace")

            ping_interval = 25.0
            ping_timeout = 20.0

            if isinstance(first_msg, str) and first_msg.startswith(EIO_OPEN):
                try:
                    eio_config = json.loads(first_msg[1:])
                    ping_interval = eio_config.get("pingInterval", 25000) / 1000.0
                    ping_timeout = eio_config.get("pingTimeout", 20000) / 1000.0
                    logger.info(
                        f"[ChzzkOfficial] EIO open 수신: "
                        f"pingInterval={ping_interval}s, pingTimeout={ping_timeout}s"
                    )
                except json.JSONDecodeError:
                    logger.warning(f"[ChzzkOfficial] EIO open 파싱 실패: {first_msg[:100]}")
            else:
                logger.warning(f"[ChzzkOfficial] 예상과 다른 첫 메시지: {str(first_msg)[:100]}")

            # Engine.IO open 이후 Socket.IO connect 패킷을 보내야
            # 서버가 SYSTEM connected 이벤트와 sessionKey를 내려준다.
            await ws.send(EIO_MESSAGE + SIO_CONNECT)
            logger.info("[ChzzkOfficial] Socket.IO connect 패킷 전송")

            # 5. Socket.IO connect 대기
            sio_connected = False
            subscription_done = asyncio.Event()

            async def _handle_message(msg: str):
                nonlocal sio_connected

                if msg == EIO_PING:
                    await ws.send(EIO_PONG)
                    return

                if msg == EIO_PONG:
                    return

                if msg == EIO_CLOSE:
                    logger.info("[ChzzkOfficial] 서버가 EIO close 전송")
                    return

                if not msg.startswith(EIO_MESSAGE):
                    logger.debug(f"[ChzzkOfficial] 알 수 없는 EIO 패킷: {msg[:50]}")
                    return

                sio_payload = msg[1:]  # "4" 제거 → Socket.IO 패킷

                # SIO connect ack
                if sio_payload.startswith(SIO_CONNECT):
                    sio_connected = True
                    logger.info("[ChzzkOfficial] Socket.IO 연결 성공")
                    return

                # SIO disconnect
                if sio_payload.startswith(SIO_DISCONNECT):
                    logger.info("[ChzzkOfficial] Socket.IO disconnect 수신")
                    sio_connected = False
                    return

                # SIO event
                if sio_payload.startswith(SIO_EVENT):
                    event_name, event_data = _parse_sio_event(sio_payload)
                    if event_name:
                        await self._handle_event(event_name, event_data, subscription_done)

            def _ws_is_open():
                """websockets 16+ 호환 연결 상태 체크."""
                try:
                    return ws.close_code is None
                except Exception:
                    return False

            # 6. ping 루프
            async def _ping_loop():
                while self._running and _ws_is_open():
                    await asyncio.sleep(ping_interval)
                    if _ws_is_open():
                        try:
                            await ws.send(EIO_PING)
                            logger.debug("[ChzzkOfficial] EIO ping 전송")
                        except Exception:
                            break

            ping_task = asyncio.create_task(_ping_loop())
            health_task = asyncio.create_task(self._subscription_health_check())

            # 7. 메시지 수신 루프
            try:
                async for msg in ws:
                    if not self._running:
                        break
                    self._debug_ws_messages += 1
                    if isinstance(msg, str):
                        await _handle_message(msg)
                    # binary 메시지는 무시
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"[ChzzkOfficial] WebSocket 연결 종료: {e}")
            except Exception as e:
                logger.error(f"[ChzzkOfficial] WebSocket 수신 오류: {e}")
            finally:
                ping_task.cancel()
                health_task.cancel()
                self._session_key = None
                await self._close_ws()

    async def _handle_event(self, event_name: str, data, subscription_done: asyncio.Event):
        """Socket.IO 이벤트 처리."""
        if event_name == "SYSTEM":
            parsed = data
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}

            event_type = parsed.get("type", "")
            logger.info(f"[ChzzkOfficial] SYSTEM 이벤트: type={event_type}")

            if event_type == "connected":
                session_key = (parsed.get("data") or {}).get("sessionKey", "")
                logger.info(f"[ChzzkOfficial] connected data keys: {list(parsed.get('data', {}).keys()) if isinstance(parsed.get('data'), dict) else 'N/A'}, sessionKey={'YES' if session_key else 'EMPTY'}")
                if session_key:
                    await self._subscribe_with_key(session_key)
                else:
                    logger.warning("[ChzzkOfficial] sessionKey 없음 → _subscribe_all_events 호출")
                    await self._subscribe_all_events()
                subscription_done.set()

        elif event_name == "CHAT":
            self._debug_chat_events += 1
            logger.debug(f"[ChzzkOfficial] CHAT 이벤트 수신 (총 {self._debug_chat_events}건)")
            self._dispatch(data, is_donation=False)

        elif event_name == "DONATION":
            logger.debug(f"[ChzzkOfficial] DONATION 이벤트 수신")
            self._dispatch(data, is_donation=True)

        elif event_name == "SUBSCRIPTION":
            logger.debug(f"[ChzzkOfficial] SUBSCRIPTION 이벤트 수신")
            self._dispatch(data, is_donation=True)

        else:
            logger.info(f"[ChzzkOfficial] 알 수 없는 이벤트: {event_name}")

    async def _close_ws(self):
        """WebSocket 연결 정리."""
        ws = self._ws
        self._ws = None
        if ws and (ws.close_code is None):
            try:
                await ws.close()
                logger.info("[ChzzkOfficial] WebSocket 연결 해제")
            except Exception as e:
                logger.debug(f"[ChzzkOfficial] WebSocket 해제 중 오류 (무시): {e}")

    async def _subscribe_with_key(self, session_key: str, max_retries: int = 3) -> None:
        """주어진 세션 키로 채팅/후원/구독 이벤트를 모두 구독."""
        self._debug_subscribe_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[ChzzkOfficial] 세션 키 {session_key[:12]}... 이벤트 구독 시도 ({attempt}/{max_retries})")
                await self._client.subscribe_chat(session_key)
                await self._client.subscribe_donation(session_key)
                await self._client.subscribe_subscription(session_key)
                logger.info("[ChzzkOfficial] 채팅/후원/구독 이벤트 구독 완료")
                self._debug_subscribe_error = "OK"
                self._session_key = session_key
                return
            except Exception as e:
                self._debug_subscribe_error = f"attempt {attempt}: {e}"
                logger.error(f"[ChzzkOfficial] 이벤트 구독 실패 (시도 {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
        logger.error("[ChzzkOfficial] 이벤트 구독 최종 실패 — 채팅 수신 불가")

    # ── 구독 헬스체크 ──────────────────────────────────────────────

    _REQUIRED_EVENTS = {"CHAT", "DONATION", "SUBSCRIPTION"}

    async def _subscription_health_check(self) -> None:
        """주기적으로 구독 상태를 확인하고 누락된 이벤트를 자동 재구독."""
        # 첫 구독 완료 후 체크 시작하도록 초기 대기
        await asyncio.sleep(self._health_check_interval)

        while self._running:
            session_key = self._session_key
            if not session_key:
                await asyncio.sleep(self._health_check_interval)
                continue

            try:
                events = await self._client.get_session_events(session_key)
                # events: list of dicts or strings — 이벤트 타입 추출
                subscribed: set = set()
                for ev in events:
                    if isinstance(ev, dict):
                        subscribed.add(ev.get("eventType", "").upper())
                    elif isinstance(ev, str):
                        subscribed.add(ev.upper())

                missing = self._REQUIRED_EVENTS - subscribed
                if not missing:
                    logger.debug(
                        f"[ChzzkOfficial] 구독 헬스체크 OK: {sorted(subscribed)}"
                    )
                else:
                    logger.warning(
                        f"[ChzzkOfficial] 구독 누락 감지: {sorted(missing)} "
                        f"(현재 구독: {sorted(subscribed)})"
                    )
                    await self._resubscribe_missing(session_key, missing)

            except Exception as e:
                logger.error(f"[ChzzkOfficial] 구독 헬스체크 오류: {e}")

            await asyncio.sleep(self._health_check_interval)

    async def _resubscribe_missing(self, session_key: str, missing: set) -> None:
        """누락된 이벤트만 선택적으로 재구독."""
        subscribe_map = {
            "CHAT": self._client.subscribe_chat,
            "DONATION": self._client.subscribe_donation,
            "SUBSCRIPTION": self._client.subscribe_subscription,
        }
        for event_type in sorted(missing):
            fn = subscribe_map.get(event_type)
            if not fn:
                continue
            try:
                await fn(session_key)
                logger.info(
                    f"[ChzzkOfficial] 재구독 성공: {event_type}"
                )
            except Exception as e:
                logger.error(
                    f"[ChzzkOfficial] 재구독 실패: {event_type} — {e}"
                )

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
        """활성 세션 중 최신 sessionKey 반환."""
        active = await self._client.get_active_sessions()
        if not active:
            return ""
        latest = max(active, key=lambda s: s.get("connectedDate", ""))
        return latest.get("sessionKey", "")

    def _dispatch(self, data, is_donation: bool) -> None:
        """수신 이벤트를 ChatMessage로 변환해 콜백 호출."""
        try:
            parsed = data
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    logger.debug(f"[ChzzkOfficial] JSON 파싱 불가: {str(data)[:100]}")
                    return

            items = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])
            for item in items:
                if isinstance(item, dict):
                    self._dispatch_single(item, is_donation)
        except Exception as e:
            logger.debug(f"[ChzzkOfficial] 이벤트 파싱 오류: {e}")

    def _dispatch_single(self, item: dict, is_donation: bool) -> None:
        """단일 이벤트 항목 파싱."""
        from app.chat_collector import ChatMessage

        # ── 봇 자신의 응답 메시지만 무시 (무한 루프 방지) ──
        # 기존: senderChannelId == channelId로 방송자의 모든 메시지를 차단 → 수정
        # 봇 응답은 "[시온]" 접두사를 가지므로, content 기반으로만 필터링한다.
        # 추가 자문자답 방지는 chat_collector._enqueue의 _is_own_message()에서 처리.
        my_channel_id = item.get("channelId", "")
        sender_id = item.get("senderChannelId", "")
        content = item.get("content", "") or ""
        if sender_id and my_channel_id and sender_id == my_channel_id:
            if content.startswith("[시온]"):
                self._debug_self_filtered += 1
                logger.debug(f"[ChzzkOfficial] 봇 응답 무시: {content[:30]}")
                return
            # 방송자 본인의 일반 채팅은 통과시킨다
            logger.debug(f"[ChzzkOfficial] 방송자 채팅 통과: {content[:30]}")

        # ── 중복 메시지 필터 (동일 이벤트 다중 수신 방지) ──
        dedup_key = hashlib.md5(
            f"{sender_id}:{item.get('content', '')}:{item.get('messageTime', '')}".encode()
        ).hexdigest()
        now = time.monotonic()
        while self._seen_messages:
            oldest_key, oldest_time = next(iter(self._seen_messages.items()))
            if now - oldest_time > self._dedup_ttl:
                self._seen_messages.pop(oldest_key)
            else:
                break
        if dedup_key in self._seen_messages:
            logger.debug(f"[ChzzkOfficial] 중복 메시지 무시: {item.get('content', '')[:30]}")
            return
        self._seen_messages[dedup_key] = now

        if is_donation:
            nickname = item.get("donatorNickname") or "익명"
            text = item.get("donationText", "")
            if not text and not item.get("payAmount"):
                nickname = item.get("subscriberNickname") or "시청자"
                text = f"구독 알림 (Tier {item.get('tierNo', '?')})"
        else:
            profile = item.get("profile") or {}
            if isinstance(profile, str):
                try:
                    profile = json.loads(profile)
                except Exception:
                    profile = {}
            nickname = (profile or {}).get("nickname", "시청자")
            text = item.get("content") or item.get("message") or item.get("msg", "")

        if not text:
            return

        self._debug_dispatched += 1
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
