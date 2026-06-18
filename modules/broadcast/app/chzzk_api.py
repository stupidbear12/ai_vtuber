# -*- coding: utf-8 -*-
"""
app/chzzk_api.py — 치지직 공식 REST API 클라이언트

지원 API:
  - Live API: 스트림키 조회, 방송 설정 조회/변경
  - Chat API: 채팅 전송, 채팅 설정 조회
  - Session API: 실시간 이벤트 수신용 소켓 세션 관리

사용 방법:
    token_manager = ChzzkTokenManager()
    client = ChzzkClient(token_manager)
    stream_key = await client.get_stream_key()
"""

import json
import logging
from typing import List, Optional

import aiohttp

from app.chzzk_auth import CHZZK_BASE_URL, ChzzkTokenManager

logger = logging.getLogger(__name__)


class ChzzkClient:
    """치지직 공식 API 클라이언트 — 모든 /open/v1/* 엔드포인트 담당."""

    def __init__(self, token_manager: ChzzkTokenManager):
        self._tm = token_manager

    # ── 내부 헬퍼 ───────────────────────────────────────────────────

    async def _headers(self) -> dict:
        """User Access Token 기반 인증 헤더 빌드."""
        token = await self._tm.get_valid_token()
        return {
            "Authorization": f"Bearer {token}",
            "Client-Id": self._tm.client_id,
            "Content-Type": "application/json",
        }

    def has_token(self) -> bool:
        """Access Token 보유 여부 (토큰 매니저에 위임)."""
        return self._tm.has_token()

    def _client_headers(self) -> dict:
        """Client 인증 헤더 (Client-Id + Client-Secret). Access Token 불필요 엔드포인트용."""
        return {
            "Client-Id": self._tm.client_id,
            "Client-Secret": self._tm.client_secret,
            "Content-Type": "application/json",
        }

    async def _get_client_auth(self, path: str) -> dict:
        """Client 인증으로 GET 요청 (Bearer 토큰 없이)."""
        headers = self._client_headers()
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{CHZZK_BASE_URL}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                logger.info(f"[ChzzkAPI] _get_client_auth {path} → HTTP {r.status}, raw={text[:500]}")
                if r.status != 200:
                    raise RuntimeError(f"GET {path} 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        content = data.get("content") or data
        logger.info(f"[ChzzkAPI] _get_client_auth {path} → content keys={list(content.keys()) if isinstance(content, dict) else type(content)}")
        return content

    async def _get(self, path: str) -> dict:
        headers = await self._headers()
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{CHZZK_BASE_URL}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"GET {path} 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        return data.get("content") or data

    async def _post(self, path: str, payload: dict) -> dict:
        headers = await self._headers()
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{CHZZK_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"POST {path} 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        return data.get("content") or data

    async def _post_query(self, path: str, params: dict) -> dict:
        """POST 요청 — payload를 Query Parameter로 전송 (치지직 Session 이벤트 구독용)."""
        headers = await self._headers()
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{CHZZK_BASE_URL}{path}",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"POST {path} 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        return data.get("content") or data

    async def _patch(self, path: str, payload: dict) -> dict:
        headers = await self._headers()
        async with aiohttp.ClientSession() as s:
            async with s.patch(
                f"{CHZZK_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"PATCH {path} 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        return data.get("content") or data

    # ── Live API ────────────────────────────────────────────────────

    async def get_stream_key(self) -> str:
        """스트림키 조회 (GET /open/v1/streams/key).

        Returns:
            streamKey 문자열
        """
        content = await self._get("/open/v1/streams/key")
        key = (content or {}).get("streamKey", "")
        logger.info(f"[ChzzkAPI] 스트림키 조회 완료: {key[:10]}...")
        return key

    async def get_live_setting(self) -> dict:
        """방송 설정 조회 (GET /open/v1/lives/setting).

        Returns:
            {defaultLiveTitle, categoryType, categoryId, tags, ...}
        """
        content = await self._get("/open/v1/lives/setting")
        logger.info("[ChzzkAPI] 방송 설정 조회 완료")
        return content or {}

    async def update_live_setting(
        self,
        title: Optional[str] = None,
        category_type: Optional[str] = None,
        category_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> dict:
        """방송 설정 변경 (PATCH /open/v1/lives/setting).

        모든 파라미터는 선택사항. 전달한 항목만 변경된다.

        Args:
            title: 방송 제목 (defaultLiveTitle)
            category_type: 카테고리 타입
            category_id: 카테고리 ID
            tags: 태그 목록

        Returns:
            변경 후 방송 설정 딕셔너리
        """
        payload: dict = {}
        if title is not None:
            payload["defaultLiveTitle"] = title
        if category_type is not None:
            payload["categoryType"] = category_type
        if category_id is not None:
            payload["categoryId"] = category_id
        if tags is not None:
            payload["tags"] = tags

        if not payload:
            raise ValueError("변경할 설정이 없습니다. 최소 하나의 파라미터를 제공하세요.")

        content = await self._patch("/open/v1/lives/setting", payload)
        logger.info(f"[ChzzkAPI] 방송 설정 변경 완료: {list(payload.keys())}")
        return content or {}

    # ── Chat API ────────────────────────────────────────────────────

    async def send_chat(self, message: str) -> dict:
        """채팅 메시지 전송 (POST /open/v1/chats/send).

        Chzzk 채팅 API 최대 길이 100자 제한을 적용한다.

        Args:
            message: 전송할 채팅 메시지 (자동으로 100자 잘림)

        Returns:
            API 응답 딕셔너리
        """
        truncated = message[:100]
        content = await self._post("/open/v1/chats/send", {"message": truncated})
        logger.debug(f"[ChzzkAPI] 채팅 전송 완료: {truncated[:30]!r}")
        return content or {}

    async def get_chat_settings(self) -> dict:
        """채팅 설정 조회 (GET /open/v1/chats/settings).

        Returns:
            채팅 설정 딕셔너리 (chatAvailableCondition 등)
        """
        content = await self._get("/open/v1/chats/settings")
        return content or {}

    # ── Session API ─────────────────────────────────────────────────

    async def get_session_url(self) -> dict:
        """세션 연결 URL 조회 (GET /open/v1/sessions/auth).

        실시간 채팅 수신용 Socket.IO 연결 URL을 반환한다.
        유저당 최대 3개 세션 유지 가능.

        Returns:
            {"url": "...", ...}
        """
        content = await self._get("/open/v1/sessions/auth")
        url = (content or {}).get("url", "")
        logger.info(f"[ChzzkAPI] 세션 URL 조회 완료: {url[:60]}...")
        return content or {}

    async def get_sessions(self, page: int = 0, size: int = 20) -> list:
        """활성 세션 목록 조회 (GET /open/v1/sessions).

        Returns:
            세션 목록 [{sessionKey, connectedDate, disconnectedDate, ...}]
        """
        headers = await self._headers()
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{CHZZK_BASE_URL}/open/v1/sessions",
                headers=headers,
                params={"page": page, "size": size},
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"세션 목록 조회 실패 (HTTP {r.status}): {text}")
                data = json.loads(text)
        content = data.get("content") or data
        return (content or {}).get("data", []) if isinstance(content, dict) else []

    async def subscribe_chat(self, session_key: str) -> dict:
        """채팅 이벤트 구독 (POST /open/v1/sessions/events/subscribe/chat).

        NOTE: 치지직 Session API는 sessionKey를 Query Parameter로 전달해야 한다.
        JSON body로 보내면 400 에러 발생.
        """
        content = await self._post_query(
            "/open/v1/sessions/events/subscribe/chat",
            {"sessionKey": session_key},
        )
        logger.info("[ChzzkAPI] 채팅 이벤트 구독 완료")
        return content or {}

    async def subscribe_donation(self, session_key: str) -> dict:
        """후원 이벤트 구독 (POST /open/v1/sessions/events/subscribe/donation)."""
        content = await self._post_query(
            "/open/v1/sessions/events/subscribe/donation",
            {"sessionKey": session_key},
        )
        logger.info("[ChzzkAPI] 후원 이벤트 구독 완료")
        return content or {}

    async def subscribe_subscription(self, session_key: str) -> dict:
        """구독 알림 이벤트 구독 (POST /open/v1/sessions/events/subscribe/subscription)."""
        content = await self._post_query(
            "/open/v1/sessions/events/subscribe/subscription",
            {"sessionKey": session_key},
        )
        logger.info("[ChzzkAPI] 구독 알림 이벤트 구독 완료")
        return content or {}

    async def get_live_status(self) -> dict:
        """내 채널 방송 상태 조회.

        NOTE: 치지직 공식 API에 '내 채널 라이브 여부' 전용 엔드포인트가 없으므로
        자동 감지가 불가능하다. 항상 is_live=False를 반환하며,
        방송 시작/종료는 수동(/broadcast/start, /broadcast/stop)으로 관리한다.

        Returns:
            {"is_live": False}
        """
        return {"is_live": False}
