# -*- coding: utf-8 -*-
"""
app/chzzk_auth.py — 치지직 공식 OAuth 토큰 관리자

OAuth 2.0 Authorization Code Grant 흐름:
  1. get_auth_url()   → 사용자에게 인증 URL 안내
  2. exchange_code()  → 인증 코드 → Access/Refresh Token 교환
  3. get_valid_token() → 만료 60초 전 자동 갱신 후 토큰 반환
  4. revoke()         → 로그아웃 (모든 토큰 파기)

토큰 저장: chzzk_tokens.json (이 파일 옆 디렉터리)
환경 변수: CHZZK_CLIENT_ID, CHZZK_CLIENT_SECRET, CHZZK_REDIRECT_URI
"""

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)

CHZZK_BASE_URL = "https://openapi.chzzk.naver.com"
CHZZK_AUTH_PAGE = "https://chzzk.naver.com/account-interlock"

TOKEN_FILE = Path(__file__).parent / "chzzk_tokens.json"


class ChzzkTokenManager:
    """치지직 공식 OAuth 토큰 관리자."""

    def __init__(self):
        self.client_id: str = os.environ.get("CHZZK_CLIENT_ID", "")
        self.client_secret: str = os.environ.get("CHZZK_CLIENT_SECRET", "")
        self.redirect_uri: str = os.environ.get(
            "CHZZK_REDIRECT_URI", "http://localhost:8003/chzzk/auth/callback"
        )

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0.0

        self._load_tokens()

    # ── 토큰 영속화 ─────────────────────────────────────────────────

    def _load_tokens(self) -> None:
        """저장 파일 → 환경변수 순서로 토큰 로드."""
        if TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
                self._access_token = data.get("access_token")
                self._refresh_token = data.get("refresh_token")
                self._expires_at = float(data.get("expires_at", 0))
                logger.info("[ChzzkAuth] 토큰 파일에서 로드 완료")
                return
            except Exception as e:
                logger.warning(f"[ChzzkAuth] 토큰 파일 로드 실패: {e}")

        self._access_token = os.environ.get("CHZZK_ACCESS_TOKEN")
        self._refresh_token = os.environ.get("CHZZK_REFRESH_TOKEN")
        try:
            self._expires_at = float(os.environ.get("CHZZK_TOKEN_EXPIRES_AT", 0))
        except ValueError:
            self._expires_at = 0.0

        if self._access_token:
            logger.info("[ChzzkAuth] 환경변수에서 토큰 로드 완료")

    def _save_tokens(self) -> None:
        """토큰을 JSON 파일에 저장."""
        try:
            TOKEN_FILE.write_text(
                json.dumps(
                    {
                        "access_token": self._access_token,
                        "refresh_token": self._refresh_token,
                        "expires_at": self._expires_at,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.debug("[ChzzkAuth] 토큰 파일 저장 완료")
        except Exception as e:
            logger.error(f"[ChzzkAuth] 토큰 파일 저장 실패: {e}")

    # ── OAuth 흐름 ──────────────────────────────────────────────────

    def get_auth_url(self, state: str = "") -> Tuple[str, str]:
        """OAuth 인증 URL 생성.

        Returns:
            (auth_url, state) 튜플 — state는 CSRF 방지용 랜덤 값
        """
        if not self.client_id:
            raise RuntimeError("CHZZK_CLIENT_ID 환경변수가 설정되지 않았습니다.")
        if not state:
            state = secrets.token_urlsafe(16)
        params = urlencode(
            {
                "clientId": self.client_id,
                "redirectUri": self.redirect_uri,
                "state": state,
            }
        )
        return f"{CHZZK_AUTH_PAGE}?{params}", state

    async def exchange_code(self, code: str, state: str = "") -> dict:
        """Authorization Code → Access Token / Refresh Token 교환.

        Args:
            code: 리다이렉트 URL에서 받은 인증 코드
            state: 요청 시 사용한 state 값

        Returns:
            {"accessToken", "refreshToken", "expiresIn", ...}

        Raises:
            RuntimeError: 교환 실패 시
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHZZK_BASE_URL}/auth/v1/token",
                json={
                    "grantType": "authorization_code",
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                    "code": code,
                    "state": state,
                },
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"토큰 교환 실패 (HTTP {resp.status}): {text}")
                data = json.loads(text)

        content = data.get("content") or data
        self._access_token = content["accessToken"]
        self._refresh_token = content["refreshToken"]
        self._expires_at = time.time() + content.get("expiresIn", 86400)
        self._save_tokens()
        logger.info("[ChzzkAuth] Access Token 발급 완료")
        return content

    async def refresh(self) -> dict:
        """Refresh Token으로 새 Access Token 발급.

        Raises:
            RuntimeError: Refresh Token 없거나 갱신 실패 시
        """
        if not self._refresh_token:
            raise RuntimeError(
                "Refresh Token이 없습니다. /chzzk/auth/url에서 OAuth 인증을 완료하세요."
            )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHZZK_BASE_URL}/auth/v1/token",
                json={
                    "grantType": "refresh_token",
                    "refreshToken": self._refresh_token,
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                },
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    # Refresh Token도 만료 → 재인증 필요
                    self._access_token = None
                    self._refresh_token = None
                    self._expires_at = 0.0
                    self._save_tokens()
                    raise RuntimeError(
                        f"토큰 갱신 실패 (HTTP {resp.status}). 재인증이 필요합니다: {text}"
                    )
                data = json.loads(text)

        content = data.get("content") or data
        self._access_token = content["accessToken"]
        self._refresh_token = content["refreshToken"]
        self._expires_at = time.time() + content.get("expiresIn", 86400)
        self._save_tokens()
        logger.info("[ChzzkAuth] Access Token 갱신 완료")
        return content

    async def revoke(self, token_type: str = "access_token") -> bool:
        """토큰 취소 (로그아웃). clientId와 user가 동일한 모든 토큰 제거.

        Args:
            token_type: "access_token" 또는 "refresh_token"

        Returns:
            취소 요청 성공 여부
        """
        token = (
            self._access_token if token_type == "access_token" else self._refresh_token
        )
        if not token:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{CHZZK_BASE_URL}/auth/v1/token/revoke",
                    json={
                        "clientId": self.client_id,
                        "clientSecret": self.client_secret,
                        "token": token,
                        "tokenTypeHint": token_type,
                    },
                    timeout=aiohttp.ClientTimeout(total=10.0),
                )
        except Exception as e:
            logger.warning(f"[ChzzkAuth] 토큰 취소 요청 실패: {e}")

        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0.0
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink(missing_ok=True)
        logger.info("[ChzzkAuth] 토큰 취소 완료 (로컬 삭제)")
        return True

    # ── 공개 헬퍼 ───────────────────────────────────────────────────

    async def get_valid_token(self) -> str:
        """유효한 Access Token 반환. 만료 60초 전에 자동 갱신.

        Raises:
            RuntimeError: 토큰 없거나 갱신 불가 시
        """
        if not self._access_token:
            raise RuntimeError(
                "Access Token이 없습니다. /chzzk/auth/url에서 OAuth 인증을 완료하세요."
            )
        if self._expires_at and time.time() >= self._expires_at - 60:
            logger.info("[ChzzkAuth] 토큰 만료 임박, 자동 갱신 시작")
            await self.refresh()
        return self._access_token

    def has_token(self) -> bool:
        """Access Token 보유 여부."""
        return bool(self._access_token)

    def get_status(self) -> dict:
        """현재 토큰 상태 반환."""
        now = time.time()
        return {
            "has_access_token": bool(self._access_token),
            "has_refresh_token": bool(self._refresh_token),
            "expires_at": self._expires_at or None,
            "is_expired": (now >= self._expires_at) if self._expires_at else None,
            "seconds_until_expiry": max(0, self._expires_at - now) if self._expires_at else None,
            "client_id_set": bool(self.client_id),
            "redirect_uri": self.redirect_uri,
        }
