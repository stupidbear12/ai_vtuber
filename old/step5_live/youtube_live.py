# ============================================================
# youtube_live.py - YouTube 라이브 방송 API 관리 모듈
# VTuber 자동화 파이프라인 Step 5 (Live)
#
# 필요 패키지:
#   pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
# ============================================================

import os
import time
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config_live import (
    CLIENT_SECRETS_FILE,
    TOKEN_FILE,
    DEFAULT_TITLE,
    DEFAULT_DESCRIPTION,
    DEFAULT_PRIVACY,
    DEFAULT_CATEGORY_ID,
    LATENCY_PREFERENCE,
    STREAM_RESOLUTION,
    STREAM_FRAME_RATE,
    YOUTUBE_SCOPES,
    TRANSITION_WAIT_SEC,
)

logger = logging.getLogger(__name__)


class YouTubeLiveError(Exception):
    """YouTube 라이브 API 오류"""
    pass


class YouTubeLiveManager:
    """
    YouTube Data API v3를 사용하여 라이브 방송 전체 생명주기를 관리합니다.

    사용 흐름:
        manager = YouTubeLiveManager()
        manager.authenticate()
        stream_key = manager.setup_and_start(title, description, privacy)
        # → OBS에 stream_key 설정 후 스트리밍 시작
        # ...방송 진행...
        manager.end_broadcast(broadcast_id)
    """

    def __init__(self):
        self._youtube = None        # googleapiclient Resource 객체
        self._credentials = None    # google.oauth2.credentials.Credentials

    # ------------------------------------------------------------------
    # 인증
    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        """
        OAuth 2.0 인증을 수행합니다.
        - 토큰 파일이 존재하면 자동으로 재사용합니다.
        - 토큰이 만료된 경우 자동 갱신합니다.
        - 최초 실행 시 브라우저 팝업으로 사용자 승인을 요청합니다.
        """
        creds = None

        # 저장된 토큰 로드
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "rb") as f:
                    creds = pickle.load(f)
                logger.info("[Auth] 저장된 토큰 로드 완료: %s", TOKEN_FILE)
            except Exception as e:
                logger.warning("[Auth] 토큰 로드 실패 (재인증 필요): %s", e)
                creds = None

        # 토큰 유효성 검사 및 갱신
        if creds and creds.valid:
            logger.info("[Auth] 유효한 토큰 사용 중")
        elif creds and creds.expired and creds.refresh_token:
            logger.info("[Auth] 토큰 만료 → 자동 갱신 중")
            try:
                creds.refresh(Request())
                logger.info("[Auth] 토큰 갱신 성공")
            except Exception as e:
                logger.warning("[Auth] 토큰 갱신 실패 (재인증 필요): %s", e)
                creds = None
        else:
            creds = None

        # 새 인증 (브라우저 팝업)
        if creds is None:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise YouTubeLiveError(
                    f"client_secrets.json 파일이 없습니다: {os.path.abspath(CLIENT_SECRETS_FILE)}\n"
                    "Google Cloud Console → API 및 서비스 → 사용자 인증 정보에서\n"
                    "OAuth 2.0 클라이언트 ID를 생성하고 JSON을 다운로드하세요."
                )
            logger.info("[Auth] 브라우저 인증 팝업 시작...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, YOUTUBE_SCOPES
            )
            creds = flow.run_local_server(port=0)
            logger.info("[Auth] 인증 완료")

        # 토큰 저장
        try:
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(creds, f)
            logger.info("[Auth] 토큰 저장 완료: %s", TOKEN_FILE)
        except Exception as e:
            logger.warning("[Auth] 토큰 저장 실패: %s", e)

        self._credentials = creds
        self._youtube = build("youtube", "v3", credentials=creds)
        logger.info("[Auth] YouTube API 클라이언트 준비 완료")

    def _ensure_authenticated(self) -> None:
        """인증 여부를 확인하고, 미인증 시 예외를 발생시킵니다."""
        if self._youtube is None:
            raise YouTubeLiveError("인증되지 않았습니다. authenticate()를 먼저 호출하세요.")

    # ------------------------------------------------------------------
    # 방송 생성
    # ------------------------------------------------------------------
    def create_broadcast(
        self,
        title: str = DEFAULT_TITLE,
        description: str = DEFAULT_DESCRIPTION,
        privacy: str = DEFAULT_PRIVACY,
        scheduled_start_time: Optional[str] = None,
    ) -> str:
        """
        YouTube 라이브 방송을 생성합니다.

        Args:
            title: 방송 제목
            description: 방송 설명
            privacy: 공개 설정 (public / unlisted / private)
            scheduled_start_time: ISO 8601 형식 시작 시간 (None이면 현재 시각 기준)

        Returns:
            broadcast_id (str)
        """
        self._ensure_authenticated()

        if scheduled_start_time is None:
            # 지금부터 1분 후를 기본 스케줄 시작 시각으로 설정
            start_dt = datetime.now(timezone.utc) + timedelta(minutes=1)
            scheduled_start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "scheduledStartTime": scheduled_start_time,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
            "contentDetails": {
                "enableAutoStart": False,
                "enableAutoStop": False,
                "enableDvr": True,
                "latencyPreference": LATENCY_PREFERENCE,
                "monitorStream": {
                    "enableMonitorStream": True,
                    "broadcastStreamDelayMs": 0,
                },
            },
        }

        try:
            response = (
                self._youtube.liveBroadcasts()
                .insert(part="snippet,status,contentDetails", body=body)
                .execute()
            )
            broadcast_id = response["id"]
            logger.info("[Broadcast] 방송 생성 완료: %s (id=%s)", title, broadcast_id)
            return broadcast_id
        except HttpError as e:
            raise YouTubeLiveError(f"방송 생성 실패: {e}") from e

    # ------------------------------------------------------------------
    # 스트림 생성
    # ------------------------------------------------------------------
    def create_stream(self, title: str = "emeth stream") -> Tuple[str, str, str]:
        """
        YouTube 라이브 스트림을 생성하고 RTMP 수신 정보를 반환합니다.

        Returns:
            (stream_id, rtmp_url, stream_key) 튜플
        """
        self._ensure_authenticated()

        body = {
            "snippet": {
                "title": title,
            },
            "cdn": {
                "frameRate": STREAM_FRAME_RATE,
                "ingestionType": "rtmp",
                "resolution": STREAM_RESOLUTION,
            },
        }

        try:
            response = (
                self._youtube.liveStreams()
                .insert(part="snippet,cdn", body=body)
                .execute()
            )
            stream_id = response["id"]
            ingestion = response["cdn"]["ingestionInfo"]
            rtmp_url = ingestion["ingestionAddress"]
            stream_key = ingestion["streamName"]

            logger.info(
                "[Stream] 스트림 생성 완료: id=%s | RTMP=%s | Key=%s...",
                stream_id,
                rtmp_url,
                stream_key[:8] if stream_key else "N/A",
            )
            return stream_id, rtmp_url, stream_key
        except HttpError as e:
            raise YouTubeLiveError(f"스트림 생성 실패: {e}") from e

    # ------------------------------------------------------------------
    # 방송-스트림 연결
    # ------------------------------------------------------------------
    def bind_broadcast(self, broadcast_id: str, stream_id: str) -> None:
        """
        방송(Broadcast)과 스트림(Stream)을 연결합니다.

        Args:
            broadcast_id: liveBroadcasts.insert에서 받은 ID
            stream_id: liveStreams.insert에서 받은 ID
        """
        self._ensure_authenticated()

        try:
            self._youtube.liveBroadcasts().bind(
                part="id,contentDetails",
                id=broadcast_id,
                streamId=stream_id,
            ).execute()
            logger.info(
                "[Bind] 방송-스트림 연결 완료: broadcast=%s ↔ stream=%s",
                broadcast_id,
                stream_id,
            )
        except HttpError as e:
            raise YouTubeLiveError(f"방송-스트림 연결 실패: {e}") from e

    # ------------------------------------------------------------------
    # 상태 전환
    # ------------------------------------------------------------------
    def transition_broadcast(self, broadcast_id: str, status: str) -> dict:
        """
        방송 상태를 전환합니다.

        Args:
            broadcast_id: 대상 방송 ID
            status: 전환할 상태 ("testing" | "live" | "complete")

        Returns:
            API 응답 딕셔너리
        """
        self._ensure_authenticated()

        valid_statuses = ("testing", "live", "complete")
        if status not in valid_statuses:
            raise ValueError(f"유효하지 않은 상태: {status}. 허용값: {valid_statuses}")

        try:
            response = (
                self._youtube.liveBroadcasts()
                .transition(
                    broadcastStatus=status,
                    id=broadcast_id,
                    part="id,status",
                )
                .execute()
            )
            logger.info(
                "[Transition] 방송 상태 전환 완료: %s → %s",
                broadcast_id,
                status,
            )
            return response
        except HttpError as e:
            raise YouTubeLiveError(f"상태 전환 실패 ({status}): {e}") from e

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------
    def get_broadcast_status(self, broadcast_id: str) -> str:
        """
        방송의 현재 lifeCycleStatus를 반환합니다.

        Returns:
            상태 문자열 (예: "ready", "testing", "live", "complete" 등)
        """
        self._ensure_authenticated()

        try:
            response = (
                self._youtube.liveBroadcasts()
                .list(part="status", id=broadcast_id)
                .execute()
            )
            items = response.get("items", [])
            if not items:
                raise YouTubeLiveError(f"방송을 찾을 수 없습니다: {broadcast_id}")
            status = items[0]["status"]["lifeCycleStatus"]
            logger.debug("[Status] broadcast=%s → %s", broadcast_id, status)
            return status
        except HttpError as e:
            raise YouTubeLiveError(f"상태 조회 실패: {e}") from e

    # ------------------------------------------------------------------
    # 방송 종료
    # ------------------------------------------------------------------
    def end_broadcast(self, broadcast_id: str) -> None:
        """방송을 complete 상태로 전환하여 종료합니다."""
        logger.info("[End] 방송 종료 중: %s", broadcast_id)
        try:
            self.transition_broadcast(broadcast_id, "complete")
            logger.info("[End] 방송 종료 완료: %s", broadcast_id)
        except YouTubeLiveError as e:
            logger.warning("[End] 방송 종료 실패 (이미 종료되었을 수 있음): %s", e)

    # ------------------------------------------------------------------
    # 통합 셋업 (방송 생성 → 스트림 생성 → 연결)
    # ------------------------------------------------------------------
    def setup_and_start(
        self,
        title: str = DEFAULT_TITLE,
        description: str = DEFAULT_DESCRIPTION,
        privacy: str = DEFAULT_PRIVACY,
    ) -> dict:
        """
        방송 생성 → 스트림 생성 → 연결 → testing 상태 전환을 자동 수행합니다.

        Returns:
            {
                "broadcast_id": str,
                "stream_id": str,
                "rtmp_url": str,
                "stream_key": str,
                "watch_url": str,
            }
        """
        self._ensure_authenticated()

        logger.info("[Setup] 방송 셋업 시작: '%s'", title)

        # 1. 방송 생성
        broadcast_id = self.create_broadcast(title, description, privacy)

        # 2. 스트림 생성
        stream_id, rtmp_url, stream_key = self.create_stream(title=f"{title} stream")

        # 3. 방송-스트림 연결
        self.bind_broadcast(broadcast_id, stream_id)

        logger.info("[Setup] 방송 셋업 완료 → broadcast_id=%s", broadcast_id)

        watch_url = f"https://www.youtube.com/watch?v={broadcast_id}"
        return {
            "broadcast_id": broadcast_id,
            "stream_id": stream_id,
            "rtmp_url": rtmp_url,
            "stream_key": stream_key,
            "watch_url": watch_url,
        }

    # ------------------------------------------------------------------
    # testing → live 전환 (OBS 스트리밍 시작 이후 호출)
    # ------------------------------------------------------------------
    def go_live(self, broadcast_id: str, wait_for_stream: bool = True) -> None:
        """
        방송을 testing 거쳐 live 상태로 전환합니다.

        Args:
            broadcast_id: 대상 방송 ID
            wait_for_stream: True이면 스트림이 active 상태가 될 때까지 대기
        """
        self._ensure_authenticated()

        if wait_for_stream:
            logger.info("[GoLive] 스트림 active 대기 중 (최대 60초)...")
            deadline = time.time() + 60
            while time.time() < deadline:
                status = self.get_broadcast_status(broadcast_id)
                if status in ("testing", "live"):
                    logger.info("[GoLive] 스트림 active 확인: %s", status)
                    break
                logger.debug("[GoLive] 현재 상태: %s, 대기 중...", status)
                time.sleep(TRANSITION_WAIT_SEC)

        # testing 전환
        logger.info("[GoLive] testing 상태 전환 중...")
        try:
            self.transition_broadcast(broadcast_id, "testing")
            time.sleep(TRANSITION_WAIT_SEC)
        except YouTubeLiveError as e:
            logger.warning("[GoLive] testing 전환 실패 (무시): %s", e)

        # live 전환
        logger.info("[GoLive] live 상태 전환 중...")
        self.transition_broadcast(broadcast_id, "live")
        logger.info("[GoLive] 방송 LIVE 시작! broadcast_id=%s", broadcast_id)
