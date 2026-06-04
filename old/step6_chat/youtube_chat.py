"""
youtube_chat.py - 유튜브 라이브 채팅 수신 모듈

설치 필요:
    pip install google-api-python-client
"""

import asyncio
import logging
import time
from typing import Callable, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config_chat import YOUTUBE_API_KEY, LIVE_CHAT_POLL_INTERVAL

logger = logging.getLogger(__name__)


class YouTubeChatReader:
    """유튜브 라이브 채팅 메시지를 수신하는 클래스."""

    def __init__(self, api_key: str = YOUTUBE_API_KEY):
        if not api_key:
            raise ValueError(
                "YOUTUBE_API_KEY가 설정되지 않았습니다. config_chat.py에 API 키를 입력해주세요."
            )
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self._running = False

    def get_live_chat_id(self, video_id: str) -> str:
        """
        방송 video_id로 liveChatId를 조회합니다.

        Args:
            video_id: 유튜브 영상 ID (예: dQw4w9WgXcQ)

        Returns:
            liveChatId 문자열

        Raises:
            ValueError: 라이브 방송이 아니거나 채팅 ID를 찾을 수 없는 경우
            HttpError: YouTube API 호출 실패
        """
        try:
            response = self.youtube.videos().list(
                part="liveStreamingDetails",
                id=video_id
            ).execute()

            items = response.get("items", [])
            if not items:
                raise ValueError(f"video_id '{video_id}'에 해당하는 영상을 찾을 수 없습니다.")

            live_details = items[0].get("liveStreamingDetails", {})
            live_chat_id = live_details.get("activeLiveChatId")

            if not live_chat_id:
                raise ValueError(
                    f"video_id '{video_id}'는 현재 라이브 방송 중이 아니거나 채팅이 비활성화되어 있습니다."
                )

            logger.info(f"liveChatId 조회 성공: {live_chat_id}")
            return live_chat_id

        except HttpError as e:
            logger.error(f"YouTube API 오류: {e}")
            raise

    def fetch_messages(
        self,
        live_chat_id: str,
        page_token: Optional[str] = None
    ) -> dict:
        """
        새 채팅 메시지를 가져옵니다.

        Args:
            live_chat_id: 라이브 채팅 ID
            page_token: 페이지 토큰 (이전 호출의 nextPageToken)

        Returns:
            {
                "messages": [{"author": str, "message": str, "timestamp": str}, ...],
                "next_page_token": str,
                "polling_interval_ms": int
            }
        """
        try:
            params = {
                "liveChatId": live_chat_id,
                "part": "snippet,authorDetails",
                "maxResults": 200,
            }
            if page_token:
                params["pageToken"] = page_token

            response = self.youtube.liveChatMessages().list(**params).execute()

            messages = []
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                author = item.get("authorDetails", {})

                # 일반 텍스트 메시지만 처리 (슈퍼챗, 멤버십 등 제외)
                if snippet.get("type") != "textMessageEvent":
                    continue

                messages.append({
                    "author": author.get("displayName", "알 수 없음"),
                    "author_channel_id": author.get("channelId", ""),
                    "message": snippet.get("textMessageDetails", {}).get("messageText", ""),
                    "timestamp": snippet.get("publishedAt", ""),
                    "is_moderator": author.get("isChatModerator", False),
                    "is_owner": author.get("isChatOwner", False),
                })

            return {
                "messages": messages,
                "next_page_token": response.get("nextPageToken"),
                "polling_interval_ms": response.get("pollingIntervalMillis", LIVE_CHAT_POLL_INTERVAL * 1000),
            }

        except HttpError as e:
            logger.error(f"채팅 메시지 조회 실패: {e}")
            raise

    async def start_polling(
        self,
        live_chat_id: str,
        callback: Callable[[dict], None]
    ) -> None:
        """
        비동기 폴링 루프를 시작합니다.
        새 메시지마다 callback(message)을 호출합니다.

        Args:
            live_chat_id: 라이브 채팅 ID
            callback: 메시지 수신 시 호출할 콜백 함수 (message dict를 인자로 받음)
        """
        self._running = True
        page_token = None
        logger.info(f"채팅 폴링 시작: {live_chat_id}")

        # 초기 메시지는 건너뜀 (폴링 시작 이후 새 메시지만 처리)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fetch_messages(live_chat_id, page_token)
            )
            page_token = result["next_page_token"]
        except Exception as e:
            logger.warning(f"초기 페이지 토큰 조회 실패 (무시): {e}")

        while self._running:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.fetch_messages(live_chat_id, page_token)
                )

                for msg in result["messages"]:
                    if self._running:
                        try:
                            await callback(msg)
                        except Exception as e:
                            logger.error(f"콜백 처리 오류: {e}")

                page_token = result["next_page_token"]

                # YouTube API 권장 폴링 간격 준수 (rate limit 자동 처리)
                interval_sec = result["polling_interval_ms"] / 1000.0
                interval_sec = max(interval_sec, LIVE_CHAT_POLL_INTERVAL)
                await asyncio.sleep(interval_sec)

            except HttpError as e:
                if e.resp.status == 403:
                    logger.error("API 할당량 초과. 60초 후 재시도합니다.")
                    await asyncio.sleep(60)
                elif e.resp.status == 404:
                    logger.error("라이브 방송이 종료되었습니다. 폴링을 중지합니다.")
                    self._running = False
                else:
                    logger.error(f"HTTP 오류 ({e.resp.status}): {e}. 10초 후 재시도.")
                    await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"폴링 오류: {e}. 10초 후 재시도.")
                await asyncio.sleep(10)

        logger.info("채팅 폴링 종료")

    def stop(self) -> None:
        """폴링을 중지합니다."""
        self._running = False
        logger.info("채팅 폴링 중지 요청")
