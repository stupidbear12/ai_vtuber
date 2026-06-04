"""
chat_pipeline.py - 유튜브 라이브 채팅 전체 흐름 오케스트레이터

사용 예시:
    import asyncio
    from chat_pipeline import ChatPipeline

    pipeline = ChatPipeline()
    asyncio.run(pipeline.start("YOUR_VIDEO_ID"))
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from config_chat import MAX_CHAT_QUEUE, RESPONSE_LOG_PATH
from youtube_chat import YouTubeChatReader
from chat_handler import ChatHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ChatPipeline:
    """유튜브 라이브 채팅 수신 + 처리 전체 파이프라인 관리."""

    def __init__(self):
        self.reader = YouTubeChatReader()
        self.handler = ChatHandler()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_CHAT_QUEUE)
        self._running = False
        self._video_id: Optional[str] = None
        self._live_chat_id: Optional[str] = None
        self._stats = {
            "received": 0,
            "responded": 0,
            "skipped": 0,
            "errors": 0,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {**self._stats, "queue_size": self._queue.qsize()}

    async def _enqueue_message(self, message: dict) -> None:
        """채팅 수신 콜백 — 큐에 메시지를 추가합니다."""
        self._stats["received"] += 1
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("처리 큐가 가득 찼습니다. 메시지를 버립니다.")
            self._stats["skipped"] += 1

    async def _process_worker(self) -> None:
        """큐에서 메시지를 꺼내 순서대로 처리하는 워커."""
        logger.info("메시지 처리 워커 시작")
        while self._running or not self._queue.empty():
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                response = await self.handler.process_message(message)
                if response:
                    self._stats["responded"] += 1
                    logger.info(
                        f"✅ [{message['author']}] {message['message'][:30]}... "
                        f"→ {response[:40]}..."
                    )
                else:
                    self._stats["skipped"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"메시지 처리 오류: {e}")
            finally:
                self._queue.task_done()

        logger.info("메시지 처리 워커 종료")

    async def start(self, video_id: str) -> None:
        """
        채팅 수신 및 처리를 시작합니다.

        Args:
            video_id: 유튜브 영상 ID (라이브 방송 중이어야 함)
        """
        if self._running:
            logger.warning("이미 실행 중입니다.")
            return

        self._video_id = video_id
        self._running = True

        logger.info(f"ChatPipeline 시작: video_id={video_id}")

        # liveChatId 조회
        try:
            self._live_chat_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.reader.get_live_chat_id(video_id)
            )
        except Exception as e:
            self._running = False
            logger.error(f"liveChatId 조회 실패: {e}")
            raise

        # 종료 시그널 처리 (Ctrl+C)
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))
            except NotImplementedError:
                # Windows에서는 add_signal_handler가 지원되지 않을 수 있음
                pass

        logger.info(f"채팅 수신 시작 (liveChatId: {self._live_chat_id})")
        logger.info("Ctrl+C로 종료할 수 있습니다.")

        # 폴링 태스크 + 워커 태스크 병렬 실행
        polling_task = asyncio.create_task(
            self.reader.start_polling(self._live_chat_id, self._enqueue_message)
        )
        worker_task = asyncio.create_task(self._process_worker())

        try:
            await asyncio.gather(polling_task, worker_task)
        except asyncio.CancelledError:
            logger.info("태스크가 취소되었습니다.")
        except Exception as e:
            logger.error(f"파이프라인 오류: {e}")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """파이프라인을 정상 종료합니다."""
        if not self._running:
            return
        logger.info("ChatPipeline 종료 중...")
        self._running = False
        self.reader.stop()

        # 남은 메시지 처리 완료 대기 (최대 5초)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("남은 메시지 처리 시간 초과")

        logger.info(f"ChatPipeline 종료 완료. 통계: {self._stats}")

    def stop(self) -> None:
        """동기식 종료 (외부에서 호출 시)."""
        self._running = False
        self.reader.stop()
        logger.info("ChatPipeline 중지 요청")


# ─── 단독 실행 ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python chat_pipeline.py <video_id>")
        print("예시:   python chat_pipeline.py dQw4w9WgXcQ")
        sys.exit(1)

    video_id = sys.argv[1]
    pipeline = ChatPipeline()

    try:
        asyncio.run(pipeline.start(video_id))
    except KeyboardInterrupt:
        print("\n종료합니다.")
