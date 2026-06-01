# ============================================================
# live_pipeline.py - 라이브 방송 시작/종료 오케스트레이터
# VTuber 자동화 파이프라인 Step 5 (Live)
#
# 흐름:
#   YouTube 방송 생성 + 스트림 키 획득
#   → OBS에 스트림 키 자동 설정
#   → OBS 스트리밍 시작
#   → 방송 상태 live 전환
#   → (종료 시) complete 전환 + OBS 스트리밍 정지
# ============================================================

import os
import sys
import time
import logging
from typing import Optional, Dict, Any

# step4_obs를 import 경로에 추가
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "step4_obs"))

from youtube_live import YouTubeLiveManager, YouTubeLiveError
from config_live import DEFAULT_TITLE, DEFAULT_DESCRIPTION, DEFAULT_PRIVACY

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# OBS 스트림 키 설정 헬퍼
# ------------------------------------------------------------------
def _set_obs_stream_key(obs_client, rtmp_url: str, stream_key: str) -> None:
    """
    OBS WebSocket v5의 SetStreamServiceSettings 요청으로
    RTMP URL과 스트림 키를 설정합니다.

    Args:
        obs_client: OBSController._client (obsws_python ReqClient)
        rtmp_url: RTMP 수신 주소 (예: rtmp://a.rtmp.youtube.com/live2)
        stream_key: YouTube 스트림 키
    """
    try:
        obs_client.set_stream_service_settings(
            stream_service_type="rtmp_custom",
            stream_service_settings={
                "server": rtmp_url,
                "key": stream_key,
            },
        )
        logger.info("[OBS] 스트림 키 설정 완료 (RTMP: %s)", rtmp_url)
    except Exception as e:
        raise RuntimeError(f"OBS 스트림 키 설정 실패: {e}") from e


# ------------------------------------------------------------------
# LivePipeline
# ------------------------------------------------------------------
class LivePipeline:
    """
    YouTube 라이브 방송과 OBS를 연동하여 방송 시작/종료를 자동화합니다.

    사용 예:
        pipeline = LivePipeline()
        result = pipeline.start_live("내 방송 제목", "방송 설명", "public")
        print("방송 URL:", result["watch_url"])
        # ...
        pipeline.end_live(result["broadcast_id"])
    """

    def __init__(self):
        self._yt_manager = YouTubeLiveManager()
        self._obs_controller = None
        self._current_broadcast_id: Optional[str] = None

    # ------------------------------------------------------------------
    # OBS 컨트롤러 초기화 (지연 로딩)
    # ------------------------------------------------------------------
    def _get_obs_controller(self):
        """OBS 컨트롤러 싱글턴 반환. 미연결 시 자동 연결 시도."""
        if self._obs_controller is None:
            try:
                from obs_controller import OBSController
                self._obs_controller = OBSController()
            except ImportError as e:
                raise RuntimeError(
                    f"obs_controller 모듈을 찾을 수 없습니다: {e}\n"
                    f"step4_obs 디렉토리가 올바르게 구성되어 있는지 확인하세요."
                ) from e

        if not self._obs_controller._connected:
            logger.info("[OBS] 연결 시도 중...")
            self._obs_controller.connect()
            logger.info("[OBS] 연결 완료")

        return self._obs_controller

    # ------------------------------------------------------------------
    # 방송 시작
    # ------------------------------------------------------------------
    def start_live(
        self,
        title: str = DEFAULT_TITLE,
        description: str = DEFAULT_DESCRIPTION,
        privacy: str = DEFAULT_PRIVACY,
    ) -> Dict[str, Any]:
        """
        전체 라이브 방송 시작 흐름을 자동화합니다.

        1. YouTube 인증
        2. 방송 + 스트림 생성 및 연결 → 스트림 키 획득
        3. OBS에 스트림 키 자동 설정
        4. OBS 스트리밍 시작
        5. 방송 상태 live 전환

        Args:
            title: 방송 제목
            description: 방송 설명
            privacy: 공개 설정 (public / unlisted / private)

        Returns:
            {
                "broadcast_id": str,
                "stream_id": str,
                "rtmp_url": str,
                "stream_key": str,
                "watch_url": str,
            }
        """
        logger.info("=" * 60)
        logger.info("[Pipeline] 라이브 방송 시작: '%s'", title)
        logger.info("=" * 60)

        # [1] YouTube 인증
        logger.info("[Step 1/5] YouTube OAuth 인증")
        self._yt_manager.authenticate()

        # [2] 방송 생성 + 스트림 생성 + 연결
        logger.info("[Step 2/5] YouTube 방송 및 스트림 생성")
        setup_result = self._yt_manager.setup_and_start(title, description, privacy)
        broadcast_id = setup_result["broadcast_id"]
        rtmp_url = setup_result["rtmp_url"]
        stream_key = setup_result["stream_key"]
        self._current_broadcast_id = broadcast_id

        logger.info(
            "[Step 2/5] 완료 → broadcast_id=%s | watch_url=%s",
            broadcast_id,
            setup_result["watch_url"],
        )

        # [3] OBS 스트림 키 설정
        logger.info("[Step 3/5] OBS 스트림 키 설정")
        obs_ctrl = self._get_obs_controller()
        _set_obs_stream_key(obs_ctrl._client, rtmp_url, stream_key)

        # [4] OBS 스트리밍 시작
        logger.info("[Step 4/5] OBS 스트리밍 시작")
        ok = obs_ctrl.start_streaming()
        if not ok:
            raise RuntimeError("OBS 스트리밍 시작에 실패했습니다.")
        logger.info("[Step 4/5] OBS 스트리밍 시작 완료")

        # [5] YouTube 방송 상태 → live
        logger.info("[Step 5/5] YouTube 방송 상태 live 전환")
        self._yt_manager.go_live(broadcast_id, wait_for_stream=True)

        logger.info("=" * 60)
        logger.info("[Pipeline] 방송 LIVE 시작 완료!")
        logger.info("[Pipeline] 시청 URL: %s", setup_result["watch_url"])
        logger.info("=" * 60)

        return setup_result

    # ------------------------------------------------------------------
    # 방송 종료
    # ------------------------------------------------------------------
    def end_live(self, broadcast_id: Optional[str] = None) -> Dict[str, Any]:
        """
        라이브 방송을 종료합니다.

        1. YouTube 방송 → complete 상태 전환
        2. OBS 스트리밍 정지

        Args:
            broadcast_id: 종료할 방송 ID.
                          None이면 마지막으로 시작한 방송 ID를 사용합니다.

        Returns:
            {"broadcast_id": str, "obs_stopped": bool, "youtube_ended": bool}
        """
        bid = broadcast_id or self._current_broadcast_id
        if not bid:
            raise ValueError(
                "broadcast_id가 없습니다. "
                "start_live() 반환값의 broadcast_id를 전달하세요."
            )

        logger.info("=" * 60)
        logger.info("[Pipeline] 라이브 방송 종료: broadcast_id=%s", bid)
        logger.info("=" * 60)

        result = {"broadcast_id": bid, "obs_stopped": False, "youtube_ended": False}

        # [1] YouTube 방송 종료 (complete)
        logger.info("[Step 1/2] YouTube 방송 complete 전환")
        try:
            self._yt_manager.end_broadcast(bid)
            result["youtube_ended"] = True
            logger.info("[Step 1/2] YouTube 방송 종료 완료")
        except Exception as e:
            logger.error("[Step 1/2] YouTube 방송 종료 실패: %s", e)

        # [2] OBS 스트리밍 정지
        logger.info("[Step 2/2] OBS 스트리밍 정지")
        try:
            obs_ctrl = self._get_obs_controller()
            ok = obs_ctrl.stop_streaming()
            result["obs_stopped"] = bool(ok)
            logger.info("[Step 2/2] OBS 스트리밍 정지 완료")
        except Exception as e:
            logger.error("[Step 2/2] OBS 스트리밍 정지 실패: %s", e)

        if bid == self._current_broadcast_id:
            self._current_broadcast_id = None

        logger.info("=" * 60)
        logger.info("[Pipeline] 방송 종료 완료: %s", result)
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # 현재 방송 상태 조회
    # ------------------------------------------------------------------
    def get_status(self, broadcast_id: Optional[str] = None) -> Dict[str, Any]:
        """
        현재 방송 상태를 조회합니다.

        Returns:
            {
                "broadcast_id": str | None,
                "youtube_status": str | None,
                "obs_streaming": bool,
            }
        """
        bid = broadcast_id or self._current_broadcast_id
        youtube_status = None
        obs_streaming = False

        if bid:
            try:
                self._yt_manager._ensure_authenticated()
                youtube_status = self._yt_manager.get_broadcast_status(bid)
            except Exception as e:
                logger.warning("[Status] YouTube 상태 조회 실패: %s", e)

        try:
            obs_ctrl = self._get_obs_controller()
            obs_info = obs_ctrl.get_status()
            obs_streaming = obs_info.get("streaming", False)
        except Exception as e:
            logger.warning("[Status] OBS 상태 조회 실패: %s", e)

        return {
            "broadcast_id": bid,
            "youtube_status": youtube_status,
            "obs_streaming": obs_streaming,
        }


# ------------------------------------------------------------------
# 직접 실행 (테스트용)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="YouTube 라이브 방송 파이프라인")
    parser.add_argument("action", choices=["start", "end", "status"], help="수행할 작업")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="방송 제목")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="방송 설명")
    parser.add_argument("--privacy", default=DEFAULT_PRIVACY, help="공개 설정")
    parser.add_argument("--broadcast-id", help="종료/상태 조회 시 방송 ID")
    args = parser.parse_args()

    pipeline = LivePipeline()

    if args.action == "start":
        result = pipeline.start_live(args.title, args.description, args.privacy)
        print(f"\n✅ 방송 시작 성공!")
        print(f"   broadcast_id: {result['broadcast_id']}")
        print(f"   시청 URL: {result['watch_url']}")

    elif args.action == "end":
        if not args.broadcast_id:
            print("❌ --broadcast-id 를 지정하세요.")
            sys.exit(1)
        result = pipeline.end_live(args.broadcast_id)
        print(f"\n✅ 방송 종료: {result}")

    elif args.action == "status":
        result = pipeline.get_status(args.broadcast_id)
        print(f"\n📊 방송 상태: {result}")
