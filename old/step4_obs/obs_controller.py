# ============================================================
# obs_controller.py - OBS WebSocket 제어 모듈
# VTuber 자동화 파이프라인 Step 4
# obsws-python v1.x (OBS WebSocket v5) 기반
# ============================================================

import time
import logging
from typing import Optional, Dict, Any

import obsws_python as obs

from config_obs import (
    OBS_HOST, OBS_PORT, OBS_PASSWORD,
    RECORDING_PATH, SCENE_MAIN, AUTO_SWITCH_SCENE,
    CONNECTION_TIMEOUT, MAX_RETRY
)

logger = logging.getLogger(__name__)


class OBSConnectionError(Exception):
    """OBS 연결 실패 예외"""
    pass


class OBSController:
    """
    OBS WebSocket v5 제어 클래스.
    obsws-python 라이브러리를 통해 OBS를 원격 제어합니다.
    """

    def __init__(self, host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD):
        self.host = host
        self.port = port
        self.password = password
        self._client = None
        self._connected = False

    def connect(self):
        for attempt in range(1, MAX_RETRY + 1):
            try:
                logger.info("[OBS] 연결 시도 %d/%d -> %s:%d", attempt, MAX_RETRY, self.host, self.port)
                self._client = obs.ReqClient(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    timeout=CONNECTION_TIMEOUT,
                )
                self._connected = True
                ver = self._client.get_version()
                logger.info("[OBS] 연결 성공 | OBS %s | WebSocket %s", ver.obs_version, ver.obs_web_socket_version)
                if RECORDING_PATH:
                    try:
                        self._client.set_record_directory(record_directory=RECORDING_PATH)
                        logger.info("[OBS] 녹화 경로 설정: %s", RECORDING_PATH)
                    except Exception as e:
                        logger.warning("[OBS] 녹화 경로 설정 실패: %s", e)
                return True
            except Exception as e:
                logger.warning("[OBS] 연결 실패 (%d/%d): %s", attempt, MAX_RETRY, e)
                self._connected = False
                if attempt < MAX_RETRY:
                    time.sleep(2)
        raise OBSConnectionError(f"OBS 연결 실패 ({self.host}:{self.port})")

    def disconnect(self):
        if self._client and self._connected:
            try:
                self._client.disconnect()
                logger.info("[OBS] 연결 해제 완료")
            except Exception as e:
                logger.warning("[OBS] 연결 해제 중 오류: %s", e)
            finally:
                self._client = None
                self._connected = False

    def _ensure_connected(self):
        if not self._connected or self._client is None:
            raise OBSConnectionError("OBS에 연결되어 있지 않습니다. connect()를 먼저 호출하세요.")

    # --- 녹화 제어 ---

    def start_recording(self):
        """OBS 녹화를 시작합니다. 반환: bool"""
        self._ensure_connected()
        try:
            status = self._client.get_record_status()
            if status.output_active:
                logger.info("[OBS] 이미 녹화 중입니다.")
                return True
            self._client.start_record()
            logger.info("[OBS] 녹화 시작")
            return True
        except Exception as e:
            logger.error("[OBS] 녹화 시작 실패: %s", e)
            return False

    def stop_recording(self):
        """OBS 녹화를 정지하고 파일 경로를 반환합니다. 반환: str | None"""
        self._ensure_connected()
        try:
            status = self._client.get_record_status()
            if not status.output_active:
                logger.info("[OBS] 녹화 중이 아닙니다.")
                return None
            resp = self._client.stop_record()
            output_path = getattr(resp, "output_path", None)
            logger.info("[OBS] 녹화 정지 | 파일: %s", output_path)
            return output_path
        except Exception as e:
            logger.error("[OBS] 녹화 정지 실패: %s", e)
            return None

    def get_recording_path(self):
        """현재 녹화 파일 경로 반환 (녹화 중일 때만 유효). 반환: str | None"""
        self._ensure_connected()
        try:
            status = self._client.get_record_status()
            if status.output_active:
                return getattr(status, "output_path", None)
            return None
        except Exception as e:
            logger.error("[OBS] 녹화 경로 조회 실패: %s", e)
            return None

    # --- 스트리밍 제어 ---

    def start_streaming(self):
        """OBS 스트리밍을 시작합니다. 반환: bool"""
        self._ensure_connected()
        try:
            status = self._client.get_stream_status()
            if status.output_active:
                logger.info("[OBS] 이미 스트리밍 중입니다.")
                return True
            self._client.start_stream()
            logger.info("[OBS] 스트리밍 시작")
            return True
        except Exception as e:
            logger.error("[OBS] 스트리밍 시작 실패: %s", e)
            return False

    def stop_streaming(self):
        """OBS 스트리밍을 정지합니다. 반환: bool"""
        self._ensure_connected()
        try:
            status = self._client.get_stream_status()
            if not status.output_active:
                logger.info("[OBS] 스트리밍 중이 아닙니다.")
                return True
            self._client.stop_stream()
            logger.info("[OBS] 스트리밍 정지")
            return True
        except Exception as e:
            logger.error("[OBS] 스트리밍 정지 실패: %s", e)
            return False

    # --- 상태 조회 ---

    def get_status(self):
        """OBS 현재 상태 반환. 반환: dict"""
        base = {
            "connected": self._connected,
            "recording": False,
            "streaming": False,
            "current_scene": None,
            "recording_path": None,
            "obs_version": None,
        }
        if not self._connected or self._client is None:
            return base
        try:
            rec = self._client.get_record_status()
            base["recording"] = bool(rec.output_active)
            if rec.output_active:
                base["recording_path"] = getattr(rec, "output_path", None)
        except Exception as e:
            logger.warning("[OBS] 녹화 상태 조회 실패: %s", e)
        try:
            st = self._client.get_stream_status()
            base["streaming"] = bool(st.output_active)
        except Exception as e:
            logger.warning("[OBS] 스트리밍 상태 조회 실패: %s", e)
        try:
            sc = self._client.get_current_program_scene()
            base["current_scene"] = getattr(sc, "current_program_scene_name", None)
        except Exception as e:
            logger.warning("[OBS] 씬 조회 실패: %s", e)
        try:
            ver = self._client.get_version()
            base["obs_version"] = ver.obs_version
        except Exception as e:
            logger.warning("[OBS] 버전 조회 실패: %s", e)
        return base

    # --- 씬 제어 ---

    def switch_scene(self, scene_name):
        """지정 씬으로 전환. 반환: bool"""
        self._ensure_connected()
        try:
            self._client.set_current_program_scene(scene_name=scene_name)
            logger.info("[OBS] 씬 전환 -> %s", scene_name)
            return True
        except Exception as e:
            logger.error("[OBS] 씬 전환 실패 (%s): %s", scene_name, e)
            return False

    def get_scene_list(self):
        """등록된 모든 씬 이름 목록 반환. 반환: list[str]"""
        self._ensure_connected()
        try:
            resp = self._client.get_scene_list()
            scenes = getattr(resp, "scenes", [])
            return [s.get("sceneName", "") for s in scenes if isinstance(s, dict)]
        except Exception as e:
            logger.error("[OBS] 씬 목록 조회 실패: %s", e)
            return []

    # --- 소스 제어 ---

    def set_source_visible(self, source_name, visible, scene_name=None):
        """소스 표시/숨김 변경. 반환: bool"""
        self._ensure_connected()
        try:
            if scene_name is None:
                sc = self._client.get_current_program_scene()
                scene_name = sc.current_program_scene_name
            items_resp = self._client.get_scene_item_list(scene_name=scene_name)
            items = getattr(items_resp, "scene_items", [])
            item_id = None
            for item in items:
                if isinstance(item, dict) and item.get("sourceName") == source_name:
                    item_id = item.get("sceneItemId")
                    break
            if item_id is None:
                logger.warning("[OBS] 소스를 찾을 수 없음: %s (씬: %s)", source_name, scene_name)
                return False
            self._client.set_scene_item_enabled(
                scene_name=scene_name,
                scene_item_id=item_id,
                scene_item_enabled=visible,
            )
            logger.info("[OBS] 소스 %s -> %s", source_name, "표시" if visible else "숨김")
            return True
        except Exception as e:
            logger.error("[OBS] 소스 상태 변경 실패 (%s): %s", source_name, e)
            return False

    # --- 리플레이 버퍼 ---

    def save_replay_buffer(self):
        """리플레이 버퍼 저장. 반환: bool"""
        self._ensure_connected()
        try:
            self._client.save_replay_buffer()
            logger.info("[OBS] 리플레이 버퍼 저장 완료")
            return True
        except Exception as e:
            logger.error("[OBS] 리플레이 버퍼 저장 실패: %s", e)
            return False

    # --- Context Manager ---

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
