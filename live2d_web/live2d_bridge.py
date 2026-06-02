# -*- coding: utf-8 -*-
"""
live2d_web/live2d_bridge.py - Python → 웹 Live2D 컨트롤러

기존 VTubeController / AnimationController와 동일한 인터페이스를 제공합니다.
REST API를 통해 브라우저의 Live2D 모델 파라미터를 제어합니다.

사용 예:
    ctrl = Live2DWebController()
    await ctrl.start_idle()
    await ctrl.set_emotion("happy")
    await ctrl.lipsync(frames)   # analyze_audio() 결과 그대로 사용
    await ctrl.trigger_reaction("nod")
    await ctrl.close()
"""

import asyncio
from typing import Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore


class Live2DWebController:
    """
    웹 브라우저에서 실행 중인 Live2D 모델을 HTTP REST로 제어.

    api/main.py 서버(기본 http://localhost:8000)에 연결합니다.
    브라우저가 /live2d/ws WebSocket에 접속 중이어야 명령이 전달됩니다.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session: Optional["aiohttp.ClientSession"] = None

    async def _get_session(self) -> "aiohttp.ClientSession":
        if aiohttp is None:
            raise RuntimeError("aiohttp가 필요합니다: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _post(self, path: str, data: dict) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        async with session.post(url, json=data) as resp:
            return await resp.json()

    # ── 공개 API ─────────────────────────────────────────────────

    async def set_params(self, params: dict) -> dict:
        """
        파라미터 직접 설정.
        params 예: {"ParamAngleX": 15.0, "ParamEyeLOpen": 0.8}
        """
        return await self._post("/live2d/params", {"params": params})

    async def set_emotion(self, emotion: str) -> dict:
        """감정 전환 (calm / happy / surprised / thinking)."""
        return await self._post("/live2d/emotion", {"emotion": emotion})

    async def set_mouth(self, value: float) -> dict:
        """입 열림 값 설정 (0.0 ~ 1.0)."""
        return await self._post("/live2d/mouth", {"value": value})

    async def clear_mouth(self) -> dict:
        """립싱크 레이어 제거 (입 닫기)."""
        return await self._post("/live2d/mouth/clear", {})

    async def trigger_reaction(self, name: str) -> dict:
        """반응 애니메이션 트리거 (nod / shake / surprised / superchat)."""
        return await self._post("/live2d/reaction", {"name": name})

    async def start_idle(self) -> dict:
        """Idle 애니메이션 시작 (호흡 + 눈 깜빡임 + 고개 흔들림)."""
        return await self._post("/live2d/idle/start", {})

    async def stop_idle(self) -> dict:
        """Idle 애니메이션 정지."""
        return await self._post("/live2d/idle/stop", {})

    async def status(self) -> dict:
        """연결된 브라우저 클라이언트 수 반환."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/live2d/status") as resp:
            return await resp.json()

    async def lipsync(self, frames: list) -> None:
        """
        립싱크 재생.

        frames 포맷은 step3_vtube/lipsync.py > analyze_audio() 반환값과 동일:
            [(timestamp_ms: float, mouth_value: float), ...]

        타임스탬프에 맞춰 mouth 파라미터를 전송하고 종료 시 입을 닫습니다.
        """
        if not frames:
            return

        loop = asyncio.get_event_loop()
        start = loop.time()
        try:
            for ts_ms, mouth_val in frames:
                target = start + ts_ms / 1000.0
                wait = target - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                await self.set_mouth(float(mouth_val))
        finally:
            await self.set_mouth(0.0)
            await asyncio.sleep(0.1)
            await self.clear_mouth()

    async def speak(self, audio_path: str, emotion: str = "calm") -> None:
        """
        말하기 시퀀스 (VTubeController.speak()와 호환).
        lipsync.analyze_audio()로 오디오를 분석한 뒤 립싱크를 재생합니다.
        """
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step3_vtube'))
            from lipsync import analyze_audio
        except ImportError:
            print("[Live2DBridge] lipsync 모듈 없음 — analyze_audio를 import할 수 없습니다")
            return

        await self.set_emotion(emotion)

        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(None, analyze_audio, audio_path)
        await self.lipsync(frames)

    async def close(self) -> None:
        """HTTP 세션 종료."""
        if self._session and not self._session.closed:
            await self._session.close()
