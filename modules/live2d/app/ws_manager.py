# -*- coding: utf-8 -*-
"""
app/ws_manager.py — WebSocket 연결 관리자

역할:
  - 브라우저(웹 뷰어, Electron 펫) WebSocket 연결을 관리
  - 연결된 모든 클라이언트에게 JSON 메시지 브로드캐스트
  - 연결이 끊긴 클라이언트 자동 정리
"""

import json
from typing import Set

from fastapi import WebSocket


class WSManager:
    """WebSocket 클라이언트 연결 풀 관리자.

    FastAPI WebSocket 엔드포인트에서 인스턴스를 공유해
    여러 브라우저 창/Electron 창에 동시에 명령을 전송할 수 있다.
    """

    def __init__(self):
        # 연결된 WebSocket 클라이언트 집합
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        """새 WebSocket 연결을 수락하고 클라이언트 풀에 추가."""
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        """클라이언트 풀에서 WebSocket 제거 (연결 해제 시 호출)."""
        self._clients.discard(ws)

    async def broadcast(self, msg: dict) -> None:
        """연결된 모든 클라이언트에게 JSON 메시지를 브로드캐스트.

        전송 실패한(연결 끊긴) 클라이언트는 자동으로 풀에서 제거된다.

        Args:
            msg: 전송할 딕셔너리 (JSON으로 직렬화됨)
        """
        if not self._clients:
            return

        payload = json.dumps(msg, ensure_ascii=False)
        dead: Set[WebSocket] = set()

        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                # 전송 실패 = 연결 끊김 → 제거 대상으로 표시
                dead.add(ws)

        # 끊어진 클라이언트 일괄 제거
        self._clients -= dead

    @property
    def count(self) -> int:
        """현재 연결된 클라이언트 수."""
        return len(self._clients)


# 모듈 전역 싱글턴 — 라우터와 main에서 동일 인스턴스 공유
ws_manager = WSManager()
