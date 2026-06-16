# -*- coding: utf-8 -*-
"""
TrackQueue — DJ 트랙 대기열 관리
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from .music_engine import GenerationParams

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """큐에 들어가는 개별 요청."""
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    params: GenerationParams = field(default_factory=GenerationParams)
    requester: Optional[str] = None
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending | generating | ready | failed
    file_path: Optional[str] = None
    error: Optional[str] = None


class TrackQueue:
    """우선순위 기반 트랙 대기열."""

    def __init__(self, max_size: int = 20):
        self._max_size = max_size
        self._items: List[QueueItem] = []
        self._lock = asyncio.Lock()

    def size(self) -> int:
        return len(self._items)

    def is_full(self) -> bool:
        return len(self._items) >= self._max_size

    async def enqueue(
        self,
        params: GenerationParams,
        requester: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        async with self._lock:
            if self.is_full():
                raise QueueFullError(f"Queue is full (max={self._max_size})")

            item = QueueItem(params=params, requester=requester, priority=priority)
            self._items.append(item)
            self._items.sort(key=lambda x: (-x.priority, x.created_at))
            logger.info("Enqueued track request %s (priority=%d)", item.item_id, priority)
            return item.item_id

    async def dequeue(self) -> Optional[QueueItem]:
        async with self._lock:
            for item in self._items:
                if item.status == "pending":
                    item.status = "generating"
                    return item
            return None

    async def mark_ready(self, item_id: str, file_path: str) -> None:
        async with self._lock:
            for item in self._items:
                if item.item_id == item_id:
                    item.status = "ready"
                    item.file_path = file_path
                    return

    async def mark_failed(self, item_id: str, error: str) -> None:
        async with self._lock:
            for item in self._items:
                if item.item_id == item_id:
                    item.status = "failed"
                    item.error = error
                    return

    async def remove(self, item_id: str) -> bool:
        async with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i.item_id != item_id]
            return len(self._items) < before

    async def move(self, item_id: str, new_position: int) -> bool:
        async with self._lock:
            idx = next((i for i, x in enumerate(self._items) if x.item_id == item_id), None)
            if idx is None:
                return False
            item = self._items.pop(idx)
            new_position = max(0, min(new_position, len(self._items)))
            self._items.insert(new_position, item)
            return True

    def list_all(self) -> List[dict]:
        return [
            {
                "item_id": item.item_id,
                "prompt": item.params.prompt,
                "genre": getattr(item.params, "genre", None),
                "bpm": item.params.bpm,
                "duration": item.params.duration,
                "requester": item.requester,
                "priority": item.priority,
                "status": item.status,
                "created_at": item.created_at.isoformat(),
                "file_path": item.file_path,
                "error": item.error,
            }
            for item in self._items
        ]

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._items)
            self._items.clear()
            return count


class QueueFullError(Exception):
    """큐가 꽉 찼을 때 발생."""
    pass
