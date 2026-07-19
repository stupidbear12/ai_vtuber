# -*- coding: utf-8 -*-
"""
app/memory.py — RAG 기반 대화 기억 시스템 (ChromaDB PersistentClient)

기능:
  - 로컬 ChromaDB PersistentClient로 과거 대화 저장/검색 (Docker 불필요)
  - chromadb.DefaultEmbeddingFunction(onnxruntime) 으로 임베딩 생성
  - 캐릭터 지식 베이스 관리 (data/knowledge/*.md)
  - 방송 시청자 이름/닉네임 매핑 기억
  - 크롤링/외부 데이터 RAG 저장

환경변수:
  CHROMA_DATA_DIR  — ChromaDB 데이터 저장 디렉토리 (기본: ./data/chromadb)
  CHAT_DISABLE_RAG — 1 이면 RAG 전체 비활성화
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_KNOWLEDGE_PATH = str(_DATA_DIR / "knowledge")
_VIEWER_CACHE_PATH = str(_DATA_DIR / "viewer_names.json")


class MemoryEngine:
    """RAG 기반 대화 기억 엔진 (ChromaDB PersistentClient).

    로컬 파일 기반 ChromaDB PersistentClient를 사용한다 (Docker 불필요).
    두 가지 컬렉션을 관리한다:
      conversations — 과거 대화 Q&A 쌍 (유사 대화 검색용)
      knowledge     — 캐릭터 지식 베이스 문서 청크 + 크롤링 데이터

    초기화는 첫 호출 시 지연 실행된다.
    """

    def __init__(self):
        self._client = None
        self._conv_collection = None
        self._knowledge_collection = None
        self._viewer_cache: dict = {}
        self._load_viewer_cache()

    # ── 초기화 (지연 로딩) ────────────────────────────────────────

    def _init_db(self):
        """ChromaDB PersistentClient 및 컬렉션 초기화."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        data_dir = os.environ.get("CHROMA_DATA_DIR", "./data/chromadb")
        # 상대 경로면 chat 모듈 기준으로 해석
        if not os.path.isabs(data_dir):
            data_dir = str(Path(__file__).parent.parent / data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=data_dir)
        logger.info(f"[Memory] ChromaDB PersistentClient 초기화: {data_dir}")

        # sentence-transformers 없이 onnxruntime 기반 경량 임베딩 사용
        ef = DefaultEmbeddingFunction()

        dist_meta = {"hnsw:space": "cosine"}

        self._conv_collection = self._client.get_or_create_collection(
            name="conversations",
            embedding_function=ef,
            metadata={**dist_meta, "description": "시온 대화 기록"},
        )
        self._knowledge_collection = self._client.get_or_create_collection(
            name="knowledge",
            embedding_function=ef,
            metadata={**dist_meta, "description": "시온 캐릭터 지식 베이스"},
        )

    # ── 시청자 이름 관리 ─────────────────────────────────────────

    def _load_viewer_cache(self):
        try:
            if os.path.exists(_VIEWER_CACHE_PATH):
                with open(_VIEWER_CACHE_PATH, "r", encoding="utf-8") as f:
                    self._viewer_cache = json.load(f)
        except Exception as e:
            logger.warning(f"[Memory] 시청자 캐시 로드 실패: {e}")
            self._viewer_cache = {}

    def _save_viewer_cache(self):
        try:
            os.makedirs(os.path.dirname(_VIEWER_CACHE_PATH), exist_ok=True)
            with open(_VIEWER_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._viewer_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Memory] 시청자 캐시 저장 실패: {e}")

    def remember_viewer(self, viewer_id: str, viewer_name: str):
        if self._viewer_cache.get(viewer_id) != viewer_name:
            self._viewer_cache[viewer_id] = viewer_name
            self._save_viewer_cache()

    def get_viewer_name(self, viewer_id: str) -> Optional[str]:
        return self._viewer_cache.get(viewer_id)

    # ── 대화 저장 ─────────────────────────────────────────────────

    def save_conversation_sync(
        self,
        user_msg: str,
        bot_reply: str,
        mode: str,
        viewer_name: Optional[str] = None,
    ):
        """대화 Q&A 쌍을 ChromaDB에 저장 (동기)."""
        try:
            self._init_db()

            combined = f"{user_msg} {bot_reply}"
            doc_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            metadata = {
                "user_msg": user_msg[:500],
                "bot_reply": bot_reply[:500],
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
            }
            if viewer_name:
                metadata["viewer_name"] = viewer_name

            self._conv_collection.add(
                ids=[doc_id],
                documents=[combined],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.warning(f"[Memory] 대화 저장 실패: {e}")

    async def save_conversation(
        self,
        user_msg: str,
        bot_reply: str,
        mode: str,
        viewer_name: Optional[str] = None,
    ):
        """대화 저장 비동기 래퍼."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.save_conversation_sync(user_msg, bot_reply, mode, viewer_name),
        )

    # ── 대화 기억 검색 ────────────────────────────────────────────

    def search_memories_sync(
        self,
        query: str,
        n_results: int = 3,
        mode: Optional[str] = None,
    ) -> List[str]:
        """과거 대화에서 관련 기억 검색 (동기)."""
        try:
            self._init_db()
            total = self._conv_collection.count()
            if total == 0:
                return []

            where = {"mode": mode} if mode else None

            results = self._conv_collection.query(
                query_texts=[query],
                n_results=min(n_results, total),
                where=where,
                include=["metadatas", "distances"],
            )

            memories = []
            if results and results["metadatas"]:
                for meta, dist in zip(
                    results["metadatas"][0], results["distances"][0]
                ):
                    # 코사인 거리 > 1.0 이면 관련성 낮음
                    if dist > 1.0:
                        continue
                    user_msg = meta.get("user_msg", "")
                    bot_reply = meta.get("bot_reply", "")
                    viewer = meta.get("viewer_name", "")
                    prefix = f"{viewer}: " if viewer else "사용자: "
                    memories.append(f"{prefix}{user_msg} → 시온: {bot_reply}")

            return memories
        except Exception as e:
            logger.warning(f"[Memory] 대화 기억 검색 실패: {e}")
            return []

    async def search_memories(
        self,
        query: str,
        n_results: int = 3,
        mode: Optional[str] = None,
        timeout: float = 2.0,
    ) -> List[str]:
        """과거 대화 검색 (비동기, 타임아웃 지원)."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.search_memories_sync(query, n_results, mode),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[Memory] 대화 기억 검색 타임아웃 ({timeout}s)")
            return []
        except Exception as e:
            logger.warning(f"[Memory] 대화 기억 검색 오류: {e}")
            return []

    # ── 지식 베이스 검색 ──────────────────────────────────────────

    def search_knowledge_sync(
        self,
        query: str,
        n_results: int = 2,
    ) -> List[str]:
        """캐릭터 지식 베이스에서 관련 내용 검색 (동기)."""
        try:
            self._init_db()
            total = self._knowledge_collection.count()
            if total == 0:
                return []

            results = self._knowledge_collection.query(
                query_texts=[query],
                n_results=min(n_results, total),
                include=["documents", "distances"],
            )

            docs = []
            if results and results["documents"]:
                for doc, dist in zip(
                    results["documents"][0], results["distances"][0]
                ):
                    # 지식은 더 엄격한 기준 (cosine_dist < 0.8)
                    if dist > 0.8:
                        continue
                    docs.append(doc)
            return docs
        except Exception as e:
            logger.warning(f"[Memory] 지식 베이스 검색 실패: {e}")
            return []

    async def search_knowledge(
        self,
        query: str,
        n_results: int = 2,
        timeout: float = 2.0,
    ) -> List[str]:
        """캐릭터 지식 베이스 검색 (비동기, 타임아웃 지원)."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.search_knowledge_sync(query, n_results),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[Memory] 지식 베이스 검색 타임아웃 ({timeout}s)")
            return []
        except Exception as e:
            logger.warning(f"[Memory] 지식 베이스 검색 오류: {e}")
            return []

    # ── 지식 베이스 로드 ──────────────────────────────────────────

    def load_knowledge_docs(self, docs_dir: str = _DEFAULT_KNOWLEDGE_PATH):
        """data/knowledge/ 의 .md/.txt 파일을 ChromaDB에 로드한다.

        서버 시작 시 1회 호출. 이미 로드된 경우 스킵.
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            logger.warning(f"[Memory] 지식 문서 디렉토리 없음: {docs_dir}")
            return

        try:
            self._init_db()

            if self._knowledge_collection.count() > 0:
                logger.info(
                    f"[Memory] 지식 베이스 이미 로드됨 "
                    f"({self._knowledge_collection.count()}개 청크) — 스킵"
                )
                return

            files = sorted(
                list(docs_path.glob("*.md")) + list(docs_path.glob("*.txt"))
            )
            if not files:
                logger.warning(f"[Memory] 지식 문서 파일 없음: {docs_dir}")
                return

            ids, documents, metadatas = [], [], []

            for file in files:
                text = file.read_text(encoding="utf-8")
                chunks = self._split_text(text, chunk_size=400, overlap=50)

                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    ids.append(f"know_{file.stem}_{i}")
                    documents.append(chunk)
                    metadatas.append({"source": file.name, "chunk": i})

            if documents:
                self._knowledge_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(
                    f"[Memory] 지식 베이스 로드 완료: "
                    f"{len(files)}개 파일, {len(documents)}개 청크"
                )
        except Exception as e:
            logger.error(f"[Memory] 지식 베이스 로드 실패: {e}")

    @staticmethod
    def _split_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """텍스트를 문단 단위로 청크 분할."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}".strip() if current else para
            else:
                if current:
                    chunks.append(current)
                    tail = current[-overlap:] if len(current) > overlap else current
                    current = f"{tail}\n\n{para}".strip()
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    # ── 크롤링/외부 데이터 저장 ─────────────────────────────────

    def store_crawl_sync(
        self,
        content: str,
        source: str = "crawl",
        metadata: Optional[dict] = None,
    ):
        """크롤링/외부 데이터를 knowledge 컬렉션에 저장 (동기)."""
        try:
            self._init_db()

            chunks = self._split_text(content, chunk_size=400, overlap=50)
            if not chunks:
                return 0

            ids, documents, metadatas = [], [], []
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                ids.append(f"crawl_{ts}_{i}")
                documents.append(chunk)
                meta = {
                    "source": source,
                    "chunk": i,
                    "timestamp": datetime.now().isoformat(),
                    "type": "crawl",
                }
                if metadata:
                    meta.update(metadata)
                metadatas.append(meta)

            if documents:
                self._knowledge_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(
                    f"[Memory] 크롤링 데이터 저장: {len(documents)}개 청크 (source={source})"
                )
            return len(documents)
        except Exception as e:
            logger.warning(f"[Memory] 크롤링 데이터 저장 실패: {e}")
            return 0

    async def store_crawl(
        self,
        content: str,
        source: str = "crawl",
        metadata: Optional[dict] = None,
    ) -> int:
        """크롤링 데이터 저장 비동기 래퍼. 저장된 청크 수를 반환."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.store_crawl_sync(content, source, metadata),
        )

    # ── 통계 ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """DB 사용 현황 통계를 반환한다."""
        data_dir = os.environ.get("CHROMA_DATA_DIR", "./data/chromadb")
        backend = f"PersistentClient({data_dir})"
        try:
            self._init_db()
            return {
                "conversations": self._conv_collection.count(),
                "knowledge_chunks": self._knowledge_collection.count(),
                "chroma_backend": backend,
                "viewer_count": len(self._viewer_cache),
            }
        except Exception as e:
            return {"error": str(e), "chroma_backend": backend}


# ── 싱글톤 ───────────────────────────────────────────────────────

_engine_instance: Optional[MemoryEngine] = None


def get_memory_engine() -> MemoryEngine:
    """MemoryEngine 싱글톤을 반환한다."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MemoryEngine()
    return _engine_instance
