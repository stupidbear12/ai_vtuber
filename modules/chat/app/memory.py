# -*- coding: utf-8 -*-
"""
app/memory.py — RAG 기반 대화 기억 시스템

기능:
  - ChromaDB 로컬 벡터 DB에 과거 대화 저장/검색
  - sentence-transformers(한국어 모델)로 임베딩 생성
  - 캐릭터 지식 베이스 관리 (data/knowledge/*.md)
  - 방송 시청자 이름/닉네임 매핑 기억

사용 모델:
  기본: jhgan/ko-sroberta-multitask (한국어 특화)
  폴백: paraphrase-multilingual-MiniLM-L12-v2 (경량 다국어)

참고: 첫 실행 시 HuggingFace에서 임베딩 모델이 자동 다운로드됩니다.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 한국어 임베딩 모델 (CPU 동작 가능)
_EMBEDDING_MODEL_PRIMARY = "jhgan/ko-sroberta-multitask"
_EMBEDDING_MODEL_FALLBACK = "paraphrase-multilingual-MiniLM-L12-v2"

# 기본 경로 (modules/chat/data/)
_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_DB_PATH = str(_DATA_DIR / "chroma_db")
_DEFAULT_KNOWLEDGE_PATH = str(_DATA_DIR / "knowledge")
_VIEWER_CACHE_PATH = str(_DATA_DIR / "viewer_names.json")


class MemoryEngine:
    """RAG 기반 대화 기억 엔진.

    ChromaDB를 영구 저장소로 사용하며 두 가지 컬렉션을 관리한다:
      conversations — 과거 대화 Q&A 쌍 (유사 대화 검색용)
      knowledge     — 캐릭터 지식 베이스 문서 청크

    ChromaDB와 임베딩 모델은 첫 호출 시 지연 초기화된다.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._db_path = db_path
        self._client = None
        self._conv_collection = None
        self._knowledge_collection = None
        self._embed_model = None
        self._embed_model_name: Optional[str] = None
        # 시청자 이름 매핑 (메모리 캐시 + JSON 파일 영속)
        self._viewer_cache: dict = {}
        self._load_viewer_cache()

    # ── 초기화 (지연 로딩) ────────────────────────────────────────

    def _init_db(self):
        """ChromaDB 클라이언트 및 컬렉션 초기화."""
        if self._client is not None:
            return

        try:
            import chromadb

            os.makedirs(self._db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._db_path)

            # 코사인 거리 사용 — 값이 낮을수록 유사 (범위: 0~2)
            dist_meta = {"hnsw:space": "cosine"}

            self._conv_collection = self._client.get_or_create_collection(
                name="conversations",
                metadata={**dist_meta, "description": "시온 대화 기록"},
            )
            self._knowledge_collection = self._client.get_or_create_collection(
                name="knowledge",
                metadata={**dist_meta, "description": "시온 캐릭터 지식 베이스"},
            )

            logger.info(f"[Memory] ChromaDB 초기화 완료: {self._db_path}")
        except ImportError:
            logger.error("[Memory] chromadb 패키지 없음 — pip install chromadb")
            raise

    def _get_embed_model(self):
        """임베딩 모델 반환 (지연 로딩, 한국어 모델 우선)."""
        if self._embed_model is not None:
            return self._embed_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("[Memory] sentence-transformers 패키지 없음")
            raise

        for model_name in (_EMBEDDING_MODEL_PRIMARY, _EMBEDDING_MODEL_FALLBACK):
            try:
                self._embed_model = SentenceTransformer(model_name)
                self._embed_model_name = model_name
                logger.info(f"[Memory] 임베딩 모델 로드: {model_name}")
                return self._embed_model
            except Exception as e:
                logger.warning(f"[Memory] 모델 '{model_name}' 로드 실패: {e}")

        raise RuntimeError("[Memory] 사용 가능한 임베딩 모델이 없습니다.")

    # ── 임베딩 ────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """텍스트 → 정규화된 임베딩 벡터."""
        model = self._get_embed_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ── 시청자 이름 관리 ─────────────────────────────────────────

    def _load_viewer_cache(self):
        """JSON 파일에서 시청자 이름 캐시 로드."""
        try:
            if os.path.exists(_VIEWER_CACHE_PATH):
                with open(_VIEWER_CACHE_PATH, "r", encoding="utf-8") as f:
                    self._viewer_cache = json.load(f)
        except Exception as e:
            logger.warning(f"[Memory] 시청자 캐시 로드 실패: {e}")
            self._viewer_cache = {}

    def _save_viewer_cache(self):
        """시청자 이름 캐시를 JSON 파일에 저장."""
        try:
            os.makedirs(os.path.dirname(_VIEWER_CACHE_PATH), exist_ok=True)
            with open(_VIEWER_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._viewer_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Memory] 시청자 캐시 저장 실패: {e}")

    def remember_viewer(self, viewer_id: str, viewer_name: str):
        """시청자 ID → 닉네임 매핑을 기억한다."""
        if self._viewer_cache.get(viewer_id) != viewer_name:
            self._viewer_cache[viewer_id] = viewer_name
            self._save_viewer_cache()

    def get_viewer_name(self, viewer_id: str) -> Optional[str]:
        """시청자 ID로 기억된 닉네임을 반환한다."""
        return self._viewer_cache.get(viewer_id)

    # ── 대화 저장 ─────────────────────────────────────────────────

    def save_conversation_sync(
        self,
        user_msg: str,
        bot_reply: str,
        mode: str,
        viewer_name: Optional[str] = None,
    ):
        """대화 Q&A 쌍을 벡터 DB에 저장 (동기)."""
        try:
            self._init_db()

            # 사용자 메시지 + 봇 응답 합산 텍스트로 임베딩
            combined = f"{user_msg} {bot_reply}"
            embedding = self._embed(combined)

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
                embeddings=[embedding],
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
        """대화 저장 비동기 래퍼 (IO 스레드풀 오프로드)."""
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
        """과거 대화에서 관련 기억 검색 (동기).

        Returns:
            "사용자: <메시지> → 시온: <응답>" 형식의 문자열 리스트
        """
        try:
            self._init_db()
            total = self._conv_collection.count()
            if total == 0:
                return []

            embedding = self._embed(query)
            where = {"mode": mode} if mode else None

            results = self._conv_collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, total),
                where=where,
                include=["metadatas", "distances"],
            )

            memories = []
            if results and results["metadatas"]:
                for meta, dist in zip(
                    results["metadatas"][0], results["distances"][0]
                ):
                    # 코사인 거리 > 1.0 이면 관련성 낮음 (cosine_sim < 0.5)
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
        """과거 대화 검색 (비동기, 타임아웃 지원).

        방송 모드에서는 응답 속도 보장을 위해 짧은 타임아웃을 권장한다.
        타임아웃 초과 시 빈 리스트 반환 (기억 없는 것처럼 동작).
        """
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

            embedding = self._embed(query)
            results = self._knowledge_collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, total),
                include=["documents", "distances"],
            )

            docs = []
            if results and results["documents"]:
                for doc, dist in zip(
                    results["documents"][0], results["distances"][0]
                ):
                    # 지식은 더 엄격한 기준 (cosine_dist < 0.8, cosine_sim > 0.6)
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
        """data/knowledge/ 의 .md/.txt 파일을 벡터 DB에 로드한다.

        서버 시작 시 1회 호출. 이미 로드된 경우 스킵.
        지식 문서를 업데이트하려면 data/chroma_db 디렉토리를 삭제 후 재시작.
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

            ids, embeddings, documents, metadatas = [], [], [], []

            for file in files:
                text = file.read_text(encoding="utf-8")
                chunks = self._split_text(text, chunk_size=400, overlap=50)

                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_id = f"know_{file.stem}_{i}"
                    ids.append(chunk_id)
                    embeddings.append(self._embed(chunk))
                    documents.append(chunk)
                    metadatas.append({"source": file.name, "chunk": i})

            if documents:
                self._knowledge_collection.add(
                    ids=ids,
                    embeddings=embeddings,
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
                    # 오버랩: 직전 청크의 끝 부분을 다음 청크 앞에 포함
                    tail = current[-overlap:] if len(current) > overlap else current
                    current = f"{tail}\n\n{para}".strip()
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    # ── 통계 ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """DB 사용 현황 통계를 반환한다."""
        try:
            self._init_db()
            return {
                "conversations": self._conv_collection.count(),
                "knowledge_chunks": self._knowledge_collection.count(),
                "embedding_model": self._embed_model_name or "미로드",
                "db_path": self._db_path,
                "viewer_count": len(self._viewer_cache),
            }
        except Exception as e:
            return {"error": str(e)}


# ── 싱글톤 ───────────────────────────────────────────────────────

_engine_instance: Optional[MemoryEngine] = None


def get_memory_engine() -> MemoryEngine:
    """MemoryEngine 싱글톤을 반환한다."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MemoryEngine()
    return _engine_instance
