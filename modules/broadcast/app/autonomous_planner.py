# -*- coding: utf-8 -*-
"""
app/autonomous_planner.py -- 자율 행동 플래너 믹스인

BroadcastChatManager에서 분리된 자율 행동 관련 메서드.
Ollama LLM을 통해 다음 행동(search, talk, music, react, topic_change)을
자율적으로 결정하고 실행한다.
"""

import asyncio
import datetime
import json
import logging
import os
import random
import time

import aiohttp

logger = logging.getLogger(__name__)

# ── 자율 행동 플래너 설정 ─────────────────────────────────────
AUTONOMOUS_ENABLED = os.environ.get("AUTONOMOUS_ENABLED", "1") == "1"
AUTONOMOUS_ACTION_DELAY_MIN = int(os.environ.get("AUTONOMOUS_ACTION_DELAY_MIN", "15"))
AUTONOMOUS_ACTION_DELAY_MAX = int(os.environ.get("AUTONOMOUS_ACTION_DELAY_MAX", "45"))
AUTONOMOUS_HISTORY_SIZE = 10
AUTONOMOUS_CRAWL_SUMMARY_SIZE = 5
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "exagirl")

# 라디오 idle 기준 (chat_collector.py의 RADIO_IDLE_SECONDS를 참조)
# 실제 값은 chat_collector.py에 정의되어 있으며, 여기서는 _autonomous_worker에서
# from app.chat_collector import RADIO_IDLE_SECONDS 로 사용한다.


class AutonomousPlannerMixin:
    """자율 행동 플래너 메서드 믹스인.

    BroadcastChatManager가 이 믹스인을 상속하여 사용한다.
    self._action_history, self._crawl_results, self._current_topic,
    self._chat_url, self._music_url, self._surf_active, self._queue,
    self._tts_lock, self._running, self._last_activity_time,
    self._autonomous_active, self._radio_active 등
    BroadcastChatManager.__init__에서 초기화된 속성에 의존한다.

    또한 TTSMixin._speak_and_broadcast 및 MusicHandlerMixin._announce_music,
    MusicHandlerMixin._handle_music_action 메서드에 의존한다.
    """

    def _enforce_action_rules(self, result: dict) -> dict:
        """LLM 응답에 코드 레벨 강제 규칙을 적용한다.

        프롬프트만으로는 LLM이 규칙을 무시할 수 있으므로,
        프로그래밍적으로 다양성과 검색 비중을 강제한다.

        Args:
            result: LLM이 반환한 {"action", "content", "reason"}

        Returns:
            규칙이 적용된 결과 dict
        """
        action = result.get("action", "talk")
        content = result.get("content", "")
        original_action = action

        last_action = self._action_history[-1]["action"] if self._action_history else ""
        last_content = self._action_history[-1]["content"] if self._action_history else ""
        has_crawl = len(self._crawl_results) > 0

        # 규칙 1: react는 크롤링 결과가 있을 때만 허용
        if action == "react" and not has_crawl:
            action = "search"
            result["reason"] = f"react→search 강제전환: 크롤링 결과 0건"
            logger.info("[ActionRules] react 차단 → search 강제 (크롤링 결과 없음)")

        # 규칙 2: 연속 동일 action 방지 (search는 예외적으로 2회 연속 허용)
        if action == last_action and action != "search":
            # 크롤링 결과 없으면 search, 있으면 react, 그 외 talk
            if not has_crawl:
                action = "search"
                result["reason"] = f"연속 {last_action} 방지 → search 전환"
            elif action != "react":
                action = "react"
                result["reason"] = f"연속 {last_action} 방지 → react 전환"
            else:
                action = "topic_change"
                result["reason"] = f"연속 react 방지 → topic_change 전환"
            logger.info("[ActionRules] 연속 '%s' 방지 → '%s'", last_action, action)

        # 규칙 3: 히스토리에서 search 비율이 너무 낮으면 강제 search
        if len(self._action_history) >= 3:
            search_count = sum(1 for h in self._action_history if h["action"] == "search")
            search_ratio = search_count / len(self._action_history)
            if search_ratio < 0.2 and action != "search" and not has_crawl:
                action = "search"
                result["reason"] = f"search 비율 {search_ratio:.0%} < 20% → search 강제"
                logger.info("[ActionRules] search 비율 %.0f%% → search 강제", search_ratio * 100)

        # 규칙 4: music에서 이전에 재생한 곡과 동일한 content 방지
        if action == "music":
            played = [h["content"].lower() for h in self._action_history if h["action"] == "music"]
            if content.lower() in played:
                # 다양한 곡 풀에서 랜덤 대체
                fallback_songs = [
                    "NewJeans - Hype Boy", "IVE - LOVE DIVE", "aespa - Supernova",
                    "BTS - Dynamite", "BLACKPINK - Pink Venom", "르세라핌 - FEARLESS",
                    "아이유 - 밤편지", "데이식스 - 한 페이지가 될 수 있게",
                    "볼빨간사춘기 - 여행", "잔나비 - 주저하는 연인들을 위해",
                    "AKMU - 어떻게 이별까지 사랑하겠어", "10cm - 봄이 좋냐",
                    "Stray Kids - MEGAVERSE", "SEVENTEEN - Super",
                    "TWICE - Feel Special", "Red Velvet - Psycho",
                ]
                available = [s for s in fallback_songs if s.lower() not in played]
                if available:
                    content = random.choice(available)
                    result["reason"] = f"중복곡 방지 → {content}"
                    logger.info("[ActionRules] 중복곡 '%s' → '%s'", result["content"], content)

        # 규칙 5: topic_change에서 이전 주제와 동일하면 다른 카테고리 강제
        if action == "topic_change":
            past = [h["content"].lower() for h in self._action_history if h["action"] == "topic_change"]
            if content.lower() in past or content.lower() == (self._current_topic or "").lower():
                fallback_topics = [
                    "요즘 핫한 AI 기술 트렌드", "세계에서 가장 맛있는 길거리 음식",
                    "가보고 싶은 일본 여행지", "인디 게임 추천", "올해 e스포츠 이슈",
                    "우주 탐사 최신 뉴스", "신기한 동물 이야기", "건강한 생활 습관",
                    "최근 유행하는 밈", "좋아하는 애니메이션 OST", "심리학 재미있는 실험",
                    "한국의 숨겨진 맛집", "프로그래밍 입문 이야기", "패션 트렌드 2024",
                    "넷플릭스 추천작", "반려동물 키우기", "역사 속 반전 사건",
                ]
                available = [t for t in fallback_topics if t.lower() not in past]
                if available:
                    content = random.choice(available)
                    result["reason"] = f"중복주제 방지 → {content}"
                    logger.info("[ActionRules] 중복주제 방지 → '%s'", content)

        # 규칙 6: 첫 행동이면 반드시 search
        if len(self._action_history) == 0 and action != "search":
            action = "search"
            if not content or content == "자유 토크":
                content = "오늘의 인기 뉴스"
            result["reason"] = "첫 행동 → search 강제"
            logger.info("[ActionRules] 첫 행동 → search 강제")

        result["action"] = action
        result["content"] = content
        if action != original_action:
            logger.info("[ActionRules] 최종 보정: %s → %s", original_action, action)
        return result

    async def _ask_llm_action(self) -> dict:
        """Ollama에 직접 질의하여 다음 자율 행동을 결정한다.

        시스템 프롬프트에 최근 행동 히스토리, 크롤링 결과, 현재 시간 등
        컨텍스트를 제공하고, JSON 형식으로 다음 행동을 응답받는다.

        Returns:
            {"action": str, "content": str, "reason": str}
        """
        time_period = self._get_time_period()
        now_str = datetime.datetime.now().strftime("%H:%M")

        # 행동 히스토리 텍스트
        if self._action_history:
            history_lines = [
                f"  - [{h['action']}] {h['content'][:60]} ({h.get('time', '')})"
                for h in self._action_history
            ]
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (아직 없음)"

        # 크롤링 결과 요약
        if self._crawl_results:
            crawl_text = "\n".join(f"  - {r[:120]}" for r in self._crawl_results)
        else:
            crawl_text = "  (아직 없음)"

        # 현재 주제
        topic_text = self._current_topic if self._current_topic else "(없음)"

        # 마지막 행동 추출 (연속 방지용)
        last_action = self._action_history[-1]["action"] if self._action_history else ""
        last_content = self._action_history[-1]["content"] if self._action_history else ""

        # 이전에 재생한 곡 목록 (중복 방지)
        played_songs = [
            h["content"] for h in self._action_history if h["action"] == "music"
        ]
        played_songs_text = ", ".join(played_songs[-5:]) if played_songs else "(없음)"

        # 이전 topic_change 주제 목록 (중복 방지)
        past_topics = [
            h["content"] for h in self._action_history if h["action"] == "topic_change"
        ]
        past_topics_text = ", ".join(past_topics[-5:]) if past_topics else "(없음)"

        # 크롤링 결과 존재 여부
        has_crawl = len(self._crawl_results) > 0

        # react 가능 여부 텍스트
        if has_crawl:
            react_line = "- react: 이전 검색 결과에 대해 감상 말하기 (content에 감상할 내용 요약)"
        else:
            react_line = "- react: [사용 불가] 검색 결과가 없으므로 선택할 수 없음. 먼저 search를 해야 함"

        prompt = (
            f"너는 AI VTuber '시온'의 자율 행동 플래너야.\n"
            f"시온이 방송 중이고, 지금 다음에 뭘 할지 결정해야 해.\n\n"
            f"현재 시각: {now_str} ({time_period})\n"
            f"현재 주제: {topic_text}\n\n"
            f"최근 행동 히스토리:\n{history_text}\n\n"
            f"이전 크롤링/검색 결과:\n{crawl_text}\n\n"
            f"이전에 재생한 곡: {played_songs_text}\n"
            f"이전 topic_change 주제: {past_topics_text}\n\n"
            f"가능한 행동:\n"
            f"- search: 흥미로운 주제를 웹에서 검색 (content에 구체적 검색 키워드)\n"
            f"- talk: 자유롭게 혼잣말/시청자와 대화 (content에 말할 주제)\n"
            f"- music: 분위기에 맞는 노래 재생 (content에 '아티스트 - 제목' 형식)\n"
            f"{react_line}\n"
            f"- topic_change: 새로운 주제로 전환 (content에 새 주제)\n\n"
            f"===== 필수 규칙 (반드시 지켜야 함) =====\n"
            f"1. 크롤링 결과가 없으면 search를 최우선으로 선택해라. 현재 결과 수: {len(self._crawl_results)}건\n"
            f"2. 직전 행동이 '{last_action}'이었으므로, 이번에는 반드시 다른 action을 선택해라\n"
            f"3. react는 크롤링 결과가 있을 때만 선택 가능 (현재: {'가능' if has_crawl else '불가능'})\n"
            f"4. music 선택 시, 이전에 재생한 곡과 완전히 다른 아티스트/장르의 곡을 골라라\n"
            f"5. topic_change 선택 시, 이전 주제와 다른 카테고리에서 선택해라\n"
            f"   카테고리 예시: 주식/증권, 재무제표/재무관리/재무회계, 경제뉴스, 사이버보안/해킹, 악성코드분석, 백신/안티바이러스, 네트워크보안, 작사/작곡, 비트메이킹/프로듀싱, 음악이론, 과학기술, IT뉴스, 게임, 우주, 역사, 심리학\n"
            f"6. content는 이전 행동의 content('{last_content[:40]}')와 다른 구체적 내용이어야 함\n"
            f"7. 추천 행동 순서: search → react → topic_change → talk → music (search를 자주 해라)\n"
            f"8. 전체 히스토리에서 search 비율이 30% 이상이 되도록 해라\n"
            f"9. 주요 관심 분야 (search 시 우선적으로 검색):\n"
            f"   - 주식/증권: 오늘의 주식 시장 동향, 코스피/코스닥/나스닥 뉴스, 종목 분석, 투자 전략, 경제 지표\n"
            f"   - 재무/회계: 재무제표 읽는 법, 재무관리 기초, 재무회계 개념, 손익계산서, 대차대조표, 현금흐름표, 원가회계, 관리회계, 기업 가치평가, ROE/PER/PBR 분석\n"
            f"   - 사이버보안: 해킹 기법, 악성코드 분석, 백신 원리, CTF, 제로데이 취약점, 랜섬웨어, 보안 뉴스\n"
            f"   - 작사/작곡/비트메이킹: 작사법, 작곡 이론, 코드 진행, 멜로디 작법, 비트메이킹 튜토리얼, DAW 사용법, 믹싱/마스터링, 프로듀싱 팁, 유명 프로듀서 인터뷰\n"
            f"   - 논문/블로그: 최신 IT 논문, 기술 블로그, 개발자 블로그 글, AI/ML 연구 동향\n"
            f"   검색 시 뉴스 사이트, 논문, 블로그 위주로 검색하라\n\n"
            f'반드시 아래 JSON 형식으로만 응답해:\n'
            f'{{"action": "행동종류", "content": "내용", "reason": "선택 이유"}}'
        )

        # ── Ollama 호출 전 채팅 큐 체크 (채팅 응답 우선) ──
        if self._queue.qsize() > 0:
            logger.info("[AutonomousPlanner] Ollama 호출 건너뜀: 채팅 큐 대기 중 (%d건)", self._queue.qsize())
            return {"action": "talk", "content": "자유 토크", "reason": "채팅 우선 양보"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "[AutonomousPlanner] Ollama 응답 오류: HTTP %s", resp.status
                        )
                        return {"action": "talk", "content": "자유 토크", "reason": "Ollama 오류 fallback"}
                    data = await resp.json()

            response_text = data.get("response", "")
            result = json.loads(response_text)

            # 유효성 검증
            valid_actions = {"search", "talk", "music", "react", "topic_change"}
            if result.get("action") not in valid_actions:
                result["action"] = "talk"
            if not result.get("content"):
                result["content"] = "자유 토크"
            if not result.get("reason"):
                result["reason"] = ""

            # ── 코드 레벨 강제 규칙 ──────────────────────────
            result = self._enforce_action_rules(result)

            return result

        except json.JSONDecodeError as e:
            logger.warning("[AutonomousPlanner] LLM JSON 파싱 실패: %s (raw: %s)", e, response_text[:200])
            return {"action": "talk", "content": "자유 토크", "reason": f"JSON 파싱 실패 fallback"}
        except Exception as e:
            logger.warning("[AutonomousPlanner] LLM 행동 결정 실패: %s", e)
            return {"action": "talk", "content": "자유 토크", "reason": f"fallback: {e}"}

    async def _execute_action(self, action_data: dict) -> None:
        """선택된 자율 행동을 실행한다.

        Args:
            action_data: {"action": str, "content": str, "reason": str}
        """
        action = action_data.get("action", "talk")
        content = action_data.get("content", "")

        handlers = {
            "search": self._action_search,
            "talk": self._action_talk,
            "music": self._action_music,
            "react": self._action_react,
            "topic_change": self._action_topic_change,
        }

        handler = handlers.get(action, self._action_talk)
        await handler(content)

    async def _action_search(self, content: str) -> None:
        """웹 검색 -> browser_agent(8007) 호출 (상위 3개 결과 순회).

        검색 중에는 _surf_active 플래그를 설정하여 TTS 충돌을 방지한다.
        검색 결과는 _crawl_results에 저장하여 이후 react 행동에서 활용한다.

        Args:
            content: 검색 키워드
        """
        surf_url = os.environ.get("AI_BROWSER_AGENT_URL", "http://localhost:8007")
        self._surf_active = True
        logger.info("[AutonomousPlanner] 웹 검색 시작 (deep): %s", content)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{surf_url}/browser/surf",
                    json={
                        "message": f"{content} 검색해줘",
                        "author": "시온",
                        "switch_scene": True,
                        "max_results": 3,
                    },
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # 전체 요약 저장
                        summary = data.get("summary", data.get("reply", data.get("url", "")))
                        if summary:
                            crawl_entry = f"[{content}] {str(summary)[:500]}"
                            self._crawl_results.append(crawl_entry)

                        # 개별 결과가 있으면 각각 RAG에 저장
                        results = data.get("results", [])
                        if results:
                            for i, r in enumerate(results):
                                r_text = r.get("text", "")
                                r_url = r.get("url", "")
                                r_title = r.get("title", "")
                                if r_text:
                                    await self._store_crawl_to_rag(
                                        content=r_text,
                                        source=f"crawl:{content}:{r_title[:30]}",
                                        url=r_url,
                                    )
                            logger.info(
                                "[AutonomousPlanner] 검색 완료 (deep): %s → %d개 결과 RAG 저장",
                                content, len(results),
                            )
                        else:
                            # 개별 결과 없으면 전체 요약이라도 저장
                            if summary:
                                await self._store_crawl_to_rag(
                                    content=str(summary),
                                    source=f"crawl:{content}",
                                    url=data.get("url", ""),
                                )
                            logger.info(
                                "[AutonomousPlanner] 검색 완료: %s → %s",
                                content, str(summary)[:80],
                            )
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[AutonomousPlanner] 검색 실패: HTTP %s: %s",
                            resp.status, body[:100],
                        )
        except Exception as e:
            logger.warning("[AutonomousPlanner] browser_agent 연결 실패: %s", e)
        finally:
            self._surf_active = False
            logger.info("[AutonomousPlanner] 웹 검색 종료 → 자율 모드 재개")

    async def _store_crawl_to_rag(
        self, content: str, source: str = "crawl", url: str = ""
    ) -> None:
        """크롤링 결과를 chat 모듈의 RAG(ChromaDB)에 저장한다.

        chat 모듈(8002)의 POST /chat/memory/store 엔드포인트를 호출한다.

        Args:
            content: 저장할 텍스트 (크롤링 요약/본문)
            source: 출처 식별자
            url: 크롤링 URL (메타데이터용)
        """
        if not content or not content.strip():
            return
        try:
            metadata = {"url": url} if url else None
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._chat_url}/chat/memory/store",
                    json={
                        "content": content,
                        "source": source,
                        "metadata": metadata,
                    },
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("stored"):
                            logger.info(
                                "[RAG] 크롤링 결과 저장 성공: %d 청크 (source=%s)",
                                data.get("chunks", 0), source,
                            )
                        else:
                            logger.warning(
                                "[RAG] 크롤링 결과 저장 실패: %s", data.get("error", "")
                            )
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[RAG] chat 모듈 저장 API 오류: HTTP %s: %s",
                            resp.status, body[:100],
                        )
        except Exception as e:
            logger.warning("[RAG] chat 모듈 연결 실패 (무시): %s", e)

    async def _action_talk(self, content: str) -> None:
        """자유 토크 -- ai_chat을 통해 대사 생성 후 TTS 방송.

        Args:
            content: 말할 주제/내용
        """
        # 채팅 큐 인터럽트 체크
        if self._queue.qsize() > 0:
            logger.info("[AutonomousPlanner] talk 건너뜀: 채팅 대기 중")
            return

        time_period = self._get_time_period()
        talk_message = (
            f"[자율 방송 모드] 시온이 혼자 방송 중입니다. "
            f"현재 시간대: {time_period}. "
            f"주제: {content}. "
            f"자연스럽게 이야기해주세요."
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._chat_url}/chat",
                    json={
                        "message": talk_message,
                        "mode": "broadcast",
                        "context": f"[자율 모드] {time_period}. 주제: {content}",
                        "viewer_name": "시온",
                        "is_donation": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("[AutonomousPlanner] talk ai_chat 오류: HTTP %s: %s", resp.status, body[:200])
                        return
                    data = await resp.json()

            if data.get("error"):
                logger.warning("[AutonomousPlanner] talk ai_chat 오류: %s", data["error"])
                return

            reply = (data.get("reply") or "").strip()
            emotion = data.get("emotion", "calm")
            action = data.get("action")

            # 자연어 음악 요청 감지
            if action and action.get("type") == "play_music" and action.get("query"):
                asyncio.create_task(self._handle_music_action(action["query"], "시온"))

            if reply:
                await self._speak_and_broadcast(reply, emotion)
                self.stats["radio_talks"] += 1
                logger.info("[AutonomousPlanner] talk 완료: [%s] %s", emotion, reply[:50])
        except Exception as e:
            logger.warning("[AutonomousPlanner] talk 실패: %s", e)

    async def _action_music(self, content: str) -> None:
        """분위기에 맞는 음악 재생.

        Args:
            content: 노래 검색어 (예: "아이유 - 밤편지")
        """
        # 채팅 큐 인터럽트 체크
        if self._queue.qsize() > 0:
            logger.info("[AutonomousPlanner] music 건너뜀: 채팅 대기 중")
            return

        if not self._music_commands_enabled:
            logger.info("[AutonomousPlanner] 음악 명령 비활성화 상태, music action 건너뜀")
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._music_url}/ymusic/play",
                    json={"query": content, "requester": "시온 자율DJ"},
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        track = data.get("track") or {}
                        title = track.get("title", content)
                        artist = track.get("artist", "")
                        song_info = f"{title} - {artist}" if artist else title
                        await self._announce_music(
                            f"이 노래 들어볼까요? {song_info}", "happy"
                        )
                        # Auto-DJ 타이머 리셋
                        if hasattr(self, '_last_music_time'):
                            self._last_music_time = time.time()
                        logger.info("[AutonomousPlanner] 음악 재생: %s", song_info)
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[AutonomousPlanner] 음악 재생 실패: HTTP %s: %s",
                            resp.status, body[:200],
                        )
        except Exception as e:
            logger.warning("[AutonomousPlanner] 음악 재생 실패: %s", e)

    async def _action_react(self, content: str) -> None:
        """이전 크롤링/검색 결과에 대한 감상을 말한다.

        Args:
            content: LLM이 생성한 감상 힌트/요약
        """
        # 채팅 큐 인터럽트 체크
        if self._queue.qsize() > 0:
            logger.info("[AutonomousPlanner] react 건너뜀: 채팅 대기 중")
            return

        # 크롤링 결과 컨텍스트 구성 (전체 내용 포함)
        if self._crawl_results:
            crawl_context = "\n".join(f"- {r}" for r in self._crawl_results)
        else:
            crawl_context = "검색 결과 없음"

        react_message = (
            f"[자율 방송 모드] 이전에 검색한 웹사이트의 구체적인 내용에 대해 감상을 말해주세요.\n"
            f"반드시 아래 검색 결과의 실제 내용을 구체적으로 언급하면서 이야기해주세요.\n"
            f"일반적인 이야기가 아닌, 검색 결과에서 발견한 구체적 사실/정보를 언급해야 합니다.\n"
            f"검색 결과:\n{crawl_context}\n"
            f"시온의 시각: {content}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._chat_url}/chat",
                    json={
                        "message": react_message,
                        "mode": "broadcast",
                        "context": f"[자율 모드] 검색 결과에 대한 감상.\n{crawl_context[:300]}",
                        "viewer_name": "시온",
                        "is_donation": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()

            reply = (data.get("reply") or "").strip()
            emotion = data.get("emotion", "calm")
            if reply:
                await self._speak_and_broadcast(reply, emotion)
                logger.info("[AutonomousPlanner] react 완료: [%s] %s", emotion, reply[:50])
        except Exception as e:
            logger.warning("[AutonomousPlanner] react 실패: %s", e)

    async def _action_topic_change(self, content: str) -> None:
        """새로운 주제로 전환하며 전환 멘트를 생성한다.

        Args:
            content: 새 주제
        """
        # 채팅 큐 인터럽트 체크
        if self._queue.qsize() > 0:
            logger.info("[AutonomousPlanner] topic_change 건너뜀: 채팅 대기 중")
            return

        self._current_topic = content
        logger.info("[AutonomousPlanner] 주제 전환: %s", content)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._chat_url}/chat",
                    json={
                        "message": (
                            f"[자율 방송 모드] 새로운 주제로 전환합니다: {content}. "
                            f"자연스럽게 주제를 바꾸며 이야기해주세요."
                        ),
                        "mode": "broadcast",
                        "context": f"[자율 모드] 주제 전환: {content}",
                        "viewer_name": "시온",
                        "is_donation": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()

            reply = (data.get("reply") or "").strip()
            emotion = data.get("emotion", "calm")
            if reply:
                await self._speak_and_broadcast(reply, emotion)
                logger.info("[AutonomousPlanner] topic_change 완료: [%s] %s", emotion, reply[:50])
        except Exception as e:
            logger.warning("[AutonomousPlanner] topic_change 실패: %s", e)

    # ── 자율 행동 메인 루프 ─────────────────────────────────────

    async def _autonomous_worker(self) -> None:
        """자율 행동 플래너 워커 -- LLM이 다음 행동을 스스로 결정한다.

        동작 흐름:
          1. 매 1초마다 idle 시간 체크
          2. RADIO_IDLE_SECONDS 경과 시 자율 행동 시작
          3. LLM(Ollama)에 다음 행동 질의 -> action 실행
          4. 적절한 대기(15~45초) 후 반복
          5. 채팅 큐에 메시지가 있으면 자율 행동 일시 중지하고 채팅 응답 우선
        """
        # RADIO_IDLE_SECONDS는 chat_collector.py에 정의되어 있음
        from app.chat_collector import RADIO_IDLE_SECONDS

        logger.info("[AutonomousPlanner] 자율 행동 워커 시작")

        while self._running:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            # 채팅 큐에 대기 중이면 자율 행동 불필요 -- 채팅 응답 우선
            if self._queue.qsize() > 0:
                continue

            # 웹서핑(browser_agent) 처리 중이면 TTS 충돌 방지
            if self._surf_active:
                continue

            # idle 시간 체크
            elapsed = time.time() - self._last_activity_time
            if elapsed < RADIO_IDLE_SECONDS:
                continue

            # ── 자율 행동 실행 ──────────────────────────────
            self._autonomous_active = True
            self._radio_active = True  # 기존 호환: radio_active도 함께 설정
            logger.info("[AutonomousPlanner] 자율 모드 진입 (idle %.0fs)", elapsed)

            try:
                # 채팅 인터럽트 체크
                if self._queue.qsize() > 0:
                    logger.info("[AutonomousPlanner] 채팅 감지 → 자율 행동 일시 중지")
                    continue

                # LLM에 다음 행동 질의
                action_data = await self._ask_llm_action()
                action = action_data.get("action", "talk")
                content = action_data.get("content", "")
                reason = action_data.get("reason", "")

                logger.info(
                    "[AutonomousPlanner] 행동 선택: %s | 내용: %s | 이유: %s",
                    action, content[:60], reason[:60],
                )

                # 행동 히스토리 기록
                self._action_history.append({
                    "action": action,
                    "content": content,
                    "reason": reason,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                })

                # 채팅 인터럽트 재확인 -- LLM 응답 대기 중 채팅이 올 수 있음
                if self._queue.qsize() > 0:
                    logger.info("[AutonomousPlanner] 채팅 감지 → action 실행 건너뜀")
                    continue

                # 행동 실행
                await self._execute_action(action_data)
                self.stats["autonomous_actions"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[AutonomousPlanner] 자율 행동 실패: %s", e, exc_info=True)
            finally:
                self._autonomous_active = False
                self._radio_active = False
                # 활동 시간 갱신 -> tight retry loop 방지
                self._last_activity_time = time.time()

            # 다음 행동까지 랜덤 대기 (15~45초)
            wait_seconds = random.randint(
                AUTONOMOUS_ACTION_DELAY_MIN, AUTONOMOUS_ACTION_DELAY_MAX
            )
            logger.debug("[AutonomousPlanner] 다음 행동까지 %d초 대기", wait_seconds)
            for _ in range(wait_seconds):
                if not self._running or self._queue.qsize() > 0 or self._surf_active:
                    break
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    return
            # 대기 중 채팅이 들어왔으면 idle 타이머 자연 리셋됨

        logger.info("[AutonomousPlanner] 자율 행동 워커 종료")

    async def _radio_worker_safe(self) -> None:
        """자율 행동 워커 래퍼 -- crash 시 자동 재시작.

        AUTONOMOUS_ENABLED=1이면 자율 행동 플래너로 동작하고,
        아니면 기존 라디오 모드(talk만 반복)로 동작한다.
        """
        while self._running:
            try:
                await self._autonomous_worker()
                break  # 정상 종료 (_running=False)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "[AutonomousPlanner] 워커 crash 감지 → 5초 후 재시작: %s",
                    e, exc_info=True,
                )
                self._autonomous_active = False
                self._radio_active = False
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break
        logger.info("[AutonomousPlanner] 워커 안전 종료")
