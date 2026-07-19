# -*- coding: utf-8 -*-
"""
app/web_surfer.py — 자율 웹서핑 모듈

기능:
  - Playwright headless 브라우저로 웹서핑
  - 실시간 스크린샷 WebSocket 스트리밍 → OBS browser_source
  - LLM(Ollama)으로 자연어 의도 파싱 + 페이지 요약
  - TTS 파이프라인 연동 (voice → live2d)
"""

import asyncio
import base64
import json
import logging
import os
from typing import Optional, Set

import httpx

logger = logging.getLogger("web_surfer")


class WebSurfer:
    """Playwright 기반 자율 웹서핑 엔진."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
        self._running = False
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._current_url = ""
        self._busy = False  # 서핑 처리 중 여부

        # WebSocket 뷰어 큐 (OBS browser_source 실시간 표시용)
        self._viewer_queues: Set[asyncio.Queue] = set()

        # 모듈 URL
        self.voice_url = os.environ.get("AI_VOICE_URL", "http://localhost:8004")
        self.live2d_url = os.environ.get("AI_LIVE2D_URL", "http://localhost:8001")
        self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "exagirl")

        # 검색 엔진: "google" 또는 "naver" (기본: google)
        self.search_engine = os.environ.get("SURF_SEARCH_ENGINE", "google").lower()

    # ── 프로퍼티 ──────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def current_url(self) -> str:
        return self._current_url

    # ── 라이프사이클 ──────────────────────────────────────────────

    async def start(self):
        """Playwright headless 브라우저 시작."""
        if self._running:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
        )
        self._page = await context.new_page()

        # Stealth: navigator.webdriver 숨기기
        await self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['ko-KR', 'ko', 'en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            window.chrome = { runtime: {} };
        """)

        home = "https://www.google.com" if self.search_engine == "google" else "https://www.naver.com"
        await self._page.goto(home, wait_until="domcontentloaded")
        self._running = True
        self._current_url = home

        # 스크린샷 스트리밍 시작
        self._stream_task = asyncio.create_task(self._screenshot_loop())
        logger.info("[WebSurfer] Playwright 브라우저 시작 (headless, 1920x1080)")

    async def stop(self):
        """브라우저 종료."""
        self._running = False
        self._streaming = False
        if self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._page = self._browser = self._playwright = None
        logger.info("[WebSurfer] 브라우저 종료")

    # ── 스크린샷 스트리밍 ─────────────────────────────────────────

    def add_viewer(self, queue: asyncio.Queue):
        self._viewer_queues.add(queue)
        logger.debug("[WebSurfer] Viewer 연결 (total=%d)", len(self._viewer_queues))

    def remove_viewer(self, queue: asyncio.Queue):
        self._viewer_queues.discard(queue)
        logger.debug("[WebSurfer] Viewer 해제 (total=%d)", len(self._viewer_queues))

    async def _screenshot_loop(self):
        """2 FPS로 스크린샷을 뷰어들에게 전송."""
        while self._running:
            if self._viewer_queues and self._page:
                try:
                    jpg = await self._page.screenshot(type="jpeg", quality=75)
                    b64 = base64.b64encode(jpg).decode()
                    dead = set()
                    for q in self._viewer_queues:
                        try:
                            # 큐가 가득 차면 오래된 프레임 버리고 새 프레임 넣기
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            q.put_nowait(b64)
                        except Exception:
                            dead.add(q)
                    self._viewer_queues -= dead
                except Exception as e:
                    logger.debug("[WebSurfer] screenshot error: %s", e)
            await asyncio.sleep(0.5)

    async def screenshot_base64(self) -> str:
        """현재 페이지 스크린샷 (단발성)."""
        if not self._page:
            raise RuntimeError("브라우저 미실행")
        jpg = await self._page.screenshot(type="jpeg", quality=80)
        return base64.b64encode(jpg).decode()

    # ── 브라우저 조작 ─────────────────────────────────────────────

    async def search(self, query: str) -> str:
        """검색 수행 → 페이지 텍스트 반환 (Google/Naver 지원)."""
        if not self._running:
            await self.start()

        import urllib.parse
        encoded = urllib.parse.quote(query)

        if self.search_engine == "google":
            url = f"https://www.google.com/search?q={encoded}&hl=ko"
        else:
            url = f"https://search.naver.com/search.naver?query={encoded}"

        await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
        self._current_url = self._page.url
        await asyncio.sleep(2)

        # reCAPTCHA 감지 → 네이버로 폴백
        page_text = await self._page.evaluate("document.body.innerText")
        if "I'm not a robot" in page_text or "reCAPTCHA" in page_text:
            logger.warning("[WebSurfer] Google reCAPTCHA 감지, 네이버로 폴백")
            url = f"https://search.naver.com/search.naver?query={encoded}"
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
            self._current_url = self._page.url
            await asyncio.sleep(2)
            page_text = await self._page.evaluate("document.body.innerText")

        return page_text[:3000]

    async def navigate(self, url: str) -> str:
        """URL 이동 → 페이지 텍스트 반환."""
        if not self._running:
            await self.start()

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            logger.warning("[WebSurfer] navigate error: %s", e)

        self._current_url = self._page.url
        await asyncio.sleep(1.5)

        text = await self._page.evaluate("document.body.innerText")
        return text[:3000]

    async def click_result(self, index: int = 0) -> str:
        """검색 결과에서 N번째 링크 클릭."""
        if not self._page:
            raise RuntimeError("브라우저 미실행")

        try:
            # Google + Naver 모두 커버하는 셀렉터
            selectors = [
                "h3",                          # Google 결과 제목
                "a.link_tit",                  # Naver 통합검색
                "a.api_txt_lines",             # Naver 지식백과
                ".total_tit a",                # Naver 웹문서
                ".news_tit",                   # Naver 뉴스
            ]
            links = []
            for sel in selectors:
                links = await self._page.query_selector_all(sel)
                if links:
                    break

            if links and index < len(links):
                await links[index].click()
                await asyncio.sleep(2)
                self._current_url = self._page.url
                text = await self._page.evaluate("document.body.innerText")
                return text[:3000]
            else:
                # 검색 결과 페이지 텍스트라도 반환
                return await self._page.evaluate("document.body.innerText")
        except Exception as e:
            return f"클릭 실패: {e}"

    async def scroll_down(self):
        """페이지 스크롤 다운."""
        if self._page:
            await self._page.mouse.wheel(0, 600)
            await asyncio.sleep(0.5)

    async def go_back(self):
        """뒤로 가기."""
        if self._page:
            await self._page.go_back()
            await asyncio.sleep(1)
            self._current_url = self._page.url

    # ── 메인 서핑 로직 ────────────────────────────────────────────

    async def surf(self, user_message: str, author: str = "시청자") -> dict:
        """
        자연어 웹서핑 요청 처리 파이프라인:
          1. LLM → 의도 파싱 (search / navigate)
          2. Playwright → 브라우저 조작
          3. 첫 번째 결과 클릭 (검색인 경우)
          4. LLM → 페이지 요약
          5. TTS → 시온이 설명
        """
        if self._busy:
            return {"error": "이미 웹서핑 중입니다."}

        self._busy = True
        try:
            if not self._running:
                await self.start()

            # 1. 의도 파싱
            intent = await self._parse_intent(user_message)
            action = intent.get("action", "search")
            logger.info("[WebSurfer] surf: action=%s, intent=%s", action, intent)

            # 2. 브라우저 조작
            if action == "navigate":
                page_text = await self.navigate(intent.get("url", ""))
            else:
                query = intent.get("query", user_message)
                page_text = await self.search(query)
                # 검색 후 첫 번째 결과 클릭
                await asyncio.sleep(1)
                detail_text = await self.click_result(0)
                if "클릭 실패" not in detail_text and "없습니다" not in detail_text:
                    page_text = detail_text

            # 3. 페이지 요약
            summary = await self._summarize(user_message, page_text)

            # 4. TTS로 말하기
            await self._speak(summary, author)

            return {
                "success": True,
                "action": action,
                "url": self._current_url,
                "summary": summary,
            }

        except Exception as e:
            logger.error("[WebSurfer] surf error: %s", e)
            error_msg = "웹서핑 중에 문제가 생겼어요. 다시 시도해볼게요!"
            await self._speak(error_msg, author)
            return {"success": False, "error": str(e)}
        finally:
            self._busy = False

    # ── LLM 연동 ──────────────────────────────────────────────────

    async def _parse_intent(self, message: str) -> dict:
        """LLM으로 자연어 → 브라우징 액션 변환."""
        prompt = (
            '사용자의 웹서핑 요청을 분석하세요.\n\n'
            f'요청: "{message}"\n\n'
            '아래 JSON 형식 중 하나로만 응답하세요 (다른 텍스트 없이):\n'
            '- 검색: {"action": "search", "query": "검색어"}\n'
            '- URL 이동: {"action": "navigate", "url": "https://..."}\n\n'
            'JSON:'
        )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 100},
                    },
                )
                result = resp.json().get("response", "").strip()
                start = result.find("{")
                end = result.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(result[start:end])
        except Exception as e:
            logger.error("[WebSurfer] intent parse error: %s", e)

        # 폴백: 원본 메시지로 검색
        return {"action": "search", "query": message}

    async def _summarize(self, question: str, page_text: str) -> str:
        """LLM으로 페이지 내용 요약 (시온 캐릭터)."""
        prompt = (
            f'너는 AI VTuber 시온이야. 시청자가 "{question}"이라고 요청해서 웹서핑을 했어.\n'
            f'아래는 방문한 페이지 내용이야. 시청자에게 친근하게 2~3문장으로 설명해줘.\n\n'
            f'페이지 내용:\n{page_text[:2000]}\n\n'
            f'시온의 설명:'
        )

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 200},
                    },
                )
                return resp.json().get("response", "검색 결과를 찾았어요!").strip()
        except Exception as e:
            logger.error("[WebSurfer] summarize error: %s", e)
            return "검색은 했는데 요약하는 데 문제가 생겼어요!"

    # ── TTS 파이프라인 ────────────────────────────────────────────

    async def _speak(self, text: str, author: str = "시청자"):
        """voice(TTS) → live2d(broadcast) 파이프라인."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Step 1: TTS 음성 생성
                tts_resp = await client.post(
                    f"{self.voice_url}/voice/tts",
                    json={"text": text, "emotion": "neutral"},
                )
                if tts_resp.status_code != 200:
                    logger.error("[WebSurfer] TTS failed: %d", tts_resp.status_code)
                    return

                audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")

                # Step 2: Live2D broadcast
                await client.post(
                    f"{self.live2d_url}/live2d/broadcast",
                    json={
                        "cmd": "speak",
                        "text": text,
                        "emotion": "neutral",
                        "audio_base64": audio_b64,
                        "author": author,
                        "platform": "chzzk",
                    },
                )
                logger.info("[WebSurfer] spoke: %s", text[:50])
        except Exception as e:
            logger.error("[WebSurfer] speak error: %s", e)


# ── 라이브 뷰어 HTML ──────────────────────────────────────────────

LIVE_VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>시온 웹서핑</title>
<style>
  * { margin: 0; padding: 0; }
  body { background: #000; overflow: hidden; width: 1920px; height: 1080px; }
  #screen {
    width: 1920px; height: 1080px;
    object-fit: contain;
    display: block;
  }
  #status {
    position: absolute; top: 10px; left: 10px;
    color: #0f0; font: 14px monospace;
    background: rgba(0,0,0,0.6); padding: 4px 8px;
    border-radius: 4px; z-index: 10;
    display: none;
  }
</style>
</head>
<body>
<div id="status">연결 중...</div>
<img id="screen" src="" alt="">
<script>
const img = document.getElementById('screen');
const status = document.getElementById('status');
let ws, reconnectTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/browser/live/ws');

  ws.onopen = () => { status.style.display = 'none'; };

  ws.onmessage = (e) => {
    img.src = 'data:image/jpeg;base64,' + e.data;
  };

  ws.onclose = () => {
    status.textContent = '재연결 중...';
    status.style.display = 'block';
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onerror = () => { ws.close(); };
}

connect();
</script>
</body>
</html>"""
