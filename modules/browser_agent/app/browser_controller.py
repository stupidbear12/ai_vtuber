# -*- coding: utf-8 -*-
"""
app/browser_controller.py — Playwright 브라우저 제어 (ACT 단계)

역할:
  - Playwright 브라우저 인스턴스 관리 (시작/종료)
  - 스크린샷 캡처 (base64)
  - 액션 실행: click, type, scroll, navigate
  - headless=False (OBS에서 브라우저 창 캡처용)
"""

import asyncio
import base64
import logging
import os
from typing import Optional

logger = logging.getLogger("browser_controller")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)


class BrowserController:
    """Playwright 기반 브라우저 세션 관리 및 조작."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Playwright 브라우저를 실행한다 (headless=False, OBS 캡처용).

        BROWSER_WINDOW_X / BROWSER_WINDOW_Y 환경변수로 창 위치를 지정할 수 있다.
        서브 모니터에 띄우려면 X를 해당 모니터의 오프셋으로 설정.
        """
        if self._running:
            raise RuntimeError("브라우저가 이미 실행 중입니다.")

        from playwright.async_api import async_playwright

        win_x = os.environ.get("BROWSER_WINDOW_X", "")
        win_y = os.environ.get("BROWSER_WINDOW_Y", "")

        chrome_args = [
            "--window-size=1280,720",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]
        if win_x and win_y:
            chrome_args.append(f"--window-position={win_x},{win_y}")

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            channel="chrome",
            args=chrome_args,
        )
        self._page = await self._browser.new_page(
            viewport={"width": 1280, "height": 720},
        )
        # 초기 페이지
        await self._page.goto("https://www.google.com", wait_until="domcontentloaded")
        self._running = True
        pos_info = f" pos=({win_x},{win_y})" if win_x else ""
        logger.info("[Browser] Playwright 브라우저 시작 완료 (1280x720%s)", pos_info)

    async def stop(self) -> None:
        """브라우저를 종료한다."""
        self._running = False
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        logger.info("[Browser] 브라우저 종료 완료")

    async def screenshot_base64(self) -> str:
        """현재 페이지의 스크린샷을 base64로 반환한다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        png_bytes = await self._page.screenshot(type="png")
        return base64.b64encode(png_bytes).decode("utf-8")

    async def screenshot_bytes(self) -> bytes:
        """현재 페이지의 스크린샷을 raw PNG bytes로 반환한다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        return await self._page.screenshot(type="png")

    async def navigate(self, url: str) -> None:
        """URL로 이동한다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        # http:// 프로토콜 자동 추가
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        logger.info("[Browser] navigate → %s", url)
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            logger.warning("[Browser] navigate timeout/error: %s", exc)
        await asyncio.sleep(1.0)  # 페이지 렌더링 대기

    async def click(self, target: str) -> None:
        """요소를 클릭한다.

        target이 CSS selector 형식이면 그대로 사용,
        아니면 텍스트 기반으로 요소를 찾는다.
        """
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        logger.info("[Browser] click → %s", target[:50])

        # CSS selector 시도
        if _looks_like_selector(target):
            try:
                await self._page.click(target, timeout=5000)
                await asyncio.sleep(0.5)
                return
            except Exception:
                logger.debug("[Browser] CSS selector 실패, 텍스트 검색으로 폴백")

        # 텍스트 기반 검색 (여러 전략)
        strategies = [
            f"text={target}",
            f"role=button[name='{target}']",
            f"role=link[name='{target}']",
            f"[placeholder*='{target}']",
            f"[aria-label*='{target}']",
            f"[title*='{target}']",
        ]
        for selector in strategies:
            try:
                await self._page.click(selector, timeout=3000)
                await asyncio.sleep(0.5)
                return
            except Exception:
                continue

        # 마지막: 부분 텍스트 매칭
        try:
            await self._page.click(f"text=/{target}/i", timeout=5000)
            await asyncio.sleep(0.5)
            return
        except Exception:
            pass

        raise RuntimeError(f"클릭할 요소를 찾을 수 없습니다: {target}")

    async def type_text(self, target: str, value: str) -> None:
        """입력 필드에 텍스트를 입력한다.

        target으로 요소를 찾고 value를 타이핑한다.
        target이 비어있으면 현재 포커스된 요소에 입력한다.
        """
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        logger.info("[Browser] type → target=%s value=%s", target[:30], value[:30])

        if target:
            # 입력 필드 찾기
            input_selectors = [
                target,
                f"[name='{target}']",
                f"[placeholder*='{target}']",
                f"[aria-label*='{target}']",
                f"input[type='text']",
                f"input[type='search']",
                f"textarea",
                f"[contenteditable='true']",
            ]
            clicked = False
            for sel in input_selectors:
                try:
                    await self._page.click(sel, timeout=3000)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                logger.warning("[Browser] 입력 필드를 찾을 수 없어 현재 포커스에 입력")

        # 기존 내용 지우고 새로 입력
        await self._page.keyboard.press("Control+a")
        await asyncio.sleep(0.1)
        await self._page.keyboard.type(value, delay=50)
        await asyncio.sleep(0.3)

    async def scroll(self, direction: str = "down") -> None:
        """페이지를 스크롤한다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        delta = -500 if direction == "up" else 500
        logger.info("[Browser] scroll %s", direction)
        await self._page.mouse.wheel(0, delta)
        await asyncio.sleep(0.5)

    async def evaluate(self, expression: str):
        """페이지에서 JavaScript를 실행한다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        return await self._page.evaluate(expression)

    async def press_enter(self) -> None:
        """Enter 키를 누른다."""
        if not self._page:
            raise RuntimeError("브라우저가 실행 중이 아닙니다.")
        await self._page.keyboard.press("Enter")
        await asyncio.sleep(1.0)


def _looks_like_selector(text: str) -> bool:
    """CSS selector처럼 보이는지 간단히 판단한다."""
    selector_chars = {".", "#", "[", ">", "~", "+", ":"}
    return any(c in text for c in selector_chars) or text.startswith("//")
