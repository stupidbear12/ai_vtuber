# -*- coding: utf-8 -*-
"""
app/album_review.py — 앨범 리뷰 모듈 (브라우저 에이전트 기반)

V3: Ollama (exagirl) 기반
  1. Ollama로 트랙리스트 조회
  2. YouTube에서 "아티스트 앨범 전곡" 검색 → 첫 번째 결과 클릭
  3. 전곡 영상이 재생되는 동안 트랙별 하이라이트 대기
  4. Ollama로 시온 스타일 리뷰 생성 → TTS
  5. 전체 앨범 총평
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import List, Optional

import aiohttp

from .browser_controller import BrowserController

logger = logging.getLogger("album_review")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "exagirl")

VOICE_URL = os.environ.get("AI_VOICE_URL", "http://localhost:8004")
LIVE2D_URL = os.environ.get("AI_LIVE2D_URL", "http://localhost:8001")
HIGHLIGHT_SEC = int(os.environ.get("ALBUM_REVIEW_HIGHLIGHT_SEC", "90"))


# ── 프롬프트 ─────────────────────────────────────────────────────

TRACKLIST_PROMPT = """\
{artist}의 정규/미니 앨범 '{album}'의 공식 트랙리스트를 정확히 알려줘.
각 트랙은 서로 다른 곡이어야 해. 리믹스, 어쿠스틱 버전, Instrumental 같은 변형은 포함하지 마.

다음 JSON 형식으로만 답해:
{{
  "tracks": [
    {{"number": 1, "title": "곡 제목"}},
    {{"number": 2, "title": "곡 제목"}}
  ]
}}

정확한 공식 트랙리스트만. JSON만 출력해."""

TRACK_REVIEW_PROMPT = """\
너는 AI VTuber '시온'이야. 음악 좋아하는 20대 여자 캐릭터야.
친구한테 얘기하듯 캐주얼하게 곡 리뷰를 해줘.

지금 듣고 있는 곡:
- 아티스트: {artist}
- 앨범: {album}
- {track_number}번 트랙: {track_title}

규칙:
- 평론가 말투 절대 금지. "이 곡은~" 같은 격식체 쓰지 마
- "야 이거~", "와 진짜~", "이거 좋다~" 같은 구어체로
- ㅋㅋ, ㅎㅎ 자연스럽게 사용
- 곡의 분위기, 느낌, 포인트 위주로
- 2~3문장으로 짧게
- 한국어로

리뷰만 출력해."""

ALBUM_SUMMARY_PROMPT = """\
너는 AI VTuber '시온'이야. 음악 좋아하는 20대 여자 캐릭터야.

방금 {artist}의 앨범 '{album}'을 전곡 들었어.
각 트랙 리뷰:
{track_reviews}

앨범 전체 총평을 3~4문장으로 해줘.
규칙:
- 친구한테 얘기하듯 캐주얼하게
- 앨범 전체 분위기, 베스트 트랙 언급
- 10점 만점 점수
- 격식체 금지, 구어체로
- 한국어로

총평만 출력해."""


class AlbumReviewer:
    """YouTube 기반 앨범 리뷰 오케스트레이터 (V3: Ollama)."""

    def __init__(self, browser: BrowserController):
        self._browser = browser
        self._status = "idle"  # idle, reviewing
        self._current_album = ""
        self._current_track = 0
        self._total_tracks = 0
        self._cancel_requested = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def progress(self) -> dict:
        return {
            "status": self._status,
            "album": self._current_album,
            "current_track": self._current_track,
            "total_tracks": self._total_tracks,
        }

    def cancel(self) -> None:
        self._cancel_requested = True

    async def review(
        self,
        artist: str,
        album: str,
        highlight_sec: int = HIGHLIGHT_SEC,
    ) -> dict:
        """앨범 리뷰를 실행한다."""
        if self._status == "reviewing":
            return {"success": False, "error": "이미 리뷰 진행 중입니다."}

        self._status = "reviewing"
        self._current_album = f"{artist} - {album}"
        self._cancel_requested = False
        reviews: List[dict] = []

        try:
            # ── 1. YouTube에서 전곡 영상 검색 + 재생 ──
            await self._commentary(
                f"자~ 오늘은 {artist}의 {album} 앨범 리뷰를 해볼게! "
                f"유튜브에서 찾아볼게~",
                "excited",
            )

            search_q = f"{artist}+{album}+전곡+듣기".replace(" ", "+")
            logger.info("[AlbumReview] YouTube 검색 시작")
            await self._browser.navigate(
                f"https://www.youtube.com/results?search_query={search_q}"
            )
            logger.info("[AlbumReview] YouTube 페이지 로드 완료")
            await asyncio.sleep(3)

            # 쿠키 팝업 처리
            await self._dismiss_popups()
            await asyncio.sleep(1)

            # 첫 번째 영상 클릭 (JavaScript)
            logger.info("[AlbumReview] 영상 클릭 시도")
            clicked = False
            try:
                result = await self._browser.evaluate("""
                    (() => {
                        const el = document.querySelector('ytd-video-renderer a#video-title');
                        if (el) { el.click(); return true; }
                        const thumb = document.querySelector('ytd-video-renderer a#thumbnail');
                        if (thumb) { thumb.click(); return true; }
                        return false;
                    })()
                """)
                if result:
                    clicked = True
            except Exception as exc:
                logger.warning("[AlbumReview] JS 클릭 실패: %s", exc)

            if not clicked:
                for text_target in ["전곡", album, artist]:
                    try:
                        await self._browser.click(f"text=/{re.escape(text_target)}/i")
                        clicked = True
                        break
                    except Exception:
                        continue

            if not clicked:
                await self._commentary("영상을 찾을 수 없어ㅠ", "worried")
                return {"success": False, "error": "YouTube 영상 클릭 실패"}

            await asyncio.sleep(5)  # 영상 로딩 대기

            # ── 2. 영상 설명에서 트랙리스트 추출 ──
            tracks = await self._extract_tracklist_from_yt()

            if not tracks:
                # YouTube 설명에서 못 찾으면 Ollama로 폴백
                logger.info("[AlbumReview] YT 설명에서 트랙리스트 없음, Ollama 폴백")
                tracklist_raw = await self._ollama_chat(
                    TRACKLIST_PROMPT.format(artist=artist, album=album),
                    json_mode=True,
                )
                tracklist_data = _parse_json(tracklist_raw)
                tracks = tracklist_data.get("tracks", [])

            if not tracks:
                await self._commentary(
                    f"음... {artist} {album} 트랙리스트를 못 찾겠어ㅠ",
                    "worried",
                )
                return {"success": False, "error": "트랙리스트 조회 실패"}

            self._total_tracks = len(tracks)
            logger.info(
                "[AlbumReview] 트랙리스트 %d곡: %s",
                len(tracks),
                [t.get("title") for t in tracks],
            )

            await self._commentary(
                f"총 {len(tracks)}곡이네! 같이 들어보자~", "happy"
            )

            # ── 3. 트랙별 리뷰 ──
            for i, track in enumerate(tracks):
                if self._cancel_requested:
                    await self._commentary(
                        "리뷰가 취소됐어! 다음에 또 하자~", "calm"
                    )
                    break

                self._current_track = i + 1
                t_title = track.get("title", f"트랙 {i+1}")
                t_num = track.get("number", i + 1)

                # 트랙 소개
                await self._commentary(
                    f"{t_num}번 트랙, '{t_title}' 나온다~",
                    "excited",
                )

                # 하이라이트 재생 대기
                logger.info(
                    "[AlbumReview] 트랙 %d/%d '%s' — %ds 대기",
                    t_num, len(tracks), t_title, highlight_sec,
                )
                await asyncio.sleep(highlight_sec)

                if self._cancel_requested:
                    break

                # 리뷰 생성 (Ollama)
                review_text = await self._gen_track_review(
                    artist, album, t_num, t_title
                )
                reviews.append({
                    "track": t_num,
                    "title": t_title,
                    "review": review_text,
                })

                # 리뷰 코멘트 TTS
                await self._commentary(review_text, "happy")
                await asyncio.sleep(3)

            # ── 4. 앨범 총평 ──
            if reviews and not self._cancel_requested:
                summary = await self._gen_album_summary(artist, album, reviews)
                await self._commentary(summary, "excited")
            else:
                summary = ""

            return {
                "success": True,
                "tracks_reviewed": len(reviews),
                "reviews": reviews,
                "summary": summary,
            }

        except asyncio.CancelledError:
            await self._commentary("리뷰가 취소됐어! 다음에 또~", "worried")
            return {"success": False, "error": "취소됨"}
        except Exception as exc:
            logger.error("[AlbumReview] 오류: %s", exc, exc_info=True)
            await self._commentary("리뷰 중에 에러가 났어ㅠ 미안해~", "worried")
            return {"success": False, "error": str(exc)}
        finally:
            self._status = "idle"
            self._current_album = ""
            self._current_track = 0
            self._total_tracks = 0

    # ── 브라우저 헬퍼 ────────────────────────────────────────────

    async def _dismiss_popups(self) -> None:
        """쿠키/로그인 팝업을 닫는다 (JavaScript로 빠르게)."""
        try:
            await self._browser.evaluate("""
                (() => {
                    const texts = ['모두 거부', 'Reject all', '동의', '닫기', 'No thanks'];
                    for (const t of texts) {
                        const btns = [...document.querySelectorAll('button, [role="button"]')];
                        const btn = btns.find(b => b.textContent.trim().includes(t));
                        if (btn) { btn.click(); return t; }
                    }
                    return null;
                })()
            """)
            await asyncio.sleep(0.5)
        except Exception:
            pass

    # ── YouTube 트랙리스트 추출 ────────────────────────────────────

    async def _extract_tracklist_from_yt(self) -> List[dict]:
        """YouTube 영상에서 트랙리스트를 추출한다.

        1순위: 챕터(자동 생성 타임스탬프)
        2순위: 설명 텍스트의 타임스탬프
        """
        try:
            # ── 방법 1: YouTube 챕터에서 추출 ──
            chapters = await self._browser.evaluate("""
                (() => {
                    // 챕터 목록 (progress bar 위 또는 설명 아래)
                    const items = document.querySelectorAll(
                        'ytd-macro-markers-list-item-renderer, '
                        + 'ytd-chapter-renderer'
                    );
                    if (items.length > 0) {
                        return [...items].map((el, i) => {
                            const title = el.querySelector(
                                '#details h4, #detail h4, .macro-markers'
                            );
                            return {
                                number: i + 1,
                                title: title ? title.textContent.trim() : ''
                            };
                        }).filter(t => t.title);
                    }
                    return [];
                })()
            """)

            if chapters and len(chapters) > 1:
                # 중복 제거 (같은 제목이 반복되는 경우)
                seen = set()
                unique = []
                for ch in chapters:
                    t = ch.get("title", "")
                    if t not in seen:
                        seen.add(t)
                        # "01. 라일락 (Lilac)" → "라일락 (Lilac)"
                        clean = re.sub(r"^\d+[\.\)\-\s]+", "", t).strip()
                        unique.append({
                            "number": len(unique) + 1,
                            "title": clean or t,
                        })
                chapters = unique
                logger.info(
                    "[AlbumReview] 챕터에서 %d곡 추출", len(chapters)
                )
                return chapters

            # ── 방법 2: 설명 펼치고 타임스탬프 파싱 ──
            # 더보기 클릭
            try:
                await self._browser.evaluate("""
                    (() => {
                        const btn = document.querySelector(
                            'tp-yt-paper-button#expand'
                        ) || document.querySelector(
                            '#expand'
                        ) || document.querySelector(
                            'ytd-text-inline-expander tp-yt-paper-button'
                        );
                        if (btn) { btn.click(); return true; }
                        // 설명 영역 자체를 클릭
                        const desc = document.querySelector(
                            'ytd-text-inline-expander'
                        );
                        if (desc) { desc.click(); return true; }
                        return false;
                    })()
                """)
                await asyncio.sleep(1.5)
            except Exception:
                pass

            # 설명 전체 텍스트 추출
            desc = await self._browser.evaluate("""
                (() => {
                    // 펼쳐진 설명
                    const expanded = document.querySelector(
                        'ytd-text-inline-expander[is-expanded] '
                        + '#attributed-snippet-text'
                    ) || document.querySelector(
                        '#description-inline-expander '
                        + '#attributed-snippet-text'
                    );
                    if (expanded) return expanded.innerText;

                    // 폴백: 전체 설명 영역
                    const desc = document.querySelector(
                        'ytd-text-inline-expander'
                    ) || document.querySelector(
                        '#description yt-formatted-string'
                    );
                    return desc ? desc.innerText : '';
                })()
            """)

            if not desc or len(desc) < 20:
                logger.info("[AlbumReview] YouTube 설명 텍스트 없음/짧음")
                return []

            logger.info(
                "[AlbumReview] YT 설명 (%d자): %s", len(desc), desc[:300]
            )

            # 타임스탬프 패턴: "0:00 곡제목" 또는 "00:00 곡제목"
            pattern = re.compile(
                r"(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)(?:\n|$)"
            )
            matches = pattern.findall(desc)

            if not matches:
                logger.info("[AlbumReview] 타임스탬프 패턴 없음")
                return []

            tracks = []
            for i, (timestamp, title) in enumerate(matches, 1):
                title = re.sub(r"^\d+[\.\)\-]\s*", "", title.strip())
                title = title.strip(" -–—|")
                if title and len(title) > 1:
                    tracks.append({"number": i, "title": title})

            logger.info(
                "[AlbumReview] YT 설명에서 %d곡 추출", len(tracks)
            )
            return tracks

        except Exception as exc:
            logger.warning("[AlbumReview] YT 트랙리스트 추출 실패: %s", exc)
            return []

    # ── Ollama API ───────────────────────────────────────────────

    async def _ollama_chat(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
    ) -> str:
        """Ollama chat API 호출."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120.0),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            "[AlbumReview] Ollama %s: %s",
                            resp.status, body[:300],
                        )
                        return ""
                    data = await resp.json()
            content = data.get("message", {}).get("content", "").strip()
            logger.info("[AlbumReview] Ollama 응답 (%d자)", len(content))
            return content
        except Exception as exc:
            logger.error("[AlbumReview] Ollama 실패: %s", exc)
            return ""

    # ── 리뷰 생성 ────────────────────────────────────────────────

    async def _gen_track_review(
        self, artist: str, album: str, track_num: int, track_title: str
    ) -> str:
        prompt = TRACK_REVIEW_PROMPT.format(
            artist=artist, album=album,
            track_number=track_num, track_title=track_title,
        )
        review = await self._ollama_chat(prompt)
        return review or f"'{track_title}' 좋은 곡이야~ 다음 곡도 기대해!"

    async def _gen_album_summary(
        self, artist: str, album: str, reviews: List[dict]
    ) -> str:
        track_reviews = "\n".join(
            f"  {r['track']}. {r['title']}: {r['review']}" for r in reviews
        )
        prompt = ALBUM_SUMMARY_PROMPT.format(
            artist=artist, album=album, track_reviews=track_reviews,
        )
        summary = await self._ollama_chat(prompt)
        return summary or f"{artist}의 {album}, 좋은 앨범이었어! 한번 들어봐~"

    # ── TTS / Live2D ─────────────────────────────────────────────

    async def _commentary(self, text: str, emotion: str = "calm") -> None:
        if not text:
            return
        logger.info("[AlbumReview] 코멘터리: %s", text[:60])

        audio_b64: Optional[str] = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{VOICE_URL}/voice/tts",
                    json={"text": text, "emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.read()
                        audio_b64 = base64.b64encode(raw).decode("utf-8")
        except Exception as exc:
            logger.warning("[AlbumReview] TTS 실패: %s", exc)

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{LIVE2D_URL}/live2d/emotion",
                    json={"emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                )
                bcast: dict = {
                    "cmd": "speak", "text": text, "emotion": emotion,
                    "author": "시온", "platform": "album_review",
                    "is_donation": False,
                }
                if audio_b64:
                    bcast["audio_base64"] = audio_b64
                await session.post(
                    f"{LIVE2D_URL}/live2d/broadcast",
                    json=bcast,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                )
        except Exception as exc:
            logger.warning("[AlbumReview] Live2D 전송 실패: %s", exc)


# ── 유틸리티 ──────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    logger.warning("[AlbumReview] JSON 파싱 실패: %s", text[:200])
    return {}
