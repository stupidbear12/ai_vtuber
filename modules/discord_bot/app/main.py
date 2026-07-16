# -*- coding: utf-8 -*-
"""
Discord 봇 — 브라우저 에이전트 & 모듈 원격 제어

관리자(DISCORD_ADMIN_ID)만 슬래시 커맨드를 사용할 수 있다.
browser_agent(8007), music(8005), broadcast(8003) 등의 API를 호출한다.

사용법:
  python -m app.main
"""

import logging
import os
import sys
from pathlib import Path

import aiohttp
import discord
from discord import app_commands

# ── 환경변수 로드 ────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env", override=True)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("discord_bot")

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
_admin_raw = os.environ.get("DISCORD_ADMIN_ID", "0")
try:
    ADMIN_ID = int(_admin_raw)
except ValueError:
    ADMIN_ID = 0
GUILD_ID = os.environ.get("DISCORD_GUILD_ID", "")
# 숫자가 아닌 플레이스홀더 값은 무시
if GUILD_ID and not GUILD_ID.isdigit():
    GUILD_ID = ""

BROWSER_AGENT = os.environ.get("AI_BROWSER_AGENT_URL", "http://localhost:8007")
MUSIC_URL = os.environ.get("AI_MUSIC_URL", "http://localhost:8005")
BROADCAST_URL = os.environ.get("AI_BROADCAST_URL", "http://localhost:8003")
CORE_URL = os.environ.get("AI_CORE_URL", "http://localhost:8000")
CHESS_URL = os.environ.get("AI_CHESS_URL", "http://localhost:8008")

if not DISCORD_TOKEN:
    logger.error("DISCORD_BOT_TOKEN이 설정되지 않았습니다. .env를 확인하세요.")
    sys.exit(1)
if not ADMIN_ID:
    logger.error("DISCORD_ADMIN_ID가 설정되지 않았습니다. .env를 확인하세요.")
    sys.exit(1)


# ── 권한 체크 ────────────────────────────────────────────────────
def admin_only(interaction: discord.Interaction) -> bool:
    """관리자만 허용."""
    return interaction.user.id == ADMIN_ID


# ── HTTP 헬퍼 ────────────────────────────────────────────────────
_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    return _session


async def _api_post(base: str, path: str, json: dict | None = None) -> dict:
    s = await _get_session()
    url = f"{base}{path}"
    async with s.post(url, json=json) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json()


async def _api_get(base: str, path: str, params: dict | None = None) -> dict:
    s = await _get_session()
    url = f"{base}{path}"
    async with s.get(url, params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json()


# ── 봇 초기화 ────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

guild_obj = discord.Object(id=int(GUILD_ID)) if GUILD_ID else None


# ── 슬래시 커맨드: 브라우저 ──────────────────────────────────────

@tree.command(name="show", description="OBS에 웹 페이지를 표시합니다")
@app_commands.describe(url="표시할 URL")
@app_commands.check(admin_only)
async def cmd_show(interaction: discord.Interaction, url: str):
    await interaction.response.defer()
    try:
        r = await _api_post(BROWSER_AGENT, "/browser/show-page", {"url": url})
        await interaction.followup.send(f"✅ 표시 중: {url}")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="play-video", description="YouTube 영상을 OBS에 재생합니다")
@app_commands.describe(video_id="YouTube 영상 ID (11자)")
@app_commands.check(admin_only)
async def cmd_play_video(interaction: discord.Interaction, video_id: str):
    await interaction.response.defer()
    try:
        r = await _api_post(BROWSER_AGENT, "/browser/play-video", {"video_id": video_id})
        await interaction.followup.send(f"▶️ 재생: https://youtube.com/watch?v={video_id}")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="stop-video", description="OBS 영상을 정지하고 Radio Mode로 전환합니다")
@app_commands.check(admin_only)
async def cmd_stop_video(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(BROWSER_AGENT, "/browser/stop-video", {})
        await interaction.followup.send("⏹️ 영상 정지, Radio Mode로 전환")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


# ── 슬래시 커맨드: 음악 ─────────────────────────────────────────

@tree.command(name="music-play", description="음악을 재생합니다")
@app_commands.describe(query="검색어 또는 video_id")
@app_commands.check(admin_only)
async def cmd_music_play(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    try:
        # 11자 video_id 패턴이면 video_id로, 아니면 query로
        import re
        if re.match(r"^[a-zA-Z0-9_-]{11}$", query):
            payload = {"video_id": query}
        else:
            payload = {"query": query}
        r = await _api_post(MUSIC_URL, "/ymusic/play", payload)
        track = r.get("track", {})
        await interaction.followup.send(
            f"🎵 재생: {track.get('title', query)} — {track.get('artist', '')}"
        )
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="music-skip", description="현재 곡을 건너뜁니다")
@app_commands.check(admin_only)
async def cmd_music_skip(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(MUSIC_URL, "/ymusic/skip", {})
        await interaction.followup.send("⏭️ 스킵 완료")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="music-stop", description="음악을 정지합니다")
@app_commands.check(admin_only)
async def cmd_music_stop(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(MUSIC_URL, "/ymusic/stop", {})
        await interaction.followup.send("⏹️ 음악 정지")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="volume", description="음악 볼륨을 조절합니다 (0.0~2.0)")
@app_commands.describe(level="볼륨 레벨 (0.0~2.0, 기본 1.0)")
@app_commands.check(admin_only)
async def cmd_volume(interaction: discord.Interaction, level: float):
    await interaction.response.defer()
    try:
        r = await _api_post(MUSIC_URL, f"/music/volume?volume={level}", None)
        await interaction.followup.send(f"🔊 볼륨: {level}")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


# ── 슬래시 커맨드: 방송 ─────────────────────────────────────────

@tree.command(name="broadcast-start", description="치지직 채팅 수집을 시작합니다")
@app_commands.check(admin_only)
async def cmd_broadcast_start(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(BROADCAST_URL, "/broadcast/start", {
            "platform": "chzzk",
            "channel_id": "256e7d2368c87da9dcfd71482109135b",
        })
        await interaction.followup.send("📡 방송 채팅 수집 시작")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="broadcast-stop", description="채팅 수집을 중지합니다")
@app_commands.check(admin_only)
async def cmd_broadcast_stop(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(BROADCAST_URL, "/broadcast/stop", {})
        await interaction.followup.send("📡 방송 채팅 수집 중지")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


# ── 슬래시 커맨드: 상태 확인 ─────────────────────────────────────

@tree.command(name="status", description="모든 모듈의 상태를 확인합니다")
@app_commands.check(admin_only)
async def cmd_status(interaction: discord.Interaction):
    await interaction.response.defer()
    modules = {
        "Core(8000)": f"{CORE_URL}/health",
        "Broadcast(8003)": f"{BROADCAST_URL}/broadcast/status",
        "Music(8005)": f"{MUSIC_URL}/music/status",
        "Browser(8007)": f"{BROWSER_AGENT}/health",
        "Chess(8008)": f"{CHESS_URL}/health",
    }
    lines = []
    s = await _get_session()
    for name, url in modules.items():
        try:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "now_playing" in data and data.get("now_playing"):
                        track = data["now_playing"].get("track", {})
                        lines.append(
                            f"✅ {name} — 🎵 {track.get('title', '?')}"
                        )
                    elif "running" in data:
                        status = "수집 중" if data["running"] else "대기"
                        lines.append(f"✅ {name} — {status}")
                    else:
                        lines.append(f"✅ {name}")
                else:
                    lines.append(f"⚠️ {name} — HTTP {resp.status}")
        except Exception:
            lines.append(f"❌ {name} — 오프라인")
    await interaction.followup.send("\n".join(lines))


# ── 슬래시 커맨드: 앨범 리뷰 ─────────────────────────────────────

@tree.command(name="album-review", description="앨범 리뷰를 시작합니다")
@app_commands.describe(artist="아티스트", album="앨범명")
@app_commands.check(admin_only)
async def cmd_album_review(interaction: discord.Interaction, artist: str, album: str):
    await interaction.response.defer()
    try:
        r = await _api_post(BROWSER_AGENT, "/browser/album-review/start", {
            "artist": artist,
            "album": album,
        })
        await interaction.followup.send(f"📀 앨범 리뷰 시작: {artist} — {album}")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


# ── 슬래시 커맨드: 체스 ──────────────────────────────────────────

@tree.command(name="chess-start", description="시청자 vs 시온 체스 대국을 시작합니다")
@app_commands.describe(
    skill="난이도 0~20 (기본 5)",
    vote_time="투표 시간 초 (기본 30)",
)
@app_commands.check(admin_only)
async def cmd_chess_start(
    interaction: discord.Interaction,
    skill: int = 5,
    vote_time: int = 30,
):
    await interaction.response.defer()
    try:
        r = await _api_post(CHESS_URL, "/chess/new", {
            "sion_color": "white",
            "skill_level": skill,
            "vote_duration": vote_time,
        })
        await interaction.followup.send(
            f"♟️ 체스 대국 시작! 난이도 {skill}/20, 투표 {vote_time}초"
        )
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="chess-stop", description="체스 대국을 중단합니다 (시온 기권)")
@app_commands.check(admin_only)
async def cmd_chess_stop(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_post(CHESS_URL, "/chess/resign", {"side": "sion"})
        await interaction.followup.send("♟️ 체스 대국 종료 (시온 기권)")
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


@tree.command(name="chess-status", description="체스 대국 상태를 확인합니다")
@app_commands.check(admin_only)
async def cmd_chess_status(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        r = await _api_get(CHESS_URL, "/chess/state")
        phase = r.get("phase", "unknown")
        move_count = r.get("move_count", 0)
        msg = f"♟️ 상태: {phase}, {move_count}수 진행"
        if r.get("is_check"):
            msg += " (체크!)"
        if r.get("result"):
            msg += f" — 결과: {r['result']}"
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"❌ 실패: {e}")


# ── 에러 핸들러 ──────────────────────────────────────────────────

@tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
):
    if isinstance(error, app_commands.CheckFailure):
        if interaction.response.is_done():
            await interaction.followup.send("🚫 관리자만 사용할 수 있습니다.", ephemeral=True)
        else:
            await interaction.response.send_message(
                "🚫 관리자만 사용할 수 있습니다.", ephemeral=True
            )
    else:
        logger.error("Command error: %s", error)
        msg = f"❌ 오류: {error}"
        if interaction.response.is_done():
            await interaction.followup.send(msg)
        else:
            await interaction.response.send_message(msg)


# ── 봇 이벤트 ────────────────────────────────────────────────────

@bot.event
async def on_ready():
    logger.info("Discord 봇 로그인: %s (ID: %s)", bot.user.name, bot.user.id)
    logger.info("관리자 ID: %d", ADMIN_ID)

    # 슬래시 커맨드 동기화
    if guild_obj:
        tree.copy_global_to(guild=guild_obj)
        await tree.sync(guild=guild_obj)
        logger.info("슬래시 커맨드 동기화 완료 (길드: %s)", GUILD_ID)
    else:
        await tree.sync()
        logger.info("슬래시 커맨드 글로벌 동기화 완료 (최대 1시간 소요)")


# ── 실행 ─────────────────────────────────────────────────────────

def main():
    bot.run(DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
