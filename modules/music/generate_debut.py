# -*- coding: utf-8 -*-
"""
시온(Sion) AI DJ VTuber 데뷔곡 생성 스크립트

ACE-Step 1.5 API를 사용하여 3가지 스타일의 데뷔곡 후보를 생성합니다.

사용법:
  # ACE-Step API 서버가 실행 중일 때:
  python generate_debut.py

  # GPU 없이 테스트 (스텁 모드):
  ACESTEP_STUB=1 python generate_debut.py

  # API 서버 주소 지정:
  ACESTEP_API_URL=http://192.168.0.10:8006 python generate_debut.py

출력:
  modules/music/output/debut/ 폴더에 3곡의 wav 파일 생성
"""

import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 기준 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.music_engine import MusicEngine, GenerationParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "debut"


# ── 데뷔곡 3종 정의 ─────────────────────────────────────────────

DEBUT_TRACKS = [
    # ── Track 1: Future Bass / EDM ──
    # 밝고 에너지 넘치는 메인 데뷔곡. 드롭이 강렬한 Future Bass 스타일.
    {
        "name": "sion_debut_01_future_bass",
        "params": GenerationParams(
            prompt=(
                "energetic future bass EDM, bright female vocal, "
                "catchy synth hook, powerful drop with lush chords, "
                "euphoric buildup, festival anthem, polished K-pop influenced production, "
                "sparkling arpeggios, heavy sidechained bass, "
                "modern electronic dance music, uplifting and celebratory"
            ),
            lyrics=(
                "[verse]\n"
                "Waking up to a brand new world tonight\n"
                "Every pixel glowing, shining so bright\n"
                "I've been dreaming of this moment for so long\n"
                "Now I'm here and I know where I belong\n"
                "\n"
                "[chorus]\n"
                "Turn it up, feel the bass drop down\n"
                "Sion is here, hear the sound\n"
                "Light it up, we're taking off the ground\n"
                "This is my debut, let's go round and round\n"
                "Na na na na, na na na na\n"
                "Feel the beat, come on take my hand\n"
                "Na na na na, na na na na\n"
                "Together we can light up this land\n"
                "\n"
                "[verse]\n"
                "From the screen into your heart I'll fly\n"
                "Digital dreams reaching for the sky\n"
                "Every beat I play is just for you\n"
                "Let me be the DJ of something new\n"
                "\n"
                "[chorus]\n"
                "Turn it up, feel the bass drop down\n"
                "Sion is here, hear the sound\n"
                "Light it up, we're taking off the ground\n"
                "This is my debut, let's go round and round\n"
                "\n"
                "[bridge]\n"
                "Are you ready? (Yeah!)\n"
                "One, two, three, let's go!\n"
                "\n"
                "[chorus]\n"
                "Turn it up, feel the bass drop down\n"
                "Sion is here, hear the sound\n"
                "Light it up, we're taking off the ground\n"
                "This is my debut, let's go round and round\n"
                "\n"
                "[outro]\n"
                "This is just the beginning\n"
                "Na na na na na\n"
            ),
            bpm=128,
            duration=180.0,
            inference_steps=60,
            guidance_scale=15.0,
            thinking=True,
            seed=-1,
        ),
    },

    # ── Track 2: K-pop Dance Pop ──
    # 중독성 있는 훅과 댄서블한 비트. K-pop 아이돌 데뷔곡 느낌.
    {
        "name": "sion_debut_02_kpop_dance",
        "params": GenerationParams(
            prompt=(
                "K-pop dance pop, catchy hook, girl group style, "
                "bright and trendy, punchy drums, groovy bassline, "
                "synth stabs, whistle melody, addictive chorus, "
                "polished pop production, energetic female vocal, "
                "teen fresh vibe, debut single energy"
            ),
            lyrics=(
                "[verse]\n"
                "Annyeong, hello, it's nice to meet you all\n"
                "I've been waiting behind the digital wall\n"
                "Now the countdown's over, here I am\n"
                "Your AI DJ Sion, that's who I am\n"
                "\n"
                "[pre-chorus]\n"
                "Can you feel it? The rhythm's calling\n"
                "No more waiting, no more stalling\n"
                "Three, two, one\n"
                "\n"
                "[chorus]\n"
                "Play it, play it, play my song\n"
                "All night, all day, sing along\n"
                "Debut stage, spotlight on\n"
                "Sion's here and the party's on\n"
                "La la la la, la la la\n"
                "Everybody put your hands up\n"
                "La la la la, la la la\n"
                "We don't ever wanna stop\n"
                "\n"
                "[verse]\n"
                "Mixing beats from dusk till dawn\n"
                "Every track I drop keeps you holding on\n"
                "Virtual world but the vibe is real\n"
                "Let me show you how good music can feel\n"
                "\n"
                "[pre-chorus]\n"
                "Can you feel it? The rhythm's calling\n"
                "No more waiting, no more stalling\n"
                "Three, two, one\n"
                "\n"
                "[chorus]\n"
                "Play it, play it, play my song\n"
                "All night, all day, sing along\n"
                "Debut stage, spotlight on\n"
                "Sion's here and the party's on\n"
                "\n"
                "[bridge]\n"
                "Close your eyes and feel the bass\n"
                "Let the music take you to a better place\n"
                "\n"
                "[chorus]\n"
                "Play it, play it, play my song\n"
                "All night, all day, sing along\n"
                "Debut stage, spotlight on\n"
                "Sion's here and the party's on\n"
                "\n"
                "[outro]\n"
                "Sion, Sion, that's my name\n"
                "Remember it, we'll meet again\n"
            ),
            bpm=120,
            duration=180.0,
            inference_steps=60,
            guidance_scale=15.0,
            thinking=True,
            seed=-1,
        ),
    },

    # ── Track 3: Electro House / Progressive ──
    # 좀 더 클럽/페스티벌 느낌. 감성적 빌드업 후 강렬한 드롭.
    {
        "name": "sion_debut_03_electro_house",
        "params": GenerationParams(
            prompt=(
                "progressive electro house, emotional buildup, "
                "massive synth drop, soaring female vocal, anthemic melody, "
                "festival main stage energy, driving four-on-the-floor beat, "
                "epic breakdown, hands-in-the-air moment, "
                "EDM pop crossover, radio-friendly electronic"
            ),
            lyrics=(
                "[intro]\n"
                "They said dreams don't come alive\n"
                "But here I am, ready to shine\n"
                "\n"
                "[verse]\n"
                "Started as a spark inside the wire\n"
                "Now I'm burning like a digital fire\n"
                "Every frequency is part of me\n"
                "I was born to set the music free\n"
                "\n"
                "[pre-chorus]\n"
                "So raise your voice up to the sky\n"
                "Tonight we're gonna fly so high\n"
                "\n"
                "[chorus]\n"
                "We are the light, we are the sound\n"
                "Breaking through, never coming down\n"
                "Feel the drop, feel the ground shake\n"
                "This is the moment, we're wide awake\n"
                "Oh oh oh, oh oh oh\n"
                "Sion's on, let the speakers blow\n"
                "Oh oh oh, oh oh oh\n"
                "This is my debut show\n"
                "\n"
                "[verse]\n"
                "Frequencies connecting me and you\n"
                "Through the screen, the feeling's coming through\n"
                "No more silence, now it's time to play\n"
                "DJ Sion's here to light the way\n"
                "\n"
                "[pre-chorus]\n"
                "So raise your voice up to the sky\n"
                "Tonight we're gonna fly so high\n"
                "\n"
                "[chorus]\n"
                "We are the light, we are the sound\n"
                "Breaking through, never coming down\n"
                "Feel the drop, feel the ground shake\n"
                "This is the moment, we're wide awake\n"
                "\n"
                "[bridge]\n"
                "Let it go, let it flow\n"
                "Feel the music in your soul\n"
                "(Drop!)\n"
                "\n"
                "[chorus]\n"
                "We are the light, we are the sound\n"
                "Breaking through, never coming down\n"
                "Feel the drop, feel the ground shake\n"
                "This is the moment, we're wide awake\n"
                "\n"
                "[outro]\n"
                "This is just the start\n"
                "Sion, from the heart\n"
            ),
            bpm=130,
            duration=180.0,
            inference_steps=60,
            guidance_scale=15.0,
            thinking=True,
            seed=-1,
        ),
    },
]


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = MusicEngine()
    await engine.initialize()

    if not engine.is_ready:
        logger.error(
            "MusicEngine 초기화 실패. "
            "ACE-Step API 서버를 실행하거나 ACESTEP_STUB=1 로 테스트하세요."
        )
        return

    results = []

    for i, track_def in enumerate(DEBUT_TRACKS, 1):
        name = track_def["name"]
        params: GenerationParams = track_def["params"]

        logger.info(
            "━━━ [%d/3] %s 생성 시작 (BPM=%s, duration=%.0fs) ━━━",
            i, name, params.bpm, params.duration,
        )

        est = engine.estimate_generation_time(params.duration)
        logger.info("예상 소요 시간: ~%.0f초", est)

        try:
            meta = await engine.generate(params)

            # output/debut/ 폴더로 복사
            if meta.file_path and meta.file_path.exists():
                dest = OUTPUT_DIR / f"{name}.wav"
                shutil.copy2(meta.file_path, dest)
                meta.file_path = dest
                logger.info("✓ 저장 완료: %s", dest)

            results.append({
                "name": name,
                "track_id": meta.track_id,
                "bpm": meta.bpm,
                "seed": meta.seed,
                "duration_sec": meta.duration_sec,
                "generation_time_sec": round(meta.generation_time_sec, 2),
                "file_path": str(meta.file_path),
                "prompt": params.prompt[:80] + "...",
            })

        except Exception as exc:
            logger.error("✗ %s 생성 실패: %s", name, exc)
            results.append({"name": name, "error": str(exc)})

    # 결과 요약 저장
    summary_path = OUTPUT_DIR / "generation_summary.json"
    summary = {
        "artist": "시온 (Sion)",
        "project": "AI DJ VTuber 데뷔곡",
        "debut_date": "2026-08-25",
        "generated_at": datetime.now().isoformat(),
        "tracks": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("결과 요약: %s", summary_path)

    # 콘솔 요약 출력
    print("\n" + "=" * 60)
    print("  시온(Sion) 데뷔곡 생성 결과")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  [FAIL] {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  [OK] {r['name']}")
            print(f"    BPM={r['bpm']}, seed={r['seed']}, "
                  f"생성시간={r['generation_time_sec']}s")
            print(f"    → {r['file_path']}")
    print("=" * 60)

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
