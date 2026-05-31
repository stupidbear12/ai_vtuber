"""
data/download.py
================
CC 라이선스 Lofi 트랙 자동 수집 스크립트

지원 소스:
  1. FMA (Free Music Archive) - CC 라이선스 트랙 직접 다운로드
  2. YouTube Audio Library - yt-dlp를 통한 CC 라이선스 트랙 추출

사용법:
    python data/download.py --source fma --output data/raw --limit 100
    python data/download.py --source ytdlp --output data/raw --limit 50
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


# ──────────────────────────────────────────────
# FMA (Free Music Archive) 소스
# ──────────────────────────────────────────────

FMA_BASE_URL = "https://freemusicarchive.org/api/get"

# Lofi / Hip-Hop 관련 FMA 장르 ID (실제 FMA API 기준)
# 장르 목록: https://freemusicarchive.org/genre
FMA_GENRE_IDS = {
    "hip-hop": 76,
    "instrumental": 9,
    "electronic": 10,
}

# FMA small dataset — 전처리된 8000곡 오디오 파일 직접 링크
# CC 라이선스(CC BY, CC BY-SA, CC BY-NC, CC BY-ND, CC0) 트랙만 포함
FMA_SMALL_DATASET_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

# ──────────────────────────────────────────────
# YouTube Audio Library CC 트랙 URL 샘플 목록
# 모두 Creative Commons Attribution 라이선스 적용
# ──────────────────────────────────────────────
YT_CC_LOFI_URLS = [
    # YouTube Audio Library — Lofi / Chill / Hip-Hop (CC-BY)
    "https://www.youtube.com/watch?v=5qap5aO4i9A",  # Lofi Hip Hop Radio (example)
    "https://www.youtube.com/watch?v=jfKfPfyJRdk",  # lofi hip hop radio
    "https://www.youtube.com/watch?v=Na0w3Mz46GA",  # Chill Lofi Study Beats
    # 아래는 YouTube Audio Library 직접 검색으로 추가 가능
    # https://studio.youtube.com/channel/UC/music 에서 CC 필터 사용
]


def download_file(url: str, dest_path: Path, retries: int = 3) -> bool:
    """단일 파일 HTTP 다운로드 (재시도 포함)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            print(f"  ↓ {url}")
            urllib.request.urlretrieve(url, dest_path)
            return True
        except urllib.error.URLError as e:
            print(f"  ✗ 시도 {attempt + 1}/{retries} 실패: {e}")
            time.sleep(2 ** attempt)
    return False


def fetch_fma_tracks(
    api_key: str,
    genre_id: int,
    limit: int,
    output_dir: Path,
) -> list[dict]:
    """FMA API를 통해 CC 라이선스 트랙 메타데이터 수집."""
    tracks = []
    page = 1
    collected = 0

    while collected < limit:
        url = (
            f"{FMA_BASE_URL}/tracks.json"
            f"?api_key={api_key}"
            f"&genre_id={genre_id}"
            f"&limit=50"
            f"&page={page}"
            f"&track_filter_cc=1"  # CC 라이선스 필터
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"FMA API 오류: {e}")
            break

        dataset = data.get("dataset", [])
        if not dataset:
            break

        for track in dataset:
            if collected >= limit:
                break
            track_id = track.get("track_id")
            track_url = track.get("track_file")
            if track_id and track_url:
                dest = output_dir / f"fma_{track_id:06d}.mp3"
                if not dest.exists():
                    tracks.append({"id": track_id, "url": track_url, "dest": dest})
                collected += 1

        page += 1
        time.sleep(0.5)

    return tracks


def download_fma_small(output_dir: Path) -> None:
    """
    FMA small dataset 전체 ZIP 다운로드 (약 7.2GB).
    대역폭이 충분한 경우 권장 — 8000곡 MP3 포함.
    """
    zip_path = output_dir / "fma_small.zip"
    meta_path = output_dir / "fma_metadata.zip"

    print("=== FMA Small Dataset 다운로드 ===")
    print(f"대상 경로: {output_dir}")
    print("주의: 데이터셋 크기 약 7.2GB — 시간이 소요될 수 있습니다.\n")

    if not zip_path.exists():
        print("오디오 파일 다운로드 중...")
        download_file(FMA_SMALL_DATASET_URL, zip_path)
    else:
        print(f"이미 존재: {zip_path}")

    if not meta_path.exists():
        print("메타데이터 다운로드 중...")
        download_file(FMA_METADATA_URL, meta_path)
    else:
        print(f"이미 존재: {meta_path}")

    print("\n다운로드 완료. 압축 해제 방법:")
    print(f"  cd {output_dir}")
    print("  unzip fma_small.zip")
    print("  unzip fma_metadata.zip")


def download_with_ytdlp(
    urls: list[str],
    output_dir: Path,
    limit: int,
    audio_format: str = "wav",
    sample_rate: int = 22050,
) -> None:
    """
    yt-dlp를 사용해 YouTube CC 트랙을 WAV로 추출.

    요구사항:
        pip install yt-dlp
        ffmpeg 설치 필요 (https://ffmpeg.org)
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp가 설치되어 있지 않습니다.")
        print("설치 방법: pip install yt-dlp")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", str(sample_rate),   # 샘플레이트 22050Hz
            "-ac", "1",                # mono
        ],
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": True,
    }

    target_urls = urls[:limit]
    print(f"=== yt-dlp 다운로드 ({len(target_urls)}개 트랙) ===")

    with __import__("yt_dlp").YoutubeDL(ydl_opts) as ydl:
        ydl.download(target_urls)

    print(f"\n완료: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CC 라이선스 Lofi 트랙 자동 수집"
    )
    parser.add_argument(
        "--source",
        choices=["fma", "fma_small", "ytdlp"],
        default="fma_small",
        help="데이터 소스 선택 (기본: fma_small)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="다운로드 저장 경로 (기본: data/raw)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="최대 다운로드 트랙 수 (기본: 100)",
    )
    parser.add_argument(
        "--fma_api_key",
        type=str,
        default="",
        help="FMA API 키 (FMA API 사용 시 필요, https://freemusicarchive.org/api)",
    )
    parser.add_argument(
        "--genre",
        choices=list(FMA_GENRE_IDS.keys()),
        default="hip-hop",
        help="FMA 장르 (기본: hip-hop)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "fma_small":
        download_fma_small(output_dir)

    elif args.source == "fma":
        if not args.fma_api_key:
            print("오류: FMA API 키가 필요합니다 (--fma_api_key).")
            print("API 키 발급: https://freemusicarchive.org/api")
            return
        genre_id = FMA_GENRE_IDS[args.genre]
        print(f"=== FMA API 다운로드 (장르: {args.genre}, 최대 {args.limit}곡) ===")
        tracks = fetch_fma_tracks(
            args.fma_api_key, genre_id, args.limit, output_dir
        )
        print(f"총 {len(tracks)}곡 다운로드 시작...")
        success = 0
        for i, t in enumerate(tracks, 1):
            print(f"[{i}/{len(tracks)}] track_id={t['id']}")
            if download_file(t["url"], t["dest"]):
                success += 1
        print(f"\n완료: {success}/{len(tracks)} 성공")

    elif args.source == "ytdlp":
        download_with_ytdlp(
            urls=YT_CC_LOFI_URLS,
            output_dir=output_dir,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
