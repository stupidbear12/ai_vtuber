"""Download COOMO (쿠모) YouTube audio for voice training"""
import os
import subprocess
import sys

OUTPUT_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS\coomo_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COOMO vocal cover videos
URLS = [
    "https://www.youtube.com/watch?v=zzRh40hBzu8",  # 사랑하긴 했었나요 cover
    "https://www.youtube.com/watch?v=fban80_SLQs",  # 만찬가 cover
    "https://www.youtube.com/watch?v=ExKhGlaS1JI",  # Barcelona cover
    "https://www.youtube.com/watch?v=2Xii3UdDKeY",  # 민수는 혼란스럽다 cover
    "https://www.youtube.com/watch?v=j7sj6xnZW8s",  # 수영해 live cover
]

print(f"Downloading {len(URLS)} COOMO videos to {OUTPUT_DIR}\n")

success = 0
for i, url in enumerate(URLS, 1):
    print(f"[{i}/{len(URLS)}] {url}")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", os.path.join(OUTPUT_DIR, f"coomo_{i:02d}.%(ext)s"),
        "--no-playlist",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print(f"  OK")
        success += 1
    else:
        print(f"  FAILED: {result.stderr[-200:]}")

print(f"\nDone: {success}/{len(URLS)} downloaded")
# List files
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    print(f"  {f} ({size_mb:.1f} MB)")
