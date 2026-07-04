"""Download extended JFla songs for improved voice training.
Existing 5 originals + 10 popular covers = 15 total songs.
"""
import yt_dlp
import os
import subprocess

output_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio_extended"
os.makedirs(output_dir, exist_ok=True)

# Check ffmpeg
ffmpeg_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
has_ffmpeg = os.path.exists(ffmpeg_path)
if not has_ffmpeg:
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        has_ffmpeg = True
        ffmpeg_dir = None
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
print(f"FFmpeg available: {has_ffmpeg}")

# === ORIGINAL 5 (from previous training) ===
# === POPULAR COVERS 10 (new additions) ===
videos = [
    # Originals (kept from previous training)
    ('https://www.youtube.com/watch?v=TZB0vwRxtgI', 'jfla_01_alien'),
    ('https://www.youtube.com/watch?v=8BWo0Xt4Bfk', 'jfla_02_take_me_back'),
    ('https://www.youtube.com/watch?v=iFDpCiaaY2Q', 'jfla_03_how_could_i'),
    ('https://www.youtube.com/watch?v=swZ9w-kwXYE', 'jfla_04_one_sweet_day'),
    ('https://www.youtube.com/watch?v=oZsa-qgs21Q', 'jfla_05_moon_eater'),
    # Popular covers (new)
    ('https://www.youtube.com/watch?v=MhQKe-aERsU', 'jfla_06_shape_of_you'),       # Ed Sheeran - 334M+ views
    ('https://www.youtube.com/watch?v=4bmUFRxNEIg', 'jfla_07_despacito'),           # Luis Fonsi - 110M+ views
    ('https://www.youtube.com/watch?v=VD1vuQJFvvY', 'jfla_08_closer'),              # Chainsmokers
    ('https://www.youtube.com/watch?v=ZIBWwdCyNro', 'jfla_09_new_rules'),           # Dua Lipa
    ('https://www.youtube.com/watch?v=BkrrbDAca9w', 'jfla_10_price_tag'),           # Jessie J
    ('https://www.youtube.com/watch?v=LouUEaweP3M', 'jfla_11_no_tears'),            # Ariana Grande
    ('https://www.youtube.com/watch?v=YopnW1cOxac', 'jfla_12_eyes_nose_lips'),      # Taeyang (Korean)
    ('https://www.youtube.com/watch?v=cr6kwdYSHJU', 'jfla_13_if_you'),              # BIGBANG (Korean)
    ('https://www.youtube.com/watch?v=55zmJLagrDQ', 'jfla_14_love_you_like'),       # Selena Gomez
    ('https://www.youtube.com/watch?v=zH7LgsZXPRs', 'jfla_15_ugly'),               # 2NE1 (Korean)
]

downloaded = 0
failed = []
for url, filename in videos:
    out_path = os.path.join(output_dir, f"{filename}.wav")
    if os.path.exists(out_path):
        print(f"[SKIP] {filename} already exists")
        downloaded += 1
        continue

    print(f"\n=== Downloading {filename} ===")
    try:
        ydl_opts = {
            'format': '140/251/bestaudio/best',
            'outtmpl': os.path.join(output_dir, f'{filename}.%(ext)s'),
            'nocheckcertificate': True,
        }
        if has_ffmpeg:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }]
            if ffmpeg_dir:
                ydl_opts['ffmpeg_location'] = ffmpeg_dir

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            downloaded += 1
            print(f"  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append(filename)

print(f"\n{'='*60}")
print(f"Downloaded: {downloaded}/{len(videos)}")
if failed:
    print(f"Failed: {', '.join(failed)}")

# List all files
print(f"\nFiles in {output_dir}:")
total_size = 0
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    total_size += size_mb
    print(f"  {f} ({size_mb:.1f} MB)")
print(f"\nTotal: {total_size:.1f} MB")
