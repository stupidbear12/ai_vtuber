"""Download JFla original songs v3 for GPT-SoVITS training.
15 originals total: 5 existing + 10 from Burn The Flower album (2023).
NO covers - only studio-quality originals for clean vocal training.
"""
import yt_dlp
import os
import subprocess
import shutil

output_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio_v3"
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

# === Step 1: Copy existing 5 originals from jfla_audio/ ===
existing_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio"
existing_map = {
    # old_filename_pattern -> new_filename
    'alien': 'jfla_v3_01_alien',
    'take_me_back': 'jfla_v3_02_take_me_back',
    'how_could_i': 'jfla_v3_03_how_could_i',
    'one_sweet_day': 'jfla_v3_04_one_sweet_day',
    'moon_eater': 'jfla_v3_05_moon_eater',
}

if os.path.exists(existing_dir):
    print("\n=== Copying existing 5 originals ===")
    for f in os.listdir(existing_dir):
        if not f.endswith('.wav'):
            continue
        for key, new_name in existing_map.items():
            if key in f.lower():
                src = os.path.join(existing_dir, f)
                dst = os.path.join(output_dir, f"{new_name}.wav")
                if os.path.exists(dst):
                    print(f"  [SKIP] {new_name}.wav already exists")
                else:
                    shutil.copy2(src, dst)
                    print(f"  [COPY] {f} -> {new_name}.wav")
                break
else:
    print(f"WARNING: {existing_dir} not found. Will download all from YouTube.")

# === Step 2: Download 10 new Burn The Flower album tracks ===
new_songs = [
    ('https://www.youtube.com/watch?v=iivRH_79JhE', 'jfla_v3_06_my_childhood_dream'),
    ('https://www.youtube.com/watch?v=LGJrotrHuTA', 'jfla_v3_07_invisible_me'),
    ('https://www.youtube.com/watch?v=PeddM8fCh-k', 'jfla_v3_08_telecaster'),
    ('https://www.youtube.com/watch?v=MezczBL2fz0', 'jfla_v3_09_a_four_leaf_clover'),
    ('https://www.youtube.com/watch?v=ZS6HeUBNWVw', 'jfla_v3_10_to_me'),
    ('https://www.youtube.com/watch?v=g_UuZAOOkiA', 'jfla_v3_11_bedroom_singer'),
    ('https://www.youtube.com/watch?v=QaOM-82fb10', 'jfla_v3_12_the_hare'),
    ('https://www.youtube.com/watch?v=gHy-005FOTE', 'jfla_v3_13_before_i_met_you'),
    ('https://music.youtube.com/watch?v=cR0JKECPyOc', 'jfla_v3_14_nineteen'),
    ('https://music.youtube.com/watch?v=s2x9WYhzesw', 'jfla_v3_15_sorry_i_made_you_wait'),
]

downloaded = 0
failed = []
for url, filename in new_songs:
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

# === Summary ===
print(f"\n{'='*60}")
print(f"New songs downloaded: {downloaded}/{len(new_songs)}")
if failed:
    print(f"Failed: {', '.join(failed)}")

# List all files
print(f"\nAll files in {output_dir}:")
total_size = 0
wav_count = 0
for f in sorted(os.listdir(output_dir)):
    if not f.endswith('.wav'):
        continue
    fpath = os.path.join(output_dir, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    total_size += size_mb
    wav_count += 1
    print(f"  {f} ({size_mb:.1f} MB)")
print(f"\nTotal: {wav_count} files, {total_size:.1f} MB")
