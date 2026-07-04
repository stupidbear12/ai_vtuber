import yt_dlp
import traceback
import sys
import os
import subprocess

output_dir = r"C:\Users\thtgg\workspace2\GPT-SoVITS\jfla_audio"
os.makedirs(output_dir, exist_ok=True)

# Check ffmpeg availability
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

# JFla songs to download (~17 min total)
videos = [
    'https://www.youtube.com/watch?v=TZB0vwRxtgI',  # Alien (247s)
    'https://www.youtube.com/watch?v=8BWo0Xt4Bfk',  # Take Me Back (209s)
    'https://www.youtube.com/watch?v=iFDpCiaaY2Q',  # How Could I Be This Into You (202s)
    'https://www.youtube.com/watch?v=swZ9w-kwXYE',  # One Sweet Day in Paris (202s)
    'https://www.youtube.com/watch?v=oZsa-qgs21Q',  # Moon Eater (187s)
]

downloaded = 0
for url in videos:
    vid_id = url.split('v=')[1]
    print(f"\n=== Downloading {vid_id} ===")
    try:
        ydl_opts = {
            'format': '140/251/bestaudio/best',  # prefer m4a 128k or opus
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
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

print(f"\n=== Done: {downloaded}/{len(videos)} downloaded ===")

# List downloaded files
files = os.listdir(output_dir)
total_size = 0
print(f"\nFiles in {output_dir}:")
for f in sorted(files):
    fpath = os.path.join(output_dir, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    total_size += size_mb
    print(f"  {f} ({size_mb:.1f} MB)")
print(f"\nTotal: {total_size:.1f} MB")

# Pick ~5 longer JFla songs for enough vocal data (~10 min total)
videos = [
    'https://www.youtube.com/watch?v=TZB0vwRxtgI',  # Alien (247s)
    'https://www.youtube.com/watch?v=8BWo0Xt4Bfk',  # Take Me Back (209s)
    'https://www.youtube.com/watch?v=iFDpCiaaY2Q',  # How Could I Be This Into You (202s)
    'https://www.youtube.com/watch?v=swZ9w-kwXYE',  # One Sweet Day in Paris (202s)
    'https://www.youtube.com/watch?v=oZsa-qgs21Q',  # Moon Eater (187s)
]

# Check ffmpeg
ffmpeg_path = r"C:\Users\thtgg\workspace2\GPT-SoVITS\ffmpeg.exe"
if not os.path.exists(ffmpeg_path):
    # Try system ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        ffmpeg_path = 'ffmpeg'
    except FileNotFoundError:
        print("WARNING: ffmpeg not found, will download raw audio without conversion")
        ffmpeg_path = None

downloaded = 0
for url in videos:
    vid_id = url.split('v=')[1]
    print(f"\n=== Downloading {vid_id} ===")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'extractor_args': {'youtube': {'player_client': ['web']}},
            'nocheckcertificate': True,
        }
        if ffmpeg_path:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }]
            if ffmpeg_path != 'ffmpeg':
                ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            downloaded += 1
            print(f"  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\n=== Done: {downloaded}/{len(videos)} downloaded ===")

# List downloaded files
files = os.listdir(output_dir)
print(f"Files in {output_dir}:")
for f in files:
    fpath = os.path.join(output_dir, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    print(f"  {f} ({size_mb:.1f} MB)")
