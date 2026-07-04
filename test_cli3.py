"""Test with adjusted parameters - try to fix short output"""
import os
import sys

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS"))
os.chdir(GPT_SOVITS_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import soundfile as sf
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

# Use PRETRAINED GPT + FINE-TUNED SoVITS
GPT_MODEL = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
SOVITS_MODEL = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2", "jfla_e20_s420.pth")

# Try using a SLICED vocal segment as reference (shorter, cleaner, speech-like)
SLICED_DIR = os.path.join(GPT_SOVITS_DIR, "jfla_sliced")
sliced_files = sorted([f for f in os.listdir(SLICED_DIR) if f.endswith('.wav')])

# Pick a segment that's about 3-5 seconds
best_ref = None
best_dur = 0
for f in sliced_files:
    data, sr = sf.read(os.path.join(SLICED_DIR, f))
    dur = len(data) / sr
    if 3 <= dur <= 6:
        if dur > best_dur:
            best_ref = f
            best_dur = dur
    elif best_ref is None and dur > 2:
        best_ref = f
        best_dur = dur

ref_audio_path = os.path.join(SLICED_DIR, best_ref) if best_ref else os.path.join(GPT_SOVITS_DIR, "jfla.wav")
ref_data, ref_sr = sf.read(ref_audio_path)
ref_dur = len(ref_data) / ref_sr
print(f"Reference: {os.path.basename(ref_audio_path)} ({ref_dur:.1f}s)")

# Get the transcript for the reference
asr_list = os.path.join(GPT_SOVITS_DIR, "jfla_asr", "jfla_sliced.list")
ref_text = "I could be an alien"  # default
with open(asr_list, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if best_ref and best_ref in parts[0]:
            ref_text = parts[3].strip() if len(parts) > 3 else ref_text
            break
print(f"Reference text: {ref_text}")

print("\nLoading pretrained GPT + fine-tuned SoVITS...")
change_gpt_weights(gpt_path=GPT_MODEL)
change_sovits_weights(sovits_path=SOVITS_MODEL)
print("Models loaded!")

output_dir = os.path.join(GPT_SOVITS_DIR, "test_output_finetuned")

test_cases = [
    ("안녕하세요, 저는 시온이에요. 반가워요!", "한국어", "sion_kr_v3_01"),
    ("오늘 방송 재미있게 할게요, 많이 응원해주세요!", "한국어", "sion_kr_v3_02"),
    ("와 진짜요? 대박이다! 너무 좋아요!", "한국어", "sion_kr_v3_03"),
    ("Hello everyone! I'm Sion, it's so nice to meet you all today!", "영어", "sion_en_v3_01"),
]

# Try with different parameters
for text, lang, filename in test_cases:
    print(f"\nGenerating [{lang}]: {text}")
    try:
        result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=i18n("영어"),
            text=text,
            text_language=i18n(lang),
            top_p=0.8,         # Lower top_p for more focused output
            temperature=0.8,    # Lower temp for more predictable output
        )
        result_list = list(result)
        if result_list:
            sr, audio = result_list[-1]
            output_path = os.path.join(output_dir, f"{filename}.wav")
            sf.write(output_path, audio, sr)
            dur = len(audio) / sr
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved: {filename}.wav ({size_kb:.0f} KB, {dur:.1f}s)")
        else:
            print(f"  No result!")
    except Exception as e:
        import traceback
        traceback.print_exc()

# Also try with pretrained SoVITS (no fine-tuning) for comparison
print(f"\n{'='*60}")
print("Now testing with FULLY PRETRAINED models (no fine-tuning)...")
PRETRAINED_SOVITS = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth")
change_sovits_weights(sovits_path=PRETRAINED_SOVITS)

for text, lang, filename_base in [
    ("안녕하세요, 저는 시온이에요. 반가워요!", "한국어", "sion_kr_pretrained_01"),
    ("Hello everyone! I'm Sion, it's so nice to meet you!", "영어", "sion_en_pretrained_01"),
]:
    print(f"\nGenerating [{lang}]: {text}")
    try:
        result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=i18n("영어"),
            text=text,
            text_language=i18n(lang),
            top_p=0.8,
            temperature=0.8,
        )
        result_list = list(result)
        if result_list:
            sr, audio = result_list[-1]
            output_path = os.path.join(output_dir, f"{filename_base}.wav")
            sf.write(output_path, audio, sr)
            dur = len(audio) / sr
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved: {filename_base}.wav ({size_kb:.0f} KB, {dur:.1f}s)")
    except Exception as e:
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("All files:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.wav'):
        data, sr = sf.read(os.path.join(output_dir, f))
        dur = len(data) / sr
        sz = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  {f}: {dur:.2f}s ({sz:.0f} KB)")
