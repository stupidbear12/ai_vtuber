"""Test fine-tuned GPT-SoVITS via direct inference (no API server)"""
import os
import sys

GPT_SOVITS_DIR = r"C:\Users\thtgg\workspace2\GPT-SoVITS"
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS"))
os.chdir(GPT_SOVITS_DIR)

# Force CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import soundfile as sf
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

GPT_MODEL = os.path.join(GPT_SOVITS_DIR, "GPT_weights_v2", "jfla-e5.ckpt")
SOVITS_MODEL = os.path.join(GPT_SOVITS_DIR, "SoVITS_weights_v2", "jfla_e20_s420.pth")
REF_AUDIO = os.path.join(GPT_SOVITS_DIR, "jfla.wav")

print("Loading models...")
change_gpt_weights(gpt_path=GPT_MODEL)
change_sovits_weights(sovits_path=SOVITS_MODEL)
print("Models loaded!")

output_dir = os.path.join(GPT_SOVITS_DIR, "test_output_finetuned")
os.makedirs(output_dir, exist_ok=True)

test_cases = [
    ("안녕하세요, 저는 시온이에요. 반가워요!", "한국어", "sion_kr_01"),
    ("오늘 방송 재미있게 할게요, 많이 응원해주세요!", "한국어", "sion_kr_02"),
    ("와 진짜요? 대박이다!", "한국어", "sion_kr_03"),
    ("Hello everyone! I'm Sion, nice to meet you!", "영어", "sion_en_01"),
]

for text, lang, filename in test_cases:
    print(f"\nGenerating [{lang}]: {text}")
    try:
        result = get_tts_wav(
            ref_wav_path=REF_AUDIO,
            prompt_text="I'm in love with the shape of you",
            prompt_language=i18n("영어"),
            text=text,
            text_language=i18n(lang),
            top_p=1,
            temperature=1,
        )
        result_list = list(result)
        if result_list:
            sr, audio = result_list[-1]
            output_path = os.path.join(output_dir, f"{filename}.wav")
            sf.write(output_path, audio, sr)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved: {filename}.wav ({size_kb:.0f} KB, {len(audio)/sr:.1f}s)")
        else:
            print(f"  No result!")
    except Exception as e:
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"Results: {output_dir}")
for f in sorted(os.listdir(output_dir)):
    sz = os.path.getsize(os.path.join(output_dir, f)) / 1024
    print(f"  {f} ({sz:.0f} KB)")
