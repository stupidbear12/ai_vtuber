"""
generate_script.py
emeth 버튜버 대본 자동 생성기

사용법:
    python generate_script.py
    python generate_script.py --topic "시편 23편과 함께하는 새벽 lofi"
    python generate_script.py --topic "주제" --model gemma3:4b

의존성 설치:
    pip install ollama
"""

import ollama
import argparse
import os
import sys
from datetime import datetime
from config import (
    MODEL_NAME, BASE_MODEL, MODELFILE_PATH,
    CHARACTER_NAME, SCRIPT_DIR, PROMPT_TEMPLATE, TARGET_LENGTH
)


def check_model_exists(model_name: str) -> bool:
    """Ollama에 해당 모델이 존재하는지 확인"""
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        return any(model_name in m for m in available)
    except Exception as e:
        print(f"[오류] Ollama 연결 실패: {e}")
        print("  → Ollama가 실행 중인지 확인해주세요: ollama serve")
        sys.exit(1)


def create_emeth_model():
    """Modelfile로 emeth 커스텀 모델 생성"""
    modelfile_path = os.path.join(os.path.dirname(__file__), MODELFILE_PATH)
    if not os.path.exists(modelfile_path):
        print(f"[오류] Modelfile을 찾을 수 없습니다: {modelfile_path}")
        sys.exit(1)

    print(f"[emeth 모델 생성 중] {MODEL_NAME} ← {BASE_MODEL}")
    with open(modelfile_path, "r", encoding="utf-8") as f:
        modelfile_content = f.read()

    try:
        for progress in ollama.create(model=MODEL_NAME, modelfile=modelfile_content, stream=True):
            status = progress.get("status", "")
            if status:
                print(f"  {status}")
        print(f"[완료] '{MODEL_NAME}' 모델 생성 완료!\n")
    except Exception as e:
        print(f"[오류] 모델 생성 실패: {e}")
        sys.exit(1)


def generate_script(topic: str, use_model: str = None) -> str:
    """주어진 주제로 대본 생성"""
    model = use_model or MODEL_NAME

    # 커스텀 모델이 없으면 생성
    if model == MODEL_NAME and not check_model_exists(MODEL_NAME):
        print(f"[안내] '{MODEL_NAME}' 모델이 없습니다. 생성을 시작합니다...")
        create_emeth_model()

    # 베이스 모델 직접 사용 시 시스템 프롬프트 포함
    if model != MODEL_NAME:
        system_prompt = f"""당신은 '{CHARACTER_NAME}'라는 이름의 버튜버입니다.
성경과 복음 기반의 lofi 음악을 소개하며, 차분하고 따뜻한 존댓말을 사용합니다.
시청자는 '여러분'이라고 부릅니다."""
    else:
        system_prompt = ""

    prompt = PROMPT_TEMPLATE.format(
        topic=topic,
        target_length=TARGET_LENGTH
    )

    print(f"\n[대본 생성 시작]")
    print(f"  모델: {model}")
    print(f"  주제: {topic}")
    print(f"  목표 분량: 약 {TARGET_LENGTH}자 이상\n")
    print("─" * 50)

    full_response = ""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
            full_response += content
    except Exception as e:
        print(f"\n[오류] 대본 생성 실패: {e}")
        sys.exit(1)

    print("\n" + "─" * 50)
    return full_response


def save_script(topic: str, script_content: str) -> str:
    """대본을 txt 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 파일명으로 쓰기 안전한 주제 문자열 처리
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:30].strip()
    filename = f"emeth_{safe_topic}_{timestamp}.txt"
    filepath = os.path.join(SCRIPT_DIR, filename)

    header = f"""# emeth 방송 대본
# 생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 주제: {topic}
# ─────────────────────────────────────────

"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + script_content)

    print(f"\n[저장 완료] {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="emeth 버튜버 대본 자동 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python generate_script.py
  python generate_script.py --topic "시편 23편과 함께하는 새벽 lofi"
  python generate_script.py --topic "주님의 평안" --model gemma3:4b
        """
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="대본 주제 (미입력 시 직접 입력 요청)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=f"사용할 모델명 (기본값: {MODEL_NAME})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="파일 저장 없이 화면에만 출력"
    )

    args = parser.parse_args()

    print("=" * 50)
    print(f"  emeth 버튜버 대본 생성기")
    print(f"  캐릭터: {CHARACTER_NAME} | 모델: {args.model or MODEL_NAME}")
    print("=" * 50)

    # 주제 입력
    topic = args.topic
    if not topic:
        print("\n오늘 방송의 주제를 입력해주세요.")
        print("예) 시편 23편과 함께하는 새벽 lofi, 주님의 평안, 은혜로운 아침")
        topic = input("주제: ").strip()
        if not topic:
            print("[오류] 주제를 입력해야 합니다.")
            sys.exit(1)

    # 대본 생성
    script = generate_script(topic, use_model=args.model)

    # 저장
    if not args.no_save:
        save_script(topic, script)
    
    print("\n[완료] 대본 생성이 끝났습니다. 방송에 활용해 보세요 :)")


if __name__ == "__main__":
    main()
