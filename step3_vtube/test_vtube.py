"""
test_vtube.py - VTube Studio 연결 / 기능 테스트 스크립트

실행 전 확인:
  1. VTube Studio 실행 중
  2. 설정 > API 탭 > WebSocket API 활성화 (포트 8001)
  3. 모델 로드 완료

실행 방법:
  cd (프로젝트 루트)
  python -m step3_vtube.test_vtube
"""

import asyncio
import math
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from step3_vtube.vtube_controller import VTubeController


async def test_connection(ctrl: VTubeController) -> bool:
    """연결 및 인증 테스트"""
    print("\n" + "=" * 50)
    print("TEST 1: VTube Studio 연결 테스트")
    print("=" * 50)
    try:
        await ctrl.connect()
        print("[OK] 연결 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 연결 실패: {e}")
        return False


async def test_model_info(ctrl: VTubeController):
    """현재 로드된 모델 정보 출력"""
    print("\n" + "=" * 50)
    print("TEST 2: 모델 정보 조회")
    print("=" * 50)
    try:
        info = await ctrl.get_current_model()
        if info["model_loaded"]:
            print("[OK] 모델 로드됨")
            print(f"     이름     : {info['model_name']}")
            print(f"     ID       : {info['model_id']}")
            print(f"     VTS 이름 : {info['vts_model_name']}")
            print(f"     파일     : {info['live2d_file']}")
        else:
            print("[WARN] 모델이 로드되어 있지 않습니다.")
    except Exception as e:
        print(f"[FAIL] 모델 정보 조회 실패: {e}")


async def test_expressions(ctrl: VTubeController):
    """표정 전환 테스트: default -> calm -> happy -> default"""
    print("\n" + "=" * 50)
    print("TEST 3: 표정 전환 테스트")
    print("=" * 50)
    sequence = [
        ("default", 1.5),
        ("calm",    2.0),
        ("happy",   2.0),
        ("default", 1.0),
    ]
    for expr_name, hold_sec in sequence:
        try:
            await ctrl.set_expression(expr_name)
            print(f"[OK] 표정: {expr_name} -> {hold_sec}초 유지")
            await asyncio.sleep(hold_sec)
        except Exception as e:
            print(f"[FAIL] 표정 전환 실패 ({expr_name}): {e}")


async def test_mouth_movement(ctrl: VTubeController):
    """입 움직임 테스트 (사인파 5초)"""
    print("\n" + "=" * 50)
    print("TEST 4: 입 움직임 테스트 (5초)")
    print("=" * 50)
    try:
        for i in range(50):
            val = (math.sin(i * 0.4) + 1.0) / 2.0
            await ctrl.set_mouth_open(val)
            await asyncio.sleep(0.1)
        await ctrl.set_mouth_open(0.0)
        print("[OK] 입 움직임 테스트 완료")
    except Exception as e:
        print(f"[FAIL] 입 움직임 오류: {e}")


async def test_idle(ctrl: VTubeController):
    """아이들 애니메이션 테스트 (5초)"""
    print("\n" + "=" * 50)
    print("TEST 5: 아이들 애니메이션 (5초)")
    print("=" * 50)
    try:
        await ctrl.idle_animation(duration=5.0)
        print("[OK] 아이들 애니메이션 완료")
    except Exception as e:
        print(f"[FAIL] 아이들 애니메이션 오류: {e}")


async def test_lipsync_file(ctrl: VTubeController, audio_path: str):
    """오디오 파일 립싱크 테스트"""
    print("\n" + "=" * 50)
    print("TEST 6: 립싱크 파일 테스트")
    print("=" * 50)
    if not os.path.exists(audio_path):
        print(f"[SKIP] 오디오 파일 없음: {audio_path}")
        print("       step2_tts/output_audio/ 에 .mp3 파일이 있으면 경로를 지정하세요.")
        return
    try:
        print(f"파일: {audio_path}")
        await ctrl.lipsync_from_audio(audio_path)
        print("[OK] 립싱크 테스트 완료")
    except Exception as e:
        print(f"[FAIL] 립싱크 오류: {e}")


async def main():
    sample_audio = os.path.join(
        ROOT_DIR, "step2_tts", "output_audio", "sample.mp3"
    )

    ctrl = VTubeController()
    connected = await test_connection(ctrl)

    if not connected:
        print("\n연결 실패 - VTube Studio가 실행 중인지, API가 활성화되어 있는지 확인하세요.")
        return

    await test_model_info(ctrl)
    await test_expressions(ctrl)
    await test_mouth_movement(ctrl)
    await test_idle(ctrl)
    await test_lipsync_file(ctrl, sample_audio)

    await ctrl.disconnect()
    print("\n" + "=" * 50)
    print("모든 테스트 완료")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
