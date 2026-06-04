# ============================================================
# test_obs.py - OBS WebSocket 연결 테스트
# VTuber 자동화 파이프라인 Step 4
# ============================================================
# 실행 방법:
#   cd C:\Users\thtgg\mydream\vtuber-auto\step4_obs
#   python test_obs.py
# ============================================================

import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, ".")

from obs_controller import OBSController, OBSConnectionError
from config_obs import OBS_HOST, OBS_PORT

SEPARATOR = "=" * 50


def test_connection(ctrl):
    """테스트 1: OBS 연결"""
    print(f"\n{SEPARATOR}")
    print("테스트 1: OBS WebSocket 연결")
    print(SEPARATOR)
    try:
        ctrl.connect()
        print(f"[OK] {OBS_HOST}:{OBS_PORT} 연결 성공")
        return True
    except OBSConnectionError as e:
        print(f"[FAIL] 연결 실패: {e}")
        print("\n※ 해결 방법:")
        print("  1. OBS Studio가 실행 중인지 확인")
        print("  2. 도구 > WebSocket 서버 설정 > 서버 활성화 체크")
        print("  3. config_obs.py의 OBS_PASSWORD 확인")
        return False


def test_scene_list(ctrl):
    """테스트 2: 씬 목록 출력"""
    print(f"\n{SEPARATOR}")
    print("테스트 2: 씬 목록 조회")
    print(SEPARATOR)
    scenes = ctrl.get_scene_list()
    if scenes:
        print(f"[OK] 총 {len(scenes)}개 씬 발견:")
        for i, s in enumerate(scenes, 1):
            print(f"  {i}. {s}")
    else:
        print("[WARN] 씬 목록이 비어있거나 조회 실패")
    return scenes


def test_status(ctrl):
    """테스트 3: 현재 상태 확인"""
    print(f"\n{SEPARATOR}")
    print("테스트 3: 현재 상태 확인")
    print(SEPARATOR)
    status = ctrl.get_status()
    print(f"  연결됨    : {status['connected']}")
    print(f"  OBS 버전  : {status.get('obs_version', 'N/A')}")
    print(f"  현재 씬   : {status['current_scene']}")
    print(f"  녹화 중   : {status['recording']}")
    print(f"  스트리밍  : {status['streaming']}")
    if status['recording_path']:
        print(f"  녹화 파일 : {status['recording_path']}")
    return status


def test_recording(ctrl):
    """테스트 4: 5초 테스트 녹화"""
    print(f"\n{SEPARATOR}")
    print("테스트 4: 5초 테스트 녹화")
    print(SEPARATOR)

    print("[1/3] 녹화 시작...")
    ok = ctrl.start_recording()
    if not ok:
        print("[FAIL] 녹화 시작 실패")
        return None

    print("[OK] 녹화 시작됨. 5초 대기...")
    for i in range(5, 0, -1):
        print(f"  {i}초 남음...", end="\r")
        time.sleep(1)
    print("  녹화 완료!          ")

    print("[2/3] 녹화 정지...")
    output_path = ctrl.stop_recording()
    if output_path:
        print(f"[OK] 녹화 파일 저장됨: {output_path}")
    else:
        print("[WARN] 녹화 파일 경로를 가져오지 못했습니다.")
        print("       (OBS 기본 녹화 경로에 저장되었을 수 있습니다)")

    print("[3/3] 최종 상태 확인...")
    status = ctrl.get_status()
    print(f"  녹화 중: {status['recording']}")
    return output_path


def main():
    print(f"\n{'#' * 50}")
    print("  OBS WebSocket 연결 테스트")
    print(f"  대상: {OBS_HOST}:{OBS_PORT}")
    print(f"{'#' * 50}")

    ctrl = OBSController()
    results = {}

    # 테스트 1: 연결
    connected = test_connection(ctrl)
    results["연결"] = "OK" if connected else "FAIL"

    if not connected:
        print("\n[중단] OBS 연결 실패로 테스트를 종료합니다.")
        print_summary(results)
        sys.exit(1)

    # 테스트 2: 씬 목록
    scenes = test_scene_list(ctrl)
    results["씬 목록"] = f"OK ({len(scenes)}개)" if scenes else "WARN (빈 목록)"

    # 테스트 3: 상태 확인
    status = test_status(ctrl)
    results["상태 조회"] = "OK"

    # 테스트 4: 테스트 녹화
    answer = input("\n5초 테스트 녹화를 진행하겠습니까? (y/N): ").strip().lower()
    if answer == "y":
        path = test_recording(ctrl)
        results["테스트 녹화"] = f"OK ({path})" if path else "WARN (경로 미확인)"
    else:
        print("테스트 녹화 건너뜀.")
        results["테스트 녹화"] = "SKIP"

    # 연결 해제
    ctrl.disconnect()
    print(f"\n[OK] OBS 연결 해제 완료")

    print_summary(results)


def print_summary(results):
    print(f"\n{SEPARATOR}")
    print("테스트 결과 요약")
    print(SEPARATOR)
    for name, result in results.items():
        icon = "[OK]  " if result.startswith("OK") else ("[WARN]" if result.startswith("WARN") else "[FAIL]")
        print(f"  {icon} {name}: {result}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
