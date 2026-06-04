# -*- coding: utf-8 -*-
"""
move_model.py - VTube Studio 모델 위치/크기 조정 스크립트

MoveModelRequest를 사용해 현재 로드된 모델을
중앙 하단에 배치하고 크기를 줄입니다.
"""

import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "step3_vtube"))

from config_vtube import VTS_HOST, VTS_PORT, PLUGIN_NAME, PLUGIN_DEVELOPER, TOKEN_FILE
import pyvts


async def get_current_position(vts) -> dict:
    resp = await vts.request(vts.vts_request.BaseRequest("CurrentModelRequest"))
    return resp.get("data", {}).get("modelPosition", {})


async def move_model(vts, pos_x: float, pos_y: float, size: float,
                     rotation: float = 0.0, duration: float = 1.0):
    """
    MoveModelRequest로 모델 이동/축소.

    Parameters
    ----------
    pos_x    : float  X 위치 (-1.0 ~ 1.0, 0 = 중앙)
    pos_y    : float  Y 위치 (-1.0 ~ 1.0, 0 = 중앙, 음수 = 아래)
    size     : float  크기 (-100 ~ 100, 0 = 기본)
    rotation : float  회전 각도 (도)
    duration : float  이동 애니메이션 시간 (초)
    """
    req = vts.vts_request.BaseRequest(
        "MoveModelRequest",
        {
            "timeInSeconds": duration,
            "valuesAreRelativeToModel": False,
            "positionX": pos_x,
            "positionY": pos_y,
            "rotation": rotation,
            "size": size,
        }
    )
    resp = await vts.request(req)
    return resp


async def main():
    vts = pyvts.vts(
        plugin_info={
            "plugin_name": PLUGIN_NAME,
            "developer":   PLUGIN_DEVELOPER,
            "authentication_token_path": TOKEN_FILE,
        },
        vts_api_info={"version": "1.0", "name": "VTubeStudioPublicAPI", "port": VTS_PORT}
    )

    print(f"VTube Studio 연결 중 ({VTS_HOST}:{VTS_PORT})...")
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()
    print("[OK] 연결 및 인증 완료")

    # ── 현재 위치 확인 ──────────────────────────────────────
    pos = await get_current_position(vts)
    print("\n=== 현재 모델 위치 ===")
    print(json.dumps(pos, indent=2, ensure_ascii=False))

    current_size = pos.get("size", 0.0)
    current_x    = pos.get("positionX", 0.0)
    current_y    = pos.get("positionY", 0.0)

    # ── 목표값 설정 ─────────────────────────────────────────
    # VTube Studio size 범위: -100(최소) ~ 100(최대)
    # 현재 -74.5인데 모델이 크게 보인다면 vtube.json 기본 스케일이 큰 것.
    # size를 -95로 설정 (현재보다 훨씬 작게) + Y를 내려서 전신 보이게
    target_size = -95.0        # VTS 최소값 근처
    target_x    =  0.0         # X 중앙
    target_y    = -0.5         # 아래로 내려 전신 표시

    print(f"\n=== 조정 계획 ===")
    print(f"  size  : {current_size:.1f} → {target_size:.1f}")
    print(f"  posX  : {current_x:.3f}  → {target_x:.3f}")
    print(f"  posY  : {current_y:.3f}  → {target_y:.3f}")
    print()

    # ── 이동 실행 ───────────────────────────────────────────
    print("모델 이동/축소 중 (1.5초)...")
    resp = await move_model(vts, target_x, target_y, target_size, duration=1.5)
    print(f"응답: {resp.get('messageType', '?')}")

    await asyncio.sleep(2.0)

    # ── 결과 확인 ───────────────────────────────────────────
    pos_after = await get_current_position(vts)
    print("\n=== 이동 후 모델 위치 ===")
    print(json.dumps(pos_after, indent=2, ensure_ascii=False))

    await vts.close()
    print("\n[완료] 연결 종료")


if __name__ == "__main__":
    asyncio.run(main())
