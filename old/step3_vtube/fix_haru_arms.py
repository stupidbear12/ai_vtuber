"""
fix_haru_arms.py - Haru 모델 팔 4개 문제 수정

원인: VTube Studio가 pose3.json Pose 시스템을 실행하지 않아
      Part01ArmRA001/Part01ArmLA001 (A세트)와
      Part01ArmRB001/Part01ArmLB001 (B세트)가 동시에 렌더링됨.

수정: ColorTintRequest로 B세트 artmesh의 alpha를 0으로 설정.

사용법:
    cd step3_vtube
    python fix_haru_arms.py [--list]  # --list 는 artmesh 목록만 출력
"""

import asyncio
import json
import sys
import websockets

VTS_URI = "ws://localhost:8001"
TOKEN_FILE = "./vts_token.json"
PLUGIN_NAME = "emeth-controller"
PLUGIN_DEVELOPER = "emeth"

# B세트 파츠를 식별하는 키워드 (artmesh 이름에 포함된 경우 숨김)
B_ARM_KEYWORDS = ["ArmRB", "ArmLB"]


def _load_token() -> str:
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # pyvts 토큰 파일 형식: {"token": "..."}  또는 {"authenticationToken": "..."}
        return data.get("token") or data.get("authenticationToken") or ""
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


async def _send(ws, msg_type: str, data: dict = None) -> dict:
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"fix_{msg_type}",
        "messageType": msg_type,
        "data": data or {},
    }
    await ws.send(json.dumps(msg))
    resp_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
    return json.loads(resp_raw)


async def run(list_only: bool = False):
    token = _load_token()

    async with websockets.connect(VTS_URI) as ws:
        # ── 인증 ──────────────────────────────────────────────
        if not token:
            print("[*] 저장된 토큰 없음 - VTube Studio 팝업에서 승인하세요")
            for attempt in range(6):
                resp = await _send(ws, "AuthenticationTokenRequest", {
                    "pluginName": PLUGIN_NAME,
                    "pluginDeveloper": PLUGIN_DEVELOPER,
                })
                err_id = resp.get("data", {}).get("errorID")
                # errorID 51 = another auth window already open; wait and retry
                if err_id == 51:
                    print(f"  VTube Studio에 다른 인증 팝업이 열려 있습니다. "
                          f"닫아주세요... ({attempt+1}/6)")
                    await asyncio.sleep(5)
                    continue
                token = resp.get("data", {}).get("authenticationToken", "")
                break
            if not token:
                print(f"[!] 토큰 발급 실패: {resp}")
                return
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump({"token": token}, f)
            print(f"[+] 토큰 저장: {TOKEN_FILE}")

        resp = await _send(ws, "AuthenticationRequest", {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEVELOPER,
            "authenticationToken": token,
        })
        if not resp.get("data", {}).get("authenticated"):
            print(f"[!] 인증 실패: {resp}")
            return
        print("[+] VTube Studio 인증 성공")

        # ── 현재 모델 확인 ─────────────────────────────────────
        resp = await _send(ws, "CurrentModelRequest")
        model_name = resp.get("data", {}).get("modelName", "(알 수 없음)")
        print(f"[*] 현재 모델: {model_name}")

        # ── Artmesh 목록 조회 ──────────────────────────────────
        resp = await _send(ws, "ArtMeshListRequest")
        all_meshes = resp.get("data", {}).get("artMeshNames", [])
        print(f"[*] 전체 artmesh 수: {len(all_meshes)}")

        if list_only:
            print("\n[전체 Artmesh 목록]")
            for m in sorted(all_meshes):
                print(f"  {m}")
            return

        # B arm 키워드로 대상 artmesh 필터링
        b_arm_meshes = [m for m in all_meshes
                        if any(kw in m for kw in B_ARM_KEYWORDS)]

        if not b_arm_meshes:
            print("\n[!] B arm artmesh를 찾지 못했습니다.")
            print("    Arm 관련 artmesh 목록:")
            arm_meshes = [m for m in all_meshes
                          if "arm" in m.lower() or "Arm" in m]
            for m in arm_meshes:
                print(f"  {m}")
            print("\n    --list 옵션으로 전체 목록을 확인하세요.")
            return

        print(f"\n[*] 숨길 B arm artmesh ({len(b_arm_meshes)}개):")
        for m in b_arm_meshes:
            print(f"  {m}")

        # ── ColorTintRequest로 alpha=0 적용 ─────────────────────
        resp = await _send(ws, "ColorTintRequest", {
            "colorTint": {
                "colorR": 255,
                "colorG": 255,
                "colorB": 255,
                "colorA": 0,          # 완전 투명
                "mixWithSceneLightingColor": 1.0,
            },
            "artMeshMatcher": {
                "tintAll": False,
                "artMeshNumber": [],
                "nameExact": b_arm_meshes,
                "nameContains": [],
            },
        })

        err = resp.get("data", {}).get("errorID")
        if err:
            print(f"[!] ColorTintRequest 오류 (errorID={err}): {resp}")
        else:
            matched = resp.get("data", {}).get("matchedArtMeshes", len(b_arm_meshes))
            print(f"[+] 완료: {matched}개 B arm artmesh 숨김 처리")
            print("    ※ 이 설정은 모델을 다시 로드하면 초기화됩니다.")
            print("    영구 적용: vtube_controller.py의 connect() 이후 hide_b_arms() 호출")

        # ── 결과 확인: Arm 관련 artmesh 현황 출력 ────────────────
        arm_meshes = [m for m in all_meshes if "arm" in m.lower() or "Arm" in m]
        print(f"\n[Arm 관련 artmesh 전체 ({len(arm_meshes)}개)]")
        for m in sorted(arm_meshes):
            tag = "← 숨김" if m in b_arm_meshes else ""
            print(f"  {m}  {tag}")


if __name__ == "__main__":
    list_only = "--list" in sys.argv
    asyncio.run(run(list_only=list_only))
