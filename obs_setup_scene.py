# -*- coding: utf-8 -*-
"""OBS 앨범 리뷰 씬 생성 스크립트"""
import obsws_python as obs
import sys
import time

OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASS = "LzDko8s5VoYj3XoA"

SCENE_NAME = "앨범 리뷰"
CANVAS_W, CANVAS_H = 1920, 1080


def main():
    cl = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

    if len(sys.argv) < 2:
        print("Usage: python obs_setup_scene.py [list|create]")
        return

    if sys.argv[1] == "list":
        scenes = cl.get_scene_list()
        print(f"Current: {scenes.current_program_scene_name}")
        for s in scenes.scenes:
            name = s["sceneName"]
            print(f"  - {name}")
            items = cl.get_scene_item_list(name)
            for item in items.scene_items:
                src = item["sourceName"]
                kind = item.get("inputKind", "")
                sid = item["sceneItemId"]
                t = item.get("sceneItemTransform", {})
                x, y = t.get("positionX", 0), t.get("positionY", 0)
                w, h = t.get("width", 0), t.get("height", 0)
                print(f"      [{kind}] {src} (id={sid}, pos={x:.0f},{y:.0f} size={w:.0f}x{h:.0f})")

    elif sys.argv[1] == "create":
        # 1. 씬 생성
        try:
            cl.create_scene(SCENE_NAME)
            print(f"[OK] Scene '{SCENE_NAME}' created")
        except Exception as e:
            if "601" in str(e):
                print(f"[OK] Scene '{SCENE_NAME}' already exists")
            else:
                print(f"[ERR] Scene create: {e}")
                return

        # 2. 브라우저 창 캡처 — Live2D와 같은 방식(browser_source)으로 YouTube 표시
        #    Playwright Chrome 창은 별도 캡처가 필요 -> window_capture 사용
        try:
            cl.create_input(
                SCENE_NAME,
                "album_yt_browser",
                "window_capture",
                {
                    "capture_method": "method_bitblt",
                    "cursor": False,
                    "window": "",  # 첫 실행 시 수동 선택 필요
                },
                True,
            )
            print("[OK] Added: album_yt_browser (window_capture)")
        except Exception as e:
            print(f"[SKIP] album_yt_browser: {e}")

        # 위치/크기 조정: 전체 화면
        _set_transform(cl, SCENE_NAME, "album_yt_browser", 0, 0, CANVAS_W, CANVAS_H)

        # 3. Live2D 오버레이 (browser_source) — 기존 live_2d 소스와 동일 URL
        try:
            cl.create_input(
                SCENE_NAME,
                "album_live2d",
                "browser_source",
                {
                    "url": "http://localhost:8001/live2d/static/index.html?transparent=1&v=3",
                    "width": 1920,
                    "height": 1080,
                    "css": "body { background: transparent; }",
                    "reroute_audio": True,
                },
                True,
            )
            print("[OK] Added: album_live2d (browser_source)")
        except Exception as e:
            print(f"[SKIP] album_live2d: {e}")

        # Live2D 우측 하단 배치 (400x500 크기, 우측 하단)
        live2d_w, live2d_h = 500, 650
        live2d_x = CANVAS_W - live2d_w - 20  # 우측 여백 20
        live2d_y = CANVAS_H - live2d_h - 20  # 하단 여백 20
        _set_transform(cl, SCENE_NAME, "album_live2d", live2d_x, live2d_y, live2d_w, live2d_h)

        # 4. 리뷰 자막 오버레이 (browser_source)
        subtitle_path = "C:/Users/thtgg/workspace2/ai_vtuber/obs/album_review_subtitle.html"
        try:
            cl.create_input(
                SCENE_NAME,
                "album_subtitle",
                "browser_source",
                {
                    "url": f"file:///{subtitle_path}",
                    "width": 1920,
                    "height": 200,
                    "css": "body { background: transparent; }",
                },
                True,
            )
            print("[OK] Added: album_subtitle (browser_source)")
        except Exception as e:
            print(f"[SKIP] album_subtitle: {e}")

        # 자막 하단 중앙 배치
        _set_transform(cl, SCENE_NAME, "album_subtitle", 0, CANVAS_H - 200, CANVAS_W, 200)

        print("\n[DONE] Scene created! In OBS:")
        print("  1. Select 'album_yt_browser' -> Properties -> choose Chrome window")
        print("  2. Adjust Live2D size if needed")
        print("  3. Switch to '앨범 리뷰' scene to preview")


def _set_transform(cl, scene, source, x, y, w, h):
    """소스의 위치와 크기를 설정한다."""
    try:
        item_id = cl.get_scene_item_id(scene, source).scene_item_id
        # 현재 소스 크기 가져오기
        transform = cl.get_scene_item_transform(scene, item_id)
        src_w = transform.scene_item_transform.get("sourceWidth", w)
        src_h = transform.scene_item_transform.get("sourceHeight", h)

        scale_x = w / src_w if src_w > 0 else 1
        scale_y = h / src_h if src_h > 0 else 1

        cl.set_scene_item_transform(scene, item_id, {
            "positionX": x,
            "positionY": y,
            "scaleX": scale_x,
            "scaleY": scale_y,
        })
        print(f"  -> {source}: pos=({x},{y}), scale=({scale_x:.2f},{scale_y:.2f})")
    except Exception as e:
        print(f"  -> Transform {source}: {e}")


if __name__ == "__main__":
    main()
