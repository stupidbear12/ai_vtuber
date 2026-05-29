"""
api/main.py - emeth 방송 자동화 파이프라인 FastAPI 서버 (v0.7.0)
실행 (vtuber-auto/ 루트에서):
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

v0.7.0 신규 엔드포인트:
  POST /vtube/animation/start  — 전체 리깅 애니메이션 시작
  POST /vtube/animation/stop   — 전체 리깅 애니메이션 정지
  POST /vtube/emotion          — 감정 상태 변경 (calm/happy/surprised/thinking)
  POST /vtube/reaction         — 즉각 반응 애니메이션 트리거
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "step1_script"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step2_tts"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step3_vtube"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step4_obs"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step5_live"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step6_chat"))

# 파이프라인 상태
pipeline_status = {
    "state": "idle",
    "step": None,
    "topic": None,
    "script_path": None,
    "audio_path": None,
    "vtube_active": False,
    "obs_recording_path": None,
    "error": None,
    "updated_at": None,
}

# 채팅 파이프라인 인스턴스 및 상태
_chat_pipeline = None
_chat_task: Optional[asyncio.Task] = None
chat_status = {
    "running": False,
    "video_id": None,
    "live_chat_id": None,
    "started_at": None,
    "stats": {"received": 0, "responded": 0, "skipped": 0, "errors": 0},
}

# OBS 컨트롤러 싱글턴
_obs_controller = None

# VTube Studio 전체 애니메이션 컨트롤러 싱글턴
_vtube_full_ctrl = None   # VTubeController 인스턴스 (전체 애니메이션용)

# 라이브 파이프라인 싱글턴 및 상태
_live_pipeline = None
live_broadcast_state = {
    "broadcast_id": None,
    "watch_url": None,
    "title": None,
    "started_at": None,
    "youtube_status": None,
    "obs_streaming": False,
}


def _get_obs_controller():
    """OBS 컨트롤러 싱글턴 반환. 미연결 시 자동 연결 시도."""
    global _obs_controller
    try:
        from obs_controller import OBSController, OBSConnectionError
        if _obs_controller is None:
            _obs_controller = OBSController()
        if not _obs_controller._connected:
            _obs_controller.connect()
        return _obs_controller
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"obs_controller 모듈 없음: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OBS 연결 실패: {e}")


def _get_live_pipeline():
    """LivePipeline 싱글턴 반환."""
    global _live_pipeline
    if _live_pipeline is None:
        try:
            from live_pipeline import LivePipeline
            _live_pipeline = LivePipeline()
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"live_pipeline 모듈 없음: {e}")
    return _live_pipeline


def update_status(**kwargs):
    pipeline_status.update(kwargs)
    pipeline_status["updated_at"] = datetime.now().isoformat()


app = FastAPI(
    title="emeth 방송 자동화 API",
    description="스크립트 생성 + TTS + VTube Studio + OBS 제어 + YouTube 라이브 방송 통합 파이프라인",
    version="0.7.0",
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------------------------------------------ 요청 모델
class ScriptRequest(BaseModel):
    topic: str
    model: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    output_filename: Optional[str] = None

class PipelineRequest(BaseModel):
    topic: str
    model: Optional[str] = None
    voice: Optional[str] = None
    rate: Optional[str] = None
    use_vtube: bool = False
    use_obs: bool = False

class VTubeControlRequest(BaseModel):
    action: str
    expression: Optional[str] = None
    mouth_value: Optional[float] = None
    audio_path: Optional[str] = None
    idle_duration: Optional[float] = None

class ChatStartRequest(BaseModel):
    video_id: str

class SwitchSceneRequest(BaseModel):
    scene_name: str

class LiveStartRequest(BaseModel):
    title: Optional[str] = "emeth lofi 방송"
    description: Optional[str] = "emeth와 함께하는 잔잔한 lofi 음악"
    privacy: Optional[str] = "public"

class LiveEndRequest(BaseModel):
    broadcast_id: Optional[str] = None

# ── v0.7.0 전체 리깅 애니메이션 요청 모델 ─────────────────────

class VTubeAnimationStartRequest(BaseModel):
    emotion: Optional[str] = "calm"      # 초기 감정 상태

class VTubeEmotionRequest(BaseModel):
    emotion: str                          # calm / happy / surprised / thinking

class VTubeReactionRequest(BaseModel):
    reaction_type: str                    # chat_superchat / surprised / nod / shake

class VTubeSpeakRequest(BaseModel):
    audio_path: str
    emotion: Optional[str] = "calm"


# ------------------------------------------------------------------ 웹 UI
_LIVE_SECTION_HTML = """
    <!-- ===== 라이브 방송 섹션 ===== -->
    <div class="section" id="live-section">
      <h2>🔴 라이브 방송</h2>

      <div class="form-group">
        <label>방송 제목</label>
        <input type="text" id="live-title" value="emeth lofi 방송" placeholder="방송 제목 입력">
      </div>
      <div class="form-group">
        <label>방송 설명</label>
        <input type="text" id="live-description" value="emeth와 함께하는 잔잔한 lofi 음악" placeholder="방송 설명 입력">
      </div>
      <div class="form-group">
        <label>공개 설정</label>
        <select id="live-privacy">
          <option value="public">공개 (public)</option>
          <option value="unlisted">링크 공유 (unlisted)</option>
          <option value="private">비공개 (private)</option>
        </select>
      </div>

      <div class="button-row">
        <button class="btn btn-danger" onclick="startLive()">▶ 방송 시작</button>
        <button class="btn btn-secondary" onclick="endLive()">■ 방송 종료</button>
        <button class="btn btn-info" onclick="getLiveStatus()">📊 상태 확인</button>
      </div>

      <div id="live-result" class="result-box" style="display:none;"></div>
      <div id="live-url-box" class="url-box" style="display:none;">
        <strong>📺 방송 URL:</strong>
        <a id="live-url-link" href="#" target="_blank"></a>
      </div>
    </div>

    <style>
      .section { background:#1e1e2e; border-radius:10px; padding:24px; margin-bottom:20px; }
      .section h2 { color:#cdd6f4; margin-bottom:16px; font-size:1.2rem; }
      .form-group { margin-bottom:12px; }
      .form-group label { display:block; color:#a6adc8; font-size:0.85rem; margin-bottom:4px; }
      .form-group input, .form-group select {
        width:100%; padding:8px 12px; background:#313244; border:1px solid #45475a;
        border-radius:6px; color:#cdd6f4; font-size:0.9rem;
      }
      .button-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
      .btn { padding:10px 20px; border:none; border-radius:6px; cursor:pointer; font-size:0.9rem; font-weight:600; }
      .btn-danger { background:#f38ba8; color:#1e1e2e; }
      .btn-secondary { background:#6c7086; color:#cdd6f4; }
      .btn-info { background:#89dceb; color:#1e1e2e; }
      .result-box { margin-top:14px; padding:12px; background:#313244; border-radius:6px;
                    color:#a6e3a1; font-size:0.85rem; white-space:pre-wrap; }
      .url-box { margin-top:10px; padding:10px 14px; background:#1e1e2e; border:1px solid #45475a;
                 border-radius:6px; color:#89b4fa; font-size:0.9rem; }
      .url-box a { color:#89dceb; text-decoration:none; }
      .url-box a:hover { text-decoration:underline; }
    </style>

    <script>
      let _currentBroadcastId = null;

      async function startLive() {
        const title = document.getElementById('live-title').value.trim();
        const description = document.getElementById('live-description').value.trim();
        const privacy = document.getElementById('live-privacy').value;
        const resultBox = document.getElementById('live-result');
        const urlBox = document.getElementById('live-url-box');

        if (!title) { alert('방송 제목을 입력하세요.'); return; }

        resultBox.style.display = 'block';
        resultBox.style.color = '#f9e2af';
        resultBox.textContent = '⏳ 방송 시작 중... (인증 및 YouTube 방송 생성 중)';
        urlBox.style.display = 'none';

        try {
          const resp = await fetch('/live/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({title, description, privacy})
          });
          const data = await resp.json();
          if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));

          _currentBroadcastId = data.broadcast_id;
          resultBox.style.color = '#a6e3a1';
          resultBox.textContent =
            '✅ 방송 시작 성공!\n' +
            'broadcast_id: ' + data.broadcast_id + '\n' +
            'stream_key: ' + (data.stream_key ? data.stream_key.substring(0, 8) + '...' : 'N/A');

          document.getElementById('live-url-link').href = data.watch_url;
          document.getElementById('live-url-link').textContent = data.watch_url;
          urlBox.style.display = 'block';
        } catch(e) {
          resultBox.style.color = '#f38ba8';
          resultBox.textContent = '❌ 오류: ' + e.message;
        }
      }

      async function endLive() {
        const resultBox = document.getElementById('live-result');
        resultBox.style.display = 'block';
        resultBox.style.color = '#f9e2af';
        resultBox.textContent = '⏳ 방송 종료 중...';

        try {
          const body = _currentBroadcastId ? {broadcast_id: _currentBroadcastId} : {};
          const resp = await fetch('/live/end', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
          });
          const data = await resp.json();
          if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));

          resultBox.style.color = '#a6e3a1';
          resultBox.textContent =
            '✅ 방송 종료 완료\n' +
            'YouTube 종료: ' + (data.youtube_ended ? '성공' : '실패') + '\n' +
            'OBS 정지: ' + (data.obs_stopped ? '성공' : '실패');
          _currentBroadcastId = null;
        } catch(e) {
          resultBox.style.color = '#f38ba8';
          resultBox.textContent = '❌ 오류: ' + e.message;
        }
      }

      async function getLiveStatus() {
        const resultBox = document.getElementById('live-result');
        resultBox.style.display = 'block';
        resultBox.style.color = '#cdd6f4';
        resultBox.textContent = '⏳ 상태 조회 중...';

        try {
          const resp = await fetch('/live/status');
          const data = await resp.json();
          resultBox.textContent = JSON.stringify(data, null, 2);
        } catch(e) {
          resultBox.style.color = '#f38ba8';
          resultBox.textContent = '❌ 오류: ' + e.message;
        }
      }
    </script>
"""

_UI_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>emeth 방송 자동화 v0.6.0</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #181825; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; padding: 20px; }
    .header { text-align: center; padding: 24px 0 20px; }
    .header h1 { font-size: 1.6rem; color: #cba6f7; }
    .header p { color: #6c7086; font-size: 0.85rem; margin-top: 6px; }
    .container { max-width: 800px; margin: 0 auto; }
    .badge { display: inline-block; background: #313244; color: #89b4fa;
             padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px; }
    a.docs-link { color: #89dceb; text-decoration: none; font-size: 0.85rem; }
    a.docs-link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎭 emeth 방송 자동화 <span class="badge">v0.6.0</span></h1>
      <p>스크립트 · TTS · VTube Studio · OBS · YouTube 라이브 방송 통합 파이프라인</p>
      <p style="margin-top:8px;"><a class="docs-link" href="/docs">📄 API 문서 보기</a></p>
    </div>
""" + _LIVE_SECTION_HTML + """
  </div>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
async def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        # 기존 index.html이 있으면 라이브 섹션을 동적으로 주입
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
        if "live-section" not in html:
            # </body> 직전에 라이브 섹션 삽입
            html = html.replace("</body>", _LIVE_SECTION_HTML + "\n</body>")
        return HTMLResponse(content=html)
    return HTMLResponse(content=_UI_HTML)


# ------------------------------------------------------------------ 기본 라우트
@app.get("/status")
async def get_status():
    return JSONResponse(content=pipeline_status)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.7.0",
        "timestamp": datetime.now().isoformat(),
    }


# ------------------------------------------------------------------ 스크립트 생성
@app.post("/generate-script")
async def generate_script_endpoint(request: ScriptRequest):
    update_status(state="running", step="스크립트 생성 중", topic=request.topic, error=None)
    try:
        from generate_script import generate_script, save_script
        loop = asyncio.get_event_loop()
        script_text = await loop.run_in_executor(
            None,
            lambda: generate_script(request.topic, use_model=request.model)
        )
        file_path = save_script(request.topic, script_text)
        update_status(state="completed", step="스크립트 생성 완료", script_path=file_path)
        return {
            "success": True,
            "topic": request.topic,
            "script": script_text,
            "file_path": file_path,
            "char_count": len(script_text),
        }
    except Exception as e:
        update_status(state="error", step="스크립트 생성 실패", error=str(e))
        raise HTTPException(status_code=500, detail=f"스크립트 생성 실패: {str(e)}")


# ------------------------------------------------------------------ TTS 생성
@app.post("/generate-tts")
async def generate_tts_endpoint(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="변환할 텍스트가 비어있습니다.")
    update_status(state="running", step="TTS 변환 중", error=None)
    try:
        from tts_engine import TTSEngine
        engine = TTSEngine()
        audio_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.synthesize(text=request.text, output_filename=request.output_filename)
        )
        update_status(state="completed", step="TTS 완료", audio_path=audio_path)
        return {
            "success": True,
            "audio_path": audio_path,
            "file_size_kb": round(os.path.getsize(audio_path) / 1024, 1),
        }
    except Exception as e:
        update_status(state="error", step="TTS 실패", error=str(e))
        raise HTTPException(status_code=500, detail=f"TTS 실패: {str(e)}")


# ------------------------------------------------------------------ VTube 제어
@app.post("/control-vtube")
async def control_vtube_endpoint(request: VTubeControlRequest):
    try:
        from vtube_controller import VTubeController
        ctrl = VTubeController()
        await ctrl.connect()
        result = {"action": request.action, "success": False}
        if request.action == "expression":
            if not request.expression:
                raise HTTPException(status_code=400, detail="expression 값이 필요합니다.")
            await ctrl.set_expression(request.expression)
            result.update({"success": True, "expression": request.expression})
        elif request.action == "mouth":
            if request.mouth_value is None:
                raise HTTPException(status_code=400, detail="mouth_value 값이 필요합니다.")
            await ctrl.set_mouth_open(request.mouth_value)
            result.update({"success": True, "mouth_value": request.mouth_value})
        elif request.action == "idle":
            await ctrl.idle_animation(duration=request.idle_duration)
            result.update({"success": True, "duration": request.idle_duration})
        elif request.action == "lipsync":
            if not request.audio_path:
                raise HTTPException(status_code=400, detail="audio_path 값이 필요합니다.")
            if not os.path.exists(request.audio_path):
                raise HTTPException(status_code=404, detail=f"오디오 파일 없음: {request.audio_path}")
            await ctrl.lipsync_from_audio(request.audio_path)
            result.update({"success": True, "audio_path": request.audio_path})
        else:
            raise HTTPException(status_code=400, detail=f"알 수 없는 action: {request.action}")
        await ctrl.disconnect()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VTube 제어 실패: {str(e)}")


# ================================================================
# OBS 제어 엔드포인트 (Step 4)
# ================================================================

@app.get("/obs/status")
async def obs_status():
    """OBS 현재 상태 조회 (녹화/스트리밍/씬 정보)."""
    ctrl = _get_obs_controller()
    status = ctrl.get_status()
    return JSONResponse(content=status)


@app.post("/obs/start-recording")
async def obs_start_recording():
    """OBS 녹화 시작."""
    ctrl = _get_obs_controller()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, ctrl.start_recording)
    if not ok:
        raise HTTPException(status_code=500, detail="OBS 녹화 시작 실패")
    return {"success": True, "message": "OBS 녹화가 시작되었습니다."}


@app.post("/obs/stop-recording")
async def obs_stop_recording():
    """OBS 녹화 정지. 녹화 파일 경로를 반환합니다."""
    ctrl = _get_obs_controller()
    loop = asyncio.get_event_loop()
    output_path = await loop.run_in_executor(None, ctrl.stop_recording)
    return {
        "success": True,
        "message": "OBS 녹화가 정지되었습니다.",
        "recording_path": output_path,
    }


@app.post("/obs/start-stream")
async def obs_start_stream():
    """OBS 스트리밍 시작."""
    ctrl = _get_obs_controller()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, ctrl.start_streaming)
    if not ok:
        raise HTTPException(status_code=500, detail="OBS 스트리밍 시작 실패")
    return {"success": True, "message": "OBS 스트리밍이 시작되었습니다."}


@app.post("/obs/stop-stream")
async def obs_stop_stream():
    """OBS 스트리밍 정지."""
    ctrl = _get_obs_controller()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, ctrl.stop_streaming)
    if not ok:
        raise HTTPException(status_code=500, detail="OBS 스트리밍 정지 실패")
    return {"success": True, "message": "OBS 스트리밍이 정지되었습니다."}


@app.post("/obs/switch-scene")
async def obs_switch_scene(request: SwitchSceneRequest):
    """OBS 씬 전환."""
    if not request.scene_name.strip():
        raise HTTPException(status_code=400, detail="scene_name이 비어있습니다.")
    ctrl = _get_obs_controller()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, lambda: ctrl.switch_scene(request.scene_name))
    if not ok:
        raise HTTPException(status_code=500, detail=f"씬 전환 실패: {request.scene_name}")
    return {
        "success": True,
        "message": f"씬이 '{request.scene_name}'으로 전환되었습니다.",
        "scene_name": request.scene_name,
    }


# ------------------------------------------------------------------ 통합 파이프라인
@app.post("/run-pipeline")
async def run_pipeline_endpoint(request: PipelineRequest):
    update_status(
        state="running",
        step="파이프라인 시작",
        topic=request.topic,
        script_path=None,
        audio_path=None,
        vtube_active=request.use_vtube,
        obs_recording_path=None,
        error=None,
    )

    obs_ctrl = None

    try:
        from generate_script import generate_script, save_script
        from tts_engine import TTSEngine
        loop = asyncio.get_event_loop()

        if request.use_obs:
            update_status(step="[OBS] 녹화 시작 중")
            try:
                obs_ctrl = _get_obs_controller()
                from config_obs import AUTO_SWITCH_SCENE, SCENE_MAIN
                if AUTO_SWITCH_SCENE:
                    await loop.run_in_executor(None, lambda: obs_ctrl.switch_scene(SCENE_MAIN))
                await loop.run_in_executor(None, obs_ctrl.start_recording)
                update_status(step="[OBS] 녹화 시작 완료")
            except Exception as e:
                update_status(step=f"[OBS] 녹화 시작 실패 (무시하고 계속): {e}")

        update_status(step="[1/3] 스크립트 생성 중")
        script_text = await loop.run_in_executor(
            None,
            lambda: generate_script(request.topic, use_model=request.model)
        )
        script_path = save_script(request.topic, script_text)
        update_status(step="[1/3] 스크립트 생성 완료", script_path=script_path)

        update_status(step="[2/3] TTS 변환 중")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"emeth_pipeline_{timestamp}.mp3"
        tts_engine = TTSEngine()
        audio_path = await loop.run_in_executor(
            None,
            lambda: tts_engine.synthesize(text=script_text, output_filename=audio_filename)
        )
        update_status(step="[2/3] TTS 완료", audio_path=audio_path)

        vtube_result = None
        if request.use_vtube:
            update_status(step="[3/3] VTube Studio 제어 중")
            try:
                from vtube_controller import VTubeController
                vtube = VTubeController()
                await vtube.connect()
                await vtube.set_expression("speaking")
                await vtube.lipsync_from_audio(audio_path)
                await vtube.set_expression("default")
                await vtube.disconnect()
                vtube_result = {"success": True, "audio_path": audio_path}
                update_status(step="[3/3] VTube Studio 완료")
            except Exception as e:
                vtube_result = {"success": False, "error": str(e)}
                update_status(step="[3/3] VTube Studio 실패 (무시하고 계속)")
        else:
            update_status(step="[3/3] VTube Studio 건너뜀 (use_vtube=False)")

        obs_result = None
        if request.use_obs and obs_ctrl is not None:
            update_status(step="[OBS] 녹화 정지 중")
            try:
                rec_path = await loop.run_in_executor(None, obs_ctrl.stop_recording)
                obs_result = {"success": True, "recording_path": rec_path}
                update_status(step="[OBS] 녹화 완료", obs_recording_path=rec_path)
            except Exception as e:
                obs_result = {"success": False, "error": str(e)}
                update_status(step=f"[OBS] 녹화 정지 실패: {e}")

        update_status(state="completed", step="파이프라인 완료")

        return {
            "success": True,
            "topic": request.topic,
            "script": script_text,
            "script_path": script_path,
            "audio_path": audio_path,
            "char_count": len(script_text),
            "file_size_kb": round(os.path.getsize(audio_path) / 1024, 1),
            "vtube": vtube_result,
            "obs": obs_result,
        }

    except Exception as e:
        if request.use_obs and obs_ctrl is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, obs_ctrl.stop_recording)
            except Exception:
                pass
        update_status(state="error", step="파이프라인 오류", error=str(e))
        raise HTTPException(status_code=500, detail=f"파이프라인 실패: {str(e)}")


# ------------------------------------------------------------------ 채팅 시스템
@app.post("/chat/start")
async def chat_start(request: ChatStartRequest, background_tasks: BackgroundTasks):
    global _chat_pipeline, _chat_task
    if chat_status["running"]:
        raise HTTPException(status_code=400, detail="채팅 시스템이 이미 실행 중입니다.")
    if not request.video_id.strip():
        raise HTTPException(status_code=400, detail="video_id가 비어있습니다.")
    try:
        from chat_pipeline import ChatPipeline
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"step6_chat 모듈 없음: {e}")
    _chat_pipeline = ChatPipeline()
    chat_status.update({
        "running": True,
        "video_id": request.video_id,
        "live_chat_id": None,
        "started_at": datetime.now().isoformat(),
        "stats": {"received": 0, "responded": 0, "skipped": 0, "errors": 0},
    })
    async def _run_chat():
        global _chat_task
        try:
            await _chat_pipeline.start(request.video_id)
        except Exception as e:
            chat_status["running"] = False
            chat_status["error"] = str(e)
        finally:
            chat_status["running"] = False
    _chat_task = asyncio.create_task(_run_chat())
    return {
        "success": True,
        "message": f"채팅 시스템이 시작되었습니다. (video_id: {request.video_id})",
        "video_id": request.video_id,
    }


@app.post("/chat/stop")
async def chat_stop():
    global _chat_pipeline, _chat_task
    if not chat_status["running"]:
        return {"success": True, "message": "채팅 시스템이 이미 정지 상태입니다."}
    if _chat_pipeline:
        _chat_pipeline.stop()
    if _chat_task and not _chat_task.done():
        _chat_task.cancel()
        try:
            await asyncio.wait_for(_chat_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    chat_status["running"] = False
    return {
        "success": True,
        "message": "채팅 시스템이 정지되었습니다.",
        "stats": _chat_pipeline.stats if _chat_pipeline else {},
    }


@app.get("/chat/status")
async def chat_status_endpoint():
    status = dict(chat_status)
    if _chat_pipeline and chat_status["running"]:
        status["stats"] = _chat_pipeline.stats
        status["live_chat_id"] = _chat_pipeline._live_chat_id
    return JSONResponse(content=status)


# ================================================================
# 라이브 방송 엔드포인트 (Step 5 - v0.6.0 신규 추가)
# ================================================================

@app.post("/live/start")
async def live_start(request: LiveStartRequest):
    """
    YouTube 라이브 방송을 시작합니다.

    흐름: YouTube 방송 생성 → 스트림 키 획득 → OBS 스트림 키 설정
          → OBS 스트리밍 시작 → 방송 live 전환

    Returns:
        broadcast_id, stream_key, watch_url 등
    """
    title = (request.title or "emeth lofi 방송").strip()
    description = (request.description or "").strip()
    privacy = (request.privacy or "public").strip()

    if privacy not in ("public", "unlisted", "private"):
        raise HTTPException(status_code=400, detail="privacy는 public / unlisted / private 중 하나여야 합니다.")

    pipeline = _get_live_pipeline()
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.start_live(title, description, privacy),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"라이브 방송 시작 실패: {str(e)}")

    # 전역 상태 업데이트
    live_broadcast_state.update({
        "broadcast_id": result["broadcast_id"],
        "watch_url": result["watch_url"],
        "title": title,
        "started_at": datetime.now().isoformat(),
        "youtube_status": "live",
        "obs_streaming": True,
    })

    return {
        "success": True,
        "broadcast_id": result["broadcast_id"],
        "stream_id": result["stream_id"],
        "rtmp_url": result["rtmp_url"],
        "stream_key": result["stream_key"],
        "watch_url": result["watch_url"],
        "title": title,
        "privacy": privacy,
        "started_at": live_broadcast_state["started_at"],
    }


@app.post("/live/end")
async def live_end(request: LiveEndRequest):
    """
    라이브 방송을 종료합니다.

    - broadcast_id를 지정하지 않으면 마지막으로 시작한 방송을 종료합니다.
    """
    bid = request.broadcast_id or live_broadcast_state.get("broadcast_id")
    if not bid:
        raise HTTPException(
            status_code=400,
            detail="broadcast_id가 없습니다. 요청 본문에 broadcast_id를 포함하거나 먼저 방송을 시작하세요.",
        )

    pipeline = _get_live_pipeline()
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.end_live(bid),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"라이브 방송 종료 실패: {str(e)}")

    # 전역 상태 초기화
    live_broadcast_state.update({
        "broadcast_id": None,
        "watch_url": None,
        "title": None,
        "started_at": None,
        "youtube_status": "complete",
        "obs_streaming": False,
    })

    return {
        "success": True,
        "broadcast_id": bid,
        "youtube_ended": result.get("youtube_ended", False),
        "obs_stopped": result.get("obs_stopped", False),
        "ended_at": datetime.now().isoformat(),
    }


@app.get("/live/status")
async def live_status():
    """
    현재 라이브 방송 상태를 조회합니다.

    YouTube 방송 lifeCycleStatus와 OBS 스트리밍 상태를 함께 반환합니다.
    """
    bid = live_broadcast_state.get("broadcast_id")
    pipeline = _get_live_pipeline()
    loop = asyncio.get_event_loop()

    try:
        live_info = await loop.run_in_executor(
            None,
            lambda: pipeline.get_status(bid),
        )
    except Exception as e:
        live_info = {
            "broadcast_id": bid,
            "youtube_status": None,
            "obs_streaming": False,
            "error": str(e),
        }

    return JSONResponse(content={
        **live_broadcast_state,
        "youtube_status": live_info.get("youtube_status"),
        "obs_streaming": live_info.get("obs_streaming", False),
    })


# ================================================================
# 전체 리깅 애니메이션 엔드포인트 (v0.7.0 신규)
# ================================================================

def _get_vtube_full_ctrl():
    """VTubeController 싱글턴 반환 (연결 안 된 경우 예외)"""
    global _vtube_full_ctrl
    if _vtube_full_ctrl is None:
        try:
            from vtube_controller import VTubeController
            _vtube_full_ctrl = VTubeController()
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"vtube_controller 모듈 없음: {e}")
    return _vtube_full_ctrl


@app.post(
    "/vtube/animation/start",
    summary="전체 리깅 애니메이션 시작",
    description=(
        "호흡 · 눈 깜빡임 · 머리 흔들림 · 머리카락 물리 · 감정 레이어를 "
        "모두 asyncio task로 동시에 시작합니다. "
        "emotion 파라미터로 초기 감정 상태를 지정할 수 있습니다."
    ),
)
async def vtube_animation_start(request: VTubeAnimationStartRequest):
    global _vtube_full_ctrl
    ctrl = _get_vtube_full_ctrl()
    try:
        if not ctrl._connected:
            await ctrl.connect()
        await ctrl.start_full_animation()
        if request.emotion and request.emotion != "calm":
            ctrl.set_emotion_layer(request.emotion)
        update_status(vtube_active=True)
        return {
            "success":  True,
            "message":  "전체 리깅 애니메이션 시스템이 시작되었습니다.",
            "emotion":  request.emotion or "calm",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"애니메이션 시작 실패: {e}")


@app.post(
    "/vtube/animation/stop",
    summary="전체 리깅 애니메이션 정지",
    description="start로 시작된 모든 애니메이션 레이어를 정지합니다.",
)
async def vtube_animation_stop():
    global _vtube_full_ctrl
    ctrl = _get_vtube_full_ctrl()
    try:
        await ctrl.stop_full_animation()
        update_status(vtube_active=False)
        return {"success": True, "message": "전체 리깅 애니메이션이 정지되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"애니메이션 정지 실패: {e}")


@app.post(
    "/vtube/emotion",
    summary="감정 상태 변경",
    description=(
        "0.8초 보간으로 자연스럽게 감정 표정을 전환합니다. "
        "지원 감정: calm / happy / surprised / thinking. "
        "전체 애니메이션 시스템(/vtube/animation/start) 이 활성화된 상태여야 합니다."
    ),
)
async def vtube_emotion(request: VTubeEmotionRequest):
    VALID = {"calm", "happy", "surprised", "thinking"}
    if request.emotion not in VALID:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 감정입니다. 가능한 값: {sorted(VALID)}",
        )
    ctrl = _get_vtube_full_ctrl()
    try:
        if not ctrl._connected:
            raise HTTPException(status_code=503, detail="VTube Studio에 연결되지 않았습니다.")
        ctrl.set_emotion_layer(request.emotion)
        return {"success": True, "emotion": request.emotion}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감정 변경 실패: {e}")


@app.post(
    "/vtube/reaction",
    summary="즉각 반응 애니메이션",
    description=(
        "최우선 레이어(priority=40)로 즉각 반응 애니메이션을 트리거합니다. "
        "지원 타입: chat_superchat / surprised / nod / shake. "
        "실행 중이던 반응이 있으면 취소하고 새 반응으로 교체합니다."
    ),
)
async def vtube_reaction(request: VTubeReactionRequest):
    VALID = {"chat_superchat", "surprised", "nod", "shake"}
    if request.reaction_type not in VALID:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 reaction_type입니다. 가능한 값: {sorted(VALID)}",
        )
    ctrl = _get_vtube_full_ctrl()
    try:
        if not ctrl._connected:
            raise HTTPException(status_code=503, detail="VTube Studio에 연결되지 않았습니다.")
        await ctrl.trigger_reaction(request.reaction_type)
        return {"success": True, "reaction_type": request.reaction_type}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"반응 애니메이션 실패: {e}")


@app.post(
    "/vtube/speak",
    summary="말하기 + 감정 동시 처리",
    description=(
        "지정된 오디오 파일로 립싱크를 실행하면서 감정 표정도 함께 변경합니다. "
        "전체 애니메이션 시스템이 비활성 상태면 자동으로 시작합니다."
    ),
)
async def vtube_speak(request: VTubeSpeakRequest):
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail=f"오디오 파일 없음: {request.audio_path}")
    ctrl = _get_vtube_full_ctrl()
    try:
        if not ctrl._connected:
            await ctrl.connect()
        await ctrl.speak(request.audio_path, emotion=request.emotion or "calm")
        return {
            "success":    True,
            "audio_path": request.audio_path,
            "emotion":    request.emotion or "calm",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"speak 실패: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
