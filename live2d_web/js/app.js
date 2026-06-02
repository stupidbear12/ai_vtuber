// app.js - PixiJS + pixi-live2d-display 메인 애플리케이션

// 기본 모델 경로 (index.html?model=path 쿼리로 변경 가능)
const DEFAULT_MODEL = 'models/Haru/Haru.model3.json';

let pixiApp   = null;
let l2dModel  = null;
let animSys   = null;
let wsClient  = null;

// ─── 로그 패널 ────────────────────────────────────────────────────
const $log = document.getElementById('log');
function addLog(msg) {
  const d = document.createElement('div');
  d.textContent = `[${new Date().toLocaleTimeString('ko-KR')}] ${msg}`;
  $log.appendChild(d);
  $log.scrollTop = $log.scrollHeight;
  if ($log.children.length > 60) $log.removeChild($log.firstChild);
}

// ─── 상태 표시 ────────────────────────────────────────────────────
function setWsStatus(s) {
  const dot   = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  const MAP = {
    connected:    ['connected', '서버 연결됨'],
    disconnected: ['',          '서버 미연결'],
    error:        ['error',     '연결 오류'],
  };
  const [cls, text] = MAP[s] ?? ['', '알 수 없음'];
  dot.className = 'dot ' + cls;
  label.textContent = text;
}

function setModelStatus(ok, text) {
  document.getElementById('model-dot').className = 'dot' + (ok ? ' connected' : ' error');
  document.getElementById('model-label').textContent = text;
}

// ─── PixiJS 초기화 (v8: Application.init()이 비동기) ─────────────
async function initPixi() {
  const wrap = document.getElementById('canvas-wrap');
  pixiApp = new PIXI.Application();
  await pixiApp.init({
    canvas:      document.getElementById('live2d-canvas'),
    width:       wrap.clientWidth,
    height:      wrap.clientHeight,
    background:  0x1a1a2e,   // v8: backgroundColor → background
    antialias:   true,
    autoDensity: true,
    resolution:  window.devicePixelRatio || 1,
  });
  window.addEventListener('resize', () => {
    pixiApp.renderer.resize(wrap.clientWidth, wrap.clientHeight);
    if (l2dModel) centerModel();
  });
}

// ─── 모델 배치 ───────────────────────────────────────────────────
// @naari3/pixi-live2d-display v1.x는 model.x/y를 CSS px가 아닌
// 물리(physical) px로 해석한다. DPR을 곱해서 보정한다.
function centerModel() {
  const { width: sw, height: sh } = pixiApp.screen;
  const dpr = window.devicePixelRatio || 1;
  const { originalWidth: mw, originalHeight: mh } = l2dModel.internalModel;
  const scale = Math.min(sw / mw, sh / mh) * 0.90;

  l2dModel.scale.set(scale);
  l2dModel.anchor.set(0, 0);
  l2dModel.pivot.set(0, 0);

  // 캔버스를 수평 중앙, 하단 정렬 (발이 화면 아래쪽에 위치)
  l2dModel.x = dpr * (sw / 2 - (mw * scale) / 2);
  l2dModel.y = dpr * (sh - mh * scale);
}

// ─── 모델 로드 ───────────────────────────────────────────────────
async function loadModel(modelPath) {
  // 이전 모델 제거
  if (l2dModel) { pixiApp.stage.removeChild(l2dModel); l2dModel.destroy(); l2dModel = null; }

  setModelStatus(false, '로딩 중…');
  addLog(`모델 로드: ${modelPath}`);
  try {
    l2dModel = await PIXI.live2d.Live2DModel.from(modelPath, {
      autoInteract: false,  // 마우스 추적 비활성 (코드로 제어)
      // autoUpdate는 기본값(true) 유지 — GL 컨텍스트 초기화에 필요
    });
    pixiApp.stage.addChild(l2dModel);

    // PixiJS v8: setRenderer()로 GL 컨텍스트 명시적 연결 (필수)
    l2dModel.setRenderer(pixiApp.renderer);

    centerModel();

    // 내장 모션/표정은 비활성화하되 기본 업데이트 루프는 유지
    l2dModel.internalModel.motionManager.stopAllMotions?.();

    setModelStatus(true, modelPath.split('/').slice(-2, -1)[0] || '모델');
    addLog('모델 로드 성공');

    // 파라미터 주입 루프: PIXI ticker에 추가 (renderTick은 파라미터만 씀)
    pixiApp.ticker.remove(renderTick);
    pixiApp.ticker.add(renderTick);
  } catch (e) {
    setModelStatus(false, '로드 실패');
    addLog(`오류: ${e.message}`);
    console.error('[Live2D]', e);
  }
}

// ─── 파라미터 주입 루프 ──────────────────────────────────────────
// autoUpdate=true이므로 pixi-live2d-display가 물리/렌더를 담당.
// 이 함수는 애니메이션 파라미터만 coreModel에 주입한다.
function renderTick() {
  if (!l2dModel || !animSys) return;

  const merged    = animSys.tick();
  const coreModel = l2dModel.internalModel.coreModel;

  for (const [id, val] of Object.entries(merged)) {
    try { coreModel.setParameterValueById(id, val, 1.0); } catch (_) {}
  }
}

// ─── 슬라이더 UI 연결 ────────────────────────────────────────────
function initSliders() {
  const rows = [
    { sid: 's-angle-x', vid: 'v-angle-x', param: PARAMS.ANGLE_X },
    { sid: 's-angle-y', vid: 'v-angle-y', param: PARAMS.ANGLE_Y },
    { sid: 's-angle-z', vid: 'v-angle-z', param: PARAMS.ANGLE_Z },
    { sid: 's-eye-l',   vid: 'v-eye-l',   param: PARAMS.EYE_L },
    { sid: 's-eye-r',   vid: 'v-eye-r',   param: PARAMS.EYE_R },
    { sid: 's-mouth',   vid: 'v-mouth',   param: PARAMS.MOUTH_OPEN },
    { sid: 's-smile',   vid: 'v-smile',   param: PARAMS.MOUTH_FORM },
    { sid: 's-breath',  vid: 'v-breath',  param: PARAMS.BREATH },
  ];

  const manual = {};
  for (const { sid, vid, param } of rows) {
    const el = document.getElementById(sid);
    const vl = document.getElementById(vid);
    if (!el) continue;
    el.addEventListener('input', () => {
      const v = parseFloat(el.value);
      vl.textContent = v.toFixed(2);
      manual[param] = v;
      animSys.setManualParams({ ...manual });
    });
  }
}

// ─── 버튼 UI 연결 ────────────────────────────────────────────────
function initButtons() {
  document.getElementById('btn-idle').addEventListener('click', () => {
    animSys.startIdle(); addLog('Idle 시작');
  });
  document.getElementById('btn-stop').addEventListener('click', () => {
    animSys.stopIdle(); addLog('Idle 정지');
  });

  document.querySelectorAll('[data-emotion]').forEach(btn => {
    btn.addEventListener('click', () => {
      const em = btn.dataset.emotion;
      animSys.setEmotion(em);
      addLog(`감정: ${em}`);
    });
  });

  document.querySelectorAll('[data-reaction]').forEach(btn => {
    btn.addEventListener('click', () => {
      const rx = btn.dataset.reaction;
      animSys.triggerReaction(rx);
      addLog(`반응: ${rx}`);
    });
  });

  // 모델 경로 변경
  const pathInput = document.getElementById('model-path');
  document.getElementById('btn-load').addEventListener('click', () => {
    const p = pathInput.value.trim() || DEFAULT_MODEL;
    loadModel(p);
  });
}

// ─── 진입점 ──────────────────────────────────────────────────────
async function main() {
  animSys  = new AnimSystem();
  wsClient = new Live2DWSClient(animSys, setWsStatus, addLog);

  await initPixi();
  initSliders();
  initButtons();

  // URL 쿼리 파라미터로 모델 경로 지정 가능: ?model=models/Hiyori/...
  const params    = new URLSearchParams(location.search);
  const modelPath = params.get('model') || DEFAULT_MODEL;
  document.getElementById('model-path').value = modelPath;

  await loadModel(modelPath);

  // 모델 로드 성공 시 Idle 자동 시작
  if (l2dModel) { animSys.startIdle(); addLog('Idle 자동 시작'); }

  wsClient.connect();
}

main().catch(console.error);
