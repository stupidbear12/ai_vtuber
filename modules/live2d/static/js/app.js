// app.js - mao_pro PixiJS + pixi-live2d-display 메인 앱

const DEFAULT_MODEL = 'models/mao_pro/runtime/mao_pro.model3.json';

// model_dict.json 기반 감정→표정 매핑
const EMOTION_TO_EXPR = {
  neutral: 'exp_01', calm: 'exp_01', default: 'exp_01',
  content: 'exp_02', gentle: 'exp_02',
  sad: 'exp_03',     sadness: 'exp_03', fear: 'exp_03',
  happy: 'exp_04',   joy: 'exp_04', excited: 'exp_04',
  annoyed: 'exp_05', frustrated: 'exp_05', thinking: 'exp_05',
  embarrassed: 'exp_06', shy: 'exp_06', flustered: 'exp_06',
  surprise: 'exp_07', surprised: 'exp_07', shocked: 'exp_07', worried: 'exp_07',
  angry: 'exp_08',   anger: 'exp_08', disgust: 'exp_08',
  smirk: 'exp_04',
};

// 감정별 모션 트리거 매핑 (group='', index)
const EMOTION_TO_MOTION = {
  happy:     { group: '', index: 0 },  // mtn_02
  excited:   { group: '', index: 3 },  // special_01
  surprised: { group: '', index: 4 },  // special_02
  angry:     { group: '', index: 2 },  // mtn_04
  shy:       { group: '', index: 1 },  // mtn_03
};

// 히트 영역 → 재생할 모션 (group, index)
const HIT_MOTIONS = {
  HitAreaHead: { group: '', index: 1 },
  HitAreaBody: { group: '', index: 2 },
};

let pixiApp  = null;
let l2dModel = null;
let animSys  = null;
let wsClient = null;
let motionRestoreTimer = null;

const DEFAULT_MOTION_MS = 5500;

// 투명 모드 (OBS 브라우저 소스)
const urlParams  = new URLSearchParams(location.search);
const isTransparent = urlParams.get('transparent') === '1';
const isChromakey = urlParams.get('chromakey') === '1';

// ── 로그 ─────────────────────────────────────────────────────────
const $log = document.getElementById('log');
function addLog(msg) {
  if (!$log) return;
  const d = document.createElement('div');
  d.textContent = `[${new Date().toLocaleTimeString('ko-KR')}] ${msg}`;
  $log.appendChild(d);
  $log.scrollTop = $log.scrollHeight;
  if ($log.children.length > 60) $log.removeChild($log.firstChild);
}

// ── 상태 표시 ────────────────────────────────────────────────────
function setWsStatus(s) {
  const dot   = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  if (!dot || !label) return;
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
  const dot = document.getElementById('model-dot');
  const lbl = document.getElementById('model-label');
  if (dot) dot.className = 'dot' + (ok ? ' connected' : ' error');
  if (lbl) lbl.textContent = text;
}

// ── PixiJS 초기화 ────────────────────────────────────────────────
async function initPixi() {
  if (isChromakey) {
    document.body.classList.add('chromakey');
    document.documentElement.classList.add('chromakey');
  } else if (isTransparent) {
    document.body.classList.add('transparent');
    document.documentElement.classList.add('transparent');
  }

  const wrap = document.getElementById('canvas-wrap');
  const canvas = document.getElementById('live2d-canvas');

  // OBS CEF에서는 DPR이 1이므로 해상도 보정 불필요
  const dpr = (isTransparent || isChromakey) ? 1 : (window.devicePixelRatio || 1);

  const pixiOpts = {
    canvas:          canvas,
    width:           wrap.clientWidth,
    height:          wrap.clientHeight,
    antialias:       true,
    autoDensity:     !(isTransparent || isChromakey),
    resolution:      dpr,
    preference:      'webgl',
  };

  if (isTransparent || isChromakey) {
    pixiOpts.backgroundAlpha = isChromakey ? 1 : 0;
    pixiOpts.background = isChromakey ? 0x00FF00 : 0x000000;
    pixiOpts.clearBeforeRender = true;
  } else {
    pixiOpts.background = 0x1a1a2e;
    pixiOpts.backgroundAlpha = 1;
  }

  pixiApp = new PIXI.Application();
  await pixiApp.init(pixiOpts);

  // ── OBS 크로마키: gl.clearColor 강제 오버라이드 ──────────────
  // PixiJS v8 또는 Live2D SDK가 clearColor를 (1,1,1,1) 흰색으로 덮어씀.
  // clearColor 호출 자체를 가로채서 항상 green으로 강제.
  if (isChromakey) {
    const gl = pixiApp.renderer.gl;
    const origClearColor = gl.clearColor.bind(gl);
    gl.clearColor = function(r, g, b, a) {
      // 모든 clearColor 호출을 green으로 강제
      origClearColor(0, 1, 0, 1);
    };
    console.log('[Live2D] Chromakey: clearColor overridden to green');
    canvas.style.cssText += ';background:#00FF00!important;';
  } else if (isTransparent) {
    canvas.style.cssText += ';background:transparent!important;';
  }

  window.addEventListener('resize', () => {
    pixiApp.renderer.resize(wrap.clientWidth, wrap.clientHeight);
    if (l2dModel) centerModel();
  });
}

// ── 모델 배치 ────────────────────────────────────────────────────
function centerModel() {
  const { width: sw, height: sh } = pixiApp.screen;
  // 투명 모드(OBS)에서는 DPR 보정 불필요 (resolution=1)
  const dpr = (isTransparent || isChromakey) ? 1 : (window.devicePixelRatio || 1);
  const { originalWidth: mw, originalHeight: mh } = l2dModel.internalModel;
  const scale = Math.min(sw / mw, sh / mh) * 0.90;

  l2dModel.scale.set(scale);
  l2dModel.anchor.set(0, 0);
  l2dModel.pivot.set(0, 0);
  l2dModel.x = dpr * (sw / 2 - (mw * scale) / 2);
  l2dModel.y = dpr * (sh - mh * scale);
}

// ── 모델 로드 ────────────────────────────────────────────────────
async function loadModel(modelPath) {
  if (l2dModel) { pixiApp.stage.removeChild(l2dModel); l2dModel.destroy(); l2dModel = null; }

  setModelStatus(false, '로딩 중…');
  addLog(`모델 로드: ${modelPath}`);
  try {
    l2dModel = await PIXI.live2d.Live2DModel.from(modelPath, {
      autoInteract: false,
    });
    pixiApp.stage.addChild(l2dModel);
    l2dModel.setRenderer(pixiApp.renderer);
    centerModel();

    // 내장 모션 정지 — AnimSystem이 파라미터 제어
    l2dModel.internalModel.motionManager.stopAllMotions?.();

    // 파라미터 인덱스 맵 구축 + draw 후크 설치
    buildParamIndexMap();
    hookDrawForParams();

    // 히트 영역 클릭 → 모션 재생
    l2dModel.on('hit', (hitAreas) => {
      for (const area of hitAreas) {
        const m = HIT_MOTIONS[area];
        if (m) { playMotion(m.group, m.index); addLog(`히트: ${area}`); break; }
      }
    });

    setModelStatus(true, modelPath.split('/').slice(-2, -1)[0] || '모델');
    addLog('모델 로드 성공');
  } catch (e) {
    setModelStatus(false, '로드 실패');
    addLog(`오류: ${e.message}`);
    console.error('[Live2D]', e);
  }
}

// ── 파라미터 인덱스 맵 ──────────────────────────────────────────
// Cubism 5 SDK 버그: getParameterIndex()가 잘못된 인덱스를 반환함.
// getParameterId(i)로 올바른 ID→인덱스 매핑을 직접 구축한다.
let _paramIndexMap = {};

function buildParamIndexMap() {
  const core = l2dModel.internalModel.coreModel;
  const count = core.getParameterCount();
  _paramIndexMap = {};
  for (let i = 0; i < count; i++) {
    try {
      const idObj = core.getParameterId(i);
      const idStr = idObj?._id?.s;
      if (idStr) _paramIndexMap[idStr] = i;
    } catch (_) {}
  }
  addLog(`파라미터 맵: ${Object.keys(_paramIndexMap).length}개`);
}

// ── 파라미터 주입 (draw 후크) ───────────────────────────────────
// Pixi v8에서 ticker 콜백이 정상 실행되지 않는 문제가 있어
// internalModel.draw()를 래핑하여 draw 직전에 파라미터를 주입한다.
function hookDrawForParams() {
  const im = l2dModel.internalModel;
  const origDraw = im.draw.bind(im);

  im.draw = function (gl) {
    if (animSys && _paramIndexMap) {
      const merged = animSys.tick();
      const core = this.coreModel;
      for (const [id, val] of Object.entries(merged)) {
        const idx = _paramIndexMap[id];
        if (idx !== undefined) {
          core.setParameterValueByIndex(idx, val);
        }
      }
      core.update();
    }
    return origDraw(gl);
  };
}

// 레거시 호환 — 외부에서 renderTick 참조 시 오류 방지
function renderTick() {}

// ── Expression 제어 (pixi-live2d-display 내장) ───────────────────
function setExpression(nameOrIndex) {
  if (!l2dModel) return;
  try {
    l2dModel.expression(nameOrIndex);
    addLog(`표정: ${nameOrIndex}`);
  } catch (e) {
    console.warn('[Expression]', e);
  }
}

// 감정 이름 → 표정 변환 후 적용 (+ 매핑된 모션 동시 재생)
function setEmotion(emotionName) {
  const expr = EMOTION_TO_EXPR[emotionName] ?? 'exp_01';
  setExpression(expr);

  const motion = EMOTION_TO_MOTION[emotionName];
  if (motion) playMotion(motion.group, motion.index);
}

// ── Motion 제어 (pixi-live2d-display 내장) ───────────────────────
function restoreAfterMotion() {
  if (motionRestoreTimer) {
    clearTimeout(motionRestoreTimer);
    motionRestoreTimer = null;
  }
  if (!l2dModel) return;
  l2dModel.internalModel.motionManager.stopAllMotions?.();
  animSys?.startIdle();
}

/**
 * @param {string} group
 * @param {number} index
 * @param {{ duration?: number, restoreIdle?: boolean }} [options]
 */
function playMotion(group, index, options = {}) {
  if (!l2dModel) return;
  const priority = PIXI.live2d.MotionPriority?.FORCE ?? 3;
  const durationMs = options.duration ?? DEFAULT_MOTION_MS;
  const restoreIdle = options.restoreIdle !== false;

  if (motionRestoreTimer) {
    clearTimeout(motionRestoreTimer);
    motionRestoreTimer = null;
  }

  // AnimSystem 파라미터가 모션 파라미터를 덮어쓰지 않도록 idle 중지
  animSys?.stopIdle();
  l2dModel.internalModel.motionManager.stopAllMotions?.();

  try {
    l2dModel.motion(group, index, priority);
    const label = group || '(default)';
    addLog(`모션: ${label}[${index}]`);
    if (restoreIdle && durationMs > 0) {
      motionRestoreTimer = setTimeout(restoreAfterMotion, durationMs);
    }
  } catch (e) {
    console.warn('[Motion]', e);
    if (restoreIdle) animSys?.startIdle();
  }
}

// ── 슬라이더 UI ─────────────────────────────────────────────────
function initSliders() {
  const rows = [
    { sid: 's-angle-x',  vid: 'v-angle-x',  param: PARAMS.ANGLE_X  },
    { sid: 's-angle-y',  vid: 'v-angle-y',  param: PARAMS.ANGLE_Y  },
    { sid: 's-angle-z',  vid: 'v-angle-z',  param: PARAMS.ANGLE_Z  },
    { sid: 's-eye-l',    vid: 'v-eye-l',    param: PARAMS.EYE_L    },
    { sid: 's-eye-r',    vid: 'v-eye-r',    param: PARAMS.EYE_R    },
    { sid: 's-mouth-a',  vid: 'v-mouth-a',  param: PARAMS.MOUTH_A  },
    { sid: 's-cheek',    vid: 'v-cheek',    param: PARAMS.CHEEK    },
    { sid: 's-breath',   vid: 'v-breath',   param: PARAMS.BREATH   },
  ];
  const manual = {};
  for (const { sid, vid, param } of rows) {
    const el = document.getElementById(sid);
    const vl = document.getElementById(vid);
    if (!el) continue;
    el.addEventListener('input', () => {
      const v = parseFloat(el.value);
      if (vl) vl.textContent = v.toFixed(2);
      manual[param] = v;
      animSys.setManualParams({ ...manual });
    });
  }
}

// ── 버튼 UI ──────────────────────────────────────────────────────
function initButtons() {
  document.getElementById('btn-idle')?.addEventListener('click', () => {
    animSys.startIdle(); addLog('Idle 시작');
  });
  document.getElementById('btn-stop')?.addEventListener('click', () => {
    animSys.stopIdle(); addLog('Idle 정지');
  });

  // Expression 버튼
  document.querySelectorAll('[data-expr]').forEach(btn => {
    btn.addEventListener('click', () => setExpression(btn.dataset.expr));
  });

  // Motion 버튼
  document.querySelectorAll('[data-motion-group]').forEach(btn => {
    btn.addEventListener('click', () => {
      const g = btn.dataset.motionGroup;
      const i = parseInt(btn.dataset.motionIndex ?? '0', 10);
      playMotion(g, i);
    });
  });

  // 반응 버튼
  document.querySelectorAll('[data-reaction]').forEach(btn => {
    btn.addEventListener('click', () => {
      const rx = btn.dataset.reaction;
      animSys.triggerReaction(rx);
      addLog(`반응: ${rx}`);
    });
  });

  // 모델 경로 변경
  const pathInput = document.getElementById('model-path');
  document.getElementById('btn-load')?.addEventListener('click', () => {
    const p = pathInput?.value.trim() || DEFAULT_MODEL;
    loadModel(p);
  });
}

// ── 진입점 ───────────────────────────────────────────────────────
async function main() {
  animSys  = new AnimSystem();
  wsClient = new Live2DWSClient(animSys, setWsStatus, addLog, setExpression, playMotion);

  await initPixi();
  initSliders();
  initButtons();

  const modelPath = urlParams.get('model') || DEFAULT_MODEL;
  if (document.getElementById('model-path')) {
    document.getElementById('model-path').value = modelPath;
  }

  await loadModel(modelPath);
  if (l2dModel) { animSys.startIdle(); addLog('Idle 자동 시작'); }

  wsClient.connect();
}

main().catch(console.error);





