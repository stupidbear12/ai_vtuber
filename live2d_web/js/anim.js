// anim.js - Live2D 파라미터 애니메이션 시스템
// animation_controller.py의 레이어 구조를 JavaScript로 포팅

const PARAMS = {
  ANGLE_X:    'ParamAngleX',      // 고개 좌우 -30~30
  ANGLE_Y:    'ParamAngleY',      // 고개 상하 -30~30
  ANGLE_Z:    'ParamAngleZ',      // 고개 기울임 -30~30
  EYE_L:      'ParamEyeLOpen',    // 왼눈 열림 0~1
  EYE_R:      'ParamEyeROpen',    // 오른눈 열림 0~1
  EYE_BALL_X: 'ParamEyeBallX',   // 시선 좌우 -1~1
  EYE_BALL_Y: 'ParamEyeBallY',   // 시선 상하 -1~1
  BROW_L:     'ParamBrowLY',      // 왼눈썹 -1~1
  BROW_R:     'ParamBrowRY',      // 오른눈썹 -1~1
  MOUTH_OPEN: 'ParamMouthOpenY',  // 입 열림 0~1
  MOUTH_FORM: 'ParamMouthForm',   // 미소 -1~1
  BODY_X:     'ParamBodyAngleX',  // 몸 좌우 -10~10
  BREATH:     'ParamBreath',      // 호흡 0~1
  TERE:       'ParamTere',        // 볼 붉힘 0~1
};

const PRIORITY = {
  BREATH:   10,
  HEAD:     15,
  EMOTION:  20,
  TALKING:  25,
  BLINK:    30,
  REACTION: 40,
  MANUAL:   50,
};

function lerp(a, b, t) { return a + (b - a) * t; }
function easeInOut(t) { return t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t; }
function easeOut(t) { return 1 - (1 - t) * (1 - t); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── 우선순위 기반 파라미터 레이어 병합 ────────────────────────────
class LayeredParamState {
  constructor() {
    this._layers = new Map(); // id → { values, priority }
  }

  set(id, values, priority) {
    this._layers.set(id, { values, priority });
  }

  remove(id) {
    this._layers.delete(id);
  }

  clear() {
    this._layers.clear();
  }

  getValues(id) {
    return this._layers.get(id)?.values ?? null;
  }

  getMerged() {
    const sorted = [...this._layers.values()]
      .sort((a, b) => a.priority - b.priority);
    const out = {};
    for (const { values } of sorted) Object.assign(out, values);
    return out;
  }
}

// ─── 호흡 레이어 ────────────────────────────────────────────────────
class BreathLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._t0 = 0;
    this._period = 4.0;
  }

  start() { this._running = true; this._t0 = performance.now() / 1000; }

  stop() { this._running = false; this._s.remove('breath'); }

  update(now) {
    if (!this._running) return;
    const phase = ((now - this._t0) % this._period) / this._period;
    const b = (Math.sin(phase * Math.PI * 2 - Math.PI / 2) + 1) / 2;
    this._s.set('breath', {
      [PARAMS.BREATH]: b,
      [PARAMS.BODY_X]: Math.sin(phase * Math.PI * 2) * 0.5,
    }, PRIORITY.BREATH);
  }
}

// ─── 자연 눈 깜빡임 레이어 ──────────────────────────────────────────
class BlinkLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._phase = 'wait'; // wait | close | open
    this._phaseT = 0;
    this._nextBlink = 0;
  }

  start() {
    this._running = true;
    this._scheduleNext(performance.now() / 1000);
  }

  stop() {
    this._running = false;
    this._phase = 'wait';
    this._s.remove('blink');
  }

  _scheduleNext(now) {
    this._nextBlink = now + 3.0 + Math.random() * 3.0;
    this._phase = 'wait';
  }

  update(now) {
    if (!this._running) return;

    if (this._phase === 'wait') {
      if (now >= this._nextBlink) { this._phase = 'close'; this._phaseT = now; }
      return;
    }
    if (this._phase === 'close') {
      const t = Math.min((now - this._phaseT) / 0.10, 1);
      this._s.set('blink', { [PARAMS.EYE_L]: 1 - t, [PARAMS.EYE_R]: 1 - t }, PRIORITY.BLINK);
      if (t >= 1) { this._phase = 'open'; this._phaseT = now; }
      return;
    }
    if (this._phase === 'open') {
      const t = Math.min((now - this._phaseT) / 0.12, 1);
      this._s.set('blink', { [PARAMS.EYE_L]: t, [PARAMS.EYE_R]: t }, PRIORITY.BLINK);
      if (t >= 1) { this._s.remove('blink'); this._scheduleNext(now); }
    }
  }
}

// ─── 고개 흔들림 레이어 ─────────────────────────────────────────────
class HeadSwayLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._t0 = 0;
  }

  start() { this._running = true; this._t0 = performance.now() / 1000; }

  stop() { this._running = false; this._s.remove('head'); }

  update(now) {
    if (!this._running) return;
    const e = now - this._t0;
    this._s.set('head', {
      [PARAMS.ANGLE_X]: Math.sin(e * Math.PI * 2 / 8.0) * 2.0,
      [PARAMS.ANGLE_Y]: Math.sin(e * Math.PI * 2 / 6.5) * 1.5,
      [PARAMS.ANGLE_Z]: Math.sin(e * Math.PI * 2 / 10.0) * 1.0,
    }, PRIORITY.HEAD);
  }
}

// ─── 감정 레이어 (보간) ─────────────────────────────────────────────
const EMOTION_PRESETS = {
  calm: {
    [PARAMS.EYE_L]: 1.0, [PARAMS.EYE_R]: 1.0,
    [PARAMS.BROW_L]: 0.0, [PARAMS.BROW_R]: 0.0,
    [PARAMS.MOUTH_FORM]: 0.1, [PARAMS.TERE]: 0.0,
  },
  happy: {
    [PARAMS.EYE_L]: 0.85, [PARAMS.EYE_R]: 0.85,
    [PARAMS.BROW_L]: 0.2, [PARAMS.BROW_R]: 0.2,
    [PARAMS.MOUTH_FORM]: 0.5, [PARAMS.TERE]: 0.3,
  },
  surprised: {
    [PARAMS.EYE_L]: 1.0, [PARAMS.EYE_R]: 1.0,
    [PARAMS.BROW_L]: 0.5, [PARAMS.BROW_R]: 0.5,
    [PARAMS.MOUTH_FORM]: -0.1, [PARAMS.TERE]: 0.0,
  },
  thinking: {
    [PARAMS.EYE_L]: 0.7, [PARAMS.EYE_R]: 0.7,
    [PARAMS.BROW_L]: -0.15, [PARAMS.BROW_R]: 0.25,
    [PARAMS.MOUTH_FORM]: -0.05, [PARAMS.TERE]: 0.0,
  },
};

class EmotionLayer {
  constructor(state) {
    this._s = state;
    this._from = { ...EMOTION_PRESETS.calm };
    this._to   = { ...EMOTION_PRESETS.calm };
    this._interpT0 = 0;
    this._interpDur = 0.5;
    this._interping = false;
  }

  setEmotion(name) {
    const preset = EMOTION_PRESETS[name] ?? EMOTION_PRESETS.calm;
    this._from = this._current();
    this._to   = preset;
    this._interpT0 = performance.now() / 1000;
    this._interping = true;
  }

  _current() {
    return { ...this._from };
  }

  update(now) {
    if (!this._interping) {
      this._s.set('emotion', this._to, PRIORITY.EMOTION);
      return;
    }
    const t = easeInOut(Math.min((now - this._interpT0) / this._interpDur, 1));
    const allKeys = new Set([...Object.keys(this._from), ...Object.keys(this._to)]);
    const merged = {};
    for (const k of allKeys) merged[k] = lerp(this._from[k] ?? 0, this._to[k] ?? 0, t);
    this._s.set('emotion', merged, PRIORITY.EMOTION);
    if (t >= 1) { this._from = { ...this._to }; this._interping = false; }
  }
}

// ─── 반응 애니메이션 시스템 ─────────────────────────────────────────
class ReactionSystem {
  constructor(state) {
    this._s = state;
    this._cancelFlag = { v: false };
  }

  async trigger(name) {
    this._cancelFlag.v = true;
    const flag = { v: false };
    this._cancelFlag = flag;

    const map = {
      nod:       () => this._nod(flag),
      shake:     () => this._shake(flag),
      surprised: () => this._surprised(flag),
      superchat: () => this._superchat(flag),
    };
    await (map[name]?.() ?? Promise.resolve());
  }

  async _interp(layerId, from, to, durSec, flag) {
    const keys = new Set([...Object.keys(from), ...Object.keys(to)]);
    const steps = Math.max(Math.round(durSec * 60), 2);
    const stepMs = (durSec * 1000) / steps;
    for (let i = 0; i <= steps; i++) {
      if (flag.v) break;
      const t = easeInOut(i / steps);
      const vals = {};
      for (const k of keys) vals[k] = lerp(from[k] ?? 0, to[k] ?? 0, t);
      this._s.set(layerId, vals, PRIORITY.REACTION);
      await sleep(stepMs);
    }
  }

  async _fadeOut(layerId, durSec, flag) {
    const cur = this._s.getValues(layerId);
    if (!cur) return;
    const zero = Object.fromEntries(Object.keys(cur).map(k => [k, 0]));
    await this._interp(layerId, cur, zero, durSec, flag);
  }

  async _nod(flag) {
    const L = 'rx_nod';
    for (let i = 0; i < 2 && !flag.v; i++) {
      await this._interp(L, { [PARAMS.ANGLE_Y]: 0 }, { [PARAMS.ANGLE_Y]: -9 }, 0.18, flag);
      await this._interp(L, { [PARAMS.ANGLE_Y]: -9 }, { [PARAMS.ANGLE_Y]: 0 }, 0.18, flag);
      await sleep(60);
    }
    this._s.remove(L);
  }

  async _shake(flag) {
    const L = 'rx_shake';
    const angles = [0, 11, -11, 8, -8, 0];
    for (let i = 0; i < angles.length - 1 && !flag.v; i++) {
      await this._interp(L,
        { [PARAMS.ANGLE_X]: angles[i] },
        { [PARAMS.ANGLE_X]: angles[i + 1] },
        0.14, flag);
    }
    this._s.remove(L);
  }

  async _surprised(flag) {
    const L = 'rx_surprised';
    await this._interp(L,
      { [PARAMS.BROW_L]: 0, [PARAMS.BROW_R]: 0, [PARAMS.MOUTH_OPEN]: 0 },
      { [PARAMS.BROW_L]: 0.5, [PARAMS.BROW_R]: 0.5, [PARAMS.MOUTH_OPEN]: 0.25 },
      0.20, flag);
    if (!flag.v) await sleep(1000);
    if (!flag.v) await this._fadeOut(L, 0.45, flag);
    this._s.remove(L);
  }

  async _superchat(flag) {
    const L = 'rx_superchat';
    await this._interp(L,
      { [PARAMS.BROW_L]: 0, [PARAMS.BROW_R]: 0, [PARAMS.MOUTH_FORM]: 0, [PARAMS.TERE]: 0 },
      { [PARAMS.BROW_L]: 0.3, [PARAMS.BROW_R]: 0.3, [PARAMS.MOUTH_FORM]: 0.4, [PARAMS.TERE]: 0.8 },
      0.45, flag);
    if (!flag.v) await sleep(1800);
    if (!flag.v) await this._fadeOut(L, 0.5, flag);
    this._s.remove(L);
  }
}

// ─── 통합 애니메이션 시스템 ─────────────────────────────────────────
class AnimSystem {
  constructor() {
    this._state    = new LayeredParamState();
    this._breath   = new BreathLayer(this._state);
    this._blink    = new BlinkLayer(this._state);
    this._head     = new HeadSwayLayer(this._state);
    this._emotion  = new EmotionLayer(this._state);
    this._reaction = new ReactionSystem(this._state);
    this._idleOn   = false;
  }

  startIdle() {
    this._idleOn = true;
    this._breath.start();
    this._blink.start();
    this._head.start();
  }

  stopIdle() {
    this._idleOn = false;
    this._breath.stop();
    this._blink.stop();
    this._head.stop();
    this._state.clear();
  }

  setEmotion(name) { this._emotion.setEmotion(name); }

  setMouth(value) {
    this._state.set('lipsync', { [PARAMS.MOUTH_OPEN]: value }, PRIORITY.TALKING);
  }

  clearMouth() { this._state.remove('lipsync'); }

  setManualParams(params) { this._state.set('manual', params, PRIORITY.MANUAL); }

  triggerReaction(name) { this._reaction.trigger(name); }

  // 60fps 틱마다 호출 - 병합된 파라미터 반환
  tick() {
    const now = performance.now() / 1000;
    if (this._idleOn) {
      this._breath.update(now);
      this._blink.update(now);
      this._head.update(now);
    }
    this._emotion.update(now);
    return this._state.getMerged();
  }
}
