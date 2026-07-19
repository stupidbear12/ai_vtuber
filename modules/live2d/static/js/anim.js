// anim.js - mao_pro Live2D 파라미터 애니메이션 시스템

// mao_pro 파라미터 ID
const PARAMS = {
  ANGLE_X:    'ParamAngleX',
  ANGLE_Y:    'ParamAngleY',
  ANGLE_Z:    'ParamAngleZ',
  EYE_L:      'ParamEyeLOpen',
  EYE_R:      'ParamEyeROpen',
  EYE_BALL_X: 'ParamEyeBallX',
  EYE_BALL_Y: 'ParamEyeBallY',
  BROW_L:     'ParamBrowLY',
  BROW_R:     'ParamBrowRY',
  MOUTH_A:    'ParamA',          // 립싱크 (모음 'ㅏ' 개구도)
  BODY_X:     'ParamBodyAngleX',
  BODY_Y:     'ParamBodyAngleY',
  BODY_Z:     'ParamBodyAngleZ',
  BREATH:     'ParamBreath',
  CHEEK:      'ParamCheek',
  EYE_L_SMILE: 'ParamEyeLSmile',
  EYE_R_SMILE: 'ParamEyeRSmile',
  EYE_EFFECT:  'ParamEyeEffect',
  HEART_HEAL_ON: 'ParamHeartHealOn',
  HEART_DRAW:    'ParamHeartDrow',
  HEART_SIZE:    'ParamHeartSize',
  AURA_ON:       'ParamAuraOn',
  AURA:          'ParamAura',
};

const PRIORITY = {
  BREATH:   10,
  HEAD:     15,
  TALKING:  25,
  BLINK:    30,
  REACTION: 40,
  MANUAL:   50,
};

function lerp(a, b, t) { return a + (b - a) * t; }
function easeInOut(t) { return t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t; }
function sleep(ms)     { return new Promise(r => setTimeout(r, ms)); }

// ── 우선순위 기반 파라미터 레이어 병합 ───────────────────────────
class LayeredParamState {
  constructor() {
    this._layers = new Map();
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
    const sorted = [...this._layers.values()].sort((a, b) => a.priority - b.priority);
    const out = {};
    for (const { values } of sorted) Object.assign(out, values);
    return out;
  }
}

// ── 호흡 레이어 ─────────────────────────────────────────────────
class BreathLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._t0 = 0;
    this._period = 4.0;
  }

  start() { this._running = true; this._t0 = performance.now() / 1000; }
  stop()  { this._running = false; this._s.remove('breath'); }

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

// ── 눈 깜빡임 레이어 ────────────────────────────────────────────
class BlinkLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._phase = 'wait';
    this._phaseT = 0;
    this._nextBlink = 0;
  }

  start() { this._running = true; this._scheduleNext(performance.now() / 1000); }
  stop()  { this._running = false; this._phase = 'wait'; this._s.remove('blink'); }

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

// ── 고개 흔들림 레이어 ──────────────────────────────────────────
class HeadSwayLayer {
  constructor(state) {
    this._s = state;
    this._running = false;
    this._t0 = 0;
  }

  start() { this._running = true; this._t0 = performance.now() / 1000; }
  stop()  { this._running = false; this._s.remove('head'); }

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

// ── 반응 애니메이션 시스템 ──────────────────────────────────────
class ReactionSystem {
  constructor(state) {
    this._s = state;
    // 반응 이름별 취소 플래그 — 서로 다른 반응(예: heart + superchat)이
    // 동시에 실행될 수 있도록 이름별로 독립 관리한다.
    this._flags = new Map();
  }

  async trigger(name) {
    const prev = this._flags.get(name);
    if (prev) prev.v = true;
    const flag = { v: false };
    this._flags.set(name, flag);
    const map = {
      nod:       () => this._nod(flag),
      shake:     () => this._shake(flag),
      surprised: () => this._surprised(flag),
      superchat: () => this._superchat(flag),
      laugh:       () => this._laugh(flag),
      greet:       () => this._greet(flag),
      embarrassed: () => this._embarrassed(flag),
      think:       () => this._think(flag),
      dance:       () => this._dance(flag),
      heart:       () => this._heart(flag),
      magic:       () => this._magic(flag),
    };
    await (map[name]?.() ?? Promise.resolve());
    if (this._flags.get(name) === flag) this._flags.delete(name);
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
      { [PARAMS.BROW_L]: 0, [PARAMS.BROW_R]: 0, [PARAMS.MOUTH_A]: 0 },
      { [PARAMS.BROW_L]: 0.5, [PARAMS.BROW_R]: 0.5, [PARAMS.MOUTH_A]: 0.4 },
      0.20, flag);
    if (!flag.v) await sleep(1000);
    if (!flag.v) await this._fadeOut(L, 0.45, flag);
    this._s.remove(L);
  }

  async _superchat(flag) {
    const L = 'rx_superchat';
    await this._interp(L,
      { [PARAMS.BROW_L]: 0, [PARAMS.BROW_R]: 0, [PARAMS.MOUTH_A]: 0, [PARAMS.CHEEK]: 0 },
      { [PARAMS.BROW_L]: 0.3, [PARAMS.BROW_R]: 0.3, [PARAMS.MOUTH_A]: 0.3, [PARAMS.CHEEK]: 0.8 },
      0.45, flag);
    if (!flag.v) await sleep(1800);
    if (!flag.v) await this._fadeOut(L, 0.5, flag);
    this._s.remove(L);
  }

  async _laugh(flag) {
    const L = 'rx_laugh';
    await this._interp(L,
      { [PARAMS.EYE_L_SMILE]: 0, [PARAMS.EYE_R_SMILE]: 0, [PARAMS.MOUTH_A]: 0, [PARAMS.BODY_Y]: 0 },
      { [PARAMS.EYE_L_SMILE]: 1, [PARAMS.EYE_R_SMILE]: 1, [PARAMS.MOUTH_A]: 1, [PARAMS.BODY_Y]: -3 },
      0.15, flag);
    for (let i = 0; i < 3 && !flag.v; i++) {
      await this._interp(L,
        { [PARAMS.MOUTH_A]: 1, [PARAMS.BODY_Y]: -3 },
        { [PARAMS.MOUTH_A]: 0.3, [PARAMS.BODY_Y]: 2 },
        0.14, flag);
      await this._interp(L,
        { [PARAMS.MOUTH_A]: 0.3, [PARAMS.BODY_Y]: 2 },
        { [PARAMS.MOUTH_A]: 1, [PARAMS.BODY_Y]: -3 },
        0.14, flag);
    }
    if (!flag.v) await this._fadeOut(L, 0.35, flag);
    this._s.remove(L);
  }

  async _greet(flag) {
    const L = 'rx_greet';
    await this._interp(L,
      { [PARAMS.BODY_X]: 0, [PARAMS.ANGLE_Y]: 0 },
      { [PARAMS.BODY_X]: -6, [PARAMS.ANGLE_Y]: -6 },
      0.25, flag);
    if (!flag.v) await this._interp(L,
      { [PARAMS.BODY_X]: -6, [PARAMS.ANGLE_Y]: -6 },
      { [PARAMS.BODY_X]: 6, [PARAMS.ANGLE_Y]: 2 },
      0.30, flag);
    if (!flag.v) await this._interp(L,
      { [PARAMS.BODY_X]: 6, [PARAMS.ANGLE_Y]: 2 },
      { [PARAMS.BODY_X]: 0, [PARAMS.ANGLE_Y]: 0 },
      0.25, flag);
    this._s.remove(L);
  }

  async _embarrassed(flag) {
    const L = 'rx_embarrassed';
    await this._interp(L,
      { [PARAMS.CHEEK]: 0, [PARAMS.BROW_L]: 0, [PARAMS.BROW_R]: 0 },
      { [PARAMS.CHEEK]: 1, [PARAMS.BROW_L]: 0.3, [PARAMS.BROW_R]: 0.3 },
      0.25, flag);
    if (!flag.v) await this._interp(L,
      { [PARAMS.EYE_L]: 1, [PARAMS.EYE_R]: 1 },
      { [PARAMS.EYE_L]: 0, [PARAMS.EYE_R]: 0 },
      0.12, flag);
    if (!flag.v) await this._interp(L,
      { [PARAMS.EYE_L]: 0, [PARAMS.EYE_R]: 0 },
      { [PARAMS.EYE_L]: 1, [PARAMS.EYE_R]: 1 },
      0.16, flag);
    if (!flag.v) await sleep(800);
    if (!flag.v) await this._fadeOut(L, 0.4, flag);
    this._s.remove(L);
  }

  async _think(flag) {
    const L = 'rx_think';
    await this._interp(L,
      { [PARAMS.EYE_BALL_X]: 0, [PARAMS.EYE_BALL_Y]: 0, [PARAMS.ANGLE_Z]: 0 },
      { [PARAMS.EYE_BALL_X]: 0.3, [PARAMS.EYE_BALL_Y]: 0.3, [PARAMS.ANGLE_Z]: 6 },
      0.30, flag);
    if (!flag.v) await sleep(1200);
    if (!flag.v) await this._fadeOut(L, 0.4, flag);
    this._s.remove(L);
  }

  async _dance(flag) {
    const L = 'rx_dance';
    const steps = [
      { [PARAMS.BODY_X]: -8, [PARAMS.ANGLE_Z]: -6 },
      { [PARAMS.BODY_X]: 8,  [PARAMS.ANGLE_Z]: 6 },
      { [PARAMS.BODY_X]: -8, [PARAMS.ANGLE_Z]: -6 },
      { [PARAMS.BODY_X]: 8,  [PARAMS.ANGLE_Z]: 6 },
    ];
    let prev = { [PARAMS.BODY_X]: 0, [PARAMS.ANGLE_Z]: 0 };
    for (const step of steps) {
      if (flag.v) break;
      await this._interp(L, prev, step, 0.22, flag);
      prev = step;
    }
    if (!flag.v) await this._interp(L, prev, { [PARAMS.BODY_X]: 0, [PARAMS.ANGLE_Z]: 0 }, 0.22, flag);
    this._s.remove(L);
  }

  async _heart(flag) {
    const L = 'rx_heart';
    await this._interp(L,
      { [PARAMS.HEART_HEAL_ON]: 0, [PARAMS.HEART_DRAW]: 0, [PARAMS.HEART_SIZE]: 0 },
      { [PARAMS.HEART_HEAL_ON]: 1, [PARAMS.HEART_DRAW]: 0, [PARAMS.HEART_SIZE]: 0 },
      0.05, flag);
    if (!flag.v) await this._interp(L,
      { [PARAMS.HEART_HEAL_ON]: 1, [PARAMS.HEART_DRAW]: 0, [PARAMS.HEART_SIZE]: 0 },
      { [PARAMS.HEART_HEAL_ON]: 1, [PARAMS.HEART_DRAW]: 1, [PARAMS.HEART_SIZE]: 1 },
      0.4, flag);
    if (!flag.v) await sleep(900);
    if (!flag.v) await this._fadeOut(L, 0.5, flag);
    this._s.remove(L);
  }

  async _magic(flag) {
    const L = 'rx_magic';
    await this._interp(L,
      { [PARAMS.AURA_ON]: 0, [PARAMS.AURA]: 0 },
      { [PARAMS.AURA_ON]: 1, [PARAMS.AURA]: 1 },
      0.5, flag);
    if (!flag.v) await sleep(1200);
    if (!flag.v) await this._fadeOut(L, 0.6, flag);
    this._s.remove(L);
  }
}

// ── 통합 애니메이션 시스템 ──────────────────────────────────────
class AnimSystem {
  constructor() {
    this._state    = new LayeredParamState();
    this._breath   = new BreathLayer(this._state);
    this._blink    = new BlinkLayer(this._state);
    this._head     = new HeadSwayLayer(this._state);
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

  // 립싱크: ParamA 직접 주입
  setMouth(value) {
    this._state.set('lipsync', { [PARAMS.MOUTH_A]: Math.max(0, Math.min(1, value)) }, PRIORITY.TALKING);
  }

  clearMouth() { this._state.remove('lipsync'); }

  setManualParams(params) { this._state.set('manual', params, PRIORITY.MANUAL); }

  triggerReaction(name) { this._reaction.trigger(name); }

  tick() {
    const now = performance.now() / 1000;
    if (this._idleOn) {
      this._breath.update(now);
      this._blink.update(now);
      this._head.update(now);
    }
    return this._state.getMerged();
  }
}
