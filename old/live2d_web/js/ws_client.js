// ws_client.js - Python 서버 ↔ 브라우저 WebSocket 브리지

class Live2DWSClient {
  /**
   * @param {AnimSystem} animSystem
   * @param {Function} onStatusChange
   * @param {Function} addLog
   * @param {Function} setExpression - app.js의 setExpression()
   * @param {Function} playMotion    - app.js의 playMotion()
   */
  constructor(animSystem, onStatusChange, addLog, setExpression, playMotion) {
    this._anim          = animSystem;
    this._onChange      = onStatusChange;
    this._log           = addLog;
    this._setExpression = setExpression;
    this._playMotion    = playMotion;
    this._ws            = null;
    this._delay         = 2000;
    this._url = `ws://${location.host}/live2d/ws`;
  }

  connect() { this._try(); }

  _try() {
    try {
      this._ws = new WebSocket(this._url);
      this._ws.onopen    = () => {
        this._onChange('connected');
        this._delay = 2000;
        this._log('서버 WebSocket 연결됨');
      };
      this._ws.onclose   = () => {
        this._onChange('disconnected');
        setTimeout(() => this._try(), this._delay);
        this._delay = Math.min(this._delay * 1.5, 15000);
      };
      this._ws.onerror   = () => { this._onChange('error'); };
      this._ws.onmessage = (e) => {
        try { this._handle(JSON.parse(e.data)); } catch (ex) { console.error('WS parse', ex); }
      };
    } catch (_) {
      setTimeout(() => this._try(), this._delay);
    }
  }

  _handle(msg) {
    switch (msg.cmd) {
      // 파라미터 직접 주입
      case 'set_params':
        this._anim.setManualParams(msg.params);
        break;

      // 감정 이름 → expression 변환
      case 'set_emotion':
        if (typeof setEmotion === 'function') {
          setEmotion(msg.emotion);
          this._log(`감정: ${msg.emotion}`);
        }
        break;

      // expression 직접 지정 (이름 또는 0-based 인덱스)
      case 'set_expression':
        this._setExpression?.(msg.expression);
        this._log(`표정: ${msg.expression}`);
        break;

      // 모션 재생 (group, index)
      case 'set_motion':
        this._playMotion?.(msg.group ?? '', msg.index ?? 0);
        this._log(`모션: ${msg.group}[${msg.index}]`);
        break;

      // 립싱크 — ParamA 직접 주입
      case 'set_mouth':
        this._anim.setMouth(msg.value);
        break;
      case 'clear_mouth':
        this._anim.clearMouth();
        break;

      // 반응 애니메이션
      case 'reaction':
        this._anim.triggerReaction(msg.name);
        this._log(`반응: ${msg.name}`);
        break;

      // Idle 제어
      case 'idle_start':
        this._anim.startIdle();
        this._log('Idle 시작 (서버)');
        break;
      case 'idle_stop':
        this._anim.stopIdle();
        this._log('Idle 정지 (서버)');
        break;
    }
  }

  send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  }
}
