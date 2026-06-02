// ws_client.js - Python 서버 ↔ 브라우저 WebSocket 브리지

class Live2DWSClient {
  constructor(animSystem, onStatusChange, addLog) {
    this._anim     = animSystem;
    this._onChange = onStatusChange;
    this._log      = addLog;
    this._ws       = null;
    this._delay    = 2000;
    // FastAPI 서버와 같은 origin이면 자동 연결, 다르면 config에서 조정
    this._url = `ws://${location.host}/live2d/ws`;
  }

  connect() { this._try(); }

  _try() {
    try {
      this._ws = new WebSocket(this._url);
      this._ws.onopen    = () => { this._onChange('connected'); this._delay = 2000; this._log('서버 WebSocket 연결됨'); };
      this._ws.onclose   = () => { this._onChange('disconnected'); setTimeout(() => this._try(), this._delay); this._delay = Math.min(this._delay * 1.5, 15000); };
      this._ws.onerror   = () => { this._onChange('error'); };
      this._ws.onmessage = (e) => { try { this._handle(JSON.parse(e.data)); } catch(ex) { console.error('WS parse', ex); } };
    } catch (_) {
      setTimeout(() => this._try(), this._delay);
    }
  }

  _handle(msg) {
    switch (msg.cmd) {
      case 'set_params':
        this._anim.setManualParams(msg.params);
        break;
      case 'set_emotion':
        this._anim.setEmotion(msg.emotion);
        this._log(`감정: ${msg.emotion}`);
        break;
      case 'set_mouth':
        this._anim.setMouth(msg.value);
        break;
      case 'clear_mouth':
        this._anim.clearMouth();
        break;
      case 'reaction':
        this._anim.triggerReaction(msg.name);
        this._log(`반응: ${msg.name}`);
        break;
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
