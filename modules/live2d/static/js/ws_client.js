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

    // speak 오디오 재생 상태
    this._currentAudio  = null;
    this._lipsyncTimer  = null;
    this._lipsyncCtx    = null;
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

      // 모션 재생 (group, index, duration ms — 선택)
      case 'set_motion':
        this._playMotion?.(msg.group ?? '', msg.index ?? 0, {
          duration: msg.duration,
        });
        break;

      case 'play_motion_once':
        this._playMotion?.(msg.group ?? '', msg.index ?? 0, {
          duration: msg.duration,
        });
        break;

      case 'play_motion':
        this._playMotion?.(msg.group ?? '', msg.index ?? 0, {
          duration: msg.duration,
        });
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

      // 이펙트 (heart, magic 등)
      case 'effect':
        this._anim.triggerReaction(msg.name);
        this._log(`이펙트: ${msg.name}`);
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

      // 방송 speak: 자막 + TTS 오디오 재생 + 감정 표정
      case 'speak':
        this._handleSpeak(msg);
        break;
    }
  }

  // ── speak 명령 처리 ─────────────────────────────────────────────

  /**
   * 방송 모듈에서 전달받은 speak 명령 처리:
   *   1. 감정 표정 변경
   *   2. 자막 UI 표시
   *   3. TTS 오디오 재생 + 립싱크
   */
  _handleSpeak(msg) {
    const { text, emotion, author, platform, is_donation, audio_base64 } = msg;

    // 1) 감정 표정 변경
    if (typeof setEmotion === 'function' && emotion) {
      setEmotion(emotion);
    }

    // 1-1) 감정별 반응 애니메이션 트리거
    const EMOTION_TO_REACTION = {
      happy: 'laugh', joy: 'laugh', excited: 'laugh',
      surprised: 'surprised', shocked: 'surprised', surprise: 'surprised',
      angry: 'shake', anger: 'shake',
      embarrassed: 'embarrassed', shy: 'embarrassed', flustered: 'embarrassed',
      thinking: 'think',
    };
    const reaction = EMOTION_TO_REACTION[emotion];
    if (reaction) this._anim.triggerReaction(reaction);

    // 1-2) 후원(도네이션)이면 하트 이펙트 + superchat 반응
    if (is_donation) {
      this._anim.triggerReaction('heart');
      this._anim.triggerReaction('superchat');
    }

    // 2) 자막 표시
    this._showSubtitle(text, author, platform, is_donation);

    // 3) TTS 오디오 재생 (base64 WAV, GPT-SoVITS)
    if (audio_base64) {
      this._playAudio(audio_base64);
    }

    this._log('[speak] ' + (text || '').slice(0, 40));
  }

  /**
   * base64 인코딩된 WAV 오디오를 재생하고 립싱크를 연동한다.
   */
  _playAudio(base64) {
    // 이전 재생 중단
    this._stopAudio();

    try {
      const blob = this._base64ToBlob(base64, 'audio/wav');
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this._currentAudio = audio;

      // 재생 시작 -> 립싱크 시작
      audio.addEventListener('play', () => {
        this._startLipsync(audio);
      });

      // 재생 종료 -> 립싱크 정지 + 자막 숨김
      audio.addEventListener('ended', () => {
        this._stopLipsync();
        this._hideSubtitle();
        URL.revokeObjectURL(url);
        this._currentAudio = null;
      });

      audio.addEventListener('error', (e) => {
        console.error('[Speak] audio error:', e);
        this._stopLipsync();
        this._currentAudio = null;
        URL.revokeObjectURL(url);
      });

      audio.play().catch(err => {
        console.warn('[Speak] autoplay blocked:', err.message);
        this._hideSubtitle();
      });

    } catch (err) {
      console.error('[Speak] audio processing error:', err);
    }
  }

  /**
   * Web Audio API AnalyserNode로 음량 측정 -> mouth 립싱크
   */
  _startLipsync(audioEl) {
    this._stopLipsync();

    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const source = ctx.createMediaElementSource(audioEl);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyser.connect(ctx.destination);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const update = () => {
        if (!this._currentAudio) return;

        analyser.getByteFrequencyData(dataArray);
        // 저주파~중주파 대역 평균 에너지 -> 0.0~1.0 범위
        let sum = 0;
        const count = Math.min(32, dataArray.length);
        for (let i = 0; i < count; i++) sum += dataArray[i];
        const avg = sum / count / 255;
        const mouth = Math.min(1.0, avg * 1.8);
        this._anim.setMouth(mouth);

        this._lipsyncTimer = requestAnimationFrame(update);
      };

      this._lipsyncTimer = requestAnimationFrame(update);
      this._lipsyncCtx = ctx;
    } catch (err) {
      // Web Audio API 미지원 시 단순 on/off 립싱크 폴백
      console.warn('[Speak] Web Audio lipsync unavailable, using timer fallback:', err);
      let open = true;
      this._lipsyncTimer = setInterval(() => {
        this._anim.setMouth(open ? 0.6 : 0.1);
        open = !open;
      }, 150);
    }
  }

  /** 립싱크 중단 + 입 닫기 */
  _stopLipsync() {
    if (this._lipsyncTimer) {
      if (typeof this._lipsyncTimer === 'number' && this._lipsyncTimer > 1000) {
        clearInterval(this._lipsyncTimer);
      } else {
        cancelAnimationFrame(this._lipsyncTimer);
      }
      this._lipsyncTimer = null;
    }
    if (this._lipsyncCtx) {
      try { this._lipsyncCtx.close(); } catch (_) {}
      this._lipsyncCtx = null;
    }
    this._anim.clearMouth();
  }

  /** 현재 재생 중인 오디오 중단 */
  _stopAudio() {
    if (this._currentAudio) {
      this._currentAudio.pause();
      this._currentAudio = null;
    }
    this._stopLipsync();
  }

  /**
   * 자막 오버레이 UI를 화면 하단에 표시한다.
   * OBS 브라우저 소스에서도 보이도록 canvas-wrap 내부에 삽입.
   */
  _showSubtitle(text, author, platform, isDonation) {
    if (!text) return;
    this._hideSubtitle();

    const wrap = document.getElementById('canvas-wrap') || document.body;

    const el = document.createElement('div');
    el.id = 'speak-subtitle';
    const bgStyle = isDonation
      ? 'background: linear-gradient(135deg, rgba(255,107,107,0.55), rgba(255,165,0,0.55)); border: 1px solid #ffa500;'
      : 'background: rgba(0, 0, 0, 0.75); border: 1px solid rgba(255,255,255,0.15);';
    el.style.cssText = [
      'position: absolute',
      'bottom: 24px',
      'left: 50%',
      'transform: translateX(-50%)',
      'max-width: 80%',
      'padding: 12px 20px',
      'border-radius: 12px',
      "font-family: 'Segoe UI', sans-serif",
      'font-size: 16px',
      'line-height: 1.5',
      'color: #fff',
      'text-align: center',
      'z-index: 9999',
      'pointer-events: none',
      'animation: subtitleFadeIn 0.3s ease',
      bgStyle,
    ].join('; ');

    // 보낸 사람 태그
    if (author) {
      const tag = platform === 'chzzk' ? '치지직' : platform === 'youtube' ? '유튜브' : '';
      const authorEl = document.createElement('div');
      authorEl.style.cssText = 'font-size: 12px; color: #aaa; margin-bottom: 4px;';
      authorEl.textContent = tag ? '[' + tag + '] ' + author : author;
      el.appendChild(authorEl);
    }

    const textEl = document.createElement('div');
    textEl.textContent = text;
    el.appendChild(textEl);

    wrap.appendChild(el);

    // CSS 애니메이션 keyframes 삽입 (1회만)
    if (!document.getElementById('subtitle-keyframes')) {
      const style = document.createElement('style');
      style.id = 'subtitle-keyframes';
      style.textContent = [
        '@keyframes subtitleFadeIn {',
        '  from { opacity: 0; transform: translateX(-50%) translateY(10px); }',
        '  to   { opacity: 1; transform: translateX(-50%) translateY(0); }',
        '}',
        '@keyframes subtitleFadeOut {',
        '  from { opacity: 1; }',
        '  to   { opacity: 0; }',
        '}',
      ].join('\n');
      document.head.appendChild(style);
    }

    // 오디오 없으면 5초 후 자동 숨김
    if (!this._currentAudio) {
      setTimeout(() => this._hideSubtitle(), 5000);
    }
  }

  /** 자막 오버레이 제거 (페이드아웃) */
  _hideSubtitle() {
    const el = document.getElementById('speak-subtitle');
    if (!el) return;
    el.style.animation = 'subtitleFadeOut 0.3s ease';
    setTimeout(() => el.remove(), 300);
  }

  /** base64 문자열 -> Blob 변환 */
  _base64ToBlob(base64, mimeType) {
    const bin = atob(base64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mimeType });
  }

  send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  }
}
