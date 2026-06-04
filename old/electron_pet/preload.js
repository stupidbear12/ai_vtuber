'use strict';
const { ipcRenderer, contextBridge } = require('electron');

// ── Window drag ───────────────────────────────────────────────
// preload 캡처 리스너가 모든 마우스 이벤트를 먼저 받아 창 이동 처리
// chat-wrap 영역 클릭은 드래그로 오인하지 않도록 제외

let dragging = false;
let moved    = false;
let lastX    = 0;
let lastY    = 0;
const DRAG_THRESHOLD = 4; // px — 이 거리 이상 이동 시 드래그로 판정

window.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  // chat 오버레이 위에서는 드래그 시작 안 함
  try {
    if (e.target?.closest?.('#chat-wrap')) return;
  } catch (_) {}
  dragging = true;
  moved    = false;
  lastX    = e.screenX;
  lastY    = e.screenY;
}, true);

window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const dx = e.screenX - lastX;
  const dy = e.screenY - lastY;
  if (!moved && Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) return;
  moved = true;
  lastX = e.screenX;
  lastY = e.screenY;
  ipcRenderer.send('drag-move', { dx, dy });
}, true);

window.addEventListener('mouseup',    () => { dragging = false; }, true);
window.addEventListener('mouseleave', () => { dragging = false; }, true);

// 더블클릭으로 마우스 통과 모드 토글
window.addEventListener('dblclick', (e) => {
  try { if (e.target?.closest?.('#chat-wrap')) return; } catch (_) {}
  ipcRenderer.send('toggle-mouse-through');
});

// ── Renderer API ──────────────────────────────────────────────
contextBridge.exposeInMainWorld('petAPI', {
  // 채팅 메시지 전송 (main.js → FastAPI /live2d/chat)
  sendChat: (msg) => ipcRenderer.send('chat-message', msg),

  // main.js가 보내는 채팅 응답 수신
  onChatReply: (cb) => ipcRenderer.on('chat-reply', (_, data) => cb(data)),

  // 마우스 통과 모드 변경 알림
  onMouseThrough: (cb) => ipcRenderer.on('mouse-through-changed', (_, enabled) => cb(enabled)),

  // 이번 mousedown-mouseup 사이에 유의미한 드래그가 있었는지
  // (클릭 vs 드래그 구분용)
  wasDragged: () => moved,
});
