'use strict';
const { app, BrowserWindow, Tray, Menu, nativeImage, ipcMain, screen } = require('electron');
const path = require('path');
const http = require('http');

// ai_live2d 서버 포트 — Python 서버와 반드시 일치해야 함
const SERVER_PORT = 8001;

let win = null;
let tray = null;
let mouseThroughEnabled = false;

// ── 메인 창 생성 ──────────────────────────────────────────────

function createWindow() {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const winW = 400, winH = 600;

  // 화면 우측 하단에 창 배치
  win = new BrowserWindow({
    width: winW,
    height: winH,
    x: sw - winW,
    y: sh - winH,
    frame: false,           // 테두리 없는 창
    transparent: true,       // 투명 창 (Live2D 모델만 표시)
    backgroundColor: '#00000000',
    alwaysOnTop: true,       // 항상 최상단 표시
    skipTaskbar: true,       // 작업 표시줄에 표시 안 함
    resizable: false,
    hasShadow: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,  // preload.js와 renderer 격리
      webSecurity: false,       // localhost 리소스 접근 허용
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  win.loadFile(path.join(__dirname, 'pet.html'));
  win.on('closed', () => { win = null; });
}

// ── 시스템 트레이 아이콘 ──────────────────────────────────────

function createTray() {
  let img;
  try {
    img = nativeImage.createFromPath(path.join(__dirname, 'icon.png'));
    if (img.isEmpty()) throw new Error('empty');
  } catch (_) {
    img = nativeImage.createEmpty();
  }
  tray = new Tray(img);
  tray.setToolTip('에메스 (ai_live2d)');
  rebuildTrayMenu();
}

function rebuildTrayMenu() {
  const menu = Menu.buildFromTemplate([
    {
      label: `마우스 통과: ${mouseThroughEnabled ? 'ON' : 'OFF'}`,
      enabled: false,
    },
    {
      label: '마우스 통과 토글  (더블클릭으로도 전환)',
      click: () => toggleMouseThrough(),
    },
    { type: 'separator' },
    {
      label: '표시 / 숨기기',
      click: () => { if (win) win.isVisible() ? win.hide() : win.show(); },
    },
    { type: 'separator' },
    { label: '종료', click: () => app.quit() },
  ]);
  tray.setContextMenu(menu);
}

function toggleMouseThrough() {
  mouseThroughEnabled = !mouseThroughEnabled;
  if (win) {
    // forward: true → 마우스 이벤트를 OS 창으로 전달하면서 renderer도 수신
    win.setIgnoreMouseEvents(mouseThroughEnabled, { forward: true });
    win.webContents.send('mouse-through-changed', mouseThroughEnabled);
  }
  rebuildTrayMenu();
}

// ── IPC 핸들러 ────────────────────────────────────────────────

// 창 드래그 이동 — preload.js에서 마우스 델타값을 IPC로 전달
ipcMain.on('drag-move', (_, { dx, dy }) => {
  if (!win) return;
  const [cx, cy] = win.getPosition();
  win.setPosition(cx + Math.round(dx), cy + Math.round(dy));
});

// 마우스 통과 모드 토글 (더블클릭 시)
ipcMain.on('toggle-mouse-through', () => toggleMouseThrough());

// 채팅 메시지 수신 → ai_live2d FastAPI 서버로 HTTP POST
ipcMain.on('chat-message', (_, msg) => {
  const body = JSON.stringify({ message: String(msg) });

  const req = http.request(
    {
      hostname: 'localhost',
      port: SERVER_PORT,              // ai_live2d 서버 포트 (8001)
      path: '/live2d/chat',           // 에메스 채팅 엔드포인트
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    },
    (res) => {
      let buf = '';
      res.on('data', (c) => { buf += c; });
      res.on('end', () => {
        try {
          const data = JSON.parse(buf);
          // renderer에 응답 전달 (표정 + 텍스트)
          if (win) win.webContents.send('chat-reply', data);
        } catch (_) {
          if (win) win.webContents.send('chat-reply', { reply: '응답 파싱 오류' });
        }
      });
    }
  );

  req.on('error', (e) => {
    if (win) win.webContents.send('chat-reply', { reply: `서버 오류: ${e.message}` });
  });

  req.write(body);
  req.end();
});

// ── 앱 생명주기 ───────────────────────────────────────────────

// Windows WebGL 투명 창 합성 문제 수정: 소프트웨어 GPU 합성 강제
app.commandLine.appendSwitch('disable-gpu-compositing');

app.whenReady().then(() => {
  createWindow();
  createTray();
});

// 트레이에 상주 — 모든 창이 닫혀도 앱 종료 안 함
app.on('window-all-closed', () => {});
