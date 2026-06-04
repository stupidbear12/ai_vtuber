'use strict';
const { app, BrowserWindow, Tray, Menu, nativeImage, ipcMain, screen } = require('electron');
const path = require('path');
const http = require('http');

let win = null;
let tray = null;
let mouseThroughEnabled = false;

// ── Window ────────────────────────────────────────────────────

function createWindow() {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const winW = 400, winH = 600;
  win = new BrowserWindow({
    width: winW,
    height: winH,
    x: sw - winW,
    y: sh - winH,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  win.loadFile(path.join(__dirname, 'pet.html'));
  win.on('closed', () => { win = null; });
}

// ── Tray ──────────────────────────────────────────────────────

function createTray() {
  let img;
  try {
    img = nativeImage.createFromPath(path.join(__dirname, 'icon.png'));
    if (img.isEmpty()) throw new Error('empty');
  } catch (_) {
    img = nativeImage.createEmpty();
  }
  tray = new Tray(img);
  tray.setToolTip('AI VTuber Desktop Pet');
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
    // forward: true → 마우스 이벤트가 창 뒤 OS 창으로 전달되지만 renderer도 수신
    win.setIgnoreMouseEvents(mouseThroughEnabled, { forward: true });
    win.webContents.send('mouse-through-changed', mouseThroughEnabled);
  }
  rebuildTrayMenu();
}

// ── IPC ───────────────────────────────────────────────────────

ipcMain.on('drag-move', (_, { dx, dy }) => {
  if (!win) return;
  const [cx, cy] = win.getPosition();
  win.setPosition(cx + Math.round(dx), cy + Math.round(dy));
});

ipcMain.on('toggle-mouse-through', () => toggleMouseThrough());

ipcMain.on('chat-message', (_, msg) => {
  const body = JSON.stringify({ message: String(msg) });
  const req = http.request(
    {
      hostname: 'localhost',
      port: 8000,
      path: '/live2d/chat',
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

// ── App lifecycle ─────────────────────────────────────────────

// Windows에서 WebGL 투명 창 합성 문제 해결: 소프트웨어 GPU 합성 강제
app.commandLine.appendSwitch('disable-gpu-compositing');

app.whenReady().then(() => {
  createWindow();
  createTray();
});

// 트레이에 상주 — 모든 창이 닫혀도 종료하지 않음
app.on('window-all-closed', () => {});
