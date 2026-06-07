# start-pet.ps1 — Electron 데스크톱 펫 실행
# 사용법: .\start-pet.ps1
# 전제: live2d 서버(8001) 실행 중 — 없으면 start-all.ps1 먼저 실행

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$PetDir = Join-Path $Root "modules\live2d\electron"

function Test-PortListen([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-PortListen 8001)) {
    Write-Host "live2d 서버(8001)가 실행 중이 아닙니다." -ForegroundColor Yellow
    Write-Host "먼저 실행: .\start-all.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $PetDir "node_modules"))) {
    Write-Host "Electron 의존성 설치 중 (npm install)..." -ForegroundColor Cyan
    Push-Location $PetDir
    npm install
    Pop-Location
}

Write-Host "Electron 데스크톱 펫 시작..." -ForegroundColor Green
Write-Host "  - 캐릭터 클릭: 채팅 열기/닫기" -ForegroundColor Gray
Write-Host "  - 더블클릭: 마우스 통과 모드" -ForegroundColor Gray
Write-Host "  - 트레이 아이콘: 표시/숨기기, 종료" -ForegroundColor Gray

Push-Location $PetDir
npm start
Pop-Location
