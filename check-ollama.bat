@echo off
chcp 65001 >nul
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

echo === Ollama 상태 확인 ===
echo.
echo cmd에서 "ollama"가 안 될 때: 이 창을 닫고 cmd를 새로 여세요.
echo.

if not exist "%OLLAMA%" (
    echo [오류] Ollama가 설치되어 있지 않습니다.
    echo https://ollama.com 에서 설치하세요.
    pause
    exit /b 1
)

echo [설치 경로] %OLLAMA%
echo.
echo [모델 목록]
"%OLLAMA%" list
echo.

powershell -NoProfile -Command "try { $r = Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3; Write-Host '[API] 실행 중 (포트 11434)' -ForegroundColor Green } catch { Write-Host '[API] 미실행 - Ollama 앱을 시작하세요' -ForegroundColor Red }"

echo.
echo .env 설정: OLLAMA_MODEL=emeth
pause
