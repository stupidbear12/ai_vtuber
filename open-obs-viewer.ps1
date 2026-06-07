# open-obs-viewer.ps1 — OBS용 브라우저 소스 URL 안내 및 선택 실행
# 사용법: .\open-obs-viewer.ps1

$Base = "http://localhost:8001/live2d/static/"

$urls = @{
    "1" = @{
        Name = "투명 배경 (OBS 추천)"
        Url  = "${Base}?transparent=1"
    }
    "2" = @{
        Name = "크로마키 (녹색 배경)"
        Url  = "${Base}?chromakey=1"
    }
    "3" = @{
        Name = "mao_pro 모델 + 투명"
        Url  = "${Base}?transparent=1&model=models/mao_pro/runtime/mao_pro.model3.json"
    }
    "4" = @{
        Name = "Haru 모델 + 투명"
        Url  = "${Base}?transparent=1&model=models/Haru/Haru.model3.json"
    }
    "5" = @{
        Name = "개발용 (패널 포함)"
        Url  = $Base
    }
}

Write-Host ""
Write-Host "=== OBS 브라우저 소스 URL ===" -ForegroundColor Cyan
Write-Host ""
foreach ($key in @("1","2","3","4","5")) {
    $item = $urls[$key]
    Write-Host "[$key] $($item.Name)" -ForegroundColor Yellow
    Write-Host "    $($item.Url)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "OBS 설정:" -ForegroundColor Cyan
Write-Host "  1. 소스 추가 -> 브라우저" -ForegroundColor Gray
Write-Host "  2. 위 URL 붙여넣기 (보통 [1] 투명 배경)" -ForegroundColor Gray
Write-Host "  3. 너비 1920 / 높이 1080 권장" -ForegroundColor Gray
Write-Host "  4. [투명 배경] 체크 (transparent=1 사용 시)" -ForegroundColor Gray
Write-Host "  5. FPS 30, 하드웨어 가속 끄기 (깜빡임 시)" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "브라우저에서 미리보기할 번호 (Enter=1, q=종료)"
if ($choice -eq "q") { exit 0 }
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }
if (-not $urls.ContainsKey($choice)) {
    Write-Host "잘못된 선택입니다." -ForegroundColor Yellow
    exit 1
}

Start-Process $urls[$choice].Url
Write-Host "브라우저에서 열었습니다." -ForegroundColor Green
