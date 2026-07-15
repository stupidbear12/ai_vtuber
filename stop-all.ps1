# stop-all.ps1 — ai_vtuber local 5-module stop
# Usage: .\stop-all.ps1  or  stop-all.bat

$Root = $PSScriptRoot
$RunDir = Join-Path $Root ".run"
$pidsFile = Join-Path $RunDir "pids.json"
$ports = @(8000, 8001, 8002, 8003, 8004, 8005, 8007)

function Stop-ProcessTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return }
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { return }
    taskkill /PID $ProcessId /T /F 2>$null | Out-Null
}

function Stop-ByPort {
    param([int]$Port)
    $ownerPids = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $ownerPids) {
        if ($ownerPid -and $ownerPid -gt 0) {
            Write-Host "  port $Port -> PID $ownerPid stopped" -ForegroundColor Gray
            Stop-ProcessTree $ownerPid
        }
    }
}

Write-Host "[STOP] ai_vtuber modules..." -ForegroundColor Cyan

if (Test-Path $pidsFile) {
    try {
        $saved = Get-Content $pidsFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $list = @($saved)
        if ($saved -is [psobject] -and $saved.name) { $list = @($saved) }
        foreach ($entry in $list) {
            if ($entry.pid) {
                Write-Host "  $($entry.name) PID $($entry.pid) stopped" -ForegroundColor Gray
                Stop-ProcessTree ([int]$entry.pid)
            }
        }
    } catch {
        Write-Host "  PID file read failed, stopping by port..." -ForegroundColor Yellow
    }
    Remove-Item $pidsFile -Force -ErrorAction SilentlyContinue
}

foreach ($port in $ports) {
    Stop-ByPort $port
}

Start-Sleep -Milliseconds 500

$remaining = @()
foreach ($port in $ports) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        $remaining += $port
    }
}

if ($remaining.Count -gt 0) {
    Write-Host ""
    Write-Host "[WARN] Ports still open: $($remaining -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "[DONE] All modules stopped." -ForegroundColor Green
}
