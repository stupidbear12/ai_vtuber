# start-all.ps1 — ai_vtuber local 5-module launcher
# Usage: .\start-all.ps1  or  start-all.bat

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$RunDir = Join-Path $Root ".run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Import-RootEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -match '^\s*([^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Get-PythonExe {
    if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) {
        return (Join-Path $Root ".venv\Scripts\python.exe")
    }
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { return "py" }
    throw "Python not found. Install python or create .venv"
}

function Test-PortInUse {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$conn
}

function Wait-Health {
    param([int]$Port, [string]$Name, [int]$TimeoutSec = 30)
    $url = "http://127.0.0.1:$Port/health"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $res = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($res.StatusCode -eq 200) {
                Write-Host "  OK  $Name (:$Port)" -ForegroundColor Green
                return $true
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "  WARN $Name (:$Port) health check failed - see logs_*.txt" -ForegroundColor Yellow
    return $false
}

Import-RootEnv (Join-Path $Root ".env")

$python = Get-PythonExe
$pyArgsPrefix = @()
if ($python -eq "py") { $pyArgsPrefix = @("-3") }

$modules = @(
    @{ Name = "chat";      Port = 8002; Dir = "modules\chat" },
    @{ Name = "live2d";    Port = 8001; Dir = "modules\live2d" },
    @{ Name = "broadcast"; Port = 8003; Dir = "modules\broadcast" },
    @{ Name = "voice";     Port = 8004; Dir = "modules\voice" },
    @{ Name = "core";      Port = 8000; Dir = "modules\core" }
)

$busy = @($modules | Where-Object { Test-PortInUse $_.Port })
if ($busy.Count -gt 0) {
    Write-Host "[INFO] Modules already running. Restarting..." -ForegroundColor Yellow
    $busy | ForEach-Object { Write-Host "  - $($_.Name) :$($_.Port)" }
    & (Join-Path $Root "stop-all.ps1")
    Start-Sleep -Seconds 2
    $stillBusy = @($modules | Where-Object { Test-PortInUse $_.Port })
    if ($stillBusy.Count -gt 0) {
        Write-Host "[ERROR] Ports still in use. Run stop-all.bat again." -ForegroundColor Red
        $stillBusy | ForEach-Object { Write-Host "  - $($_.Name) :$($_.Port)" }
        exit 1
    }
}

Write-Host "[START] ai_vtuber modules..." -ForegroundColor Cyan
$pids = @()

foreach ($mod in $modules) {
    $workDir = Join-Path $Root $mod.Dir
    $stdout = Join-Path $Root "logs_$($mod.Name).txt"
    $stderr = Join-Path $Root "logs_$($mod.Name)_err.txt"

    $args = $pyArgsPrefix + @(
        "-m", "uvicorn", "app.main:app",
        "--reload", "--host", "0.0.0.0", "--port", "$($mod.Port)"
    )

    Write-Host "  starting $($mod.Name) (:$($mod.Port))" -ForegroundColor Gray

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $args `
        -WorkingDirectory $workDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden

    $pids += [ordered]@{
        name = $mod.Name
        port = $mod.Port
        pid  = $proc.Id
        dir  = $mod.Dir
    }

    Start-Sleep -Seconds 1
}

$pids | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $RunDir "pids.json") -Encoding UTF8

Write-Host ""
Write-Host "[HEALTH CHECK]" -ForegroundColor Cyan
foreach ($mod in $modules) {
    Wait-Health -Port $mod.Port -Name $mod.Name | Out-Null
}

Write-Host ""
Write-Host "[DONE] All modules started." -ForegroundColor Green
Write-Host "  Dashboard  http://localhost:8000"
Write-Host "  Live2D     http://localhost:8001/live2d/static/"
Write-Host "  Status     http://localhost:8000/status"
Write-Host "  Logs       logs_<module>.txt"
Write-Host "  Stop       stop-all.bat"
