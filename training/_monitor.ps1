# Monitor finetune training - writes status to monitor.txt every 5 minutes
$logFile = "C:\Users\thtgg\workspace2\ai_vtuber\training\finetune_capture.log"
$monFile = "C:\Users\thtgg\workspace2\ai_vtuber\training\monitor.txt"
$checkpointDir = "C:\Users\thtgg\workspace2\ai_vtuber\training\sion-finetuned"

while ($true) {
    $ts = Get-Date -Format "HH:mm:ss"
    $lastLines = Get-Content $logFile -Tail 10 -ErrorAction SilentlyContinue
    $gpu = nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader 2>$null
    $checkpoints = cmd /c dir "$checkpointDir" /b 2>$null

    $status = @"
=== $ts ===
GPU: $gpu
Checkpoints: $checkpoints
Last log:
$($lastLines -join "`n")
"@

    Set-Content $monFile $status -Encoding UTF8
    Start-Sleep 300
}
