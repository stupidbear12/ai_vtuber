$logFile = "C:\Users\thtgg\workspace2\ai_vtuber\training\finetune_capture.log"
$statusFile = "C:\Users\thtgg\workspace2\ai_vtuber\training\milestone.txt"

while ($true) {
    Start-Sleep 60
    $content = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
    if ($content -match "EXIT CODE") {
        Set-Content $statusFile "DONE at $(Get-Date -Format 'HH:mm:ss')`n$($content | Select-String 'loss' | Select-Object -Last 5 | ForEach-Object { $_.Line })" -Encoding UTF8
        break
    }
    # Find latest step number
    $steps = [regex]::Matches($content, '(\d+)/876')
    if ($steps.Count -gt 0) {
        $lastStep = [int]$steps[$steps.Count-1].Groups[1].Value
        # Find all loss lines
        $lossLines = $content | Select-String "'loss'"
        $lastLoss = if ($lossLines) { ($lossLines | Select-Object -Last 1).Line } else { "none" }
        $gpu = nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>$null
        $checkpoints = cmd /c dir "C:\Users\thtgg\workspace2\ai_vtuber\training\sion-finetuned" /b 2>$null
        
        Set-Content $statusFile "Step: $lastStep/876 at $(Get-Date -Format 'HH:mm:ss')`nGPU: $gpu`nCheckpoints: $checkpoints`nLast loss: $lastLoss" -Encoding UTF8
    }
}
