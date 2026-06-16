# train_pipeline.ps1
# 시온(sion) v2 데이터셋 전체 파이프라인
#   1) QLoRA 파인튜닝 (170,559개, 1 epoch)
#   2) LoRA 병합 (CPU, bf16)
#   3) GGUF 변환 (Q8_0)
#   4) Ollama 등록 (sion:latest + sion-chat)
#
# 실행:
#   cd C:\Users\thtgg\workspace2\ai_vtuber
#   .\modules\chat\scripts\train_pipeline.ps1
#
# 로그: modules\chat\output\pipeline_log.txt

$ErrorActionPreference = "Continue"
$ROOT  = "C:\Users\thtgg\workspace2\ai_vtuber"
$OUT   = "$ROOT\modules\chat\output"
$GGUF  = "$OUT\sion-llama31-q8_0.gguf"
$LOG   = "$OUT\pipeline_log.txt"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts | $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

New-Item -ItemType Directory -Force -Path $OUT | Out-Null
"" | Set-Content -Path $LOG   # 로그 초기화

Set-Location $ROOT

# ─── STEP 1: QLoRA 파인튜닝 ───────────────────────────────────────────
Log "===== STEP 1/4: QLoRA 파인튜닝 시작 ====="
Log "데이터: modules\chat\data\sion_dataset_v2.jsonl (170,559개)"
Log "설정  : 1 epoch / batch 1 / grad_accum 8 / lr 2e-4 / seq 1024"

$t0 = Get-Date
python modules\chat\scripts\finetune.py 2>&1 | Tee-Object -FilePath "$OUT\finetune_log_v2.txt"
if ($LASTEXITCODE -ne 0) { Log "ERROR: 파인튜닝 실패 (exit $LASTEXITCODE)"; exit 1 }

$elapsed = [int]((Get-Date) - $t0).TotalMinutes
Log "===== STEP 1 완료 (${elapsed}분 소요) ====="


# ─── STEP 2: LoRA 병합 ───────────────────────────────────────────────
Log "===== STEP 2/4: LoRA 병합 시작 ====="
Log "RAM 약 16~20 GB 필요 (CPU 병합)"

$t0 = Get-Date
python modules\chat\scripts\merge_and_export.py 2>&1 | Tee-Object -FilePath "$OUT\merge_log.txt"
if ($LASTEXITCODE -ne 0) { Log "ERROR: LoRA 병합 실패 (exit $LASTEXITCODE)"; exit 1 }

$elapsed = [int]((Get-Date) - $t0).TotalMinutes
Log "===== STEP 2 완료 (${elapsed}분 소요) ====="


# ─── STEP 3: GGUF 변환 ───────────────────────────────────────────────
Log "===== STEP 3/4: GGUF 변환 시작 (Q8_0) ====="

$MERGED  = "$OUT\sion-llama31-merged"
$LLAMA   = "$ROOT\llama.cpp\convert_hf_to_gguf.py"

if (-not (Test-Path $MERGED)) {
    Log "ERROR: 병합 모델 없음: $MERGED"; exit 1
}
if (-not (Test-Path $LLAMA)) {
    Log "ERROR: convert_hf_to_gguf.py 없음: $LLAMA"; exit 1
}

$t0 = Get-Date
python $LLAMA $MERGED --outtype q8_0 --outfile $GGUF 2>&1 | Tee-Object -FilePath "$OUT\gguf_log.txt"
if ($LASTEXITCODE -ne 0) { Log "ERROR: GGUF 변환 실패 (exit $LASTEXITCODE)"; exit 1 }

$size = [int]((Get-Item $GGUF).Length / 1GB * 10) / 10
$elapsed = [int]((Get-Date) - $t0).TotalMinutes
Log "===== STEP 3 완료 — $GGUF (${size} GB, ${elapsed}분 소요) ====="


# ─── STEP 4: Ollama 등록 ─────────────────────────────────────────────
Log "===== STEP 4/4: Ollama 등록 시작 ====="

$MF_MAIN = "$ROOT\modules\chat\scripts\Modelfile"
$MF_CHAT = "$ROOT\modules\chat\scripts\Modelfile.chat"

# Modelfile의 FROM 경로를 절대 경로로 치환
$mfContent = Get-Content $MF_MAIN -Raw
$mfContent = $mfContent -replace 'FROM\s+.*\.gguf', "FROM $GGUF"
$mfContent | Set-Content -Path $MF_MAIN -NoNewline

$mfChatContent = Get-Content $MF_CHAT -Raw
$mfChatContent = $mfChatContent -replace 'FROM\s+.*\.gguf', "FROM $GGUF"
$mfChatContent | Set-Content -Path $MF_CHAT -NoNewline

Log "sion:latest 등록 중..."
ollama create sion -f $MF_MAIN
if ($LASTEXITCODE -ne 0) { Log "ERROR: sion:latest 등록 실패"; exit 1 }

Log "sion-chat 등록 중..."
ollama create sion-chat -f $MF_CHAT
if ($LASTEXITCODE -ne 0) { Log "ERROR: sion-chat 등록 실패"; exit 1 }

Log "===== STEP 4 완료 ====="
Log ""
Log "============================================================"
Log "  파인튜닝 파이프라인 완료!"
Log "  GGUF  : $GGUF"
Log "  모델  : sion:latest / sion-chat"
Log "  테스트: ollama run sion '안녕!'"
Log "============================================================"
