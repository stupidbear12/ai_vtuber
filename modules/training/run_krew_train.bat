@echo off
chcp 65001 > nul
echo ============================================================
echo   시온(sion) KREW 롤플레이 학습 파이프라인
echo ============================================================
echo.

REM ── 설정 ────────────────────────────────────────────────────
set DATA_DIR=.\data_krew
set OUTPUT_DIR=.\output_krew
set EPOCHS=2
set BATCH_SIZE=2
set GRAD_ACCUM=8
set LR=1e-4
set MAX_SEQ_LENGTH=1024
set SAVE_STEPS=200

REM ── Step 1: 데이터 전처리 ──────────────────────────────────
echo [Step 1/3] KREW 데이터 전처리
echo.
python prepare_krew_data.py ^
    --output_dir %DATA_DIR% ^
    --cache_dir %DATA_DIR%\krew_cache

if %errorlevel% neq 0 (
    echo [오류] 데이터 전처리 실패
    pause
    exit /b 1
)
echo.

REM ── Step 2: LoRA 학습 ──────────────────────────────────────
echo [Step 2/3] LoRA 학습 시작
echo   - epochs: %EPOCHS%
echo   - batch: %BATCH_SIZE% x %GRAD_ACCUM% = %BATCH_SIZE% * %GRAD_ACCUM%
echo   - lr: %LR%
echo   - max_seq_length: %MAX_SEQ_LENGTH%
echo.
python train_lora.py ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --grad_accum %GRAD_ACCUM% ^
    --lr %LR% ^
    --max_seq_length %MAX_SEQ_LENGTH% ^
    --save_steps %SAVE_STEPS% ^
    --use_eval

if %errorlevel% neq 0 (
    echo [오류] 학습 실패
    pause
    exit /b 1
)
echo.

REM ── Step 3: Ollama 변환 ────────────────────────────────────
echo [Step 3/3] Ollama 모델 변환 및 등록
echo.
python export_ollama.py ^
    --lora_path %OUTPUT_DIR%\sion_lora ^
    --output_dir %OUTPUT_DIR% ^
    --model_name sion ^
    --quantization q4_k_m

if %errorlevel% neq 0 (
    echo [오류] Ollama 변환 실패
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   파이프라인 완료!
echo   테스트: ollama run sion "시온아 안녕! 오늘 뭐 했어?"
echo   롤백:  ollama cp sion-backup sion
echo ============================================================
pause
