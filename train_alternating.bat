@echo off
REM ========================================
REM Train Alternating Spatio-Temporal Model
REM ========================================

echo ========================================
echo   Alternating ST Architecture Training
echo ========================================
echo.
echo Configuration: parameters/PEMS03_alternating.yaml
echo Model: AlternatingSTModel
echo Dataset: PEMS03
echo.
echo Features:
echo   - Separate Temporal and Spatial Encoders
echo   - Gated Fusion Mechanism
echo   - 2-Stage Alternating Encoding
echo   - Mixed Precision Training (AMP)
echo   - Denoising Module
echo.
echo Expected improvements:
echo   - Better performance (target MAE ^< 15)
echo   - Faster training (30-50%% speedup with AMP)
echo.
echo ========================================
echo.

python main.py ^
    --cfg parameters/PEMS03_alternating.yaml ^
    --gpus 0 ^
    --workers 8 ^
    --tqdm_mode enabled

echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause
