@echo off
REM Direct Forecasting训练脚本 - 使用自适应图学习的端到端训练

echo ======================================
echo AGPST Training with Adaptive Graph
echo ======================================

python main.py ^
    --config ./parameters/PEMS03_direct_forecasting.yaml ^
    --device cuda ^
    --mode train ^
    --swanlab_mode online ^
    --test_mode 0

echo.
echo Training completed!
pause
