@echo off
REM SwanLab 训练示例脚本 (Windows)

echo ===================================
echo AGPST 训练 with SwanLab 监控
echo ===================================

REM 检查 SwanLab 安装
echo 检查 SwanLab 安装...
pip show swanlab >nul 2>&1
if errorlevel 1 (
    echo SwanLab 未安装，正在安装...
    pip install swanlab
) else (
    echo SwanLab 已安装 ✓
)

echo.
echo ===================================
echo 实验 1: 单尺度基线
echo ===================================
python main.py ^
    --config parameters/PEMS03.yaml ^
    --pretrain_epochs 50 ^
    --finetune_epochs 50 ^
    --mask_ratio 0.25

echo.
echo ===================================
echo 实验 2: 多尺度 [6, 12, 24]
echo ===================================
python main.py ^
    --config parameters/PEMS03_multiscale.yaml ^
    --pretrain_epochs 50 ^
    --finetune_epochs 50 ^
    --mask_ratio 0.25

echo.
echo ===================================
echo 实验 3: 多尺度 + 高 mask ratio
echo ===================================
python main.py ^
    --config parameters/PEMS03_multiscale.yaml ^
    --pretrain_epochs 50 ^
    --finetune_epochs 50 ^
    --mask_ratio 0.5

echo.
echo ===================================
echo 训练完成！
echo ===================================
echo 查看实验结果：
echo   方式1: swanlab watch
echo   方式2: 访问 https://swanlab.cn
echo ===================================
pause
