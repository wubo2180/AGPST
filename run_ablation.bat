@echo off
REM ========================================
REM 消融实验批量运行脚本
REM ========================================

setlocal enabledelayedexpansion

set EPOCHS=100
set DATASET=PEMS03

echo.
echo ========================================
echo 开始消融实验 - %DATASET%
echo 训练轮数: %EPOCHS%
echo ========================================
echo.

REM 记录开始时间
set START_TIME=%time%

REM ----------------------------------------
REM 实验 1: 完整模型 (Baseline)
REM ----------------------------------------
echo [1/6] 运行完整模型 (Baseline)...
echo 配置: 时间编码器 ✓  空间编码器 (Hybrid) ✓  Stage 2 ✓  去噪 ✓
python main.py --cfg parameters/ablation/full_model.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo ERROR: 完整模型训练失败!
    exit /b 1
)
echo [1/6] 完成!
echo.

REM ----------------------------------------
REM 实验 2: 无时间编码器
REM ----------------------------------------
echo [2/6] 运行无时间编码器实验...
echo 配置: 时间编码器 ✗  空间编码器 (Hybrid) ✓  Stage 2 ✓  去噪 ✓
python main.py --cfg parameters/ablation/wo_temporal.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo WARNING: 无时间编码器实验失败,继续...
)
echo [2/6] 完成!
echo.

REM ----------------------------------------
REM 实验 3: 无空间编码器
REM ----------------------------------------
echo [3/6] 运行无空间编码器实验...
echo 配置: 时间编码器 ✓  空间编码器 ✗  Stage 2 ✓  去噪 ✓
python main.py --cfg parameters/ablation/wo_spatial.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo WARNING: 无空间编码器实验失败,继续...
)
echo [3/6] 完成!
echo.

REM ----------------------------------------
REM 实验 4: 无第二阶段
REM ----------------------------------------
echo [4/6] 运行无第二阶段实验...
echo 配置: 时间编码器 ✓  空间编码器 (Hybrid) ✓  Stage 2 ✗  去噪 ✓
python main.py --cfg parameters/ablation/wo_stage2.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo WARNING: 无第二阶段实验失败,继续...
)
echo [4/6] 完成!
echo.

REM ----------------------------------------
REM 实验 5: 仅嵌入层
REM ----------------------------------------
echo [5/6] 运行仅嵌入层实验...
echo 配置: 时间编码器 ✗  空间编码器 ✗  Stage 2 ✗  去噪 ✓
python main.py --cfg parameters/ablation/embedding_only.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo WARNING: 仅嵌入层实验失败,继续...
)
echo [5/6] 完成!
echo.

REM ----------------------------------------
REM 实验 6: 无去噪模块
REM ----------------------------------------
echo [6/6] 运行无去噪模块实验...
echo 配置: 时间编码器 ✓  空间编码器 (Hybrid) ✓  Stage 2 ✓  去噪 ✗
python main.py --cfg parameters/ablation/wo_denoising.yaml --epochs %EPOCHS%
if %ERRORLEVEL% neq 0 (
    echo WARNING: 无去噪模块实验失败,继续...
)
echo [6/6] 完成!
echo.

REM 记录结束时间
set END_TIME=%time%

echo.
echo ========================================
echo 所有消融实验完成!
echo ========================================
echo 开始时间: %START_TIME%
echo 结束时间: %END_TIME%
echo.
echo 实验结果保存在: checkpoints/%DATASET%/ablation/
echo.
echo 下一步:
echo 1. 查看训练日志分析结果
echo 2. 运行结果可视化脚本: python analyze_ablation.py
echo 3. 生成论文表格和图表
echo ========================================

endlocal
