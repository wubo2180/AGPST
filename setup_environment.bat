@echo off
REM AGPST 环境快速设置脚本 (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo AGPST 环境设置脚本
echo ========================================
echo.

REM 检查 conda 是否安装
where conda >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda 未安装
    echo 请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✓ 检测到 Conda
echo.

REM 询问用户选择安装方式
echo 请选择安装方式:
echo 1) 使用 environment_agpst.yaml (推荐，包含 PyTorch GPU 支持)
echo 2) 使用 requirements.txt (手动安装 PyTorch)
echo.
set /p choice="请输入选项 [1-2]: "

if "%choice%"=="1" goto env_yaml
if "%choice%"=="2" goto requirements
echo 无效选项
pause
exit /b 1

:env_yaml
echo.
echo ========================================
echo 使用 environment_agpst.yaml 创建环境
echo ========================================

REM 检查环境是否已存在
conda env list | findstr /C:"agpst" >nul 2>&1
if not errorlevel 1 (
    echo.
    set /p rebuild="环境 'agpst' 已存在，是否删除并重建? [y/N]: "
    if /i "!rebuild!"=="y" (
        echo 删除旧环境...
        conda env remove -n agpst -y
    ) else (
        echo 取消安装
        pause
        exit /b 0
    )
)

echo.
echo 创建新环境 (这可能需要几分钟)...
conda env create -f environment_agpst.yaml

echo.
echo ✓ 环境创建成功！
goto verify

:requirements
echo.
echo ========================================
echo 使用 requirements.txt 安装
echo ========================================

REM 创建基础环境
conda env list | findstr /C:"agpst" >nul 2>&1
if not errorlevel 1 (
    echo.
    set /p use_existing="环境 'agpst' 已存在，是否使用现有环境? [Y/n]: "
    if /i "!use_existing!"=="n" (
        echo 删除旧环境...
        conda env remove -n agpst -y
        echo 创建新环境...
        conda create -n agpst python=3.8 -y
    ) else (
        echo 使用现有环境...
    )
) else (
    echo 创建新环境...
    conda create -n agpst python=3.8 -y
)

REM 激活环境
call conda activate agpst

echo.
echo 请选择 PyTorch 安装选项:
echo 1) CUDA 11.7
echo 2) CUDA 11.8
echo 3) CPU only
echo.
set /p cuda_choice="请输入选项 [1-3]: "

if "%cuda_choice%"=="1" (
    echo 安装 PyTorch with CUDA 11.7...
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
) else if "%cuda_choice%"=="2" (
    echo 安装 PyTorch with CUDA 11.8...
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
) else if "%cuda_choice%"=="3" (
    echo 安装 PyTorch (CPU only)...
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
) else (
    echo 无效选项，默认安装 CUDA 11.7
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
)

echo.
echo 安装其他依赖...
pip install -r requirements.txt

echo.
echo ✓ 依赖安装成功！

:verify
echo.
echo ========================================
echo 验证安装
echo ========================================

REM 激活环境并验证
call conda activate agpst

echo.
echo Python 版本:
python --version

echo.
echo PyTorch 信息:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 激活环境:
echo   conda activate agpst
echo.
echo 运行训练:
echo   python main.py --config parameters/PEMS03_multiscale.yaml
echo.
echo 查看 SwanLab 结果:
echo   swanlab watch
echo.
echo ========================================
pause
