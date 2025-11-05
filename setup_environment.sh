#!/bin/bash
# AGPST 环境快速设置脚本 (Linux/Mac)

set -e

echo "========================================"
echo "AGPST 环境设置脚本"
echo "========================================"
echo ""

# 检查 conda 是否安装
if ! command -v conda &> /dev/null
then
    echo "❌ Conda 未安装"
    echo "请先安装 Anaconda 或 Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ 检测到 Conda"
echo ""

# 询问用户选择安装方式
echo "请选择安装方式:"
echo "1) 使用 environment_agpst.yaml (推荐，包含 PyTorch GPU 支持)"
echo "2) 使用 requirements.txt (手动安装 PyTorch)"
echo ""
read -p "请输入选项 [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "========================================"
        echo "使用 environment_agpst.yaml 创建环境"
        echo "========================================"
        
        # 检查环境是否已存在
        if conda env list | grep -q "^agpst "; then
            echo ""
            read -p "环境 'agpst' 已存在，是否删除并重建? [y/N]: " rebuild
            if [[ $rebuild =~ ^[Yy]$ ]]; then
                echo "删除旧环境..."
                conda env remove -n agpst -y
            else
                echo "取消安装"
                exit 0
            fi
        fi
        
        echo ""
        echo "创建新环境 (这可能需要几分钟)..."
        conda env create -f environment_agpst.yaml
        
        echo ""
        echo "✓ 环境创建成功！"
        ;;
        
    2)
        echo ""
        echo "========================================"
        echo "使用 requirements.txt 安装"
        echo "========================================"
        
        # 创建基础环境
        if conda env list | grep -q "^agpst "; then
            echo ""
            read -p "环境 'agpst' 已存在，是否使用现有环境? [Y/n]: " use_existing
            if [[ ! $use_existing =~ ^[Nn]$ ]]; then
                echo "使用现有环境..."
            else
                echo "删除旧环境..."
                conda env remove -n agpst -y
                echo "创建新环境..."
                conda create -n agpst python=3.8 -y
            fi
        else
            echo "创建新环境..."
            conda create -n agpst python=3.8 -y
        fi
        
        # 激活环境
        eval "$(conda shell.bash hook)"
        conda activate agpst
        
        echo ""
        echo "请选择 PyTorch 安装选项:"
        echo "1) CUDA 11.7"
        echo "2) CUDA 11.8"
        echo "3) CPU only"
        echo ""
        read -p "请输入选项 [1-3]: " cuda_choice
        
        case $cuda_choice in
            1)
                echo "安装 PyTorch with CUDA 11.7..."
                conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
                ;;
            2)
                echo "安装 PyTorch with CUDA 11.8..."
                conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
                ;;
            3)
                echo "安装 PyTorch (CPU only)..."
                conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
                ;;
            *)
                echo "无效选项，默认安装 CUDA 11.7"
                conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
                ;;
        esac
        
        echo ""
        echo "安装其他依赖..."
        pip install -r requirements.txt
        
        echo ""
        echo "✓ 依赖安装成功！"
        ;;
        
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "验证安装"
echo "========================================"

# 激活环境并验证
eval "$(conda shell.bash hook)"
conda activate agpst

echo ""
echo "Python 版本:"
python --version

echo ""
echo "PyTorch 信息:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "激活环境:"
echo "  conda activate agpst"
echo ""
echo "运行训练:"
echo "  python main.py --config parameters/PEMS03_multiscale.yaml"
echo ""
echo "查看 SwanLab 结果:"
echo "  swanlab watch"
echo ""
echo "========================================"
