#!/bin/bash
# ============================================================================
# METR-LA 问题诊断脚本
# 快速测试不同配置找出问题
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}    🔍 METR-LA 问题诊断工具${NC}"
echo -e "${BLUE}============================================================================${NC}\n"

# 创建诊断目录
DIAG_DIR="diagnosis/metr_la_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DIAG_DIR}"

# 基础配置
BASE_CONFIG="parameters/METR-LA_alternating.yaml"
GPU_ID=0

echo -e "${YELLOW}将测试以下配置变化：${NC}"
echo -e "  1. 学习率调整 (0.001 → 0.0005 → 0.002)"
echo -e "  2. Batch size 调整 (32 → 16 → 64)"
echo -e "  3. 模型维度调整 (96 → 64 → 128)"
echo -e "  4. Dropout 调整 (0.1 → 0.05 → 0.2)"
echo -e "  5. 使用 AGPST 原始架构对比\n"

# ============================================================================
# 测试 1: 学习率调整
# ============================================================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}测试 1: 降低学习率 (0.001 → 0.0005)${NC}"
echo -e "${CYAN}原因: METR-LA 可能需要更细致的优化${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# 创建临时配置
TEST_CONFIG="${DIAG_DIR}/test1_lr0.0005.yaml"
cp "${BASE_CONFIG}" "${TEST_CONFIG}"
sed -i 's/lr: 0.001/lr: 0.0005/g' "${TEST_CONFIG}"

echo -e "${YELLOW}运行 10 epochs 快速测试...${NC}"
# 修改 epochs 为 10 用于快速测试
sed -i 's/epochs: 100/epochs: 10/g' "${TEST_CONFIG}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --config "${TEST_CONFIG}" \
    --device cuda \
    --swanlab_mode disabled \
    --tqdm_mode enabled \
    > "${DIAG_DIR}/test1_lr0.0005.log" 2>&1 &

PID1=$!
echo -e "${GREEN}✓ 测试 1 已启动 (PID: $PID1)${NC}\n"
sleep 2

# ============================================================================
# 测试 2: 增大学习率
# ============================================================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}测试 2: 增大学习率 (0.001 → 0.002)${NC}"
echo -e "${CYAN}原因: 检查是否学习率过小导致收敛慢${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

TEST_CONFIG="${DIAG_DIR}/test2_lr0.002.yaml"
cp "${BASE_CONFIG}" "${TEST_CONFIG}"
sed -i 's/lr: 0.001/lr: 0.002/g' "${TEST_CONFIG}"
sed -i 's/epochs: 100/epochs: 10/g' "${TEST_CONFIG}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --config "${TEST_CONFIG}" \
    --device cuda \
    --swanlab_mode disabled \
    --tqdm_mode enabled \
    > "${DIAG_DIR}/test2_lr0.002.log" 2>&1 &

PID2=$!
echo -e "${GREEN}✓ 测试 2 已启动 (PID: $PID2)${NC}\n"
sleep 2

# ============================================================================
# 测试 3: 减小 Batch Size
# ============================================================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}测试 3: 减小 batch size (32 → 16)${NC}"
echo -e "${CYAN}原因: 小 batch 可能提供更好的梯度估计${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

TEST_CONFIG="${DIAG_DIR}/test3_bs16.yaml"
cp "${BASE_CONFIG}" "${TEST_CONFIG}"
sed -i 's/batch_size: 32/batch_size: 16/g' "${TEST_CONFIG}"
sed -i 's/epochs: 100/epochs: 10/g' "${TEST_CONFIG}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --config "${TEST_CONFIG}" \
    --device cuda \
    --swanlab_mode disabled \
    --tqdm_mode enabled \
    > "${DIAG_DIR}/test3_bs16.log" 2>&1 &

PID3=$!
echo -e "${GREEN}✓ 测试 3 已启动 (PID: $PID3)${NC}\n"
sleep 2

# ============================================================================
# 测试 4: 使用 AGPST 原始架构
# ============================================================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}测试 4: 使用 AGPST 原始架构${NC}"
echo -e "${CYAN}原因: 验证是否 AlternatingSTModel 不适合 METR-LA${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

TEST_CONFIG="${DIAG_DIR}/test4_agpst.yaml"
cp "parameters/METR-LA.yaml" "${TEST_CONFIG}" 2>/dev/null || {
    echo -e "${RED}METR-LA.yaml not found, skipping AGPST test${NC}"
    PID4=""
}

if [ -f "${TEST_CONFIG}" ]; then
    sed -i 's/epochs: 100/epochs: 10/g' "${TEST_CONFIG}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --config "${TEST_CONFIG}" \
        --device cuda \
        --swanlab_mode disabled \
        --tqdm_mode enabled \
        > "${DIAG_DIR}/test4_agpst.log" 2>&1 &
    
    PID4=$!
    echo -e "${GREEN}✓ 测试 4 已启动 (PID: $PID4)${NC}\n"
fi

# ============================================================================
# 等待所有测试完成
# ============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}等待所有测试完成...${NC}"
echo -e "${YELLOW}预计时间: ~10-20 分钟 (10 epochs)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# 等待所有进程
wait $PID1 2>/dev/null
echo -e "${GREEN}✓ 测试 1 完成${NC}"

wait $PID2 2>/dev/null
echo -e "${GREEN}✓ 测试 2 完成${NC}"

wait $PID3 2>/dev/null
echo -e "${GREEN}✓ 测试 3 完成${NC}"

if [ ! -z "$PID4" ]; then
    wait $PID4 2>/dev/null
    echo -e "${GREEN}✓ 测试 4 完成${NC}"
fi

# ============================================================================
# 分析结果
# ============================================================================
echo -e "\n${BLUE}============================================================================${NC}"
echo -e "${BLUE}    📊 诊断结果分析${NC}"
echo -e "${BLUE}============================================================================${NC}\n"

echo -e "${CYAN}最终 Validation MAE 对比:${NC}\n"

extract_mae() {
    local LOG_FILE=$1
    local TEST_NAME=$2
    
    if [ -f "$LOG_FILE" ]; then
        MAE=$(grep "Val MAE:" "$LOG_FILE" | tail -1 | grep -oP "Val MAE: \K[\d\.]+")
        if [ ! -z "$MAE" ]; then
            echo -e "${GREEN}${TEST_NAME}${NC}: MAE = ${YELLOW}${MAE}${NC}"
        else
            echo -e "${RED}${TEST_NAME}${NC}: No MAE found"
        fi
    else
        echo -e "${RED}${TEST_NAME}${NC}: Log file not found"
    fi
}

extract_mae "${DIAG_DIR}/test1_lr0.0005.log" "测试 1 (lr=0.0005)"
extract_mae "${DIAG_DIR}/test2_lr0.002.log" "测试 2 (lr=0.002) "
extract_mae "${DIAG_DIR}/test3_bs16.log" "测试 3 (bs=16)    "
extract_mae "${DIAG_DIR}/test4_agpst.log" "测试 4 (AGPST)   "

echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📋 建议:${NC}"
echo -e "  1. 选择 MAE 最低的配置进行完整训练 (100 epochs)"
echo -e "  2. 查看各测试的详细日志: ${CYAN}${DIAG_DIR}/${NC}"
echo -e "  3. 如果所有配置都不理想,考虑 Phase 2/3 优化"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${BLUE}诊断完成! 结果保存在: ${GREEN}${DIAG_DIR}/${NC}\n"
