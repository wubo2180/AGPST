#!/bin/bash
# ============================================================================
# Real-time Monitoring for Parallel Training
# Shows GPU usage and training progress
# Usage: bash monitor_training.sh
# ============================================================================

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Find the latest results directory
RESULTS_DIR=$(ls -td results/parallel_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo -e "${RED}No parallel training results found!${NC}"
    echo -e "${YELLOW}Start training first: bash train_all_parallel.sh${NC}"
    exit 1
fi

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}    📊 Real-time Training Monitor${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "Results directory: ${GREEN}${RESULTS_DIR}${NC}\n"

# Dataset to GPU mapping
declare -A DATASET_GPU_MAP=(
    ["PEMS03"]="0"
    ["PEMS04"]="1"
    ["PEMS07"]="2"
    ["PEMS08"]="3"
    ["METR-LA"]="4"
    ["PEMS-BAY"]="5"
)

# Function to get latest epoch from log
get_latest_epoch() {
    local LOG_FILE=$1
    if [ -f "$LOG_FILE" ]; then
        grep -oP "Epoch \K\d+(?=/\d+)" "$LOG_FILE" | tail -1
    else
        echo "0"
    fi
}

# Function to get latest MAE
get_latest_mae() {
    local LOG_FILE=$1
    if [ -f "$LOG_FILE" ]; then
        grep "Val MAE:" "$LOG_FILE" | tail -1 | grep -oP "Val MAE: \K[\d\.]+"
    else
        echo "N/A"
    fi
}

# Monitor loop
while true; do
    clear
    
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}    📊 Training Progress - $(date +'%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
    
    # Show GPU usage
    echo -e "${CYAN}⚡ GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
    while IFS=, read -r idx name util mem_used mem_total temp; do
        # Color code based on utilization
        if [ $util -gt 80 ]; then
            COLOR=$GREEN
        elif [ $util -gt 50 ]; then
            COLOR=$YELLOW
        else
            COLOR=$RED
        fi
        printf "${COLOR}GPU %s${NC}: %s | Util: %3s%% | Mem: %5s/%5s MB | Temp: %2s°C\n" \
               "$idx" "$(echo $name | cut -c1-15)" "$util" "$mem_used" "$mem_total" "$temp"
    done
    
    echo ""
    echo -e "${CYAN}📈 Training Progress:${NC}"
    printf "${YELLOW}%-12s %-6s %-10s %-15s %-10s${NC}\n" "Dataset" "GPU" "Epoch" "Val MAE" "Status"
    echo "------------------------------------------------------------"
    
    for DATASET in PEMS03 PEMS04 PEMS07 PEMS08 METR-LA PEMS-BAY; do
        GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
        LOG_FILE="${RESULTS_DIR}/${DATASET}_gpu${GPU_ID}.log"
        
        if [ -f "$LOG_FILE" ]; then
            EPOCH=$(get_latest_epoch "$LOG_FILE")
            MAE=$(get_latest_mae "$LOG_FILE")
            
            # Check if process is still running
            if pgrep -f "python main.py.*${DATASET}" > /dev/null; then
                STATUS="${GREEN}Running${NC}"
            elif grep -q "Training completed" "$LOG_FILE" 2>/dev/null; then
                STATUS="${BLUE}Completed${NC}"
            else
                STATUS="${RED}Stopped${NC}"
            fi
            
            printf "%-12s %-6s %-10s %-15s %b\n" "$DATASET" "$GPU_ID" "$EPOCH" "$MAE" "$STATUS"
        else
            printf "%-12s %-6s %-10s %-15s ${RED}No log${NC}\n" "$DATASET" "$GPU_ID" "N/A" "N/A"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  View individual log: ${CYAN}tail -f ${RESULTS_DIR}/PEMS03_gpu0.log${NC}"
    echo -e "  Stop all training:   ${CYAN}bash stop_training.sh${NC}"
    echo -e "  Exit monitor:        ${RED}Ctrl+C${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    
    # Update every 5 seconds
    sleep 5
done
