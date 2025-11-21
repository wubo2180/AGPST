#!/bin/bash
# ============================================================================
# Parallel Training Script for 6 Datasets on 6 GPUs
# Each GPU runs one dataset simultaneously
# Hardware: 6x RTX 5090
# Usage: bash train_all_parallel.sh
# ============================================================================

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
WORKERS="8"
SWANLAB_MODE="online"  # SwanLab online mode for experiment tracking
TQDM_MODE="enabled"

# Dataset to GPU mapping (6 datasets on 6 GPUs)
declare -A DATASET_GPU_MAP=(
    ["PEMS03"]="0"
    ["PEMS04"]="1"
    ["PEMS07"]="2"
    ["PEMS08"]="3"
    ["METR-LA"]="4"
    ["PEMS-BAY"]="5"
)

# Get dataset list
DATASETS=(${!DATASET_GPU_MAP[@]})

# Print banner
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}    🚀 Parallel Training on 6x RTX 5090 GPUs${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}Hardware: 6x RTX 5090${NC}"
echo -e "${GREEN}Mode: Parallel Training (Each GPU runs 1 dataset)${NC}"
echo -e "${GREEN}SwanLab: ${SWANLAB_MODE} (Each dataset = 1 experiment)${NC}"
echo -e "${GREEN}Workers per GPU: ${WORKERS}${NC}"
echo ""
echo -e "${CYAN}GPU Assignment:${NC}"
for DATASET in "${DATASETS[@]}"; do
    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
    echo -e "  ${YELLOW}GPU ${GPU_ID}${NC}: ${GREEN}${DATASET}${NC}"
done
echo -e "${BLUE}============================================================================${NC}\n"

# Create results directory
RESULTS_DIR="results/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# Log file
MAIN_LOG="${RESULTS_DIR}/parallel_training.log"
echo "Parallel training started at $(date)" > "${MAIN_LOG}"

# Store PIDs for background processes
declare -A PIDS

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Received interrupt signal. Cleaning up...${NC}"
    for DATASET in "${!PIDS[@]}"; do
        PID=${PIDS[$DATASET]}
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Killing process for ${DATASET} (PID: $PID)${NC}"
            kill $PID 2>/dev/null || true
        fi
    done
    exit 1
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Launch training for each dataset in parallel
echo -e "${MAGENTA}============================================================================${NC}"
echo -e "${MAGENTA}📊 Launching Parallel Training...${NC}"
echo -e "${MAGENTA}============================================================================${NC}\n"

for DATASET in "${DATASETS[@]}"; do
    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
    CONFIG_FILE="parameters/${DATASET}_alternating.yaml"
    DATASET_LOG="${RESULTS_DIR}/${DATASET}_gpu${GPU_ID}.log"
    
    # Check if config file exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}" | tee -a "${MAIN_LOG}"
        continue
    fi
    
    echo -e "${CYAN}[GPU ${GPU_ID}] Starting ${DATASET}${NC}"
    echo -e "  Config: ${CONFIG_FILE}"
    echo -e "  Log: ${DATASET_LOG}"
    echo -e "  SwanLab Experiment: ${GREEN}${DATASET}_AlternatingST_*${NC}\n"
    
    # Launch training in background
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --config "${CONFIG_FILE}" \
        --device cuda \
        --swanlab_mode "${SWANLAB_MODE}" \
        --tqdm_mode "${TQDM_MODE}" \
        > "${DATASET_LOG}" 2>&1 &
    
    # Store PID
    PIDS[$DATASET]=$!
    
    echo "[GPU ${GPU_ID}] ${DATASET}: PID ${PIDS[$DATASET]}" >> "${MAIN_LOG}"
    
    # Small delay to avoid simultaneous startup issues
    sleep 2
done

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}✅ All 6 training processes launched!${NC}"
echo -e "${GREEN}============================================================================${NC}\n"

# Print monitoring information
echo -e "${YELLOW}📊 Monitoring Information:${NC}"
echo -e "  Main log: ${CYAN}${MAIN_LOG}${NC}"
echo -e "  Results directory: ${CYAN}${RESULTS_DIR}${NC}"
echo ""
echo -e "${YELLOW}🔍 Monitor individual dataset progress:${NC}"
for DATASET in "${DATASETS[@]}"; do
    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
    echo -e "  ${GREEN}${DATASET}${NC} (GPU ${GPU_ID}): tail -f ${RESULTS_DIR}/${DATASET}_gpu${GPU_ID}.log"
done
echo ""
echo -e "${YELLOW}⚡ Monitor GPU usage:${NC}"
echo -e "  watch -n 1 nvidia-smi"
echo ""
echo -e "${YELLOW}🛑 To stop all training:${NC}"
echo -e "  Press ${RED}Ctrl+C${NC} or run: ${CYAN}pkill -f 'python main.py'${NC}"
echo -e "${GREEN}============================================================================${NC}\n"

# Wait for all processes and track status
echo -e "${CYAN}Waiting for all training processes to complete...${NC}\n"

declare -A STATUS
declare -A DURATIONS
START_TIME=$(date +%s)

# Monitor processes
while true; do
    RUNNING=0
    for DATASET in "${DATASETS[@]}"; do
        PID=${PIDS[$DATASET]}
        
        if [ -z "${STATUS[$DATASET]}" ]; then
            if kill -0 $PID 2>/dev/null; then
                RUNNING=$((RUNNING + 1))
            else
                # Process finished, check exit code
                wait $PID
                EXIT_CODE=$?
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                DURATIONS[$DATASET]=$DURATION
                
                if [ $EXIT_CODE -eq 0 ]; then
                    STATUS[$DATASET]="SUCCESS"
                    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
                    echo -e "${GREEN}✅ [GPU ${GPU_ID}] ${DATASET} completed successfully!${NC}"
                    echo "[GPU ${GPU_ID}] ${DATASET}: SUCCESS (${DURATION}s)" >> "${MAIN_LOG}"
                else
                    STATUS[$DATASET]="FAILED"
                    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
                    echo -e "${RED}❌ [GPU ${GPU_ID}] ${DATASET} failed with exit code ${EXIT_CODE}${NC}"
                    echo "[GPU ${GPU_ID}] ${DATASET}: FAILED (exit code ${EXIT_CODE}, ${DURATION}s)" >> "${MAIN_LOG}"
                fi
            fi
        fi
    done
    
    # Break if all processes finished
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    # Update status every 10 seconds
    sleep 10
done

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Print summary
echo -e "\n${BLUE}============================================================================${NC}"
echo -e "${BLUE}                    📊 Training Summary${NC}"
echo -e "${BLUE}============================================================================${NC}"

SUCCESSFUL=0
FAILED=0

for DATASET in "${DATASETS[@]}"; do
    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
    if [ "${STATUS[$DATASET]}" == "SUCCESS" ]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
        DURATION=${DURATIONS[$DATASET]}
        DUR_H=$((DURATION / 3600))
        DUR_M=$(((DURATION % 3600) / 60))
        DUR_S=$((DURATION % 60))
        echo -e "${GREEN}✅ [GPU ${GPU_ID}] ${DATASET}: SUCCESS (${DUR_H}h ${DUR_M}m ${DUR_S}s)${NC}"
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}❌ [GPU ${GPU_ID}] ${DATASET}: FAILED${NC}"
    fi
done

echo -e "${BLUE}============================================================================${NC}"
echo -e "Total datasets: ${YELLOW}6${NC}"
echo -e "${GREEN}Successful: ${SUCCESSFUL}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo -e "Total parallel time: ${GREEN}${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo -e "Results directory: ${CYAN}${RESULTS_DIR}${NC}"
echo -e "Main log: ${CYAN}${MAIN_LOG}${NC}"
echo -e "Finished at: ${GREEN}$(date)${NC}"
echo -e "${BLUE}============================================================================${NC}\n"

# Save summary to file
{
    echo ""
    echo "============================================================================"
    echo "Parallel Training Summary"
    echo "============================================================================"
    for DATASET in "${DATASETS[@]}"; do
        GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
        echo "[GPU ${GPU_ID}] ${DATASET}: ${STATUS[$DATASET]}"
    done
    echo "============================================================================"
    echo "Total datasets: 6"
    echo "Successful: ${SUCCESSFUL}"
    echo "Failed: ${FAILED}"
    echo "Total parallel time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Finished at: $(date)"
    echo "============================================================================"
} >> "${MAIN_LOG}"

# Extract and display final results
echo -e "${YELLOW}📈 Extracting Final Test Results...${NC}\n"
for DATASET in "${DATASETS[@]}"; do
    GPU_ID="${DATASET_GPU_MAP[$DATASET]}"
    DATASET_LOG="${RESULTS_DIR}/${DATASET}_gpu${GPU_ID}.log"
    
    if [ -f "${DATASET_LOG}" ] && [ "${STATUS[$DATASET]}" == "SUCCESS" ]; then
        echo -e "${CYAN}--- ${DATASET} (GPU ${GPU_ID}) ---${NC}"
        grep -E "Overall - Test MAE|Best model saved" "${DATASET_LOG}" | tail -3 || echo "No results found"
        echo ""
    fi
done

# AutoDL auto-shutdown if all successful
if [ ${FAILED} -eq 0 ]; then
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}🎉 All datasets trained successfully!${NC}"
    echo -e "${GREEN}============================================================================${NC}\n"
    
    # AutoDL auto-shutdown
    echo -e "${YELLOW}============================================================================${NC}"
    echo -e "${YELLOW}🔌 AutoDL Auto-Shutdown Initiated${NC}"
    echo -e "${YELLOW}============================================================================${NC}"
    echo -e "${GREEN}All 6 datasets completed successfully!${NC}"
    echo -e "${YELLOW}System will shutdown in 5 minutes to save costs...${NC}"
    echo -e "${RED}Press Ctrl+C within 5 minutes to cancel shutdown!${NC}"
    echo -e "${YELLOW}============================================================================${NC}\n"
    
    # Countdown
    for i in {300..1}; do
        printf "\r${YELLOW}Shutdown in: %02d:%02d (Press Ctrl+C to cancel)${NC}" $((i/60)) $((i%60))
        sleep 1
    done
    
    echo -e "\n\n${RED}Executing shutdown now...${NC}"
    /usr/bin/shutdown
    
    exit 0
else
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}⚠️  Some datasets failed. Skipping auto-shutdown.${NC}"
    echo -e "${RED}============================================================================${NC}\n"
    exit 1
fi
