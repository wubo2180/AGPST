#!/bin/bash
# ============================================================================
# Training Script for All Datasets (Linux/macOS)
# Trains AlternatingSTModel on 6 datasets sequentially
# Usage: bash train_all_datasets.sh
# ============================================================================

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEVICE="cuda"
GPU_ID="0"
WORKERS="8"
SWANLAB_MODE="online"  # Change to "disabled" to disable SwanLab logging
TQDM_MODE="enabled"    # Change to "disabled" to hide progress bars

# Dataset configurations
declare -a DATASETS=(
    "PEMS03"
    "PEMS04"
    "PEMS07"
    "PEMS08"
    "METR-LA"
    "PEMS-BAY"
)

# Print banner
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}    Training Alternating Spatio-Temporal Model on All Datasets${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "Device: ${GREEN}${DEVICE}${NC}"
echo -e "GPU ID: ${GREEN}${GPU_ID}${NC}"
echo -e "Workers: ${GREEN}${WORKERS}${NC}"
echo -e "SwanLab Mode: ${GREEN}${SWANLAB_MODE}${NC}"
echo -e "Datasets: ${GREEN}${DATASETS[@]}${NC}"
echo -e "${BLUE}============================================================================${NC}\n"

# Create results directory
RESULTS_DIR="results/alternating_st_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# Log file
LOG_FILE="${RESULTS_DIR}/training_log.txt"
echo "Training started at $(date)" > "${LOG_FILE}"

# Training loop
TOTAL_DATASETS=${#DATASETS[@]}
SUCCESSFUL=0
FAILED=0

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    CONFIG_FILE="parameters/${DATASET}_alternating.yaml"
    CURRENT=$((i + 1))
    
    echo -e "\n${YELLOW}============================================================================${NC}"
    echo -e "${YELLOW}[${CURRENT}/${TOTAL_DATASETS}] Training on ${DATASET}${NC}"
    echo -e "${YELLOW}============================================================================${NC}"
    echo -e "Config: ${GREEN}${CONFIG_FILE}${NC}"
    echo -e "Start time: ${GREEN}$(date)${NC}\n"
    
    # Check if config file exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}" | tee -a "${LOG_FILE}"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Start training
    START_TIME=$(date +%s)
    
    if CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --config "${CONFIG_FILE}" \
        --device "${DEVICE}" \
        --swanlab_mode "${SWANLAB_MODE}" \
        --tqdm_mode "${TQDM_MODE}" 2>&1 | tee -a "${RESULTS_DIR}/${DATASET}_train.log"; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))
        
        echo -e "\n${GREEN}✅ ${DATASET} training completed!${NC}"
        echo -e "Duration: ${GREEN}${HOURS}h ${MINUTES}m ${SECONDS}s${NC}\n"
        echo "[${CURRENT}/${TOTAL_DATASETS}] ${DATASET}: SUCCESS (${HOURS}h ${MINUTES}m ${SECONDS}s)" >> "${LOG_FILE}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo -e "\n${RED}❌ ${DATASET} training failed!${NC}"
        echo -e "Duration before failure: ${RED}${DURATION}s${NC}\n"
        echo "[${CURRENT}/${TOTAL_DATASETS}] ${DATASET}: FAILED (${DURATION}s)" >> "${LOG_FILE}"
        FAILED=$((FAILED + 1))
    fi
    
    echo -e "${YELLOW}============================================================================${NC}\n"
    
    # Optional: Add a small delay between datasets
    sleep 2
done

# Print summary
echo -e "\n${BLUE}============================================================================${NC}"
echo -e "${BLUE}                        Training Summary${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "Total datasets: ${TOTAL_DATASETS}"
echo -e "${GREEN}Successful: ${SUCCESSFUL}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo -e "Results directory: ${GREEN}${RESULTS_DIR}${NC}"
echo -e "Log file: ${GREEN}${LOG_FILE}${NC}"
echo -e "Finished at: ${GREEN}$(date)${NC}"
echo -e "${BLUE}============================================================================${NC}\n"

# Save summary to file
{
    echo ""
    echo "============================================================================"
    echo "Training Summary"
    echo "============================================================================"
    echo "Total datasets: ${TOTAL_DATASETS}"
    echo "Successful: ${SUCCESSFUL}"
    echo "Failed: ${FAILED}"
    echo "Finished at: $(date)"
    echo "============================================================================"
} >> "${LOG_FILE}"

# Display individual results
echo -e "\n${BLUE}Individual Results:${NC}"
for DATASET in "${DATASETS[@]}"; do
    if [ -f "${RESULTS_DIR}/${DATASET}_train.log" ]; then
        echo -e "\n${YELLOW}--- ${DATASET} ---${NC}"
        # Extract final test results
        grep -E "(Overall - Test MAE|Best validation loss)" "${RESULTS_DIR}/${DATASET}_train.log" | tail -5 || echo "No results found"
    fi
done

# Exit with appropriate code
if [ ${FAILED} -gt 0 ]; then
    echo -e "\n${RED}Some datasets failed to train. Check logs for details.${NC}"
    echo -e "\n${YELLOW}⚠️  Training incomplete. Skipping auto-shutdown.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All datasets trained successfully!${NC}"
    
    # AutoDL auto-shutdown: Wait 5 minutes then shutdown
    echo -e "\n${YELLOW}============================================================================${NC}"
    echo -e "${YELLOW}🔌 AutoDL Auto-Shutdown Initiated${NC}"
    echo -e "${YELLOW}============================================================================${NC}"
    echo -e "${GREEN}All training completed successfully!${NC}"
    echo -e "${YELLOW}System will shutdown in 5 minutes to save costs...${NC}"
    echo -e "${RED}Press Ctrl+C within 5 minutes to cancel shutdown!${NC}"
    echo -e "${YELLOW}============================================================================${NC}\n"
    
    # Countdown
    for i in {300..1}; do
        printf "\r${YELLOW}Shutdown in: %02d:%02d (Press Ctrl+C to cancel)${NC}" $((i/60)) $((i%60))
        sleep 1
    done
    
    echo -e "\n\n${RED}Executing shutdown now...${NC}"
    
    # Use AutoDL shutdown command (best practice with complete path)
    /usr/bin/shutdown
    
    exit 0
fi
