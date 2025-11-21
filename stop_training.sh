#!/bin/bash
# ============================================================================
# Stop All Parallel Training Processes
# Usage: bash stop_training.sh
# ============================================================================

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}============================================================================${NC}"
echo -e "${RED}    🛑 Stopping All Training Processes${NC}"
echo -e "${RED}============================================================================${NC}\n"

# Find all Python training processes
PIDS=$(pgrep -f "python main.py" | tr '\n' ' ')

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}No training processes found.${NC}"
else
    echo -e "${YELLOW}Found training processes: ${PIDS}${NC}"
    
    # Ask for confirmation
    read -p "$(echo -e ${RED}Are you sure you want to stop all training? [y/N]: ${NC})" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Kill all training processes
        pkill -f "python main.py"
        
        echo -e "${GREEN}✅ All training processes stopped.${NC}"
        
        # Cancel any scheduled shutdown
        if sudo -n shutdown -c 2>/dev/null; then
            echo -e "${GREEN}✅ Scheduled shutdown cancelled.${NC}"
        fi
    else
        echo -e "${YELLOW}Operation cancelled.${NC}"
        exit 0
    fi
fi

# Show remaining processes
sleep 1
REMAINING=$(pgrep -f "python main.py" | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo -e "\n${GREEN}============================================================================${NC}"
    echo -e "${GREEN}✅ All training processes successfully stopped!${NC}"
    echo -e "${GREEN}============================================================================${NC}\n"
else
    echo -e "\n${RED}============================================================================${NC}"
    echo -e "${RED}⚠️  Some processes may still be running. Force kill?${NC}"
    echo -e "${RED}============================================================================${NC}\n"
    
    read -p "$(echo -e ${RED}Force kill remaining processes? [y/N]: ${NC})" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -9 -f "python main.py"
        echo -e "${GREEN}✅ Force killed all remaining processes.${NC}"
    fi
fi

echo -e "\n${YELLOW}Current GPU status:${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
           --format=csv,noheader | head -6

echo ""
