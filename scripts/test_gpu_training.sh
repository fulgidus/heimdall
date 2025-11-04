#!/bin/bash
# GPU Training Optimization Test Script
# Tests GPU-cached dataset with RTX 3090

set -e

echo "🚀 GPU Training Optimization Test"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if backend is running
echo -e "${YELLOW}1. Checking backend service...${NC}"
if ! docker compose ps backend | grep -q "running"; then
    echo -e "${RED}❌ Backend not running. Starting...${NC}"
    docker compose up -d backend
    sleep 5
fi
echo -e "${GREEN}✅ Backend running${NC}"
echo ""

# Check if training service is running
echo -e "${YELLOW}2. Checking training service...${NC}"
if ! docker compose ps training | grep -q "running"; then
    echo -e "${RED}❌ Training not running. Starting...${NC}"
    docker compose up -d training
    sleep 10
fi
echo -e "${GREEN}✅ Training service running${NC}"
echo ""

# Check GPU availability
echo -e "${YELLOW}3. Checking GPU availability...${NC}"
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}❌ NVIDIA GPU not detected!${NC}"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}✅ GPU detected${NC}"
echo ""

# Get dataset ID
echo -e "${YELLOW}4. Fetching available datasets...${NC}"
DATASET_ID=$(docker compose exec -T postgres psql -U heimdall_user -d heimdall -t -c \
    "SELECT id FROM heimdall.synthetic_datasets ORDER BY num_samples DESC LIMIT 1;" | tr -d ' \n')

if [ -z "$DATASET_ID" ]; then
    echo -e "${RED}❌ No datasets found. Please create a synthetic dataset first.${NC}"
    exit 1
fi

DATASET_INFO=$(docker compose exec -T postgres psql -U heimdall_user -d heimdall -t -c \
    "SELECT name, num_samples FROM heimdall.synthetic_datasets WHERE id='$DATASET_ID';" | tr -s ' ')
echo "Selected dataset: $DATASET_INFO"
echo -e "${GREEN}✅ Dataset found: $DATASET_ID${NC}"
echo ""

# Create training job
echo -e "${YELLOW}5. Creating GPU-cached training job...${NC}"
JOB_RESPONSE=$(curl -s -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"job_name\": \"GPU Cache Test - $(date +%Y%m%d_%H%M%S)\",
    \"config\": {
      \"dataset_ids\": [\"$DATASET_ID\"],
      \"batch_size\": 256,
      \"epochs\": 1000,
      \"learning_rate\": 0.001,
      \"preload_to_gpu\": true,
      \"accelerator\": \"auto\",
      \"early_stop_patience\": 10
    }
  }")

JOB_ID=$(echo $JOB_RESPONSE | grep -o '"id":"[^"]*' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}❌ Failed to create job${NC}"
    echo "Response: $JOB_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✅ Job created: $JOB_ID${NC}"
echo ""

# Monitor GPU utilization
echo -e "${YELLOW}6. Monitoring training...${NC}"
echo "Job ID: $JOB_ID"
echo ""
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "=========================================="
echo ""

# Function to check job status
check_status() {
    STATUS=$(docker compose exec -T postgres psql -U heimdall_user -d heimdall -t -c \
        "SELECT status FROM heimdall.training_jobs WHERE id='$JOB_ID';" | tr -d ' \n')
    echo "$STATUS"
}

# Function to get job progress
get_progress() {
    docker compose exec -T postgres psql -U heimdall_user -d heimdall -t -c \
        "SELECT current_epoch, progress_percent, val_loss, val_accuracy 
         FROM heimdall.training_jobs WHERE id='$JOB_ID';"
}

# Monitor loop
COUNTER=0
while true; do
    STATUS=$(check_status)
    
    case "$STATUS" in
        "completed")
            echo -e "${GREEN}✅ Training completed!${NC}"
            get_progress
            break
            ;;
        "failed")
            echo -e "${RED}❌ Training failed!${NC}"
            docker compose logs --tail=50 training
            exit 1
            ;;
        "running")
            if [ $((COUNTER % 5)) -eq 0 ]; then
                echo -e "${YELLOW}[$(date +%H:%M:%S)] Training in progress...${NC}"
                get_progress
                
                # Show GPU stats
                nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
                    --format=csv,noheader,nounits | \
                    awk -F', ' '{printf "GPU: %s%% | VRAM: %s%% (%s MB) | Power: %s W\n", $1, $2, $3, $4}'
                echo ""
            fi
            ;;
        "pending"|"queued")
            echo -e "${YELLOW}[$(date +%H:%M:%S)] Job queued, waiting to start...${NC}"
            if [ $COUNTER -eq 0 ]; then
                echo "Tailing training logs..."
                docker compose logs --tail=20 training
                echo ""
            fi
            ;;
    esac
    
    sleep 2
    COUNTER=$((COUNTER + 1))
    
    # Safety timeout (10 minutes)
    if [ $COUNTER -gt 300 ]; then
        echo -e "${RED}⚠️  Timeout reached (10 minutes)${NC}"
        break
    fi
done

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Test Complete!${NC}"
echo ""
echo "View logs:"
echo "  docker compose logs -f training"
echo ""
echo "Check job status:"
echo "  curl http://localhost:8001/api/v1/training/jobs/$JOB_ID"
echo ""
echo "Monitor GPU:"
echo "  watch -n 1 nvidia-smi"
