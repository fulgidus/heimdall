#!/bin/bash

# Test script to validate the "continue until N valid samples" fix
# This script creates a small dataset generation job and monitors its progress

echo "=== Testing Valid Samples Fix ==="
echo ""
echo "Creating a test job: 50 valid samples with GDOP=150"
echo "With ~20-40% success rate, this should attempt 125-250 samples"
echo ""

# Create a small test job
RESPONSE=$(curl -s -X POST "http://localhost:8001/api/v1/training/jobs/generate-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "valid_samples_test",
    "description": "Test job to verify continue-until-valid-samples fix",
    "num_samples": 50,
    "dataset_type": "feature_based",
    "frequency_mhz": 145.0,
    "tx_power_dbm": 10.0,
    "min_snr_db": 5.0,
    "min_receivers": 3,
    "max_gdop": 150.0,
    "use_random_receivers": false,
    "use_srtm_terrain": true
  }')

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

if [ "$JOB_ID" == "null" ] || [ -z "$JOB_ID" ]; then
  echo "Error: Failed to create job"
  echo "$RESPONSE" | jq '.'
  exit 1
fi

echo "✓ Job created: $JOB_ID"
echo ""
echo "Monitoring progress (Ctrl+C to stop)..."
echo "----------------------------------------"

# Monitor the job
while true; do
  STATUS=$(curl -s "http://localhost:8001/api/v1/training/jobs/$JOB_ID" | jq -r '.')
  
  CURRENT=$(echo "$STATUS" | jq -r '.current // 0')
  TOTAL=$(echo "$STATUS" | jq -r '.total // 0')
  MESSAGE=$(echo "$STATUS" | jq -r '.message // "Processing..."')
  STATUS_STATE=$(echo "$STATUS" | jq -r '.status')
  
  echo -ne "\r[$(date +%H:%M:%S)] $MESSAGE | Status: $STATUS_STATE    "
  
  if [ "$STATUS_STATE" == "completed" ] || [ "$STATUS_STATE" == "failed" ]; then
    echo ""
    echo ""
    echo "=== Job Complete ==="
    echo "$STATUS" | jq '{
      status,
      current,
      total,
      progress_percent,
      message,
      completed_at
    }'
    
    # Check if we got all 50 samples
    if [ "$CURRENT" -ge 50 ]; then
      echo ""
      echo "✓ SUCCESS: Generated $CURRENT valid samples (target was 50)"
      exit 0
    else
      echo ""
      echo "✗ FAILURE: Only generated $CURRENT valid samples (target was 50)"
      exit 1
    fi
  fi
  
  sleep 2
done
