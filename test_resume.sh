#!/bin/bash
# Test resume endpoint
JOB_ID="c930e109-db84-4629-886f-56db1bae7042"
API_URL="http://localhost/api/v1/training/jobs/${JOB_ID}/resume"

echo "Testing resume endpoint for job ${JOB_ID}..."
echo "URL: ${API_URL}"
echo ""
echo "1. Checking current job status..."
curl -s "http://localhost/api/v1/training/jobs/${JOB_ID}" | python3 -m json.tool | grep -E "status|pause_checkpoint"
echo ""
echo "2. Attempting to resume job..."
curl -X POST -s "${API_URL}"
echo ""
