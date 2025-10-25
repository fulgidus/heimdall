#!/bin/bash
# Test script to verify WebSDR API endpoints

set -e

API_GATEWAY_URL="${API_GATEWAY_URL:-http://localhost:8000}"
RF_ACQUISITION_URL="${RF_ACQUISITION_URL:-http://localhost:8001}"

echo "=========================================="
echo "Testing WebSDR API Endpoints"
echo "=========================================="
echo ""

# Test 1: API Gateway Health
echo "Test 1: API Gateway Health Check"
echo "---"
curl -s "${API_GATEWAY_URL}/health" | jq '.' || echo "FAILED: API Gateway not responding"
echo ""
echo ""

# Test 2: RF Acquisition Health
echo "Test 2: RF Acquisition Service Health Check"
echo "---"
curl -s "${RF_ACQUISITION_URL}/health" | jq '.' || echo "FAILED: RF Acquisition not responding"
echo ""
echo ""

# Test 3: Get WebSDRs List
echo "Test 3: Get WebSDRs Configuration"
echo "---"
curl -s "${API_GATEWAY_URL}/api/v1/acquisition/websdrs" | jq '.' || echo "FAILED: Cannot fetch WebSDRs"
echo ""
echo ""

# Test 4: Check WebSDR Health (this may take time)
echo "Test 4: Check WebSDR Health Status (may take 30-60 seconds...)"
echo "---"
curl -s -m 90 "${API_GATEWAY_URL}/api/v1/acquisition/websdrs/health" | jq '.' || echo "FAILED: Health check failed or timed out"
echo ""
echo ""

# Test 5: Get Service Configuration
echo "Test 5: Get Service Configuration"
echo "---"
curl -s "${API_GATEWAY_URL}/api/v1/acquisition/config" | jq '.' || echo "FAILED: Cannot fetch config"
echo ""
echo ""

echo "=========================================="
echo "Testing Complete"
echo "=========================================="
echo ""
echo "If all tests passed, the backend is ready for frontend integration."
echo "You can now test the WebSDR management page at:"
echo "  http://localhost:3001/websdrs"
echo ""
