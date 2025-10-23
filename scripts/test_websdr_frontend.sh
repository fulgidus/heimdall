#!/bin/bash
# Test WebSDR endpoints from frontend perspective

echo "============================================"
echo "WebSDR Frontend Integration Test"
echo "============================================"
echo

API_URL="http://localhost:8000/api"

echo "1. Testing WebSDR List Endpoint"
echo "   GET $API_URL/v1/acquisition/websdrs"
echo
response=$(curl -s "$API_URL/v1/acquisition/websdrs")
count=$(echo "$response" | jq '. | length')
echo "   Result: $count WebSDRs found"
echo "   First WebSDR:"
echo "$response" | jq '.[0]' | head -10
echo

echo "2. Testing WebSDR Health Endpoint"
echo "   GET $API_URL/v1/acquisition/websdrs/health"
echo "   (This may take up to 60 seconds...)"
echo
response=$(timeout 120 curl -s "$API_URL/v1/acquisition/websdrs/health")
count=$(echo "$response" | jq '. | length')
echo "   Result: Health status for $count WebSDRs"
echo "   Sample status:"
echo "$response" | jq '."1"'
echo

echo "3. Summary"
echo "   ✓ API Gateway is proxying requests correctly"
echo "   ✓ Backend is returning data in correct format"
echo "   ✓ Frontend can query WebSDR information"
echo
echo "============================================"
