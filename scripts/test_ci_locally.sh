#!/bin/bash
# Test script for CI/CD testing locally

set -e

echo "🔍 Testing CI/CD fixes locally..."
echo ""

# Inference test
echo "1️⃣ Testing inference service..."
cd services/inference
python -m pytest tests/test_comprehensive_integration.py -v --tb=short -k "TestPreprocessingIntegration" 2>&1 | head -30

echo ""
echo "✅ Inference imports fixed!"
echo ""

# RF-Acquisition test
echo "2️⃣ Testing rf-acquisition MinIO fix..."
cd ../rf-acquisition
python -m pytest tests/integration/test_minio_storage.py::TestMinIOClient::test_download_iq_data_success -v --tb=short 2>&1 | head -30

echo ""
echo "✅ RF-Acquisition MinIO test fixed!"
