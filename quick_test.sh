#!/bin/bash
# Quick Start Guide for Data Ingestion Frontend Testing

echo "🚀 Data Ingestion Frontend - Quick Start"
echo "========================================"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."
echo ""

echo "✓ Checking Docker Compose..."
if docker-compose ps > /dev/null 2>&1; then
    echo "  ✅ Docker Compose is running"
else
    echo "  ❌ Docker Compose not running!"
    echo "  Run: docker-compose up -d"
    exit 1
fi

echo ""
echo "✓ Checking services..."

# Check each service
services=(
    "postgres:5432:PostgreSQL"
    "rabbitmq:5672:RabbitMQ"
    "redis:6379:Redis"
    "minio:9000:MinIO"
)

for service in "${services[@]}"; do
    IFS=':' read -r host port name <<< "$service"
    if nc -z localhost $port 2>/dev/null; then
        echo "  ✅ $name is running"
    else
        echo "  ❌ $name is NOT running"
    fi
done

echo ""
echo "========================================"
echo "🧪 Testing Data Ingestion Flow"
echo "========================================"
echo ""

# Step 1: Check Backend
echo "1️⃣  Testing Backend API..."
echo ""

# Create a session
echo "   Creating a session..."
response=$(curl -s -X POST http://localhost:8000/api/sessions/create \
  -H "Content-Type: application/json" \
  -d '{
    "session_name": "Test Session '$(date +%H:%M:%S)'",
    "frequency_mhz": 145.500,
    "duration_seconds": 30
  }')

session_id=$(echo $response | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)

if [ ! -z "$session_id" ]; then
    echo "   ✅ Session created: #$session_id"
    echo ""
    echo "   Full response:"
    echo "   $response" | python -m json.tool 2>/dev/null || echo "   $response"
else
    echo "   ❌ Failed to create session"
    echo "   Response: $response"
    exit 1
fi

echo ""
echo "2️⃣  Checking session status..."
echo ""

# Immediately check status
status=$(curl -s http://localhost:8000/api/sessions/$session_id/status)
echo "   Initial status:"
echo "   $status" | python -m json.tool 2>/dev/null || echo "   $status"

echo ""
echo "3️⃣  Waiting for processing..."
echo ""

# Poll for 3 minutes max
max_attempts=90
attempt=0

while [ $attempt -lt $max_attempts ]; do
    sleep 2
    attempt=$((attempt + 1))
    
    current_status=$(curl -s http://localhost:8000/api/sessions/$session_id/status)
    status_value=$(echo $current_status | grep -o '"status":"[^"]*' | cut -d'"' -f4)
    progress=$(echo $current_status | grep -o '"progress":[0-9]*' | cut -d: -f2)
    
    echo "   [$attempt/90] Status: $status_value | Progress: $progress%"
    
    if [ "$status_value" = "completed" ] || [ "$status_value" = "failed" ]; then
        break
    fi
done

echo ""
echo "4️⃣  Final status:"
echo ""

final_status=$(curl -s http://localhost:8000/api/sessions/$session_id)
echo "   $final_status" | python -m json.tool 2>/dev/null || echo "   $final_status"

echo ""
echo "========================================"
echo "🎯 Results"
echo "========================================"
echo ""

# Extract final status
final_status_value=$(echo $final_status | grep -o '"status":"[^"]*' | cut -d'"' -f4)
minio_path=$(echo $final_status | grep -o '"minio_path":"[^"]*' | cut -d'"' -f4)

if [ "$final_status_value" = "completed" ]; then
    echo "✅ Session completed successfully!"
    echo "   Session ID: $session_id"
    echo "   Status: $final_status_value"
    if [ ! -z "$minio_path" ]; then
        echo "   MinIO Path: $minio_path"
    fi
elif [ "$final_status_value" = "processing" ]; then
    echo "⏳ Session still processing (RF acquisition takes 30-70 seconds)"
    echo "   Check again in a few moments"
else
    echo "❌ Session failed"
    echo "   Status: $final_status_value"
    error=$(echo $final_status | grep -o '"error_message":"[^"]*' | cut -d'"' -f4)
    if [ ! -z "$error" ]; then
        echo "   Error: $error"
    fi
fi

echo ""
echo "========================================"
echo "🖥️  Frontend Access"
echo "========================================"
echo ""
echo "📱 Open your browser and navigate to:"
echo "   http://localhost:5173"
echo ""
echo "Then:"
echo "   1. Click 'Data Ingestion' in the sidebar"
echo "   2. You should see the session in the queue"
echo "   3. Watch it change from PENDING → PROCESSING → COMPLETED"
echo ""

echo "========================================"
echo "📊 Database Check"
echo "========================================"
echo ""
echo "View all sessions in database:"
echo "   docker exec -it heimdall-postgres psql -U heimdall_user -d heimdall"
echo "   SELECT id, session_name, status, created_at FROM recording_sessions ORDER BY created_at DESC;"
echo ""

echo "========================================"
echo "💾 MinIO Check"
echo "========================================"
echo ""
echo "View IQ data files in MinIO:"
echo "   Open: http://localhost:9001"
echo "   Login: minioadmin / minioadmin"
echo "   Navigate: heimdall-raw-iq → sessions"
echo ""

echo "========================================"
echo "✨ All Set!"
echo "========================================"
echo ""
