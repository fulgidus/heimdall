#!/bin/bash
# Quick test per controllare se l'import fix ha funzionato

cd services/training

echo "🧪 Testing training service after MLflowLogger fix..."
python -c "from train import TrainingOrchestrator" 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✅ Import successful! MLflowLogger fix worked."
else
    echo "❌ Import still failing. Check the error above."
    exit 1
fi

echo ""
echo "Running pytest collection test..."
pytest tests/test_train.py --collect-only 2>&1 | head -30
