#!/usr/bin/env python3
"""
Quick Start Guide - MLflow Training Pipeline

Run this file to verify MLflow integration is working correctly.
"""

import sys
from pathlib import Path

# Add services/training to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "training"))

from src.config import settings
from src.mlflow_setup import initialize_mlflow
import structlog

logger = structlog.get_logger(__name__)


def verify_configuration():
    """Verify MLflow configuration is correct."""
    
    logger.info("===== Verifying MLflow Configuration =====")
    
    print(f"\n📋 MLflow Settings:")
    print(f"  Tracking URI: {settings.mlflow_tracking_uri}")
    print(f"  Artifact URI: {settings.mlflow_artifact_uri}")
    print(f"  S3 Endpoint: {settings.mlflow_s3_endpoint_url}")
    print(f"  Experiment Name: {settings.mlflow_experiment_name}")
    print(f"  Backend Store: {settings.mlflow_backend_store_uri}")
    
    return True


def verify_mlflow_initialization():
    """Verify MLflow tracker can be initialized."""
    
    logger.info("===== Testing MLflow Initialization =====")
    
    try:
        tracker = initialize_mlflow(settings)
        
        print(f"\n✅ MLflow Tracker Initialized")
        print(f"  Experiment ID: {tracker.experiment_id}")
        print(f"  Experiment Name: {tracker.experiment_name}")
        
        return True
    
    except Exception as e:
        logger.error(
            "mlflow_initialization_failed",
            error=str(e),
            exc_info=True,
        )
        print(f"\n❌ MLflow Initialization Failed: {e}")
        return False


def test_basic_run():
    """Test starting and ending a run."""
    
    logger.info("===== Testing Basic Run =====")
    
    try:
        tracker = initialize_mlflow(settings)
        
        # Start run
        run_id = tracker.start_run(
            run_name="quickstart-test",
            tags={'test': 'true', 'phase': '5.6'},
        )
        
        print(f"\n✅ Run Started")
        print(f"  Run ID: {run_id}")
        
        # Log some parameters
        tracker.log_params({
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 100,
        })
        
        print(f"✅ Parameters Logged")
        
        # Log some metrics
        tracker.log_metrics({
            'epoch_1_loss': 0.523,
            'epoch_1_mae': 12.3,
        }, step=1)
        
        print(f"✅ Metrics Logged")
        
        # End run
        tracker.end_run("FINISHED")
        
        print(f"✅ Run Finished")
        
        return True
    
    except Exception as e:
        logger.error(
            "basic_run_test_failed",
            error=str(e),
            exc_info=True,
        )
        print(f"\n❌ Basic Run Test Failed: {e}")
        return False


def show_next_steps():
    """Show next steps for using MLflow."""
    
    print("\n" + "="*60)
    print("📚 NEXT STEPS")
    print("="*60)
    
    print("""
1. **View MLflow UI**:
   mlflow ui --backend-store-uri postgresql://heimdall:heimdall@postgres:5432/mlflow_db

2. **Run Training Pipeline**:
   cd services/training
   python train.py \\
     --backbone CONVNEXT_LARGE \\
     --learning-rate 1e-3 \\
     --batch-size 32 \\
     --epochs 100 \\
     --run-name my-experiment

3. **Access MLflow Web Interface**:
   http://localhost:5000

4. **Query Best Run**:
   tracker = initialize_mlflow(settings)
   best_run = tracker.get_best_run(metric="val/loss", compare_fn=min)
   print(best_run['run_id'])

5. **Register Model**:
   version = tracker.register_model(
       model_name="heimdall-localization",
       model_uri=f"runs://{run_id}/model",
       description="RF localization model",
   )

6. **Transition to Production**:
   tracker.transition_model_stage(
       model_name="heimdall-localization",
       version=version,
       stage="Production",
   )
""")


def main():
    """Run all verification tests."""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║     🧠 Heimdall Phase 5.6 - MLflow Tracking Quick Start        ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    results = {
        "Configuration": verify_configuration(),
        "Initialization": verify_mlflow_initialization(),
        "Basic Run": test_basic_run(),
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! MLflow is ready to use.")
        show_next_steps()
        return 0
    else:
        print("\n❌ Some tests failed. Check configuration and try again.")
        return 1


if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit_code = main()
    sys.exit(exit_code)
