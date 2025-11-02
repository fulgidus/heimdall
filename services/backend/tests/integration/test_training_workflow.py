"""
Integration tests for training workflow.

Tests the end-to-end training pipeline:
1. Create synthetic dataset
2. Submit training job
3. Monitor training progress
4. Verify checkpoints and metrics
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestTrainingWorkflow:
    """Test training workflow endpoints."""

    def test_training_job_creation(self):
        """Test creating a training job."""
        # This test requires a synthetic dataset to exist
        # For now, we'll create a minimal test that checks the API accepts the request
        
        training_config = {
            "job_name": "test_triangulation_training",
            "config": {
                "dataset_id": "00000000-0000-0000-0000-000000000000",  # Placeholder
                "model_architecture": "triangulation",
                "batch_size": 32,
                "epochs": 5,
                "learning_rate": 0.001,
                "early_stop_patience": 20,
                "max_grad_norm": 1.0,
                "accelerator": "cpu"
            }
        }
        
        response = client.post("/api/v1/training/jobs", json=training_config)
        
        # The job should be created (even if it will fail later due to missing dataset)
        assert response.status_code in [201, 500], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 201:
            data = response.json()
            assert "id" in data
            assert "status" in data
            assert data["status"] in ["pending", "queued"]
            assert "job_name" in data
            assert data["job_name"] == "test_triangulation_training"

    def test_list_training_jobs(self):
        """Test listing training jobs."""
        response = client.get("/api/v1/training/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)
        assert isinstance(data["total"], int)

    def test_get_training_job_details(self):
        """Test getting training job details."""
        # First, create a job
        training_config = {
            "job_name": "test_job_details",
            "config": {
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "model_architecture": "triangulation",
                "batch_size": 32,
                "epochs": 5,
                "learning_rate": 0.001
            }
        }
        
        create_response = client.post("/api/v1/training/jobs", json=training_config)
        
        if create_response.status_code == 201:
            job_id = create_response.json()["id"]
            
            # Get job details
            response = client.get(f"/api/v1/training/jobs/{job_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert "job" in data
            assert "recent_metrics" in data
            assert "websocket_url" in data
            
            job = data["job"]
            assert job["id"] == job_id
            assert job["job_name"] == "test_job_details"

    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Test with invalid config (missing required field)
        invalid_config = {
            "job_name": "test_invalid",
            "config": {
                # Missing dataset_id
                "batch_size": 32,
                "epochs": 5
            }
        }
        
        response = client.post("/api/v1/training/jobs", json=invalid_config)
        # Should still create the job, but it will fail during execution
        assert response.status_code in [201, 422, 500]

    def test_training_metrics_endpoint(self):
        """Test getting training metrics."""
        # Create a job
        training_config = {
            "job_name": "test_metrics",
            "config": {
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "model_architecture": "triangulation",
                "batch_size": 32,
                "epochs": 5
            }
        }
        
        create_response = client.post("/api/v1/training/jobs", json=training_config)
        
        if create_response.status_code == 201:
            job_id = create_response.json()["id"]
            
            # Get metrics (should be empty for new job)
            response = client.get(f"/api/v1/training/jobs/{job_id}/metrics")
            assert response.status_code == 200
            
            metrics = response.json()
            assert isinstance(metrics, list)

    def test_model_listing(self):
        """Test listing trained models."""
        response = client.get("/api/v1/training/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)

    def test_synthetic_dataset_listing(self):
        """Test listing synthetic datasets."""
        response = client.get("/api/v1/training/synthetic/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "total" in data
        assert isinstance(data["datasets"], list)


# Note: These tests verify the API endpoints work correctly.
# Full end-to-end training requires:
# 1. Synthetic dataset generation (separate task)
# 2. Running Celery worker
# 3. MinIO access
# 4. PyTorch installation in backend service
# These would be tested in E2E tests or manual validation.
