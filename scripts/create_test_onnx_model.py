"""
Create a minimal ONNX model for testing export/import functionality.

This script creates a simple dummy ONNX model and registers it in the database
with a proper ONNX location in MinIO. This allows us to test the .heimdall
export/import workflow without needing to run a full training pipeline.
"""

import io
import os
import sys
import uuid
from pathlib import Path

import boto3
import onnx
import psycopg2
import torch
import torch.nn as nn
from onnx import helper, TensorProto

# Database connection
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "heimdall")
DB_USER = os.getenv("POSTGRES_USER", "heimdall_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "heimdall_password")

# MinIO connection
MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "heimdall-models"


def create_simple_onnx_model() -> bytes:
    """Create a minimal ONNX model with the expected LocalizationNet structure."""
    # Input: mel_spectrogram (batch, 3, 128, 32)
    input_tensor = helper.make_tensor_value_info(
        "mel_spectrogram", TensorProto.FLOAT, [None, 3, 128, 32]
    )
    
    # Outputs: positions (batch, 2), uncertainties (batch, 2)
    positions_output = helper.make_tensor_value_info(
        "positions", TensorProto.FLOAT, [None, 2]
    )
    uncertainties_output = helper.make_tensor_value_info(
        "uncertainties", TensorProto.FLOAT, [None, 2]
    )
    
    # Create a simple linear transformation (dummy model)
    # This is just for testing - it doesn't do real localization
    
    # Flatten input
    flatten_node = helper.make_node(
        "Flatten",
        inputs=["mel_spectrogram"],
        outputs=["flattened"],
        axis=1
    )
    
    # Create weight matrices
    weight_pos = helper.make_tensor(
        "weight_positions",
        TensorProto.FLOAT,
        [2, 3 * 128 * 32],  # Output 2 positions
        [0.0001] * (2 * 3 * 128 * 32)
    )
    
    weight_unc = helper.make_tensor(
        "weight_uncertainties",
        TensorProto.FLOAT,
        [2, 3 * 128 * 32],  # Output 2 uncertainties
        [0.0001] * (2 * 3 * 128 * 32)
    )
    
    # MatMul for positions
    matmul_pos = helper.make_node(
        "MatMul",
        inputs=["flattened", "weight_positions"],
        outputs=["positions"]
    )
    
    # MatMul for uncertainties (with abs to ensure positive)
    matmul_unc = helper.make_node(
        "MatMul",
        inputs=["flattened", "weight_uncertainties"],
        outputs=["uncertainties_raw"]
    )
    
    abs_node = helper.make_node(
        "Abs",
        inputs=["uncertainties_raw"],
        outputs=["uncertainties"]
    )
    
    # Create graph
    graph_def = helper.make_graph(
        [flatten_node, matmul_pos, matmul_unc, abs_node],
        "test_localization_net",
        [input_tensor],
        [positions_output, uncertainties_output],
        [weight_pos, weight_unc]
    )
    
    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="heimdall_test",
        opset_imports=[helper.make_opsetid("", 14)]
    )
    
    # Check model
    onnx.checker.check_model(model_def)
    
    # Serialize to bytes
    return model_def.SerializeToString()


def upload_to_minio(onnx_bytes: bytes, object_name: str) -> str:
    """Upload ONNX model to MinIO."""
    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1"
    )
    
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
    except:
        s3_client.create_bucket(Bucket=MINIO_BUCKET)
    
    # Upload
    s3_client.put_object(
        Bucket=MINIO_BUCKET,
        Key=object_name,
        Body=onnx_bytes,
        ContentType="application/octet-stream"
    )
    
    s3_uri = f"s3://{MINIO_BUCKET}/{object_name}"
    print(f"✅ Uploaded ONNX to {s3_uri}")
    return s3_uri


def register_model_in_db(onnx_location: str) -> str:
    """Register the test model in the database with associated training job."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    model_name = f"test_export_model_{uuid.uuid4().hex[:8]}"
    job_name = f"test_training_job_{uuid.uuid4().hex[:8]}"
    
    try:
        with conn.cursor() as cur:
            # First, create a training_job record
            cur.execute("""
                INSERT INTO heimdall.training_jobs (
                    id,
                    job_name,
                    status,
                    config,
                    total_epochs,
                    train_samples,
                    val_samples,
                    current_epoch,
                    train_loss,
                    val_loss,
                    train_accuracy,
                    val_accuracy,
                    learning_rate,
                    model_architecture,
                    job_type,
                    progress_percent,
                    completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                job_id,
                job_name,
                'completed',
                '{"epochs": 50, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam", "loss_function": "gaussian_nll", "validation_split": 0.2}',
                50,  # total_epochs
                1000,  # train_samples
                200,  # val_samples
                50,  # current_epoch (completed all)
                0.0123,  # train_loss
                0.0145,  # val_loss
                100.5,  # train_accuracy
                95.2,  # val_accuracy
                0.001,  # learning_rate
                'LocalizationNet',
                'training',
                100.0,  # progress_percent
                'now()'  # completed_at
            ))
            
            # Now create the model linked to this training job
            cur.execute("""
                INSERT INTO heimdall.models (
                    id,
                    model_name,
                    model_type,
                    onnx_model_location,
                    accuracy_meters,
                    accuracy_sigma_meters,
                    loss_value,
                    epoch,
                    version,
                    hyperparameters,
                    training_metrics,
                    test_metrics,
                    is_active,
                    trained_by_job_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                model_id,
                model_name,
                "LocalizationNet",
                onnx_location,
                100.5,  # Test accuracy
                25.3,   # Test sigma
                0.0123,  # Test loss
                50,  # epoch (final epoch)
                1,  # version
                '{"architecture": "convnext_large", "num_parameters": 200000000, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam", "loss_function": "gaussian_nll"}',
                '{"final_train_loss": 0.0123, "final_val_loss": 0.0145, "train_accuracy": 100.5, "val_accuracy": 95.2}',
                '{"test_loss": 0.0150, "test_accuracy": 100.5}',
                False,
                job_id  # Link to training job
            ))
            conn.commit()
        
        print(f"✅ Registered training job: {job_id}")
        print(f"✅ Registered model in database: {model_id}")
        print(f"   Model name: {model_name}")
        return model_id
        
    finally:
        conn.close()


def main():
    """Main workflow: create ONNX → upload → register."""
    print("Creating test ONNX model for export/import testing...")
    print()
    
    # Step 1: Create ONNX model
    print("1. Creating minimal ONNX model...")
    onnx_bytes = create_simple_onnx_model()
    print(f"   ✅ Created ONNX model ({len(onnx_bytes)} bytes)")
    
    # Step 2: Upload to MinIO
    print("\n2. Uploading to MinIO...")
    object_name = f"test-models/test_export_{uuid.uuid4().hex[:8]}.onnx"
    s3_uri = upload_to_minio(onnx_bytes, object_name)
    
    # Step 3: Register in database
    print("\n3. Registering in database...")
    model_id = register_model_in_db(s3_uri)
    
    print("\n" + "="*60)
    print("✅ Test model created successfully!")
    print("="*60)
    print(f"\nModel ID: {model_id}")
    print(f"ONNX Location: {s3_uri}")
    print("\nYou can now test the export endpoint with:")
    print(f"  curl -o test.heimdall 'http://localhost:8001/api/v1/training/models/{model_id}/export'")
    print()


if __name__ == "__main__":
    main()
