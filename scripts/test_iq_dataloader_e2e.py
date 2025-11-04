#!/usr/bin/env python3
"""
End-to-end test for IQ dataloader.

This script:
1. Generates a small IQ dataset (100 samples) if none exists
2. Tests loading with TriangulationIQDataset
3. Verifies tensor shapes and data integrity

Usage:
    python scripts/test_iq_dataloader_e2e.py
"""

import os
import sys
import time
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "training" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "backend" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "common"))

import torch
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from data.triangulation_dataloader import create_iq_dataloader, SPECTROGRAM_CONFIG
from storage.minio_client import MinIOClient

logger = structlog.get_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://heimdall:heimdall_pass@localhost:5432/heimdall"
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_test_dataset(db_session):
    """Get existing IQ dataset or prompt to create one."""
    query = text("""
        SELECT id, name, num_samples
        FROM heimdall.synthetic_datasets 
        WHERE dataset_type = 'iq_raw'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    
    result = db_session.execute(query).fetchone()
    
    if result is None:
        print("❌ No IQ dataset found.")
        print("\nTo generate a test dataset, run:")
        print("\n  curl -X POST http://localhost:8083/api/training/synthetic/generate \\")
        print("       -H 'Content-Type: application/json' \\")
        print("       -d '{")
        print('         "name": "test_iq_small",')
        print('         "description": "Test IQ dataset for dataloader validation",')
        print('         "dataset_type": "iq_raw",')
        print('         "num_samples": 100,')
        print('         "min_receivers_count": 5,')
        print('         "max_receivers_count": 10')
        print("       }'\n")
        return None
    
    dataset_id, name, num_samples = result
    print(f"✓ Found IQ dataset: {name} (ID: {dataset_id}, {num_samples} samples)")
    return str(dataset_id)


def test_dataloader_creation(db_session, minio_client, dataset_id):
    """Test 1: Dataloader creation."""
    print("\n" + "=" * 70)
    print("TEST 1: DATALOADER CREATION")
    print("=" * 70)
    
    try:
        dataloader = create_iq_dataloader(
            dataset_ids=[dataset_id],
            split='train',
            db_session=db_session,
            minio_client=minio_client,
            batch_size=4,
            num_workers=0,  # Avoid multiprocessing for testing
            shuffle=False,
            max_receivers=10,
            use_cache=False
        )
        
        print(f"✓ Dataloader created successfully")
        print(f"  - Total samples: {len(dataloader.dataset)}")
        print(f"  - Batches: {len(dataloader)}")
        print(f"  - Batch size: {dataloader.batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_shapes(db_session, minio_client, dataset_id):
    """Test 2: Batch tensor shapes."""
    print("\n" + "=" * 70)
    print("TEST 2: BATCH TENSOR SHAPES")
    print("=" * 70)
    
    try:
        batch_size = 4
        max_receivers = 10
        
        dataloader = create_iq_dataloader(
            dataset_ids=[dataset_id],
            split='train',
            db_session=db_session,
            minio_client=minio_client,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            max_receivers=max_receivers,
            use_cache=False
        )
        
        # Get first batch
        print("Loading first batch...")
        start = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start
        
        # Check keys
        required_keys = ["iq_spectrograms", "receiver_positions", "signal_mask", "target_position", "metadata"]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing key: {key}")
        
        # Extract tensors
        iq_spectrograms = batch["iq_spectrograms"]
        receiver_positions = batch["receiver_positions"]
        signal_mask = batch["signal_mask"]
        target_position = batch["target_position"]
        
        # Print shapes
        print(f"\n✓ Batch loaded in {load_time:.2f}s")
        print(f"\nTensor shapes:")
        print(f"  - iq_spectrograms:     {tuple(iq_spectrograms.shape)}")
        print(f"  - receiver_positions:  {tuple(receiver_positions.shape)}")
        print(f"  - signal_mask:         {tuple(signal_mask.shape)}")
        print(f"  - target_position:     {tuple(target_position.shape)}")
        
        # Verify shapes
        assert iq_spectrograms.shape[0] == batch_size, "Batch size mismatch"
        assert iq_spectrograms.shape[1] == max_receivers, "Receiver count mismatch"
        assert iq_spectrograms.shape[2] == 2, "Should have 2 channels (real, imag)"
        
        # Frequency bins (129 for onesided STFT with n_fft=256)
        expected_freq_bins = SPECTROGRAM_CONFIG['n_fft'] // 2 + 1
        assert iq_spectrograms.shape[3] == expected_freq_bins, \
            f"Expected {expected_freq_bins} freq bins, got {iq_spectrograms.shape[3]}"
        
        assert iq_spectrograms.shape[4] > 100, "Time bins should be > 100"
        
        assert receiver_positions.shape == (batch_size, max_receivers, 2), "Receiver positions shape wrong"
        assert signal_mask.shape == (batch_size, max_receivers), "Signal mask shape wrong"
        assert target_position.shape == (batch_size, 2), "Target position shape wrong"
        
        print(f"\n✓ All shapes correct!")
        print(f"  - Frequency bins: {iq_spectrograms.shape[3]} (expected: {expected_freq_bins})")
        print(f"  - Time bins: {iq_spectrograms.shape[4]}")
        print(f"  - IQ spectrogram size: ~{iq_spectrograms.numel() * 4 / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_integrity(db_session, minio_client, dataset_id):
    """Test 3: Data integrity checks."""
    print("\n" + "=" * 70)
    print("TEST 3: DATA INTEGRITY")
    print("=" * 70)
    
    try:
        dataloader = create_iq_dataloader(
            dataset_ids=[dataset_id],
            split='train',
            db_session=db_session,
            minio_client=minio_client,
            batch_size=4,
            num_workers=0,
            shuffle=False,
            max_receivers=10,
            use_cache=False
        )
        
        batch = next(iter(dataloader))
        
        iq_spectrograms = batch["iq_spectrograms"]
        receiver_positions = batch["receiver_positions"]
        signal_mask = batch["signal_mask"]
        target_position = batch["target_position"]
        metadata = batch["metadata"]
        
        # Check for NaN/Inf
        assert not torch.isnan(iq_spectrograms).any(), "IQ spectrograms contain NaN"
        assert not torch.isinf(iq_spectrograms).any(), "IQ spectrograms contain Inf"
        assert not torch.isnan(receiver_positions).any(), "Receiver positions contain NaN"
        assert not torch.isnan(target_position).any(), "Target positions contain NaN"
        
        # Check coordinate ranges
        assert (target_position[:, 0] >= -90).all() and (target_position[:, 0] <= 90).all(), "Invalid latitudes"
        assert (target_position[:, 1] >= -180).all() and (target_position[:, 1] <= 180).all(), "Invalid longitudes"
        
        # Check signal mask is boolean
        assert signal_mask.dtype == torch.bool, "Signal mask should be boolean"
        
        # Check metadata
        assert "sample_ids" in metadata, "Missing sample_ids"
        assert "gdop" in metadata, "Missing gdop"
        assert "num_receivers" in metadata, "Missing num_receivers"
        
        print("✓ Data integrity verified")
        print(f"\nMetadata:")
        print(f"  - Sample IDs: {len(metadata['sample_ids'])} samples")
        print(f"  - GDOP range: {metadata['gdop'].min():.2f} - {metadata['gdop'].max():.2f}")
        print(f"  - Num receivers: {metadata['num_receivers'].tolist()}")
        print(f"  - Masked positions per sample: {signal_mask.sum(dim=1).tolist()}")
        
        print(f"\nData statistics:")
        print(f"  - Spectrogram mean: {iq_spectrograms.mean():.4f}")
        print(f"  - Spectrogram std: {iq_spectrograms.std():.4f}")
        print(f"  - Spectrogram min/max: {iq_spectrograms.min():.4f} / {iq_spectrograms.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_performance(db_session, minio_client, dataset_id):
    """Test 4: Cache performance."""
    print("\n" + "=" * 70)
    print("TEST 4: CACHE PERFORMANCE")
    print("=" * 70)
    
    try:
        # First pass (no cache)
        print("First pass (populating cache)...")
        start = time.time()
        
        dataloader1 = create_iq_dataloader(
            dataset_ids=[dataset_id],
            split='train',
            db_session=db_session,
            minio_client=minio_client,
            batch_size=4,
            num_workers=0,
            shuffle=False,
            max_receivers=10,
            use_cache=True
        )
        
        batches_loaded = 0
        for batch in dataloader1:
            batches_loaded += 1
            if batches_loaded >= 5:
                break
        
        time_no_cache = time.time() - start
        print(f"  - Loaded {batches_loaded} batches in {time_no_cache:.2f}s")
        
        # Second pass (with cache)
        print("\nSecond pass (using cache)...")
        start = time.time()
        
        dataloader2 = create_iq_dataloader(
            dataset_ids=[dataset_id],
            split='train',
            db_session=db_session,
            minio_client=minio_client,
            batch_size=4,
            num_workers=0,
            shuffle=False,
            max_receivers=10,
            use_cache=True
        )
        
        batches_loaded = 0
        for batch in dataloader2:
            batches_loaded += 1
            if batches_loaded >= 5:
                break
        
        time_with_cache = time.time() - start
        print(f"  - Loaded {batches_loaded} batches in {time_with_cache:.2f}s")
        
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0
        print(f"\n✓ Cache speedup: {speedup:.1f}x")
        
        if speedup < 2.0:
            print(f"  ⚠️  Warning: Cache speedup only {speedup:.1f}x (expected >2x)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("IQ DATALOADER END-TO-END TEST")
    print("=" * 70)
    
    # Setup database
    print("\nConnecting to database...")
    engine = create_engine(DB_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    print(f"✓ Connected to: {DB_URL}")
    
    # Setup MinIO
    print("\nConnecting to MinIO...")
    minio_client = MinIOClient(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
        bucket_name="heimdall-synthetic-iq"
    )
    minio_client.ensure_bucket_exists()
    print(f"✓ Connected to: {MINIO_ENDPOINT}")
    
    # Get test dataset
    print("\nLooking for test dataset...")
    dataset_id = get_or_create_test_dataset(db_session)
    
    if dataset_id is None:
        print("\n❌ Cannot proceed without a test dataset.")
        return 1
    
    # Run tests
    results = []
    
    results.append(("Dataloader Creation", test_dataloader_creation(db_session, minio_client, dataset_id)))
    results.append(("Batch Tensor Shapes", test_batch_shapes(db_session, minio_client, dataset_id)))
    results.append(("Data Integrity", test_data_integrity(db_session, minio_client, dataset_id)))
    results.append(("Cache Performance", test_cache_performance(db_session, minio_client, dataset_id)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
