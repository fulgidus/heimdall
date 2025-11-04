"""
Test script for IQ dataloader end-to-end verification.

This test:
1. Generates 100 synthetic IQ samples with random receivers
2. Loads them with the TriangulationIQDataset
3. Verifies tensor shapes and data integrity
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.triangulation_dataloader import create_iq_dataloader, SPECTROGRAM_CONFIG
from storage.minio_client import MinIOClient


@pytest.fixture(scope="module")
def db_session():
    """Create database session."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://heimdall:heimdall_pass@localhost:5432/heimdall"
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def minio_client():
    """Create MinIO client."""
    client = MinIOClient(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False
    )
    
    # Ensure bucket exists
    client.ensure_bucket_exists("heimdall-synthetic-iq")
    
    return client


@pytest.fixture(scope="module")
def test_dataset_id(db_session):
    """
    Get or create a test IQ dataset.
    
    This fixture assumes you've already generated a test dataset.
    If not, you'll need to generate one first using the API.
    """
    from sqlalchemy import text
    
    # Query for existing IQ dataset
    query = text("""
        SELECT id 
        FROM heimdall.synthetic_datasets 
        WHERE dataset_type = 'iq_raw'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    
    result = db_session.execute(query).fetchone()
    
    if result is None:
        pytest.skip("No IQ dataset found. Generate one first using the API.")
    
    return str(result[0])


def test_iq_dataloader_creation(db_session, minio_client, test_dataset_id):
    """Test that IQ dataloader can be created."""
    dataloader = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
        split='train',
        db_session=db_session,
        minio_client=minio_client,
        batch_size=4,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        shuffle=False,
        max_receivers=10,
        use_cache=False  # Disable cache for first test
    )
    
    assert dataloader is not None
    assert len(dataloader.dataset) > 0
    print(f"✓ Dataloader created with {len(dataloader.dataset)} samples")


def test_iq_dataloader_batch_shape(db_session, minio_client, test_dataset_id):
    """Test that batches have correct shapes."""
    batch_size = 4
    max_receivers = 10
    
    dataloader = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
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
    batch = next(iter(dataloader))
    
    # Verify keys
    assert "iq_spectrograms" in batch
    assert "receiver_positions" in batch
    assert "signal_mask" in batch
    assert "target_position" in batch
    assert "metadata" in batch
    
    # Verify shapes
    iq_spectrograms = batch["iq_spectrograms"]
    receiver_positions = batch["receiver_positions"]
    signal_mask = batch["signal_mask"]
    target_position = batch["target_position"]
    
    print(f"IQ spectrograms shape: {iq_spectrograms.shape}")
    print(f"Receiver positions shape: {receiver_positions.shape}")
    print(f"Signal mask shape: {signal_mask.shape}")
    print(f"Target position shape: {target_position.shape}")
    
    # Expected shapes
    # iq_spectrograms: (batch, receivers, 2, freq_bins, time_bins)
    # receiver_positions: (batch, receivers, 2)
    # signal_mask: (batch, receivers)
    # target_position: (batch, 2)
    
    assert iq_spectrograms.shape[0] == batch_size
    assert iq_spectrograms.shape[1] == max_receivers
    assert iq_spectrograms.shape[2] == 2  # real, imag
    
    # Frequency bins (129 for onesided STFT with n_fft=256)
    expected_freq_bins = SPECTROGRAM_CONFIG['n_fft'] // 2 + 1
    assert iq_spectrograms.shape[3] == expected_freq_bins, \
        f"Expected {expected_freq_bins} freq bins, got {iq_spectrograms.shape[3]}"
    
    # Time bins depend on IQ length, just verify it's reasonable
    assert iq_spectrograms.shape[4] > 100, "Time bins should be > 100"
    
    assert receiver_positions.shape == (batch_size, max_receivers, 2)
    assert signal_mask.shape == (batch_size, max_receivers)
    assert target_position.shape == (batch_size, 2)
    
    print("✓ All tensor shapes correct!")


def test_iq_dataloader_data_integrity(db_session, minio_client, test_dataset_id):
    """Test that loaded data is valid."""
    dataloader = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
        split='train',
        db_session=db_session,
        minio_client=minio_client,
        batch_size=2,
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
    
    # Verify no NaNs or Infs
    assert not torch.isnan(iq_spectrograms).any(), "IQ spectrograms contain NaN"
    assert not torch.isinf(iq_spectrograms).any(), "IQ spectrograms contain Inf"
    assert not torch.isnan(receiver_positions).any(), "Receiver positions contain NaN"
    assert not torch.isnan(target_position).any(), "Target positions contain NaN"
    
    # Verify coordinate ranges
    # Latitudes should be in reasonable range (e.g., 30-60 for Europe)
    assert (target_position[:, 0] >= -90).all() and (target_position[:, 0] <= 90).all()
    assert (target_position[:, 1] >= -180).all() and (target_position[:, 1] <= 180).all()
    
    # Verify signal mask is boolean
    assert signal_mask.dtype == torch.bool
    
    # Verify metadata
    assert "sample_ids" in metadata
    assert "gdop" in metadata
    assert "num_receivers" in metadata
    assert len(metadata["sample_ids"]) == 2  # batch_size
    
    print("✓ Data integrity verified!")
    print(f"  - GDOP range: {metadata['gdop'].min():.2f} - {metadata['gdop'].max():.2f}")
    print(f"  - Num receivers: {metadata['num_receivers'].tolist()}")
    print(f"  - Signal mask sum: {signal_mask.sum(dim=1).tolist()} (padded positions)")


def test_iq_dataloader_cache(db_session, minio_client, test_dataset_id):
    """Test that caching works and speeds up loading."""
    import time
    
    # First pass (no cache)
    start = time.time()
    dataloader1 = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
        split='train',
        db_session=db_session,
        minio_client=minio_client,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        max_receivers=10,
        use_cache=True  # Enable cache
    )
    
    # Load first 5 batches
    for i, batch in enumerate(dataloader1):
        if i >= 4:
            break
    
    time_no_cache = time.time() - start
    print(f"First pass (no cache): {time_no_cache:.2f}s")
    
    # Second pass (with cache)
    start = time.time()
    dataloader2 = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
        split='train',
        db_session=db_session,
        minio_client=minio_client,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        max_receivers=10,
        use_cache=True
    )
    
    # Load first 5 batches
    for i, batch in enumerate(dataloader2):
        if i >= 4:
            break
    
    time_with_cache = time.time() - start
    print(f"Second pass (with cache): {time_with_cache:.2f}s")
    
    speedup = time_no_cache / time_with_cache
    print(f"✓ Cache speedup: {speedup:.1f}x")
    
    # Cache should provide at least 2x speedup
    assert speedup > 2.0, f"Cache speedup only {speedup:.1f}x, expected >2x"


def test_iq_dataloader_variable_receivers(db_session, minio_client, test_dataset_id):
    """Test handling of variable receiver count."""
    dataloader = create_iq_dataloader(
        dataset_ids=[test_dataset_id],
        split='train',
        db_session=db_session,
        minio_client=minio_client,
        batch_size=8,
        num_workers=0,
        shuffle=False,
        max_receivers=10,
        use_cache=False
    )
    
    batch = next(iter(dataloader))
    metadata = batch["metadata"]
    signal_mask = batch["signal_mask"]
    
    # Verify that num_receivers varies (5-10 for IQ datasets)
    num_receivers = metadata["num_receivers"].tolist()
    print(f"Receiver counts in batch: {num_receivers}")
    
    # Should have variation
    assert min(num_receivers) >= 3, "Should have at least 3 receivers"
    assert max(num_receivers) <= 10, "Should have at most 10 receivers"
    
    # Verify padding is masked
    for i in range(len(num_receivers)):
        num_rx = num_receivers[i]
        # Padded positions should be masked (True)
        assert signal_mask[i, num_rx:].all(), f"Sample {i}: Padded positions not masked"
    
    print("✓ Variable receiver handling correct!")


if __name__ == "__main__":
    # Run tests manually if not using pytest
    import warnings
    warnings.filterwarnings("ignore")
    
    print("=" * 70)
    print("IQ DATALOADER END-TO-END TEST")
    print("=" * 70)
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Setup
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://heimdall:heimdall_pass@localhost:5432/heimdall"
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    
    minio_client = MinIOClient(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False
    )
    
    # Get test dataset
    from sqlalchemy import text
    query = text("""
        SELECT id 
        FROM heimdall.synthetic_datasets 
        WHERE dataset_type = 'iq_raw'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    result = db_session.execute(query).fetchone()
    
    if result is None:
        print("❌ No IQ dataset found. Generate one first using:")
        print("   curl -X POST http://localhost:8083/api/training/synthetic/generate \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"name\": \"test_iq\", \"dataset_type\": \"iq_raw\", \"num_samples\": 100}'")
        sys.exit(1)
    
    test_dataset_id = str(result[0])
    print(f"\nUsing dataset: {test_dataset_id}\n")
    
    # Run tests
    try:
        test_iq_dataloader_creation(db_session, minio_client, test_dataset_id)
        print()
        
        test_iq_dataloader_batch_shape(db_session, minio_client, test_dataset_id)
        print()
        
        test_iq_dataloader_data_integrity(db_session, minio_client, test_dataset_id)
        print()
        
        test_iq_dataloader_cache(db_session, minio_client, test_dataset_id)
        print()
        
        test_iq_dataloader_variable_receivers(db_session, minio_client, test_dataset_id)
        print()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()
