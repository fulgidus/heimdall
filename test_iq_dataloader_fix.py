"""
Test script to verify IQ dataloader now includes z-score standardization parameters.

This script creates an IQ dataloader and checks that batch metadata contains:
- coord_mean_lat_meters
- coord_mean_lon_meters
- coord_std_lat_meters
- coord_std_lon_meters

Usage:
    python test_iq_dataloader_fix.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "services", "training", "src"))

from src.data.triangulation_dataloader import create_iq_dataloader
from src.utils.db_manager import DatabaseManager
from src.utils.minio_manager import MinioManager
from sqlalchemy import text

def test_iq_dataloader_standardization():
    """Test that IQ dataloader includes standardization parameters."""
    
    print("=" * 80)
    print("Testing IQ Dataloader Z-Score Standardization Parameters")
    print("=" * 80)
    
    # Initialize DB and MinIO
    db_manager = DatabaseManager()
    minio_manager = MinioManager()
    
    # Find a synthetic IQ dataset
    with db_manager.get_session() as session:
        query = text("""
            SELECT DISTINCT dataset_id 
            FROM heimdall.synthetic_iq_samples 
            LIMIT 1
        """)
        result = session.execute(query).fetchone()
        
        if result is None:
            print("❌ No IQ datasets found in database")
            print("   Run dataset generation first: python scripts/generate_synthetic_dataset.py --iq")
            return False
        
        dataset_id = str(result[0])
        print(f"✅ Found IQ dataset: {dataset_id}")
    
    # Create IQ dataloader
    print("\n📦 Creating IQ dataloader...")
    with db_manager.get_session() as session:
        try:
            train_loader = create_iq_dataloader(
                dataset_ids=[dataset_id],
                split="train",
                db_session=session,
                minio_client=minio_manager,
                batch_size=4,
                num_workers=0,
                shuffle=False,
                use_cache=False  # Disable cache for clean test
            )
            print(f"✅ IQ dataloader created: {len(train_loader.dataset)} samples")
        except Exception as e:
            print(f"❌ Failed to create IQ dataloader: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Get first batch
    print("\n🔍 Fetching first batch...")
    try:
        batch = next(iter(train_loader))
        print(f"✅ Got batch with keys: {batch.keys()}")
    except Exception as e:
        print(f"❌ Failed to fetch batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check metadata contains standardization params
    print("\n🧪 Checking batch metadata...")
    metadata = batch.get("metadata", {})
    
    required_params = [
        "coord_mean_lat_meters",
        "coord_mean_lon_meters",
        "coord_std_lat_meters",
        "coord_std_lon_meters"
    ]
    
    all_present = True
    for param in required_params:
        if param in metadata:
            value = metadata[param]
            print(f"✅ {param}: {value}")
        else:
            print(f"❌ {param}: MISSING")
            all_present = False
    
    # Also check centroids still present
    if "centroids" in metadata:
        print(f"✅ centroids: shape {metadata['centroids'].shape}")
    else:
        print(f"❌ centroids: MISSING")
        all_present = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_present:
        print("✅ SUCCESS: All standardization parameters present in IQ batch metadata!")
        print("   The IQ dataloader now supports z-score coordinate denormalization.")
        return True
    else:
        print("❌ FAILURE: Some standardization parameters missing!")
        print("   Training with IQ dataloader will fail with KeyError.")
        return False

if __name__ == "__main__":
    success = test_iq_dataloader_standardization()
    sys.exit(0 if success else 1)
