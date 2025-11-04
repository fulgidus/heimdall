"""
DataLoader for triangulation model with variable-length receiver lists.

Handles:
- Loading synthetic training samples from TimescaleDB
- Collating variable-length receiver measurements
- Creating attention masks
- Batching for training
"""

import torch
import numpy as np
import structlog
import pickle
import hashlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import json

logger = structlog.get_logger(__name__)

# Global cache directory for preprocessed samples
CACHE_DIR = Path("/tmp/heimdall_training_cache")


class SyntheticTriangulationDataset(Dataset):
    """Dataset for synthetic triangulation training samples."""
    
    def __init__(
        self,
        dataset_ids: List[str],
        split: str,
        db_session,
        max_receivers: int = 7,
        cache_size: int = 10000,
        use_cache: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_ids: List of UUIDs of synthetic datasets to merge
            split: 'train', 'val', or 'test'
            db_session: Database session (only used during __init__, not stored)
            max_receivers: Maximum number of receivers (for padding)
            cache_size: Number of samples to cache in memory (deprecated)
            use_cache: Enable file-based caching for faster loading
        """
        self.dataset_ids = dataset_ids
        self.split = split
        self.max_receivers = max_receivers
        self.cache_size = cache_size
        self.use_cache = use_cache
        
        # Store the session (safe since we're using num_workers=0)
        # If we ever use num_workers>0, we'd need to store only the URL and create new sessions
        self.db_session = db_session
        
        # Setup cache directory
        if self.use_cache:
            # Create cache dir based on dataset IDs hash
            datasets_hash = hashlib.md5("_".join(sorted(dataset_ids)).encode()).hexdigest()[:8]
            self.cache_dir = CACHE_DIR / f"{datasets_hash}_{split}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"File cache enabled: {self.cache_dir}")
        else:
            self.cache_dir = None
        
        # Load sample IDs using the provided session (only during init)
        self.sample_ids = self._load_sample_ids(db_session)
        
        # Store DB connection info for creating new sessions in workers
        # We'll get this from the session's engine
        self._db_url = str(db_session.get_bind().url)
        
        # Count existing cache files
        cached_count = 0
        if self.use_cache and self.cache_dir:
            cached_count = len(list(self.cache_dir.glob("*.pkl")))
            if cached_count > 0:
                cache_pct = (cached_count / len(self.sample_ids)) * 100
                logger.info(f"⚡ Cache hit: {cached_count}/{len(self.sample_ids)} samples ({cache_pct:.1f}%) - FAST loading!")
        
        logger.info(
            "SyntheticTriangulationDataset initialized",
            dataset_ids=dataset_ids,
            split=split,
            num_samples=len(self.sample_ids),
            max_receivers=max_receivers,
            cache_enabled=self.use_cache,
            cached_samples=cached_count,
            cache_dir=str(self.cache_dir) if self.cache_dir else None
        )
    
    def _load_sample_ids(self, db_session) -> List[str]:
        """Load sample IDs from database, merging all specified datasets."""
        from sqlalchemy import text
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
        from sqlalchemy import bindparam
        
        # Query measurement_features table for all dataset_ids
        # Use PostgreSQL's = ANY() with properly typed array parameter
        query = text("""
            SELECT recording_session_id
            FROM heimdall.measurement_features
            WHERE dataset_id = ANY(CAST(:dataset_ids AS uuid[]))
            ORDER BY timestamp
        """)
        
        result = db_session.execute(
            query,
            {"dataset_ids": self.dataset_ids}
        )
        
        all_sample_ids = [str(row[0]) for row in result]
        
        # Implement deterministic 80/20 train/val split based on hash of UUID
        # This ensures consistent splits across runs
        if self.split == 'train':
            # Take first 80% of sorted samples
            split_idx = int(len(all_sample_ids) * 0.8)
            sample_ids = all_sample_ids[:split_idx]
        elif self.split == 'val':
            # Take last 20% of sorted samples
            split_idx = int(len(all_sample_ids) * 0.8)
            sample_ids = all_sample_ids[split_idx:]
        elif self.split == 'test':
            # For test split, use last 10% (overlaps with val, but separate use case)
            split_idx = int(len(all_sample_ids) * 0.9)
            sample_ids = all_sample_ids[split_idx:]
        else:
            raise ValueError(f"Invalid split: {self.split}, must be 'train', 'val', or 'test'")
        
        return sample_ids
    
    def _get_db_session(self):
        """Create a new database session for this worker process."""
        # Import here to avoid issues with pickling
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create engine and session for this worker
        engine = create_engine(self._db_url, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        return Session()
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sample_ids)
    
    def _get_cache_path(self, idx: int) -> Optional[Path]:
        """Get cache file path for a sample."""
        if not self.use_cache or not self.cache_dir:
            return None
        # Use index as filename (deterministic)
        return self.cache_dir / f"sample_{idx:06d}.pkl"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample at index.
        
        Uses file cache for 10-50x speedup on subsequent epochs.
        
        Returns:
            Dict with:
            - receiver_features: (num_receivers, 6) - [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
            - signal_mask: (num_receivers,) - Boolean mask (True = no signal)
            - target_position: (2,) - [lat, lon]
            - metadata: Dict with additional info
        """
        # Try to load from cache first
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed for idx {idx}: {e}, loading from DB")
                # Fall through to DB loading
        
        sample_id = self.sample_ids[idx]
        
        # Use the stored session (works with num_workers=0)
        # Note: With num_workers>0 we'd need to create new sessions per worker
        session = self.db_session
        
        try:
            # Load sample from measurement_features table
            from sqlalchemy import text
            
            query = text("""
                SELECT tx_latitude, tx_longitude, receiver_features, gdop, num_receivers_detected
                FROM heimdall.measurement_features
                WHERE recording_session_id = :sample_id
            """)
            
            result = session.execute(query, {"sample_id": sample_id}).fetchone()
            
            if result is None:
                raise ValueError(f"Sample {sample_id} not found")
            
            tx_lat, tx_lon, receiver_features_jsonb, gdop, num_receivers_detected = result
            
            # Extract receiver features from JSONB array
            # Each receiver has extensive features, we need: snr_db (mean), psd_dbm_per_hz (mean), 
            # frequency_offset_hz (mean), rx_lat, rx_lon, signal_present
            receiver_features = []
            signal_mask = []
            
            for rx in receiver_features_jsonb:
                # Extract mean values from feature distributions
                snr = rx['snr_db']['mean'] if 'snr_db' in rx and rx['snr_db'] else 0.0
                psd = rx['psd_dbm_per_hz']['mean'] if 'psd_dbm_per_hz' in rx and rx['psd_dbm_per_hz'] else -100.0
                freq_offset = rx['frequency_offset_hz']['mean'] if 'frequency_offset_hz' in rx and rx['frequency_offset_hz'] else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                # Features: [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
                features = [
                    snr,
                    psd,
                    freq_offset,
                    rx_lat,
                    rx_lon,
                    float(signal_present)
                ]
                receiver_features.append(features)
                signal_mask.append(not signal_present)  # True if no signal
            
            # Pad to max_receivers if needed
            num_receivers = len(receiver_features)
            if num_receivers < self.max_receivers:
                # Pad with zeros
                padding = [[0.0] * 6 for _ in range(self.max_receivers - num_receivers)]
                receiver_features.extend(padding)
                signal_mask.extend([True] * (self.max_receivers - num_receivers))  # Mask padded positions
            
            # Convert to tensors
            receiver_features_tensor = torch.tensor(receiver_features, dtype=torch.float32)
            signal_mask_tensor = torch.tensor(signal_mask, dtype=torch.bool)
            target_position_tensor = torch.tensor([tx_lat, tx_lon], dtype=torch.float32)
            
            sample_data = {
                "receiver_features": receiver_features_tensor,
                "signal_mask": signal_mask_tensor,
                "target_position": target_position_tensor,
                "metadata": {
                    "sample_id": sample_id,
                    "gdop": gdop if gdop else 50.0,  # Default GDOP if missing
                    "num_receivers": num_receivers
                }
            }
            
            # Save to cache for next epoch (10-50x speedup!)
            if cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.warning(f"Cache write failed for idx {idx}: {e}")
            
            return sample_data
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length receiver lists.
    
    Args:
        batch: List of sample dicts
    
    Returns:
        Batched dict with stacked tensors
    """
    # Stack all tensors
    receiver_features = torch.stack([sample["receiver_features"] for sample in batch])
    signal_mask = torch.stack([sample["signal_mask"] for sample in batch])
    target_position = torch.stack([sample["target_position"] for sample in batch])
    
    # Collect metadata
    metadata = {
        "sample_ids": [sample["metadata"]["sample_id"] for sample in batch],
        "gdop": torch.tensor([sample["metadata"]["gdop"] for sample in batch], dtype=torch.float32),
        "num_receivers": torch.tensor([sample["metadata"]["num_receivers"] for sample in batch], dtype=torch.long)
    }
    
    return {
        "receiver_features": receiver_features,
        "signal_mask": signal_mask,
        "target_position": target_position,
        "metadata": metadata
    }


def create_triangulation_dataloader(
    dataset_ids: List[str],
    split: str,
    db_session,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    max_receivers: int = 7,
    use_cache: bool = True
) -> DataLoader:
    """
    Create DataLoader for triangulation training.
    
    Args:
        dataset_ids: List of UUIDs of synthetic datasets to merge
        split: 'train', 'val', or 'test'
        db_session: Database session
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        max_receivers: Maximum number of receivers
        use_cache: Enable file-based caching (10-50x speedup on epoch 2+)
    
    Returns:
        DataLoader instance
    """
    dataset = SyntheticTriangulationDataset(
        dataset_ids=dataset_ids,
        split=split,
        db_session=db_session,
        use_cache=use_cache,
        max_receivers=max_receivers
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(
        "DataLoader created",
        split=split,
        num_samples=len(dataset),
        batch_size=batch_size,
        num_batches=len(dataloader),
        num_workers=num_workers
    )
    
    return dataloader


class TriangulationMetrics:
    """Calculate evaluation metrics for triangulation model."""
    
    @staticmethod
    def calculate_distance_error(
        predicted_lat: torch.Tensor,
        predicted_lon: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate distance error in kilometers using Haversine formula.
        
        Args:
            predicted_lat, predicted_lon: Predicted coordinates (batch_size,)
            true_lat, true_lon: True coordinates (batch_size,)
        
        Returns:
            Distance error in km (batch_size,)
        """
        from ..models.triangulator import haversine_distance_torch
        return haversine_distance_torch(predicted_lat, predicted_lon, true_lat, true_lon)
    
    @staticmethod
    def calculate_percentiles(errors: torch.Tensor) -> Dict[str, float]:
        """
        Calculate error percentiles.
        
        Args:
            errors: Distance errors in km (num_samples,)
        
        Returns:
            Dict with median, p68, p95 percentiles
        """
        errors_np = errors.cpu().numpy()
        
        return {
            "median": float(np.median(errors_np)),
            "p68": float(np.percentile(errors_np, 68)),
            "p95": float(np.percentile(errors_np, 95)),
            "mean": float(np.mean(errors_np)),
            "std": float(np.std(errors_np))
        }
    
    @staticmethod
    def calculate_calibration(
        predicted_positions: torch.Tensor,
        log_variances: torch.Tensor,
        true_positions: torch.Tensor,
        num_bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Calculate calibration curve for uncertainty estimates.
        
        Args:
            predicted_positions: (num_samples, 2)
            log_variances: (num_samples, 1)
            true_positions: (num_samples, 2)
            num_bins: Number of bins for calibration curve
        
        Returns:
            Dict with expected_errors and actual_errors lists
        """
        # Import constant
        from ..models.triangulator import DEGREES_TO_KM
        
        # Calculate actual errors
        predicted_lat = predicted_positions[:, 0]
        predicted_lon = predicted_positions[:, 1]
        true_lat = true_positions[:, 0]
        true_lon = true_positions[:, 1]
        
        actual_errors = TriangulationMetrics.calculate_distance_error(
            predicted_lat, predicted_lon, true_lat, true_lon
        )
        
        # Calculate predicted uncertainty (in km)
        variances = torch.exp(log_variances).squeeze()
        predicted_std_km = torch.sqrt(variances) * DEGREES_TO_KM  # Approximate conversion
        
        # Sort by predicted uncertainty
        sorted_indices = torch.argsort(predicted_std_km)
        sorted_predicted = predicted_std_km[sorted_indices]
        sorted_actual = actual_errors[sorted_indices]
        
        # Create bins
        bin_size = len(sorted_predicted) // num_bins
        expected_errors = []
        actual_errors_binned = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_predicted)
            
            expected_errors.append(float(sorted_predicted[start_idx:end_idx].mean()))
            actual_errors_binned.append(float(sorted_actual[start_idx:end_idx].mean()))
        
        return {
            "expected_errors": expected_errors,
            "actual_errors": actual_errors_binned
        }
