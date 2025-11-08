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
import io
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import json

logger = structlog.get_logger(__name__)

# Global cache directory for preprocessed samples
CACHE_DIR = Path("/tmp/heimdall_training_cache")

# Spectrogram parameters for IQ-raw datasets
SPECTROGRAM_CONFIG = {
    'n_fft': 256,         # FFT bins
    'hop_length': 128,    # Hop size
    'win_length': 256,    # Window size
    'window': 'hann',     # Window function
    'center': True,
    'normalized': True,
    'onesided': True      # Only positive frequencies for real signals
}

# Normalization constants for feature scaling (Italy bounds)
# These prevent numerical instability by normalizing coordinates to [0, 1] range
NORMALIZATION_PARAMS = {
    'lat_min': 43.0,       # Southern Italy bound
    'lat_max': 47.0,       # Northern Italy bound  
    'lon_min': 6.0,        # Western Italy bound
    'lon_max': 11.0,       # Eastern Italy bound
    'snr_min': -20.0,      # Typical SNR lower bound (dB)
    'snr_max': 60.0,       # Typical SNR upper bound (dB)
    'psd_min': -100.0,     # Typical PSD lower bound (dBm/Hz)
    'psd_max': -70.0,      # Typical PSD upper bound (dBm/Hz)
    'freq_offset_min': -50.0,  # Typical frequency offset lower bound (Hz)
    'freq_offset_max': 50.0    # Typical frequency offset upper bound (Hz)
}


def normalize_feature(value: float, min_val: float, max_val: float, clip: bool = True) -> float:
    """
    Normalize a feature to [0, 1] range.
    
    Args:
        value: Raw feature value
        min_val: Minimum expected value
        max_val: Maximum expected value
        clip: Whether to clip to [0, 1] range
        
    Returns:
        Normalized value in [0, 1]
    """
    # Check for invalid values
    if not np.isfinite(value):
        logger.warning(f"Non-finite value encountered: {value}, replacing with 0.0")
        return 0.0
        
    normalized = (value - min_val) / (max_val - min_val)
    
    if clip:
        normalized = max(0.0, min(1.0, normalized))
    
    return normalized


class SyntheticTriangulationDataset(Dataset):
    """Dataset for synthetic triangulation training samples."""
    
    def __init__(
        self,
        dataset_ids: List[str],
        split: str,
        db_session,
        max_receivers: int = 10,
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
                snr_raw = rx['snr_db']['mean'] if 'snr_db' in rx and rx['snr_db'] else 0.0
                psd_raw = rx['psd_dbm_per_hz']['mean'] if 'psd_dbm_per_hz' in rx and rx['psd_dbm_per_hz'] else -100.0
                freq_offset_raw = rx['frequency_offset_hz']['mean'] if 'frequency_offset_hz' in rx and rx['frequency_offset_hz'] else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                # Normalize signal quality features to [0, 1] range to prevent numerical instability
                # BUT: Do NOT normalize coordinates (lat/lon) - model needs real spatial relationships
                snr = normalize_feature(snr_raw, NORMALIZATION_PARAMS['snr_min'], NORMALIZATION_PARAMS['snr_max'])
                psd = normalize_feature(psd_raw, NORMALIZATION_PARAMS['psd_min'], NORMALIZATION_PARAMS['psd_max'])
                freq_offset = normalize_feature(freq_offset_raw, NORMALIZATION_PARAMS['freq_offset_min'], NORMALIZATION_PARAMS['freq_offset_max'])
                
                # Validate coordinates for NaN/Inf
                if not (np.isfinite(rx_lat) and np.isfinite(rx_lon)):
                    logger.warning(f"Invalid receiver coordinates: lat={rx_lat}, lon={rx_lon}, replacing with defaults")
                    rx_lat = 45.0  # Default to center of Italy
                    rx_lon = 9.0
                
                # Features: [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
                # Signal quality features normalized to [0,1], coordinates in real degrees
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
            
            # Calculate centroid from VALID receivers (exclude masked/padded)
            # This creates a local reference frame for each sample
            valid_positions = []
            for i in range(len(receiver_features)):
                # receiver_features[i] = [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
                if not signal_mask[i]:  # signal_present == True (not masked)
                    rx_lat_abs = receiver_features[i][3]
                    rx_lon_abs = receiver_features[i][4]
                    valid_positions.append([rx_lat_abs, rx_lon_abs])
            
            if len(valid_positions) > 0:
                # Use mean of valid receiver positions as centroid
                centroid_lat = np.mean([pos[0] for pos in valid_positions])
                centroid_lon = np.mean([pos[1] for pos in valid_positions])
            else:
                # Fallback: use all receivers (shouldn't happen in normal training)
                logger.warning(f"Sample {sample_id}: No valid receivers for centroid, using all positions")
                centroid_lat = np.mean([receiver_features[i][3] for i in range(len(receiver_features))])
                centroid_lon = np.mean([receiver_features[i][4] for i in range(len(receiver_features))])
            
            centroid = np.array([centroid_lat, centroid_lon])
            
            # Transform coordinates to DELTAS relative to centroid
            # This puts all coordinates in a local reference frame with scale [-2, +2] degrees
            # instead of absolute [43-47, 6-11] which causes gradient instability
            for i in range(len(receiver_features)):
                receiver_features[i][3] -= centroid_lat  # delta_lat = rx_lat - centroid_lat
                receiver_features[i][4] -= centroid_lon  # delta_lon = rx_lon - centroid_lon
            
            # Pad to max_receivers if needed
            num_receivers = len(receiver_features)
            if num_receivers < self.max_receivers:
                # Pad with zeros
                padding = [[0.0] * 6 for _ in range(self.max_receivers - num_receivers)]
                receiver_features.extend(padding)
                signal_mask.extend([True] * (self.max_receivers - num_receivers))  # Mask padded positions
            
            # Transform target to DELTA coordinates (relative to centroid)
            target_delta_lat = tx_lat - centroid_lat
            target_delta_lon = tx_lon - centroid_lon
            
            # Convert to tensors
            # NOTE: Target positions are now DELTA coordinates (model predicts offsets from centroid)
            receiver_features_tensor = torch.tensor(receiver_features, dtype=torch.float32)
            signal_mask_tensor = torch.tensor(signal_mask, dtype=torch.bool)
            target_position_tensor = torch.tensor([target_delta_lat, target_delta_lon], dtype=torch.float32)
            
            sample_data = {
                "receiver_features": receiver_features_tensor,
                "signal_mask": signal_mask_tensor,
                "target_position": target_position_tensor,
                "metadata": {
                    "sample_id": sample_id,
                    "gdop": gdop if gdop else 50.0,  # Default GDOP if missing
                    "num_receivers": num_receivers,
                    "centroid": centroid.tolist()  # Store for reconstruction in training/inference
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
    
    # Collect metadata including centroids for coordinate reconstruction
    # Centroids are needed to convert predicted DELTA coordinates back to absolute lat/lon
    metadata = {
        "sample_ids": [sample["metadata"]["sample_id"] for sample in batch],
        "gdop": torch.tensor([sample["metadata"]["gdop"] for sample in batch], dtype=torch.float32),
        "num_receivers": torch.tensor([sample["metadata"]["num_receivers"] for sample in batch], dtype=torch.long),
        "centroids": torch.tensor([sample["metadata"]["centroid"] for sample in batch], dtype=torch.float32)  # [batch, 2] (lat, lon)
    }
    
    return {
        "receiver_features": receiver_features,
        "signal_mask": signal_mask,
        "target_position": target_position,
        "metadata": metadata
    }


def collate_iq_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching IQ samples (raw time-series).
    
    Args:
        batch: List of sample dicts from TriangulationIQDataset
    
    Returns:
        Batched dict with stacked tensors
    """
    # Stack all tensors
    iq_samples = torch.stack([sample["iq_samples"] for sample in batch])
    receiver_positions = torch.stack([sample["receiver_positions"] for sample in batch])
    signal_mask = torch.stack([sample["signal_mask"] for sample in batch])
    target_position = torch.stack([sample["target_position"] for sample in batch])
    
    # Collect metadata including centroids for coordinate reconstruction
    # Centroids are needed to convert predicted DELTA coordinates back to absolute lat/lon
    metadata = {
        "sample_ids": [sample["metadata"]["sample_id"] for sample in batch],
        "gdop": torch.tensor([sample["metadata"]["gdop"] for sample in batch], dtype=torch.float32),
        "num_receivers": torch.tensor([sample["metadata"]["num_receivers"] for sample in batch], dtype=torch.long),
        "centroids": torch.tensor([sample["metadata"]["centroid"] for sample in batch], dtype=torch.float32)  # [batch, 2] (lat, lon)
    }
    
    return {
        "iq_samples": iq_samples,  # Changed key from iq_spectrograms to iq_samples
        "receiver_positions": receiver_positions,
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
    max_receivers: int = 10,
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


def create_iq_dataloader(
    dataset_ids: List[str],
    split: str,
    db_session,
    minio_client,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    max_receivers: int = 10,
    use_cache: bool = True
) -> DataLoader:
    """
    Create DataLoader for IQ-raw CNN training.
    
    Args:
        dataset_ids: List of UUIDs of IQ datasets to merge
        split: 'train', 'val', or 'test'
        db_session: Database session
        minio_client: MinIO client for loading IQ data
        batch_size: Batch size (smaller than feature-based due to memory)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        max_receivers: Maximum number of receivers (for padding)
        use_cache: Enable file-based caching (recommended for IQ data)
    
    Returns:
        DataLoader instance
    """
    dataset = TriangulationIQDataset(
        dataset_ids=dataset_ids,
        split=split,
        db_session=db_session,
        minio_client=minio_client,
        max_receivers=max_receivers,
        use_cache=use_cache
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_iq_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(
        "IQ DataLoader created",
        split=split,
        num_samples=len(dataset),
        batch_size=batch_size,
        num_batches=len(dataloader),
        num_workers=num_workers,
        spectrogram_config=SPECTROGRAM_CONFIG
    )
    
    return dataloader


class TriangulationIQDataset(Dataset):
    """
    Dataset for IQ-raw samples for CNN-based triangulation.
    
    Loads raw IQ samples from MinIO and computes spectrograms on-the-fly during training.
    Supports variable receiver count (5-10 per sample).
    """
    
    def __init__(
        self,
        dataset_ids: List[str],
        split: str,
        db_session,
        minio_client,
        max_receivers: int = 10,
        use_cache: bool = True
    ):
        """
        Initialize IQ dataset.
        
        Args:
            dataset_ids: List of UUIDs of IQ datasets to merge
            split: 'train', 'val', or 'test'
            db_session: Database session
            minio_client: MinIO client for loading IQ data
            max_receivers: Maximum number of receivers (for padding)
            use_cache: Enable file-based caching
        """
        self.dataset_ids = dataset_ids
        self.split = split
        self.max_receivers = max_receivers
        self.use_cache = use_cache
        self.minio_client = minio_client
        
        # Store the session
        self.db_session = db_session
        
        # Setup cache directory
        if self.use_cache:
            datasets_hash = hashlib.md5("_".join(sorted(dataset_ids)).encode()).hexdigest()[:8]
            self.cache_dir = CACHE_DIR / f"iq_{datasets_hash}_{split}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"IQ file cache enabled: {self.cache_dir}")
        else:
            self.cache_dir = None
        
        # Load sample IDs
        self.sample_ids = self._load_sample_ids(db_session)
        
        # Store DB connection info
        self._db_url = str(db_session.get_bind().url)
        
        # Count cached samples
        cached_count = 0
        if self.use_cache and self.cache_dir:
            cached_count = len(list(self.cache_dir.glob("*.pkl")))
            if cached_count > 0:
                cache_pct = (cached_count / len(self.sample_ids)) * 100
                logger.info(f"⚡ IQ Cache hit: {cached_count}/{len(self.sample_ids)} samples ({cache_pct:.1f}%)")
        
        logger.info(
            "TriangulationIQDataset initialized",
            dataset_ids=dataset_ids,
            split=split,
            num_samples=len(self.sample_ids),
            max_receivers=max_receivers,
            cache_enabled=self.use_cache,
            cached_samples=cached_count
        )
    
    def _load_sample_ids(self, db_session) -> List[str]:
        """Load sample IDs from synthetic_iq_samples table."""
        from sqlalchemy import text
        
        query = text("""
            SELECT id
            FROM heimdall.synthetic_iq_samples
            WHERE dataset_id = ANY(CAST(:dataset_ids AS uuid[]))
            ORDER BY timestamp, sample_idx
        """)
        
        result = db_session.execute(query, {"dataset_ids": self.dataset_ids})
        all_sample_ids = [str(row[0]) for row in result]
        
        # Apply train/val/test split
        if self.split == 'train':
            split_idx = int(len(all_sample_ids) * 0.8)
            sample_ids = all_sample_ids[:split_idx]
        elif self.split == 'val':
            split_idx = int(len(all_sample_ids) * 0.8)
            sample_ids = all_sample_ids[split_idx:]
        elif self.split == 'test':
            split_idx = int(len(all_sample_ids) * 0.9)
            sample_ids = all_sample_ids[split_idx:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return sample_ids
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sample_ids)
    
    def _get_cache_path(self, idx: int) -> Optional[Path]:
        """Get cache file path for a sample."""
        if not self.use_cache or not self.cache_dir:
            return None
        return self.cache_dir / f"iq_sample_{idx:06d}.pkl"
    
    def _compute_spectrogram(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Compute spectrogram from IQ data using STFT.
        
        Args:
            iq_data: Complex IQ samples (num_samples,)
        
        Returns:
            Spectrogram tensor: (2, freq_bins, time_bins) - [real, imag]
        """
        # Convert to torch complex tensor
        iq_tensor = torch.from_numpy(iq_data).to(torch.complex64)
        
        # Compute STFT
        stft_result = torch.stft(
            iq_tensor,
            n_fft=SPECTROGRAM_CONFIG['n_fft'],
            hop_length=SPECTROGRAM_CONFIG['hop_length'],
            win_length=SPECTROGRAM_CONFIG['win_length'],
            window=torch.hann_window(SPECTROGRAM_CONFIG['win_length']),
            center=SPECTROGRAM_CONFIG['center'],
            normalized=SPECTROGRAM_CONFIG['normalized'],
            onesided=SPECTROGRAM_CONFIG['onesided'],
            return_complex=True
        )
        
        # Stack real and imaginary parts: (freq_bins, time_bins) -> (2, freq_bins, time_bins)
        spectrogram = torch.stack([stft_result.real, stft_result.imag], dim=0)
        
        return spectrogram
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get IQ sample at index.
        
        Returns:
            Dict with:
            - iq_samples: (num_receivers, 2, seq_len) - [I, Q] raw IQ time-series
            - receiver_positions: (num_receivers, 2) - [lat, lon]
            - signal_mask: (num_receivers,) - Boolean mask (True = no signal/padding)
            - target_position: (2,) - [lat, lon]
            - metadata: Dict with additional info
        """
        # Try cache first
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"IQ cache read failed for idx {idx}: {e}, loading from DB")
        
        sample_id = self.sample_ids[idx]
        session = self.db_session
        
        try:
            # Load sample metadata from database
            from sqlalchemy import text
            
            query = text("""
                SELECT 
                    tx_lat, tx_lon, 
                    receivers_metadata, 
                    num_receivers,
                    iq_storage_paths,
                    gdop
                FROM heimdall.synthetic_iq_samples
                WHERE id = :sample_id
            """)
            
            result = session.execute(query, {"sample_id": sample_id}).fetchone()
            
            if result is None:
                raise ValueError(f"IQ sample {sample_id} not found")
            
            tx_lat, tx_lon, receivers_metadata, num_receivers, iq_storage_paths, gdop = result
            
            # Load RAW IQ data from MinIO (no spectrogram computation!)
            iq_samples_list = []
            receiver_positions = []
            signal_mask = []
            
            for rx_meta in receivers_metadata:
                rx_id = rx_meta['rx_id']
                rx_lat = rx_meta['lat']
                rx_lon = rx_meta['lon']
                signal_present = rx_meta.get('signal_present', True)
                
                # Get MinIO path for this receiver
                minio_path = iq_storage_paths.get(rx_id)
                
                if minio_path and signal_present:
                    # Load RAW IQ data from MinIO
                    try:
                        response = self.minio_client.s3_client.get_object(
                            Bucket="heimdall-synthetic-iq",
                            Key=minio_path
                        )
                        iq_bytes = response['Body'].read()
                        
                        # Load numpy array (complex IQ samples)
                        iq_data = np.load(io.BytesIO(iq_bytes))
                        
                        # Convert complex array to [I, Q] channels: (2, seq_len)
                        # I = real part, Q = imaginary part
                        iq_tensor = torch.tensor(np.stack([iq_data.real, iq_data.imag]), dtype=torch.float32)
                        iq_samples_list.append(iq_tensor)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load IQ for {rx_id}: {e}, using zeros")
                        # Use zero IQ if loading fails (1024 samples default)
                        iq_tensor = torch.zeros(2, 1024, dtype=torch.float32)
                        iq_samples_list.append(iq_tensor)
                        signal_present = False
                else:
                    # No signal or missing path
                    iq_tensor = torch.zeros(2, 1024, dtype=torch.float32)
                    iq_samples_list.append(iq_tensor)
                    signal_present = False
                
                receiver_positions.append([rx_lat, rx_lon])
                signal_mask.append(not signal_present)
            
            # Determine actual sequence length from first valid sample
            seq_len = iq_samples_list[0].shape[1] if len(iq_samples_list) > 0 else 1024
            
            # Pad to max_receivers if needed
            current_receivers = len(iq_samples_list)
            if current_receivers < self.max_receivers:
                padding_count = self.max_receivers - current_receivers
                
                # Add zero IQ samples
                for _ in range(padding_count):
                    iq_samples_list.append(torch.zeros(2, seq_len, dtype=torch.float32))
                    receiver_positions.append([0.0, 0.0])
                    signal_mask.append(True)  # Mask padded positions
            
            # Calculate centroid from VALID receivers (exclude masked/padded)
            # This creates a local reference frame for each sample
            valid_positions = []
            for i in range(len(receiver_positions)):
                if not signal_mask[i]:  # Signal present (not masked)
                    valid_positions.append(receiver_positions[i])
            
            if len(valid_positions) > 0:
                # Use mean of valid receiver positions as centroid
                centroid_lat = np.mean([pos[0] for pos in valid_positions])
                centroid_lon = np.mean([pos[1] for pos in valid_positions])
            else:
                # Fallback: use all receivers (shouldn't happen in normal training)
                logger.warning(f"Sample {sample_id}: No valid receivers for centroid, using all positions")
                centroid_lat = np.mean([pos[0] for pos in receiver_positions])
                centroid_lon = np.mean([pos[1] for pos in receiver_positions])
            
            centroid = np.array([centroid_lat, centroid_lon])
            
            # Transform coordinates to DELTAS relative to centroid
            # This puts all coordinates in a local reference frame
            receiver_positions_delta = []
            for i in range(len(receiver_positions)):
                rx_lat_delta = receiver_positions[i][0] - centroid_lat
                rx_lon_delta = receiver_positions[i][1] - centroid_lon
                receiver_positions_delta.append([rx_lat_delta, rx_lon_delta])
            
            # Transform target to DELTA coordinates (relative to centroid)
            target_delta_lat = tx_lat - centroid_lat
            target_delta_lon = tx_lon - centroid_lon
            
            # Stack into tensors
            iq_samples = torch.stack(iq_samples_list)  # (num_receivers, 2, seq_len) - RAW IQ!
            receiver_positions = torch.tensor(receiver_positions_delta, dtype=torch.float32)  # (num_receivers, 2) DELTA coordinates
            signal_mask = torch.tensor(signal_mask, dtype=torch.bool)  # (num_receivers,)
            target_position = torch.tensor([target_delta_lat, target_delta_lon], dtype=torch.float32)  # (2,) DELTA coordinates
            
            sample_data = {
                "iq_samples": iq_samples,  # Changed key from iq_spectrograms to iq_samples
                "receiver_positions": receiver_positions,
                "signal_mask": signal_mask,
                "target_position": target_position,
                "metadata": {
                    "sample_id": sample_id,
                    "gdop": gdop if gdop else 50.0,
                    "num_receivers": num_receivers,
                    "centroid": centroid.tolist()  # Store for reconstruction in training/inference
                }
            }
            
            # Save to cache
            if cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.warning(f"IQ cache write failed for idx {idx}: {e}")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error loading IQ sample {idx}: {e}")
            raise


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
