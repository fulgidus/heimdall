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
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import json

logger = structlog.get_logger(__name__)


class SyntheticTriangulationDataset(Dataset):
    """Dataset for synthetic triangulation training samples."""
    
    def __init__(
        self,
        dataset_id: str,
        split: str,
        db_session,
        max_receivers: int = 7,
        cache_size: int = 10000
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_id: UUID of synthetic dataset
            split: 'train', 'val', or 'test'
            db_session: Database session
            max_receivers: Maximum number of receivers (for padding)
            cache_size: Number of samples to cache in memory
        """
        self.dataset_id = dataset_id
        self.split = split
        self.db_session = db_session
        self.max_receivers = max_receivers
        self.cache_size = cache_size
        
        # Load sample IDs
        self.sample_ids = self._load_sample_ids()
        
        logger.info(
            "SyntheticTriangulationDataset initialized",
            dataset_id=dataset_id,
            split=split,
            num_samples=len(self.sample_ids),
            max_receivers=max_receivers
        )
    
    def _load_sample_ids(self) -> List[str]:
        """Load sample IDs from database."""
        from sqlalchemy import text
        
        # Query measurement_features table instead of synthetic_training_samples
        query = text("""
            SELECT recording_session_id
            FROM heimdall.measurement_features
            WHERE dataset_id = :dataset_id
            ORDER BY timestamp
        """)
        
        result = self.db_session.execute(
            query,
            {"dataset_id": self.dataset_id}
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
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample at index.
        
        Returns:
            Dict with:
            - receiver_features: (num_receivers, 6) - [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
            - signal_mask: (num_receivers,) - Boolean mask (True = no signal)
            - target_position: (2,) - [lat, lon]
            - metadata: Dict with additional info
        """
        sample_id = self.sample_ids[idx]
        
        # Load sample from measurement_features table
        from sqlalchemy import text
        
        query = text("""
            SELECT tx_latitude, tx_longitude, receiver_features, gdop, num_receivers_detected
            FROM heimdall.measurement_features
            WHERE recording_session_id = :sample_id
        """)
        
        result = self.db_session.execute(query, {"sample_id": sample_id}).fetchone()
        
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
        
        return {
            "receiver_features": receiver_features_tensor,
            "signal_mask": signal_mask_tensor,
            "target_position": target_position_tensor,
            "metadata": {
                "sample_id": sample_id,
                "gdop": gdop if gdop else 50.0,  # Default GDOP if missing
                "num_receivers": num_receivers
            }
        }


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
    dataset_id: str,
    split: str,
    db_session,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    max_receivers: int = 7
) -> DataLoader:
    """
    Create DataLoader for triangulation training.
    
    Args:
        dataset_id: UUID of synthetic dataset
        split: 'train', 'val', or 'test'
        db_session: Database session
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        max_receivers: Maximum number of receivers
    
    Returns:
        DataLoader instance
    """
    dataset = SyntheticTriangulationDataset(
        dataset_id=dataset_id,
        split=split,
        db_session=db_session,
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
