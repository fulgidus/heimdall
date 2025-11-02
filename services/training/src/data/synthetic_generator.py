"""
Synthetic training data generator for RF source localization.

Generates realistic synthetic training samples using:
- Random transmitter placement (70% inside network, 30% outside)
- RF propagation simulation per receiver
- Quality filtering (GDOP, SNR thresholds)
- Train/val/test splitting
"""

import uuid
import numpy as np
import structlog
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .config import TrainingConfig
from .propagation import RFPropagationModel, calculate_psd, calculate_frequency_offset, calculate_gdop
from .terrain import TerrainLookup

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticSample:
    """Single synthetic training sample."""
    
    # Ground truth
    tx_lat: float
    tx_lon: float
    tx_power_dbm: float
    frequency_hz: int
    
    # Receiver measurements
    receivers: List[dict]  # [{rx_id, lat, lon, snr, psd, freq_offset, signal_present}, ...]
    
    # Quality metrics
    gdop: float
    num_receivers: int
    
    # Split assignment
    split: str  # 'train', 'val', 'test'


class SyntheticDataGenerator:
    """Generate synthetic training data for RF localization."""
    
    def __init__(
        self,
        training_config: TrainingConfig,
        propagation_model: Optional[RFPropagationModel] = None,
        terrain_lookup: Optional[TerrainLookup] = None
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            training_config: Training area configuration with receiver locations
            propagation_model: RF propagation model (default: new instance)
            terrain_lookup: Terrain elevation lookup (default: simplified model)
        """
        self.config = training_config
        self.propagation = propagation_model or RFPropagationModel()
        self.terrain = terrain_lookup or TerrainLookup(use_srtm=False)
        
        logger.info(
            "Initialized synthetic data generator",
            num_receivers=len(self.config.receivers),
            training_area_km2=self.config.training_bbox.width_km() * self.config.training_bbox.height_km()
        )
    
    def generate_samples(
        self,
        num_samples: int,
        inside_ratio: float = 0.7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        frequency_mhz: float = 145.0,
        tx_power_dbm: float = 37.0,
        min_snr_db: float = 3.0,
        min_receivers: int = 3,
        max_gdop: float = 10.0,
        progress_callback=None
    ) -> List[SyntheticSample]:
        """
        Generate synthetic training samples.
        
        Args:
            num_samples: Number of samples to generate
            inside_ratio: Ratio of transmitters inside receiver network (0.7 = 70%)
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            frequency_mhz: Transmission frequency in MHz
            tx_power_dbm: Transmitter power in dBm
            min_snr_db: Minimum SNR threshold for receiver
            min_receivers: Minimum number of receivers with signal
            max_gdop: Maximum GDOP for good geometry
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of SyntheticSample objects that pass quality filters
        """
        samples = []
        attempts = 0
        max_attempts = num_samples * 100  # Allow many retries for quality filtering (low success rate expected)

        # Track rejection reasons
        rejected_min_receivers = 0
        rejected_gdop = 0

        logger.info(
            "Starting synthetic data generation",
            num_samples=num_samples,
            inside_ratio=inside_ratio,
            splits=f"train:{train_ratio}, val:{val_ratio}, test:{test_ratio}",
            min_snr_db=min_snr_db,
            min_receivers=min_receivers,
            max_gdop=max_gdop
        )
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate transmitter location
            tx_lat, tx_lon = self._sample_tx_location(inside_ratio)
            tx_alt = self.terrain.get_elevation(tx_lat, tx_lon)
            
            # Generate receiver measurements
            receiver_measurements = []
            receivers_with_signal = 0
            
            for rx in self.config.receivers:
                # Calculate received power and SNR
                rx_power, snr_db, details = self.propagation.calculate_received_power(
                    tx_power_dbm=tx_power_dbm,
                    tx_lat=tx_lat,
                    tx_lon=tx_lon,
                    tx_alt=tx_alt,
                    rx_lat=rx.latitude,
                    rx_lon=rx.longitude,
                    rx_alt=rx.altitude,
                    frequency_mhz=frequency_mhz,
                    terrain_lookup=self.terrain
                )
                
                # Check if signal is detectable
                signal_present = snr_db >= min_snr_db
                if signal_present:
                    receivers_with_signal += 1
                
                # Calculate derived metrics
                psd = calculate_psd(snr_db) if signal_present else -999.0
                freq_offset = calculate_frequency_offset(details['distance_km']) if signal_present else 0
                
                receiver_measurements.append({
                    "rx_id": rx.name,
                    "lat": rx.latitude,
                    "lon": rx.longitude,
                    "alt": rx.altitude,
                    "snr": snr_db if signal_present else -999.0,
                    "psd": psd,
                    "freq_offset": freq_offset,
                    "signal_present": 1 if signal_present else 0,
                    "distance_km": details['distance_km']
                })
            
            # Quality check: minimum receivers
            if receivers_with_signal < min_receivers:
                rejected_min_receivers += 1
                continue

            # Calculate GDOP
            receiver_positions = [
                (m['lat'], m['lon'])
                for m in receiver_measurements
                if m['signal_present'] == 1
            ]
            gdop = calculate_gdop(receiver_positions, (tx_lat, tx_lon))

            # Quality check: GDOP
            if gdop > max_gdop:
                rejected_gdop += 1
                continue
            
            # Assign to split
            split = self._assign_split(len(samples), num_samples, train_ratio, val_ratio, test_ratio)
            
            # Create sample
            sample = SyntheticSample(
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_power_dbm=tx_power_dbm,
                frequency_hz=int(frequency_mhz * 1e6),
                receivers=receiver_measurements,
                gdop=gdop,
                num_receivers=receivers_with_signal,
                split=split
            )
            
            samples.append(sample)
            
            # Progress callback
            if progress_callback and len(samples) % 100 == 0:
                progress_callback(len(samples), num_samples, f"Generated {len(samples)}/{num_samples} samples")
        
        logger.info(
            "Synthetic data generation complete",
            samples_generated=len(samples),
            attempts=attempts,
            success_rate=f"{len(samples)/attempts*100:.1f}%",
            rejected_min_receivers=rejected_min_receivers,
            rejected_gdop=rejected_gdop,
            rejection_rate_receivers=f"{rejected_min_receivers/attempts*100:.1f}%",
            rejection_rate_gdop=f"{rejected_gdop/attempts*100:.1f}%"
        )
        
        return samples
    
    def _sample_tx_location(self, inside_ratio: float) -> Tuple[float, float]:
        """
        Sample random transmitter location.
        
        Args:
            inside_ratio: Probability of sampling inside receiver network
        
        Returns:
            (latitude, longitude) tuple
        """
        bbox = self.config.receiver_bbox if np.random.random() < inside_ratio else self.config.training_bbox
        
        lat = np.random.uniform(bbox.lat_min, bbox.lat_max)
        lon = np.random.uniform(bbox.lon_min, bbox.lon_max)
        
        return lat, lon
    
    def _assign_split(
        self,
        current_idx: int,
        total: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> str:
        """
        Assign sample to train/val/test split.
        
        Args:
            current_idx: Current sample index
            total: Total number of samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            'train', 'val', or 'test'
        """
        # Deterministic split based on index
        normalized_idx = current_idx / total
        
        if normalized_idx < train_ratio:
            return 'train'
        elif normalized_idx < train_ratio + val_ratio:
            return 'val'
        else:
            return 'test'


def save_samples_to_db(samples: List[SyntheticSample], dataset_id: uuid.UUID, db_session) -> int:
    """
    Save synthetic samples to database.
    
    Args:
        samples: List of synthetic samples
        dataset_id: UUID of the dataset
        db_session: Database session
    
    Returns:
        Number of samples saved
    """
    from sqlalchemy import text
    
    insert_query = text("""
        INSERT INTO heimdall.synthetic_training_samples (
            timestamp, dataset_id, tx_lat, tx_lon, tx_power_dbm, frequency_hz,
            receivers, gdop, num_receivers, split, created_at
        ) VALUES (
            :timestamp, :dataset_id, :tx_lat, :tx_lon, :tx_power_dbm, :frequency_hz,
            CAST(:receivers AS jsonb), :gdop, :num_receivers, :split, :created_at
        )
    """)
    
    now = datetime.now(timezone.utc)
    
    for i, sample in enumerate(samples):
        import json
        
        # Insert sample
        db_session.execute(
            insert_query,
            {
                "timestamp": now,
                "dataset_id": str(dataset_id),
                "tx_lat": sample.tx_lat,
                "tx_lon": sample.tx_lon,
                "tx_power_dbm": sample.tx_power_dbm,
                "frequency_hz": sample.frequency_hz,
                "receivers": json.dumps(sample.receivers),
                "gdop": sample.gdop,
                "num_receivers": sample.num_receivers,
                "split": sample.split,
                "created_at": now
            }
        )
        
        if (i + 1) % 1000 == 0:
            db_session.commit()
            logger.info(f"Saved {i + 1}/{len(samples)} samples to database")
    
    db_session.commit()
    logger.info(f"Saved all {len(samples)} samples to database")
    
    return len(samples)


def calculate_quality_metrics(samples: List[SyntheticSample]) -> dict:
    """
    Calculate quality metrics for generated dataset.
    
    Args:
        samples: List of synthetic samples
    
    Returns:
        Dict with quality metrics
    """
    if not samples:
        return {}
    
    snr_values = []
    gdop_values = []
    num_receivers_list = []
    distances = []
    
    for sample in samples:
        snr_values.extend([r['snr'] for r in sample.receivers if r['signal_present'] == 1])
        gdop_values.append(sample.gdop)
        num_receivers_list.append(sample.num_receivers)
        distances.extend([r['distance_km'] for r in sample.receivers if r['signal_present'] == 1])
    
    metrics = {
        "avg_snr_db": np.mean(snr_values) if snr_values else 0.0,
        "min_snr_db": np.min(snr_values) if snr_values else 0.0,
        "max_snr_db": np.max(snr_values) if snr_values else 0.0,
        "avg_gdop": np.mean(gdop_values),
        "min_gdop": np.min(gdop_values),
        "max_gdop": np.max(gdop_values),
        "avg_receivers": np.mean(num_receivers_list),
        "avg_distance_km": np.mean(distances) if distances else 0.0,
        "max_distance_km": np.max(distances) if distances else 0.0
    }
    
    return metrics
