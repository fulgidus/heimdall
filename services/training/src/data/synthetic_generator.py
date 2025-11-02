"""
Synthetic training data generator for RF source localization.

Generates realistic synthetic training samples using:
- IQ sample generation with realistic RF effects
- Feature extraction from IQ samples
- Random transmitter placement (70% inside network, 30% outside)
- RF propagation simulation per receiver
- Quality filtering (GDOP, SNR thresholds)
- Multiprocessing for parallel generation
"""

import uuid
import io
import json
import os
import numpy as np
import structlog
import multiprocessing as mp
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import TrainingConfig
from .propagation import RFPropagationModel, calculate_psd, calculate_frequency_offset, calculate_gdop
from .terrain import TerrainLookup
from .iq_generator import SyntheticIQGenerator, SyntheticIQSample

# Import from common module (PYTHONPATH=/app in Dockerfile)
from common.feature_extraction import RFFeatureExtractor, IQSample

logger = structlog.get_logger(__name__)


def _generate_single_sample(args):
    """
    Generate a single synthetic sample (for multiprocessing).

    This function must be at module level for pickle serialization.

    Args:
        args: Tuple of (sample_index, receivers_config, training_config, config, seed)

    Returns:
        Tuple of (sample_idx, receiver_features, extraction_metadata, quality_metrics, iq_samples_dict, tx_position)
    """
    sample_idx, receivers_list, training_config_dict, config, base_seed = args

    # Create unique seed for this sample
    sample_seed = base_seed + sample_idx if base_seed else None
    rng = np.random.default_rng(sample_seed)

    # Initialize generators
    iq_generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,  # 200 kHz
        duration_ms=1000.0,       # 1 second
        seed=sample_seed
    )
    feature_extractor = RFFeatureExtractor(sample_rate_hz=200_000)
    propagation = RFPropagationModel()

    # Extract config parameters
    frequency_mhz = config.get('frequency_mhz', 145.0)
    tx_power_dbm = config.get('tx_power_dbm', 37.0)
    inside_ratio = config.get('inside_ratio', 0.7)
    min_snr_db = config.get('min_snr_db', 3.0)

    # Generate random TX position
    # Reconstruct bounding boxes from config dict
    if rng.random() < inside_ratio:
        # Inside receiver network
        lat_min = training_config_dict['receiver_bbox']['lat_min']
        lat_max = training_config_dict['receiver_bbox']['lat_max']
        lon_min = training_config_dict['receiver_bbox']['lon_min']
        lon_max = training_config_dict['receiver_bbox']['lon_max']
    else:
        # Training area (larger)
        lat_min = training_config_dict['training_bbox']['lat_min']
        lat_max = training_config_dict['training_bbox']['lat_max']
        lon_min = training_config_dict['training_bbox']['lon_min']
        lon_max = training_config_dict['training_bbox']['lon_max']

    tx_lat = rng.uniform(lat_min, lat_max)
    tx_lon = rng.uniform(lon_min, lon_max)
    tx_alt = 300.0  # Simplified altitude (meters ASL)

    # Generate IQ samples for each receiver
    iq_samples = {}
    receiver_features_list = []
    receivers_with_signal = 0
    propagation_snr_values = []  # Track SNR from propagation model

    for receiver in receivers_list:
        rx_id = receiver['name']
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']
        rx_alt = receiver['altitude']

        # Calculate received power and SNR using propagation model
        rx_power_dbm, snr_db, details = propagation.calculate_received_power(
            tx_power_dbm=tx_power_dbm,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=frequency_mhz,
            terrain_lookup=None
        )

        # Check if signal is detectable
        signal_present = snr_db >= min_snr_db
        if signal_present:
            receivers_with_signal += 1
            propagation_snr_values.append(snr_db)  # Store propagation SNR

        # Generate IQ sample
        noise_floor_dbm = -120.0
        frequency_offset_hz = rng.uniform(-50, 50)  # Doppler/oscillator drift
        
        iq_sample = iq_generator.generate_iq_sample(
            center_frequency_hz=frequency_mhz * 1e6,
            signal_power_dbm=rx_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            snr_db=snr_db,
            frequency_offset_hz=frequency_offset_hz,
            bandwidth_hz=12500.0,  # FM signal bandwidth
            rx_id=rx_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=float(sample_idx)
        )

        # Store IQ sample (for first 100 samples only)
        if sample_idx < 100:
            iq_samples[rx_id] = iq_sample

        # Convert SyntheticIQSample to IQSample for feature extraction
        iq_sample_for_extraction = IQSample(
            samples=iq_sample.samples,
            sample_rate_hz=int(iq_sample.sample_rate_hz),
            center_frequency_hz=int(iq_sample.center_frequency_hz),
            rx_id=iq_sample.rx_id,
            rx_lat=iq_sample.rx_lat,
            rx_lon=iq_sample.rx_lon,
            timestamp=iq_sample.to_datetime()
        )

        # Extract features (chunked: 1000ms → 5×200ms with aggregation)
        features_dict = feature_extractor.extract_features_chunked(
            iq_sample_for_extraction,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        # Add receiver metadata to features
        features_dict['rx_id'] = rx_id
        features_dict['rx_lat'] = rx_lat
        features_dict['rx_lon'] = rx_lon
        features_dict['distance_km'] = details['distance_km']

        # Override signal_present with propagation model result
        # (feature extractor may incorrectly detect no signal due to fading/multipath)
        features_dict['signal_present'] = signal_present

        receiver_features_list.append(features_dict)

    # Calculate overall quality metrics
    # Use propagation model SNR (not feature extractor SNR which can be negative due to fading)
    mean_snr_db = float(np.mean(propagation_snr_values)) if propagation_snr_values else 0.0

    # Calculate overall confidence (weighted average)
    confidence_scores = [f.get('delay_spread_confidence', {}).get('mean', 0.8)
                        for f in receiver_features_list if f.get('signal_present')]
    overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    # Calculate GDOP
    receiver_positions = [
        (f['rx_lat'], f['rx_lon'])
        for f in receiver_features_list
        if f.get('signal_present')
    ]
    gdop = calculate_gdop(receiver_positions, (tx_lat, tx_lon)) if len(receiver_positions) >= 3 else 999.0

    extraction_metadata = {
        'extraction_method': 'synthetic',
        'iq_duration_ms': 1000.0,
        'sample_rate_hz': 200_000,
        'num_chunks': 5,
        'chunk_duration_ms': 200.0,
        'generated_at': sample_idx,
        'frequency_mhz': frequency_mhz,
        'tx_power_dbm': tx_power_dbm
    }

    quality_metrics = {
        'overall_confidence': overall_confidence,
        'mean_snr_db': mean_snr_db,
        'num_receivers_detected': receivers_with_signal,
        'gdop': gdop
    }

    tx_position = {
        'tx_lat': tx_lat,
        'tx_lon': tx_lon,
        'tx_alt': tx_alt,
        'tx_power_dbm': tx_power_dbm
    }

    return (sample_idx, receiver_features_list, extraction_metadata, quality_metrics, iq_samples, tx_position)


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
        terrain_lookup: Optional[TerrainLookup] = None,
        use_srtm_terrain: bool = False
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            training_config: Training area configuration with receiver locations
            propagation_model: RF propagation model (default: new instance)
            terrain_lookup: Terrain elevation lookup (default: simplified model)
            use_srtm_terrain: If True, create TerrainLookup with SRTM support
        """
        self.config = training_config
        self.propagation = propagation_model or RFPropagationModel()
        
        # Initialize terrain lookup
        if terrain_lookup is not None:
            self.terrain = terrain_lookup
        elif use_srtm_terrain:
            # Create TerrainLookup with SRTM support and MinIO client
            try:
                import sys
                sys.path.insert(0, '/app/backend/src')
                from storage.minio_client import MinIOClient
                from config import settings as backend_settings
                
                minio_client = MinIOClient(
                    endpoint_url=backend_settings.minio_url,
                    access_key=backend_settings.minio_access_key,
                    secret_key=backend_settings.minio_secret_key,
                    bucket_name="heimdall-terrain"
                )
                self.terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)
                logger.info("Using SRTM terrain data for synthetic generation")
            except Exception as e:
                logger.warning(f"Failed to initialize SRTM terrain: {e}, using simplified model")
                self.terrain = TerrainLookup(use_srtm=False)
        else:
            self.terrain = TerrainLookup(use_srtm=False)
        
        logger.info(
            "Initialized synthetic data generator",
            num_receivers=len(self.config.receivers),
            training_area_km2=self.config.training_bbox.width_km() * self.config.training_bbox.height_km(),
            using_srtm=self.terrain.use_srtm
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


async def generate_synthetic_data_with_iq(
    dataset_id: uuid.UUID,
    num_samples: int,
    receivers_config: list,
    training_config: TrainingConfig,
    config: dict,
    conn,
    progress_callback=None,
    seed: Optional[int] = None
) -> dict:
    """
    Generate synthetic training data with IQ samples and feature extraction.

    Args:
        dataset_id: Dataset UUID
        num_samples: Number of samples to generate
        receivers_config: List of receiver configurations
        training_config: Training configuration with bounding boxes
        config: Generation parameters (frequency, power, SNR thresholds, etc.)
        conn: Database connection (async)
        progress_callback: Optional callback for progress updates
        seed: Random seed for reproducibility

    Returns:
        dict with generation statistics
    """
    logger.info(f"Starting synthetic data generation with IQ: {num_samples} samples")

    # Extract generation parameters
    min_snr_db = config.get('min_snr_db', 3.0)
    min_receivers = config.get('min_receivers', 3)
    max_gdop = config.get('max_gdop', 10.0)

    # Determine number of worker threads
    # For I/O-bound tasks (which include numpy operations), use more threads than CPUs
    # Rule of thumb: 2-4x CPU count for mixed CPU/I/O workloads
    cpu_count = mp.cpu_count()

    # Check if we can get thread limit from environment or system
    try:
        # Try to get from system limit
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        # Use a reasonable fraction of the process limit
        max_threads_from_limit = min(soft_limit // 4, 128) if soft_limit != resource.RLIM_INFINITY else 128
    except:
        max_threads_from_limit = 128

    # For ThreadPoolExecutor: use 2-3x CPU count (good for numpy/scipy workloads)
    # but cap at reasonable maximum to avoid thread explosion
    num_workers = min(cpu_count * 3, max_threads_from_limit, 72)

    logger.info(f"Using {num_workers} worker threads for parallel generation (CPUs: {cpu_count})")

    # Convert training config to dict for serialization
    training_config_dict = {
        'receiver_bbox': {
            'lat_min': training_config.receiver_bbox.lat_min,
            'lat_max': training_config.receiver_bbox.lat_max,
            'lon_min': training_config.receiver_bbox.lon_min,
            'lon_max': training_config.receiver_bbox.lon_max
        },
        'training_bbox': {
            'lat_min': training_config.training_bbox.lat_min,
            'lat_max': training_config.training_bbox.lat_max,
            'lon_min': training_config.training_bbox.lon_min,
            'lon_max': training_config.training_bbox.lon_max
        }
    }

    # Convert receivers to simple dicts
    receivers_list = [
        {
            'name': r.name,
            'latitude': r.latitude,
            'longitude': r.longitude,
            'altitude': r.altitude
        }
        for r in receivers_config
    ]

    # Prepare arguments for parallel processing
    args_list = [
        (i, receivers_list, training_config_dict, config, seed)
        for i in range(num_samples)
    ]

    # Generate samples in parallel using threads
    # Note: ThreadPoolExecutor is used instead of ProcessPoolExecutor because
    # Celery worker processes are daemon processes and cannot spawn child processes
    generated_samples = []
    iq_samples_to_save = {}  # First 100 IQ samples

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single_sample, args): args[0]
                  for args in args_list}

        completed = 0
        processed = 0  # Track total samples processed (including rejected)

        for future in as_completed(futures):
            sample_idx = futures[future]
            processed += 1  # Increment for every sample processed, regardless of validation

            try:
                sample_idx_ret, receiver_features, extraction_metadata, quality_metrics, iq_samples, tx_position = future.result()

                # Validation checks
                if quality_metrics['num_receivers_detected'] < min_receivers:
                    logger.debug(f"Sample {sample_idx}: rejected (receivers={quality_metrics['num_receivers_detected']} < {min_receivers})")
                    # Still update progress for rejected samples
                    if progress_callback and processed % 10 == 0:
                        await progress_callback(processed, num_samples)
                        logger.info(f"Progress: {processed}/{num_samples} samples processed ({len(generated_samples)} valid)")
                    continue

                if quality_metrics['mean_snr_db'] < min_snr_db:
                    logger.debug(f"Sample {sample_idx}: rejected (SNR={quality_metrics['mean_snr_db']:.1f} < {min_snr_db})")
                    # Still update progress for rejected samples
                    if progress_callback and processed % 10 == 0:
                        await progress_callback(processed, num_samples)
                        logger.info(f"Progress: {processed}/{num_samples} samples processed ({len(generated_samples)} valid)")
                    continue

                if quality_metrics['gdop'] > max_gdop:
                    logger.debug(f"Sample {sample_idx}: rejected (GDOP={quality_metrics['gdop']:.1f} > {max_gdop})")
                    # Still update progress for rejected samples
                    if progress_callback and processed % 10 == 0:
                        await progress_callback(processed, num_samples)
                        logger.info(f"Progress: {processed}/{num_samples} samples processed ({len(generated_samples)} valid)")
                    continue

                # Store sample
                generated_samples.append({
                    'sample_idx': sample_idx_ret,
                    'receiver_features': receiver_features,
                    'extraction_metadata': extraction_metadata,
                    'quality_metrics': quality_metrics,
                    'tx_position': tx_position
                })

                # Store IQ samples (first 100 only)
                if sample_idx < 100 and iq_samples:
                    iq_samples_to_save[sample_idx] = iq_samples

                completed += 1

                # Progress update (for valid samples)
                if progress_callback and processed % 10 == 0:
                    await progress_callback(processed, num_samples)
                    logger.info(f"Progress: {processed}/{num_samples} samples processed ({completed} valid)")

            except Exception as e:
                logger.error(f"Error generating sample {sample_idx}: {e}")
                # Still update progress even on error
                if progress_callback and processed % 10 == 0:
                    await progress_callback(processed, num_samples)
                    logger.info(f"Progress: {processed}/{num_samples} samples processed ({completed} valid)")
                continue

    logger.info(f"Generated {len(generated_samples)} valid samples (success rate: {len(generated_samples)/num_samples*100:.1f}%)")

    # Save features to database
    await save_features_to_db(dataset_id, generated_samples, conn)

    # Save IQ samples to MinIO (first 100 only)
    if iq_samples_to_save:
        await save_iq_to_minio(dataset_id, iq_samples_to_save)

    return {
        'total_generated': len(generated_samples),
        'total_attempted': num_samples,
        'success_rate': len(generated_samples) / num_samples if num_samples > 0 else 0,
        'iq_samples_saved': len(iq_samples_to_save)
    }


async def save_features_to_db(
    dataset_id: uuid.UUID,
    samples: list[dict],
    conn
) -> None:
    """
    Save extracted features to measurement_features table.

    Args:
        dataset_id: Dataset UUID
        samples: List of sample dicts with receiver_features, metadata, quality_metrics
        conn: Database connection (async)
    """
    from sqlalchemy import text
    
    logger.info(f"Saving {len(samples)} feature samples to database")
    
    insert_features_query = text("""
        INSERT INTO heimdall.measurement_features (
            recording_session_id, timestamp, receiver_features, tx_latitude,
            tx_longitude, tx_altitude_m, tx_power_dbm, tx_known,
            extraction_metadata, overall_confidence, mean_snr_db,
            num_receivers_detected, gdop, extraction_failed, created_at
        )
        VALUES (
            :recording_session_id, NOW(), CAST(:receiver_features AS jsonb[]),
            :tx_latitude, :tx_longitude, :tx_altitude_m, :tx_power_dbm, TRUE,
            CAST(:extraction_metadata AS jsonb), :overall_confidence,
            :mean_snr_db, :num_receivers_detected, :gdop, FALSE, NOW()
        )
    """)

    for sample in samples:
        # Generate unique recording_session_id for synthetic sample
        recording_session_id = uuid.uuid4()

        # Convert receiver features list to PostgreSQL array of JSONB
        receiver_features_json_list = [
            json.dumps(rf) for rf in sample['receiver_features']
        ]

        # Insert features
        await conn.execute(
            insert_features_query,
            {
                'recording_session_id': str(recording_session_id),
                'receiver_features': f'{{{",".join(receiver_features_json_list)}}}',  # PostgreSQL array syntax
                'tx_latitude': sample['tx_position']['tx_lat'],
                'tx_longitude': sample['tx_position']['tx_lon'],
                'tx_altitude_m': sample['tx_position']['tx_alt'],
                'tx_power_dbm': sample['tx_position']['tx_power_dbm'],
                'extraction_metadata': json.dumps(sample['extraction_metadata']),
                'overall_confidence': sample['quality_metrics']['overall_confidence'],
                'mean_snr_db': sample['quality_metrics']['mean_snr_db'],
                'num_receivers_detected': sample['quality_metrics']['num_receivers_detected'],
                'gdop': sample['quality_metrics']['gdop']
            }
        )

    logger.info(f"Saved {len(samples)} feature samples to database")


async def save_iq_to_minio(
    dataset_id: uuid.UUID,
    iq_samples_dict: dict[int, dict[str, SyntheticIQSample]]
) -> None:
    """
    Save IQ samples to MinIO.

    Args:
        dataset_id: Dataset UUID
        iq_samples_dict: Dict mapping sample_idx to dict of {rx_id: SyntheticIQSample}
    """
    # Import MinIO client
    import sys
    sys.path.insert(0, '/app/backend/src')
    from storage.minio_client import MinIOClient
    from config import settings as backend_settings

    minio_client = MinIOClient(
        endpoint_url=backend_settings.minio_url,
        access_key=backend_settings.minio_access_key,
        secret_key=backend_settings.minio_secret_key,
        bucket_name="heimdall-synthetic-iq"
    )
    
    # Ensure bucket exists
    minio_client.ensure_bucket_exists()

    logger.info(f"Saving {len(iq_samples_dict)} IQ samples to MinIO")

    for sample_idx, receivers_iq in iq_samples_dict.items():
        for rx_id, iq_sample in receivers_iq.items():
            # Binary format: complex64 array
            buffer = io.BytesIO()
            np.save(buffer, iq_sample.samples)
            buffer.seek(0)

            object_name = f"synthetic/{dataset_id}/{sample_idx}/{rx_id}.npy"

            minio_client.s3_client.put_object(
                Bucket="heimdall-synthetic-iq",
                Key=object_name,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )

    logger.info(f"Saved {len(iq_samples_dict)} IQ samples to MinIO bucket 'heimdall-synthetic-iq'")
