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
from .iq_generator import SyntheticIQGenerator, SyntheticIQSample

# Import terrain from common module (shared across services)
from common.terrain import TerrainLookup

# Import from common module (PYTHONPATH=/app in Dockerfile)
from common.feature_extraction import RFFeatureExtractor, IQSample

logger = structlog.get_logger(__name__)


def _generate_random_receivers(
    num_receivers: int,
    area_lat_min: float,
    area_lat_max: float,
    area_lon_min: float,
    area_lon_max: float,
    terrain_lookup: Optional[TerrainLookup],
    rng: np.random.Generator
) -> list[dict]:
    """
    Generate random receiver positions within specified area.
    
    Args:
        num_receivers: Number of receivers to generate
        area_lat_min: Minimum latitude
        area_lat_max: Maximum latitude
        area_lon_min: Minimum longitude
        area_lon_max: Maximum longitude
        terrain_lookup: Terrain lookup for elevation (optional)
        rng: Random number generator
    
    Returns:
        List of receiver dicts with name, latitude, longitude, altitude
    """
    receivers = []
    for i in range(num_receivers):
        lat = rng.uniform(area_lat_min, area_lat_max)
        lon = rng.uniform(area_lon_min, area_lon_max)
        
        # Get altitude from terrain if available, otherwise use default
        if terrain_lookup:
            try:
                alt = terrain_lookup.get_elevation(lat, lon)
                # Add 10-20m above ground for antenna height
                alt += rng.uniform(10.0, 20.0)
            except Exception as e:
                logger.warning(f"Failed to get terrain elevation for ({lat}, {lon}): {e}, using default")
                alt = 300.0
        else:
            alt = 300.0
        
        receivers.append({
            'name': f'RX_{i:03d}',
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        })
    
    return receivers


def _generate_single_sample(args):
    """
    Generate a single synthetic sample (for multiprocessing).

    This function must be at module level for pickle serialization.

    Args:
        args: Tuple of (sample_index, receivers_config, training_config, config, seed, dataset_type, use_random_receivers)

    Returns:
        Tuple of (sample_idx, receiver_features, extraction_metadata, quality_metrics, iq_samples_dict, tx_position, num_receivers_in_sample)
    """
    sample_idx, receivers_list, training_config_dict, config, base_seed, dataset_type, use_random_receivers = args

    # Create unique seed for this sample
    sample_seed = base_seed + sample_idx if base_seed else None
    rng = np.random.default_rng(sample_seed)
    
    # Generate random receivers if requested (for iq_raw datasets)
    if use_random_receivers:
        min_rx = config.get('min_receivers_count', 5)
        max_rx = config.get('max_receivers_count', 10)
        num_receivers = rng.integers(min_rx, max_rx + 1)  # Inclusive upper bound
        
        # Initialize terrain lookup for random receiver generation
        terrain_lookup = TerrainLookup(use_srtm=False)  # Simplified model for speed
        
        receivers_list = _generate_random_receivers(
            num_receivers=num_receivers,
            area_lat_min=config.get('area_lat_min', 44.0),
            area_lat_max=config.get('area_lat_max', 46.0),
            area_lon_min=config.get('area_lon_min', 7.0),
            area_lon_max=config.get('area_lon_max', 10.0),
            terrain_lookup=terrain_lookup,
            rng=rng
        )
        logger.debug(f"Sample {sample_idx}: Generated {len(receivers_list)} random receivers")
    
    num_receivers_in_sample = len(receivers_list)

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

        # Store IQ sample
        # For iq_raw datasets: store ALL samples
        # For feature_based datasets: store first 100 only (legacy behavior)
        if dataset_type == 'iq_raw' or sample_idx < 100:
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
        features_dict['num_receivers_in_sample'] = num_receivers_in_sample  # For iq_raw datasets

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

    return (sample_idx, receiver_features_list, extraction_metadata, quality_metrics, iq_samples, tx_position, num_receivers_in_sample)


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
                sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
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
        frequency_mhz: float = 145.0,
        tx_power_dbm: float = 37.0,
        min_snr_db: float = 3.0,
        min_receivers: int = 3,
        max_gdop: float = 10.0,
        progress_callback=None
    ) -> List[SyntheticSample]:
        """
        Generate synthetic training samples.

        Data splits (train/val/test) are NOT assigned here - they will be
        calculated at training time to allow flexible dataset combination.

        Args:
            num_samples: Number of samples to generate
            inside_ratio: Ratio of transmitters inside receiver network (0.7 = 70%)
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

            # Create sample (no split assignment - will be done at training time)
            sample = SyntheticSample(
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_power_dbm=tx_power_dbm,
                frequency_hz=int(frequency_mhz * 1e6),
                receivers=receiver_measurements,
                gdop=gdop,
                num_receivers=receivers_with_signal,
                split="train"  # Placeholder - actual split assigned at training time
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
    seed: Optional[int] = None,
    job_id: Optional[str] = None,
    dataset_type: str = 'feature_based'
) -> dict:
    """
    Generate synthetic training data with IQ samples and feature extraction.

    Args:
        dataset_id: Dataset UUID
        num_samples: Number of samples to generate
        receivers_config: List of receiver configurations (ignored if dataset_type='iq_raw')
        training_config: Training configuration with bounding boxes
        config: Generation parameters (frequency, power, SNR thresholds, etc.)
        conn: Database connection (async)
        progress_callback: Optional callback for progress updates
        seed: Random seed for reproducibility
        job_id: Optional job ID for cancellation detection
        dataset_type: Dataset type ('feature_based' or 'iq_raw')

    Returns:
        dict with generation statistics
    """
    logger.info(f"Starting synthetic data generation with IQ: {num_samples} samples (type: {dataset_type})")

    # Extract generation parameters
    min_snr_db = config.get('min_snr_db', 3.0)
    min_receivers = config.get('min_receivers', 3)
    max_gdop = config.get('max_gdop', 10.0)
    
    # For iq_raw datasets, use random receivers
    use_random_receivers = (dataset_type == 'iq_raw') or config.get('use_random_receivers', False)
    
    # For iq_raw with random receivers, GDOP filtering is less useful
    # (random geometry often has high GDOP, but that's OK for training)
    # Relax GDOP constraint to allow more diverse geometries
    if dataset_type == 'iq_raw' and use_random_receivers:
        if max_gdop < 200:
            logger.warning(f"IQ-raw with random receivers: relaxing max_gdop from {max_gdop} to 200 for better success rate")
            max_gdop = 200.0

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
        (i, receivers_list, training_config_dict, config, seed, dataset_type, use_random_receivers)
        for i in range(num_samples)
    ]

    # Generate samples in parallel using threads
    # Note: ThreadPoolExecutor is used instead of ProcessPoolExecutor because
    # Celery worker processes are daemon processes and cannot spawn child processes
    generated_samples = []
    iq_samples_to_save = {}  # First 100 IQ samples

    # Track time for progress updates (every 1 second)
    import time
    last_progress_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single_sample, args): args[0]
                  for args in args_list}

        completed = 0
        processed = 0  # Track total samples processed (including rejected)

        for future in as_completed(futures):
            sample_idx = futures[future]
            processed += 1  # Increment for every sample processed, regardless of validation

            # Check for cancellation every 100 samples
            if job_id and processed % 100 == 0:
                from sqlalchemy import text
                check_query = text("""
                    SELECT status FROM heimdall.training_jobs WHERE id = :job_id
                """)
                result = await conn.execute(check_query, {"job_id": job_id})
                row = result.fetchone()  # fetchone() returns immediately, no await needed
                if row and row[0] == 'cancelled':
                    logger.warning(f"Job {job_id} was cancelled, stopping generation at {processed}/{num_samples}")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    # Break out of loop gracefully
                    break

            try:
                sample_idx_ret, receiver_features, extraction_metadata, quality_metrics, iq_samples, tx_position, num_receivers_in_sample = future.result()

                # Validation checks
                if quality_metrics['num_receivers_detected'] < min_receivers:
                    logger.debug(f"Sample {sample_idx}: rejected (receivers={quality_metrics['num_receivers_detected']} < {min_receivers})")
                    continue

                if quality_metrics['mean_snr_db'] < min_snr_db:
                    logger.debug(f"Sample {sample_idx}: rejected (SNR={quality_metrics['mean_snr_db']:.1f} < {min_snr_db})")
                    continue

                if quality_metrics['gdop'] > max_gdop:
                    logger.debug(f"Sample {sample_idx}: rejected (GDOP={quality_metrics['gdop']:.1f} > {max_gdop})")
                    continue

                # Store sample
                generated_samples.append({
                    'sample_idx': sample_idx_ret,
                    'receiver_features': receiver_features,
                    'extraction_metadata': extraction_metadata,
                    'quality_metrics': quality_metrics,
                    'tx_position': tx_position
                })

                # Store IQ samples
                # For iq_raw: store ALL samples
                # For feature_based: store first 100 only (legacy behavior)
                if iq_samples and (dataset_type == 'iq_raw' or sample_idx < 100):
                    iq_samples_to_save[sample_idx] = iq_samples

                completed += 1

                # Incremental save every 10 valid samples
                if completed % 10 == 0 and len(generated_samples) >= 10:
                    # Save last 10 samples to database
                    samples_to_save = generated_samples[-10:]
                    try:
                        await save_features_to_db(dataset_id, samples_to_save, conn)
                        logger.info(f"Incrementally saved 10 samples (total saved: {completed})")
                    except Exception as e:
                        logger.error(f"Failed to incrementally save samples: {e}")

            except Exception as e:
                logger.error(f"Error generating sample {sample_idx}: {e}")
                continue
            
            # Progress update every 1 second (time-based instead of sample-based)
            current_time = time.time()
            time_elapsed = current_time - last_progress_time
            logger.debug(f"Progress check: callback={progress_callback is not None}, time_elapsed={time_elapsed:.2f}s, condition={(progress_callback is not None) and (time_elapsed >= 1.0)}")
            if progress_callback and (current_time - last_progress_time) >= 1.0:
                logger.info(f"[PROGRESS DEBUG] Calling progress_callback with processed={processed}, num_samples={num_samples}")
                try:
                    await progress_callback(processed, num_samples)
                    logger.info(f"[PROGRESS DEBUG] Callback completed successfully")
                except Exception as e:
                    logger.error(f"[PROGRESS DEBUG] Callback failed: {e}", exc_info=True)
                logger.info(f"[PROGRESS] Progress: {processed}/{num_samples} samples processed ({completed} valid)")
                last_progress_time = current_time

    logger.info(f"Generated {len(generated_samples)} valid samples (success rate: {len(generated_samples)/num_samples*100:.1f}%)")

    # Save remaining samples to database (those not saved incrementally)
    remaining_samples = len(generated_samples) % 10
    if remaining_samples > 0:
        samples_to_save = generated_samples[-remaining_samples:]
        await save_features_to_db(dataset_id, samples_to_save, conn)
        logger.info(f"Saved final {remaining_samples} remaining samples")

    # Save IQ samples to MinIO and database
    # For feature_based: save first 100 only
    # For iq_raw: save ALL samples
    if iq_samples_to_save:
        storage_paths = await save_iq_to_minio(dataset_id, iq_samples_to_save)
        
        # For iq_raw datasets, also save metadata to synthetic_iq_samples table
        if dataset_type == 'iq_raw':
            # Filter generated_samples to only include those with IQ storage paths
            samples_with_iq = [s for s in generated_samples if s['sample_idx'] in storage_paths]
            await save_iq_metadata_to_db(dataset_id, samples_with_iq, storage_paths, conn)

    return {
        'total_generated': len(generated_samples),
        'total_attempted': num_samples,
        'success_rate': len(generated_samples) / num_samples if num_samples > 0 else 0,
        'iq_samples_saved': len(iq_samples_to_save),
        'dataset_type': dataset_type
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
    logger.info(f"Saving {len(samples)} feature samples to database")

    # Get the underlying asyncpg connection from SQLAlchemy
    # This bypasses SQLAlchemy's parameter handling which has issues with jsonb[]
    raw_conn = await conn.get_raw_connection()
    asyncpg_conn = raw_conn.driver_connection

    insert_features_query = """
        INSERT INTO heimdall.measurement_features (
            recording_session_id, dataset_id, timestamp, receiver_features, tx_latitude,
            tx_longitude, tx_altitude_m, tx_power_dbm, tx_known,
            extraction_metadata, overall_confidence, mean_snr_db,
            num_receivers_detected, gdop, extraction_failed, created_at
        )
        VALUES (
            $1, $2, NOW(), $3::jsonb[],
            $4, $5, $6, $7, TRUE,
            $8::jsonb, $9,
            $10, $11, $12, FALSE, NOW()
        )
    """

    for sample in samples:
        # For synthetic data: generate unique UUID for each sample, but track dataset_id
        # recording_session_id = unique UUID (PRIMARY KEY)
        # dataset_id = synthetic dataset UUID (for filtering/grouping)
        recording_session_id = uuid.uuid4()

        # Convert receiver features to list of JSON strings
        # Raw asyncpg can handle this directly without PostgreSQL array literal format
        receiver_features_json_strings = [json.dumps(feat) for feat in sample['receiver_features']]

        # Use raw asyncpg connection execute (NOT SQLAlchemy wrapper)
        await asyncpg_conn.execute(
            insert_features_query,
            str(recording_session_id),                               # $1 - Unique UUID for this sample
            str(dataset_id),                                         # $2 - Dataset ID for filtering
            receiver_features_json_strings,                          # $3 - List of JSON strings (raw asyncpg handles this)
            sample['tx_position']['tx_lat'],                         # $4
            sample['tx_position']['tx_lon'],                         # $5
            sample['tx_position']['tx_alt'],                         # $6
            sample['tx_position']['tx_power_dbm'],                   # $7
            json.dumps(sample['extraction_metadata']),               # $8
            sample['quality_metrics']['overall_confidence'],         # $9
            sample['quality_metrics']['mean_snr_db'],                # $10
            sample['quality_metrics']['num_receivers_detected'],     # $11
            sample['quality_metrics']['gdop']                        # $12
        )

    logger.info(f"Saved {len(samples)} feature samples to database")


async def save_iq_to_minio(
    dataset_id: uuid.UUID,
    iq_samples_dict: dict[int, dict[str, SyntheticIQSample]]
) -> dict[int, dict[str, str]]:
    """
    Save IQ samples to MinIO.

    Args:
        dataset_id: Dataset UUID
        iq_samples_dict: Dict mapping sample_idx to dict of {rx_id: SyntheticIQSample}
    
    Returns:
        Dict mapping sample_idx to dict of {rx_id: minio_path}
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

    storage_paths = {}  # {sample_idx: {rx_id: path}}
    
    for sample_idx, receivers_iq in iq_samples_dict.items():
        storage_paths[sample_idx] = {}
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
            
            storage_paths[sample_idx][rx_id] = object_name

    logger.info(f"Saved {len(iq_samples_dict)} IQ samples to MinIO bucket 'heimdall-synthetic-iq'")
    
    return storage_paths


async def save_iq_metadata_to_db(
    dataset_id: uuid.UUID,
    samples: list[dict],
    storage_paths: dict[int, dict[str, str]],
    conn
) -> None:
    """
    Save IQ sample metadata to synthetic_iq_samples table.
    
    Args:
        dataset_id: Dataset UUID
        samples: List of sample dicts with metadata
        storage_paths: Dict mapping sample_idx to {rx_id: minio_path}
        conn: Database connection (async)
    """
    logger.info(f"Saving {len(samples)} IQ metadata samples to database")

    # Get raw asyncpg connection
    raw_conn = await conn.get_raw_connection()
    asyncpg_conn = raw_conn.driver_connection

    insert_query = """
        INSERT INTO heimdall.synthetic_iq_samples (
            timestamp, dataset_id, sample_idx, tx_lat, tx_lon, tx_alt, tx_power_dbm,
            frequency_hz, receivers_metadata, num_receivers, gdop, mean_snr_db,
            overall_confidence, iq_metadata, iq_storage_paths, created_at
        )
        VALUES (
            NOW(), $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13::jsonb, $14::jsonb, NOW()
        )
        ON CONFLICT (dataset_id, sample_idx) DO NOTHING
    """

    for sample in samples:
        sample_idx = sample['sample_idx']
        
        # Skip if no IQ storage paths (shouldn't happen for iq_raw datasets)
        if sample_idx not in storage_paths:
            logger.warning(f"Sample {sample_idx}: No IQ storage paths found, skipping metadata save")
            continue
        
        # Build receivers_metadata from receiver_features
        receivers_metadata = []
        for feat in sample['receiver_features']:
            receivers_metadata.append({
                'rx_id': feat['rx_id'],
                'lat': feat['rx_lat'],
                'lon': feat['rx_lon'],
                'alt': feat.get('rx_alt', 300.0),  # Fallback to default
                'distance_km': feat['distance_km'],
                'snr_db': feat.get('snr_db', 0.0),
                'signal_present': feat['signal_present']
            })
        
        # Build IQ metadata from extraction metadata
        iq_metadata = {
            'sample_rate_hz': sample['extraction_metadata'].get('sample_rate_hz', 200000),
            'duration_ms': sample['extraction_metadata'].get('iq_duration_ms', 1000.0),
            'center_frequency_hz': int(sample['extraction_metadata'].get('frequency_mhz', 145.0) * 1e6)
        }

        await asyncpg_conn.execute(
            insert_query,
            str(dataset_id),                                      # $1
            sample_idx,                                           # $2
            sample['tx_position']['tx_lat'],                      # $3
            sample['tx_position']['tx_lon'],                      # $4
            sample['tx_position']['tx_alt'],                      # $5
            sample['tx_position']['tx_power_dbm'],                # $6
            int(sample['extraction_metadata'].get('frequency_mhz', 145.0) * 1e6),  # $7
            json.dumps(receivers_metadata),                       # $8
            len(receivers_metadata),                              # $9
            sample['quality_metrics']['gdop'],                    # $10
            sample['quality_metrics']['mean_snr_db'],             # $11
            sample['quality_metrics']['overall_confidence'],      # $12
            json.dumps(iq_metadata),                              # $13
            json.dumps(storage_paths[sample_idx])                 # $14
        )

    logger.info(f"Saved {len(samples)} IQ metadata samples to database")
