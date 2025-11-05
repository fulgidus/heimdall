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
import threading
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import TrainingConfig, TxAntennaDistribution, RxAntennaDistribution
from .propagation import (
    RFPropagationModel, 
    calculate_psd, 
    calculate_frequency_offset, 
    calculate_gdop,
    AntennaType,
    AntennaPattern
)
from .iq_generator import SyntheticIQGenerator, SyntheticIQSample

# Import terrain from common module (shared across services)
from common.terrain import TerrainLookup

# Import from common module (PYTHONPATH=/app in Dockerfile)
from common.feature_extraction import RFFeatureExtractor, IQSample

logger = structlog.get_logger(__name__)

# Thread-local storage for reusable extractors (avoid re-creating for each sample)
_thread_local = threading.local()


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


def _select_tx_antenna(rng: np.random.Generator, tx_antenna_dist=None) -> AntennaPattern:
    """
    Select a realistic TX antenna based on configurable distribution.
    
    Default distribution (if tx_antenna_dist not provided):
    - 90% WHIP (mobile vehicle antenna)
    - 8% RUBBER_DUCK (handheld radio)
    - 2% PORTABLE_DIRECTIONAL (portable beam)
    
    Args:
        rng: Random number generator
        tx_antenna_dist: TxAntennaDistribution instance or dict (optional, uses defaults if None)
    
    Returns:
        AntennaPattern instance for TX
    """
    # Use provided distribution or defaults
    if tx_antenna_dist is not None:
        # Convert dict to dataclass if needed
        if isinstance(tx_antenna_dist, dict):
            tx_antenna_dist = TxAntennaDistribution(**tx_antenna_dist)
        antenna_probs = [tx_antenna_dist.whip, tx_antenna_dist.rubber_duck, tx_antenna_dist.portable_directional]
    else:
        antenna_probs = [0.90, 0.08, 0.02]  # Default distribution
    
    antenna_types = [AntennaType.WHIP, AntennaType.RUBBER_DUCK, AntennaType.PORTABLE_DIRECTIONAL]
    
    antenna_type = rng.choice(antenna_types, p=antenna_probs)
    
    # For directional portable antennas, randomize pointing direction
    if antenna_type == AntennaType.PORTABLE_DIRECTIONAL:
        pointing_azimuth = rng.uniform(0.0, 360.0)
    else:
        pointing_azimuth = 0.0  # Omnidirectional, doesn't matter
    
    return AntennaPattern(antenna_type, pointing_azimuth)


def _select_rx_antenna(rng: np.random.Generator, rx_antenna_dist=None) -> AntennaPattern:
    """
    Select a realistic RX antenna based on configurable distribution.
    
    Default distribution (if rx_antenna_dist not provided):
    - 80% OMNI_VERTICAL (most common for monitoring stations)
    - 15% YAGI (directional, pointed at specific area)
    - 5% COLLINEAR (high-gain omnidirectional)
    
    Args:
        rng: Random number generator
        rx_antenna_dist: RxAntennaDistribution instance or dict (optional, uses defaults if None)
    
    Returns:
        AntennaPattern instance for RX
    """
    # Use provided distribution or defaults
    if rx_antenna_dist is not None:
        # Convert dict to dataclass if needed
        if isinstance(rx_antenna_dist, dict):
            rx_antenna_dist = RxAntennaDistribution(**rx_antenna_dist)
        antenna_probs = [rx_antenna_dist.omni_vertical, rx_antenna_dist.yagi, rx_antenna_dist.collinear]
    else:
        antenna_probs = [0.80, 0.15, 0.05]  # Default distribution
    
    antenna_types = [AntennaType.OMNI_VERTICAL, AntennaType.YAGI, AntennaType.COLLINEAR]
    
    antenna_type = rng.choice(antenna_types, p=antenna_probs)
    
    # For directional antennas (Yagi), randomize pointing direction
    if antenna_type == AntennaType.YAGI:
        pointing_azimuth = rng.uniform(0.0, 360.0)
    else:
        pointing_azimuth = 0.0  # Omnidirectional, doesn't matter
    
    return AntennaPattern(antenna_type, pointing_azimuth)


def _generate_single_sample_no_features(args):
    """
    Generate a single synthetic sample WITHOUT feature extraction (for batch processing).
    
    This function generates IQ samples and metadata but does NOT extract features.
    Feature extraction is deferred to allow batch processing of multiple samples at once,
    which amortizes GPU initialization overhead.

    This function must be at module level for pickle serialization.

    Args:
        args: Tuple of (sample_index, receivers_config, training_config, config, seed, dataset_type, use_random_receivers)

    Returns:
        Tuple of (sample_idx, iq_samples_for_extraction, receivers_list, distance_kms, signal_presents, 
                  tx_position, propagation_metadata, num_receivers_in_sample, iq_samples_dict)
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

    # Use thread-local storage to reuse extractors instead of creating new ones for each sample
    # This reduces GPU initialization overhead from ~100ms to ~1ms per sample
    if not hasattr(_thread_local, 'iq_generator'):
        # Initialize generators with GPU acceleration (ONCE per thread)
        # Check if GPU is available (use torch as proxy since it's always available in training service)
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        
        _thread_local.iq_generator = SyntheticIQGenerator(
            sample_rate_hz=200_000,  # 200 kHz
            duration_ms=1000.0,       # 1 second
            seed=None,  # Will be reset per sample below
            use_gpu=use_gpu
        )
        _thread_local.propagation = RFPropagationModel()
        
        # Initialize terrain lookup (ONCE per thread)
        # Get terrain configuration from config dict
        use_srtm = config.get('use_srtm_terrain', False)
        if use_srtm:
            try:
                # Recreate MinIO client in worker thread (cannot be pickled/shared)
                import sys
                import os
                sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
                from storage.minio_client import MinIOClient
                from config import settings as backend_settings
                
                minio_client = MinIOClient(
                    endpoint_url=backend_settings.minio_url,
                    access_key=backend_settings.minio_access_key,
                    secret_key=backend_settings.minio_secret_key,
                    bucket_name="heimdall-terrain"
                )
                _thread_local.terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)
                logger.info(f"Thread {threading.current_thread().name}: Initialized SRTM terrain lookup")
            except Exception as e:
                logger.warning(f"Thread {threading.current_thread().name}: Failed to initialize SRTM terrain: {e}, using simplified model")
                _thread_local.terrain = TerrainLookup(use_srtm=False)
        else:
            _thread_local.terrain = TerrainLookup(use_srtm=False)
        
        logger.info(f"Thread {threading.current_thread().name}: Initialized reusable generators (GPU={use_gpu}, SRTM={use_srtm})")
    
    # Reuse thread-local instances
    iq_generator = _thread_local.iq_generator
    propagation = _thread_local.propagation
    terrain = _thread_local.terrain
    
    # Reset RNG seed for this sample (iq_generator supports seed reset)
    iq_generator.rng = np.random.default_rng(sample_seed)

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

    # PRE-CHECK GEOMETRY: Calculate GDOP BEFORE generating IQ samples
    # This avoids wasting time on samples that will be rejected
    receiver_positions_precheck = [(rx['latitude'], rx['longitude']) for rx in receivers_list]
    gdop_precheck = calculate_gdop(receiver_positions_precheck, (tx_lat, tx_lon)) if len(receiver_positions_precheck) >= 3 else 999.0
    
    # Get max_gdop and min_receivers from config
    max_gdop_threshold = config.get('max_gdop', 150.0)
    min_receivers_threshold = config.get('min_receivers', 3)
    
    # Early rejection: if GDOP is already too high, skip IQ generation
    if gdop_precheck > max_gdop_threshold or len(receiver_positions_precheck) < min_receivers_threshold:
        logger.debug(f"Sample {sample_idx}: PRE-REJECTED (GDOP={gdop_precheck:.1f} > {max_gdop_threshold} or receivers={len(receiver_positions_precheck)} < {min_receivers_threshold})")
        # Return None to signal rejection (will be filtered out later)
        return None

    # Step 1: Select antenna patterns for TX and RX stations
    # Extract antenna distributions from config (if provided)
    tx_antenna_dist = config.get('tx_antenna_dist', None)
    rx_antenna_dist = config.get('rx_antenna_dist', None)
    tx_antenna = _select_tx_antenna(rng, tx_antenna_dist)
    
    # Step 2: Calculate propagation for all receivers (vectorizable in future)
    num_receivers = len(receivers_list)
    rx_powers_dbm = []
    snr_dbs = []
    distance_kms = []
    signal_presents = []
    
    for receiver in receivers_list:
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']
        rx_alt = receiver['altitude']
        
        # Each receiver has its own antenna (with random type and pointing direction)
        rx_antenna = _select_rx_antenna(rng, rx_antenna_dist)

        # Calculate received power and SNR using propagation model (with antenna gains)
        rx_power_dbm, snr_db, details = propagation.calculate_received_power(
            tx_power_dbm=tx_power_dbm,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=frequency_mhz,
            terrain_lookup=terrain,
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna
        )
        
        rx_powers_dbm.append(rx_power_dbm)
        snr_dbs.append(snr_db)
        distance_kms.append(details['distance_km'])
        signal_presents.append(snr_db >= min_snr_db)
    
    # Step 3: Prepare batch parameters for GPU vectorized IQ generation
    frequency_offsets = rng.uniform(-50, 50, num_receivers).astype(np.float32)
    bandwidths = np.full(num_receivers, 12500.0, dtype=np.float32)  # FM signal bandwidth
    signal_powers_dbm = np.array(rx_powers_dbm, dtype=np.float32)
    noise_floors_dbm = np.full(num_receivers, -120.0, dtype=np.float32)
    snr_dbs_array = np.array(snr_dbs, dtype=np.float32)
    
    # Step 3: Generate ALL IQ samples at once using GPU batch processing
    iq_batch = iq_generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs_array,
        batch_size=num_receivers
    )  # Returns shape: (num_receivers, num_samples)
    
    # Step 4: Process batch results - create IQ samples and prepare for batched feature extraction
    iq_samples = {}
    iq_samples_for_extraction = []  # Collect IQ samples for batch processing
    receivers_with_signal = 0
    propagation_snr_values = []  # Track SNR from propagation model
    
    # First pass: Create IQ samples for all receivers
    for i, receiver in enumerate(receivers_list):
        rx_id = receiver['name']
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']
        rx_alt = receiver['altitude']
        
        signal_present = signal_presents[i]
        snr_db = snr_dbs[i]
        
        if signal_present:
            receivers_with_signal += 1
            propagation_snr_values.append(snr_db)
        
        # Extract IQ data for this receiver from batch
        iq_data = iq_batch[i]  # Shape: (num_samples,) complex64
        
        # Create SyntheticIQSample object (for storage/metadata)
        iq_sample = SyntheticIQSample(
            samples=iq_data,
            sample_rate_hz=200_000.0,
            duration_ms=1000.0,  # Fixed 1 second duration
            center_frequency_hz=frequency_mhz * 1e6,
            rx_id=rx_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=float(sample_idx)
        )
        
        # Store IQ sample (for MinIO storage)
        if dataset_type == 'iq_raw' or sample_idx < 100:
            iq_samples[rx_id] = iq_sample
        
        # Convert to IQSample for feature extraction
        iq_sample_for_extraction = IQSample(
            samples=iq_sample.samples,
            sample_rate_hz=int(iq_sample.sample_rate_hz),
            center_frequency_hz=int(iq_sample.center_frequency_hz),
            rx_id=iq_sample.rx_id,
            rx_lat=iq_sample.rx_lat,
            rx_lon=iq_sample.rx_lon,
            timestamp=iq_sample.to_datetime()
        )
        iq_samples_for_extraction.append(iq_sample_for_extraction)
    
    # NO FEATURE EXTRACTION HERE - deferred to batch processing
    
    # Build propagation metadata for later feature extraction
    propagation_metadata = {
        'frequency_mhz': frequency_mhz,
        'tx_power_dbm': tx_power_dbm,
        'sample_rate_hz': 200_000,
        'iq_duration_ms': 1000.0,
        'num_chunks': 5,
        'chunk_duration_ms': 200.0,
        'generated_at': sample_idx,
        'propagation_snr_values': propagation_snr_values,
        'receivers_with_signal': receivers_with_signal
    }

    tx_position = {
        'tx_lat': tx_lat,
        'tx_lon': tx_lon,
        'tx_alt': tx_alt,
        'tx_power_dbm': tx_power_dbm
    }

    return (sample_idx, iq_samples_for_extraction, receivers_list, distance_kms, 
            signal_presents, tx_position, propagation_metadata, num_receivers_in_sample, iq_samples)


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

    # Use thread-local storage to reuse extractors instead of creating new ones for each sample
    # This reduces GPU initialization overhead from ~100ms to ~1ms per sample
    if not hasattr(_thread_local, 'iq_generator'):
        # Initialize generators with GPU acceleration (ONCE per thread)
        # Check if GPU is available (use torch as proxy since it's always available in training service)
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        
        _thread_local.iq_generator = SyntheticIQGenerator(
            sample_rate_hz=200_000,  # 200 kHz
            duration_ms=1000.0,       # 1 second
            seed=None,  # Will be reset per sample below
            use_gpu=use_gpu
        )
        # Create feature extractor with GPU acceleration (reusable)
        _thread_local.feature_extractor = RFFeatureExtractor(
            sample_rate_hz=200_000,
            use_gpu=use_gpu
        )
        _thread_local.propagation = RFPropagationModel()
        
        # Initialize terrain lookup (ONCE per thread)
        # Get terrain configuration from config dict
        use_srtm = config.get('use_srtm_terrain', False)
        if use_srtm:
            try:
                # Recreate MinIO client in worker thread (cannot be pickled/shared)
                import sys
                import os
                sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
                from storage.minio_client import MinIOClient
                from config import settings as backend_settings
                
                minio_client = MinIOClient(
                    endpoint_url=backend_settings.minio_url,
                    access_key=backend_settings.minio_access_key,
                    secret_key=backend_settings.minio_secret_key,
                    bucket_name="heimdall-terrain"
                )
                _thread_local.terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)
                logger.info(f"Thread {threading.current_thread().name}: Initialized SRTM terrain lookup")
            except Exception as e:
                logger.warning(f"Thread {threading.current_thread().name}: Failed to initialize SRTM terrain: {e}, using simplified model")
                _thread_local.terrain = TerrainLookup(use_srtm=False)
        else:
            _thread_local.terrain = TerrainLookup(use_srtm=False)
        
        logger.info(f"Thread {threading.current_thread().name}: Initialized reusable generators (GPU={use_gpu}, SRTM={use_srtm})")
    
    # Reuse thread-local instances
    iq_generator = _thread_local.iq_generator
    feature_extractor = _thread_local.feature_extractor
    propagation = _thread_local.propagation
    terrain = _thread_local.terrain
    
    # Reset RNG seed for this sample (iq_generator supports seed reset)
    iq_generator.rng = np.random.default_rng(sample_seed)

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

    # PRE-CHECK GEOMETRY: Calculate GDOP BEFORE generating IQ samples
    # This avoids wasting time on samples that will be rejected
    receiver_positions_precheck = [(rx['latitude'], rx['longitude']) for rx in receivers_list]
    gdop_precheck = calculate_gdop(receiver_positions_precheck, (tx_lat, tx_lon)) if len(receiver_positions_precheck) >= 3 else 999.0
    
    # Get max_gdop and min_receivers from config
    max_gdop_threshold = config.get('max_gdop', 150.0)
    min_receivers_threshold = config.get('min_receivers', 3)
    
    # Early rejection: if GDOP is already too high, skip IQ generation
    if gdop_precheck > max_gdop_threshold or len(receiver_positions_precheck) < min_receivers_threshold:
        logger.debug(f"Sample {sample_idx}: PRE-REJECTED (GDOP={gdop_precheck:.1f} > {max_gdop_threshold} or receivers={len(receiver_positions_precheck)} < {min_receivers_threshold})")
        # Return None to signal rejection (will be filtered out later)
        return None

    # Step 1: Select TX antenna (once per sample)
    # Extract antenna distributions from config (if provided)
    tx_antenna_dist = config.get('tx_antenna_dist', None)
    rx_antenna_dist = config.get('rx_antenna_dist', None)
    tx_antenna = _select_tx_antenna(rng, tx_antenna_dist)
    
    # Step 2: Calculate propagation for all receivers (vectorizable in future)
    num_receivers = len(receivers_list)
    rx_powers_dbm = []
    snr_dbs = []
    distance_kms = []
    signal_presents = []
    
    for receiver in receivers_list:
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']
        rx_alt = receiver['altitude']

        # Select RX antenna for this receiver
        rx_antenna = _select_rx_antenna(rng, rx_antenna_dist)

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
            terrain_lookup=terrain,
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna
        )
        
        rx_powers_dbm.append(rx_power_dbm)
        snr_dbs.append(snr_db)
        distance_kms.append(details['distance_km'])
        signal_presents.append(snr_db >= min_snr_db)
    
    # Step 2: Prepare batch parameters for GPU vectorized IQ generation
    frequency_offsets = rng.uniform(-50, 50, num_receivers).astype(np.float32)
    bandwidths = np.full(num_receivers, 12500.0, dtype=np.float32)  # FM signal bandwidth
    signal_powers_dbm = np.array(rx_powers_dbm, dtype=np.float32)
    noise_floors_dbm = np.full(num_receivers, -120.0, dtype=np.float32)
    snr_dbs_array = np.array(snr_dbs, dtype=np.float32)
    
    # Step 3: Generate ALL IQ samples at once using GPU batch processing
    iq_batch = iq_generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs_array,
        batch_size=num_receivers
    )  # Returns shape: (num_receivers, num_samples)
    
    # Step 4: Process batch results - create IQ samples and prepare for batched feature extraction
    iq_samples = {}
    iq_samples_for_extraction = []  # Collect IQ samples for batch processing
    receivers_with_signal = 0
    propagation_snr_values = []  # Track SNR from propagation model
    
    # First pass: Create IQ samples for all receivers
    for i, receiver in enumerate(receivers_list):
        rx_id = receiver['name']
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']
        rx_alt = receiver['altitude']
        
        signal_present = signal_presents[i]
        snr_db = snr_dbs[i]
        
        if signal_present:
            receivers_with_signal += 1
            propagation_snr_values.append(snr_db)
        
        # Extract IQ data for this receiver from batch
        iq_data = iq_batch[i]  # Shape: (num_samples,) complex64
        
        # Create SyntheticIQSample object (for storage/metadata)
        iq_sample = SyntheticIQSample(
            samples=iq_data,
            sample_rate_hz=200_000.0,
            duration_ms=1000.0,  # Fixed 1 second duration
            center_frequency_hz=frequency_mhz * 1e6,
            rx_id=rx_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=float(sample_idx)
        )
        
        # Store IQ sample (for MinIO storage)
        if dataset_type == 'iq_raw' or sample_idx < 100:
            iq_samples[rx_id] = iq_sample
        
        # Convert to IQSample for feature extraction
        iq_sample_for_extraction = IQSample(
            samples=iq_sample.samples,
            sample_rate_hz=int(iq_sample.sample_rate_hz),
            center_frequency_hz=int(iq_sample.center_frequency_hz),
            rx_id=iq_sample.rx_id,
            rx_lat=iq_sample.rx_lat,
            rx_lon=iq_sample.rx_lon,
            timestamp=iq_sample.to_datetime()
        )
        iq_samples_for_extraction.append(iq_sample_for_extraction)
    
    # Step 5: Extract features from ALL receivers at once (GPU batched processing)
    # This processes one chunk at a time across all 7 receivers to avoid OOM
    features_dicts = feature_extractor.extract_features_batch_conservative(
        iq_samples_list=iq_samples_for_extraction,
        chunk_duration_ms=200.0,
        num_chunks=5
    )
    
    # Step 6: Add receiver metadata to each feature dict
    receiver_features_list = []
    for i, features_dict in enumerate(features_dicts):
        receiver = receivers_list[i]
        
        # Add receiver metadata
        features_dict['rx_id'] = receiver['name']
        features_dict['rx_lat'] = receiver['latitude']
        features_dict['rx_lon'] = receiver['longitude']
        features_dict['distance_km'] = distance_kms[i]
        features_dict['num_receivers_in_sample'] = num_receivers_in_sample
        
        # Override signal_present with propagation model result
        features_dict['signal_present'] = signal_presents[i]
        
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
        
        # Validate SRTM tiles if using SRTM terrain
        if self.terrain.use_srtm and hasattr(self, 'config'):
            missing_tiles = self._validate_srtm_tiles()
            if missing_tiles:
                logger.warning(
                    "Missing SRTM tiles detected",
                    missing_tiles=missing_tiles,
                    message="Download tiles before generation or enable fallback_to_simplified_terrain"
                )
        
        logger.info(
            "Initialized synthetic data generator",
            num_receivers=len(self.config.receivers),
            training_area_km2=self.config.training_bbox.width_km() * self.config.training_bbox.height_km(),
            using_srtm=self.terrain.use_srtm
        )
    
    def _validate_srtm_tiles(self) -> List[str]:
        """
        Validate that required SRTM tiles are available in MinIO.
        
        Returns:
            List of missing tile names (e.g., ['N44E007', 'N45E008'])
        """
        if not hasattr(self.terrain, 'minio_client') or self.terrain.minio_client is None:
            logger.warning("Cannot validate SRTM tiles: MinIO client not available")
            return []
        
        missing_tiles = []
        for lat, lon in self.config.srtm_tiles:
            # Format tile name (e.g., N44E007)
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"{lat_str}{lon_str}"
            
            # Check if tile exists in MinIO
            tile_key = f"srtm/{tile_name}.hgt"
            try:
                # Check if object exists
                self.terrain.minio_client.client.stat_object(
                    self.terrain.minio_client.bucket_name,
                    tile_key
                )
            except Exception:
                missing_tiles.append(tile_name)
        
        return missing_tiles
    
    def generate_samples(
        self,
        num_samples: int,
        inside_ratio: float = 0.7,
        frequency_mhz: float = 145.0,
        tx_power_dbm: float = 37.0,
        min_snr_db: float = 3.0,
        min_receivers: int = 3,
        max_gdop: float = 150.0,
        tx_antenna_dist=None,
        rx_antenna_dist=None,
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
            
            # Select TX antenna (once per sample)
            tx_antenna = _select_tx_antenna(self.rng, tx_antenna_dist)
            
            # Generate receiver measurements
            receiver_measurements = []
            receivers_with_signal = 0
            
            for rx in self.config.receivers:
                # Select RX antenna for this receiver
                rx_antenna = _select_rx_antenna(self.rng, rx_antenna_dist)
                
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
                    terrain_lookup=self.terrain,
                    tx_antenna=tx_antenna,
                    rx_antenna=rx_antenna
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
    max_gdop = config.get('max_gdop', 150.0)
    
    # For iq_raw datasets, use random receivers
    use_random_receivers = (dataset_type == 'iq_raw') or config.get('use_random_receivers', False)
    
    # For iq_raw with random receivers, GDOP filtering is less useful
    # (random geometry often has high GDOP, but that's OK for training)
    # Relax GDOP constraint to allow more diverse geometries
    if dataset_type == 'iq_raw' and use_random_receivers:
        if max_gdop < 200:
            logger.warning(f"IQ-raw with random receivers: relaxing max_gdop from {max_gdop} to 200 for better success rate")
            max_gdop = 200.0

    # Use GPU batch processing instead of thread pool
    # Batch size: Process all samples in one batch to amortize GPU initialization overhead
    # For small datasets (<100 samples), use single batch
    # For larger datasets, use mini-batches optimized for GPU memory:
    # - Fixed receivers (7): batch_size=800 → ~5,600 IQ samples (13.9% GPU mem)
    # - Random receivers (7-10 avg): batch_size=200 → ~1,600 IQ samples (safe for 24GB GPU)
    if use_random_receivers:
        # Random receivers: variable count per sample, use smaller batch
        batch_size = min(200, num_samples) if num_samples > 10 else num_samples
        logger.info(f"Random receivers mode: using batch_size={batch_size} (GPU memory conservative)")
    else:
        # Fixed receivers: consistent count per sample, use larger batch
        batch_size = min(800, num_samples) if num_samples > 10 else num_samples
        logger.info(f"Fixed receivers mode: using batch_size={batch_size} (GPU memory optimized)")
    
    # Check GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info(f"GPU batch processing enabled: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available, falling back to CPU batch processing")
    except ImportError:
        use_gpu = False
        logger.warning("PyTorch not available, falling back to CPU batch processing")
    
    logger.info(f"Using batch size: {batch_size} samples per batch (REAL batching enabled)")

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
    
    # TRUE BATCH PROCESSING: Process samples in batches to amortize GPU initialization overhead
    # Strategy: Generate geometries in parallel, then extract features for entire batch at once
    # This reduces GPU overhead from ~4.5s per sample to ~4.5s per batch_size samples
    num_workers = min(8, mp.cpu_count())  # More workers for geometry generation (no GPU contention)
    
    logger.info(f"TRUE BATCH MODE: Generating {num_samples} VALID samples in batches of {batch_size}")
    
    # Rename for clarity: num_samples is the TARGET number of VALID samples
    target_valid_samples = num_samples
    valid_samples_collected = 0
    total_attempted = 0
    generated_samples = []
    iq_samples_to_save = {}
    
    # Safety limits to prevent infinite loops with impossible parameters
    max_attempts = num_samples * 20  # Allow down to 5% success rate
    max_consecutive_failures = 5  # Stop if 5 consecutive batches have 0 valid samples
    consecutive_failures = 0
    
    batch_number = 0
    
    # Process batches until we have enough VALID samples
    logger.info(f"Target: {target_valid_samples} valid samples, max attempts: {max_attempts}")
    
    while valid_samples_collected < target_valid_samples and total_attempted < max_attempts:
        batch_number += 1
        batch_start = total_attempted
        batch_end = batch_start + batch_size
        
        # Generate args for this batch (indices continue from total_attempted)
        batch_args = [
            (batch_start + i, receivers_list, training_config_dict, config, seed, dataset_type, use_random_receivers)
            for i in range(batch_size)
        ]
        
        logger.info(f"Processing batch {batch_number}: attempting samples {batch_start} to {batch_end-1} (valid so far: {valid_samples_collected}/{target_valid_samples})")
        
        # Step 1: Generate geometries + IQ samples in parallel (NO feature extraction yet)
        batch_raw_results = []  # Store raw results without features
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_single_sample_no_features, args): args[0]
                      for args in batch_args}

            for future in as_completed(futures):
                sample_idx = futures[future]
                total_attempted += 1

                # Check for cancellation
                if job_id and total_attempted % 10 == 0:
                    from sqlalchemy import text
                    check_query = text("SELECT status FROM heimdall.training_jobs WHERE id = :job_id")
                    result_check = await conn.execute(check_query, {"job_id": job_id})
                    row = result_check.fetchone()
                    if row and row[0] == 'cancelled':
                        logger.warning(f"Job cancelled at {valid_samples_collected} valid samples ({total_attempted} attempted)")
                        for f in futures:
                            f.cancel()
                        break

                try:
                    result = future.result()
                    if result is None:
                        continue
                    
                    # Unpack results (no features yet)
                    sample_idx_ret, iq_samples_for_extraction, receivers_list, distance_kms, signal_presents, tx_position, propagation_metadata, num_receivers_in_sample, iq_samples = result

                    # Store raw result for batch feature extraction
                    batch_raw_results.append({
                        'sample_idx': sample_idx_ret,
                        'iq_samples_for_extraction': iq_samples_for_extraction,
                        'receivers_list': receivers_list,
                        'distance_kms': distance_kms,
                        'signal_presents': signal_presents,
                        'tx_position': tx_position,
                        'propagation_metadata': propagation_metadata,
                        'num_receivers_in_sample': num_receivers_in_sample,
                        'iq_samples': iq_samples
                    })

                except Exception as e:
                    logger.error(f"Error generating sample {sample_idx}: {e}")
                    continue
        
        logger.info(f"Batch IQ generation complete: {len(batch_raw_results)} samples ready for feature extraction")
        
        # Step 2: Extract features for ENTIRE batch at once (amortize GPU initialization overhead)
        # Collect ALL IQ samples from batch into a single list
        all_iq_samples = []
        sample_to_iq_indices = {}  # Map sample_idx -> (start_idx, end_idx) in all_iq_samples
        
        current_idx = 0
        for raw_result in batch_raw_results:
            sample_idx = raw_result['sample_idx']
            iq_samples_list = raw_result['iq_samples_for_extraction']
            num_iq = len(iq_samples_list)
            
            sample_to_iq_indices[sample_idx] = (current_idx, current_idx + num_iq)
            all_iq_samples.extend(iq_samples_list)
            current_idx += num_iq
        
        logger.info(f"Extracting features for {len(all_iq_samples)} IQ samples ({len(batch_raw_results)} samples × ~{len(all_iq_samples)//len(batch_raw_results) if batch_raw_results else 0} receivers)")
        
        # Extract features for ALL IQ samples at once (GPU batch processing)
        # Initialize feature extractor (reuse if already exists)
        if not hasattr(_thread_local, 'feature_extractor'):
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            _thread_local.feature_extractor = RFFeatureExtractor(
                sample_rate_hz=200_000,
                use_gpu=use_gpu
            )
            logger.info(f"Initialized batch feature extractor (GPU={use_gpu})")
        
        feature_extractor = _thread_local.feature_extractor
        
        # Extract features for entire batch (ONE GPU init for all samples!)
        all_features_dicts = feature_extractor.extract_features_batch_conservative(
            iq_samples_list=all_iq_samples,
            chunk_duration_ms=200.0,
            num_chunks=5
        )
        
        logger.info(f"Batch feature extraction complete: {len(all_features_dicts)} feature dicts extracted")
        
        # Step 3: Reconstruct samples with features and validate
        batch_results = []
        for raw_result in batch_raw_results:
            sample_idx = raw_result['sample_idx']
            start_idx, end_idx = sample_to_iq_indices[sample_idx]
            
            # Extract features for this sample
            sample_features_dicts = all_features_dicts[start_idx:end_idx]
            
            # Add receiver metadata to each feature dict
            receiver_features_list = []
            for i, features_dict in enumerate(sample_features_dicts):
                receiver = raw_result['receivers_list'][i]
                
                # Add receiver metadata
                features_dict['rx_id'] = receiver['name']
                features_dict['rx_lat'] = receiver['latitude']
                features_dict['rx_lon'] = receiver['longitude']
                features_dict['distance_km'] = raw_result['distance_kms'][i]
                features_dict['num_receivers_in_sample'] = raw_result['num_receivers_in_sample']
                
                # Override signal_present with propagation model result
                features_dict['signal_present'] = raw_result['signal_presents'][i]
                
                receiver_features_list.append(features_dict)
            
            # Calculate quality metrics
            propagation_snr_values = raw_result['propagation_metadata']['propagation_snr_values']
            mean_snr_db = float(np.mean(propagation_snr_values)) if propagation_snr_values else 0.0
            
            confidence_scores = [f.get('delay_spread_confidence', {}).get('mean', 0.8)
                                for f in receiver_features_list if f.get('signal_present')]
            overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
            
            receiver_positions = [
                (f['rx_lat'], f['rx_lon'])
                for f in receiver_features_list
                if f.get('signal_present')
            ]
            gdop = calculate_gdop(receiver_positions, (raw_result['tx_position']['tx_lat'], raw_result['tx_position']['tx_lon'])) if len(receiver_positions) >= 3 else 999.0
            
            extraction_metadata = {
                'extraction_method': 'synthetic',
                'iq_duration_ms': raw_result['propagation_metadata']['iq_duration_ms'],
                'sample_rate_hz': raw_result['propagation_metadata']['sample_rate_hz'],
                'num_chunks': raw_result['propagation_metadata']['num_chunks'],
                'chunk_duration_ms': raw_result['propagation_metadata']['chunk_duration_ms'],
                'generated_at': raw_result['propagation_metadata']['generated_at'],
                'frequency_mhz': raw_result['propagation_metadata']['frequency_mhz'],
                'tx_power_dbm': raw_result['propagation_metadata']['tx_power_dbm']
            }
            
            quality_metrics = {
                'overall_confidence': overall_confidence,
                'mean_snr_db': mean_snr_db,
                'num_receivers_detected': raw_result['propagation_metadata']['receivers_with_signal'],
                'gdop': gdop
            }
            
            # Validation
            if quality_metrics['num_receivers_detected'] < min_receivers:
                continue
            if quality_metrics['mean_snr_db'] < min_snr_db:
                continue
            if quality_metrics['gdop'] > max_gdop:
                continue
            
            # Store valid sample
            batch_results.append({
                'sample_idx': sample_idx,
                'receiver_features': receiver_features_list,
                'extraction_metadata': extraction_metadata,
                'quality_metrics': quality_metrics,
                'tx_position': raw_result['tx_position']
            })
            
            # Store IQ samples for MinIO
            if raw_result['iq_samples'] and (dataset_type == 'iq_raw' or sample_idx < 100):
                iq_samples_to_save[sample_idx] = raw_result['iq_samples']
        
        # Batch complete - update counters
        valid_in_batch = len(batch_results)
        valid_samples_collected += valid_in_batch
        generated_samples.extend(batch_results)
        
        # Track consecutive failures for safety
        if valid_in_batch == 0:
            consecutive_failures += 1
            logger.warning(f"Batch {batch_number} produced 0 valid samples (consecutive failures: {consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Stopping: {consecutive_failures} consecutive batches with 0 valid samples. Parameters may be too strict (GDOP={max_gdop}, min_receivers={min_receivers}, min_snr={min_snr_db})")
                break
        else:
            consecutive_failures = 0  # Reset on success
        
        success_rate = valid_samples_collected / total_attempted * 100 if total_attempted > 0 else 0
        logger.info(f"Batch {batch_number} complete: {valid_in_batch} valid samples | Total: {valid_samples_collected}/{target_valid_samples} valid ({total_attempted} attempted, {success_rate:.1f}% success rate)")
        
        # Save batch to database
        if batch_results:
            try:
                await save_features_to_db(dataset_id, batch_results, conn)
                logger.info(f"Saved batch to database ({valid_samples_collected} total saved)")
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
        
        # Progress update after each batch
        if progress_callback:
            try:
                logger.info(f"[PROGRESS DEBUG] Calling progress_callback with valid={valid_samples_collected}, target={target_valid_samples}, attempted={total_attempted}")
                await progress_callback(valid_samples_collected, target_valid_samples, total_attempted)
                logger.info(f"[PROGRESS DEBUG] Callback completed successfully")
                logger.info(f"[PROGRESS] Progress: {valid_samples_collected}/{target_valid_samples} valid samples ({total_attempted} attempted)")
            except Exception as e:
                logger.error(f"[PROGRESS DEBUG] Callback failed: {e}", exc_info=True)
    
    # All batches complete
    final_success_rate = valid_samples_collected / total_attempted * 100 if total_attempted > 0 else 0
    reached_target = valid_samples_collected >= target_valid_samples
    
    if reached_target:
        logger.info(f"✓ Target reached: {valid_samples_collected}/{target_valid_samples} valid samples generated ({total_attempted} attempted, {final_success_rate:.1f}% success rate)")
    elif total_attempted >= max_attempts:
        logger.warning(f"✗ Max attempts reached: {valid_samples_collected}/{target_valid_samples} valid samples ({total_attempted} attempted, {final_success_rate:.1f}% success rate)")
    elif consecutive_failures >= max_consecutive_failures:
        logger.warning(f"✗ Too many consecutive failures: {valid_samples_collected}/{target_valid_samples} valid samples ({total_attempted} attempted)")
    else:
        logger.warning(f"✗ Generation stopped early: {valid_samples_collected}/{target_valid_samples} valid samples ({total_attempted} attempted)")

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
        'total_generated': valid_samples_collected,
        'total_attempted': total_attempted,
        'success_rate': final_success_rate / 100,  # Convert to 0-1 range
        'target_samples': target_valid_samples,
        'reached_target': reached_target,
        'iq_samples_saved': len(iq_samples_to_save),
        'dataset_type': dataset_type,
        'stopped_reason': 'target_reached' if reached_target else ('max_attempts' if total_attempted >= max_attempts else ('consecutive_failures' if consecutive_failures >= max_consecutive_failures else 'unknown'))
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
