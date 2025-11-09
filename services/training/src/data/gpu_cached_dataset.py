"""
GPU-CACHED DATASET - Tutto in VRAM, velocità supersonica
Con 24GB di VRAM possiamo caricare TUTTO sulla GPU!
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)

# Import normalization utilities from triangulation_dataloader
from .triangulation_dataloader import normalize_feature, NORMALIZATION_PARAMS

# Coordinate conversion constants for Italy (approx 45° latitude)
# These convert delta degrees to meters for proper neural network scaling
METERS_PER_DEG_LAT = 111000.0  # meters per degree latitude (constant)
METERS_PER_DEG_LON = 78000.0   # meters per degree longitude at ~45° (Italy average)
                                # Note: cos(45°) ≈ 0.707, so 111km * 0.707 ≈ 78km

# Standardization parameters (z-score normalization: (x - mean) / std)
# These are computed dynamically from the actual dataset - NO HARDCODED RANGES!
# Computed per-split (train/val) to avoid data leakage
# NOTE: These are placeholder values. Actual values are computed as instance variables
# (self.coord_mean_lat_meters, self.coord_std_lat_meters, etc.) during dataset initialization


class GPUCachedDataset(Dataset):
    """
    Dataset con opzione di preload in VRAM.
    
    preload_to_gpu=True (RECOMMENDED for 24GB VRAM):
    - Carica TUTTO in VRAM all'inizio (1 volta)
    - GPU al 100%, zero I/O wait
    - ~100MB per 1000 samples
    - Con 24GB VRAM → 200k+ samples!
    
    preload_to_gpu=False:
    - Carica in RAM, copia a GPU batch per batch (normale)
    """
    
    def __init__(
        self, 
        dataset_ids: List[str], 
        split: str, 
        db_session, 
        device, 
        max_receivers: int = 10,
        preload_to_gpu: bool = True,
        min_snr_db: float = -999.0,
        max_gdop: float = 999.0
    ):
        self.dataset_ids = dataset_ids
        self.split = split
        self.max_receivers = max_receivers
        self.device = device
        self.preload_to_gpu = preload_to_gpu
        self.min_snr_db = min_snr_db
        self.max_gdop = max_gdop
        
        # Log filtering parameters
        if min_snr_db > -999.0 or max_gdop < 999.0:
            logger.info(f"📊 Quality filters: min_snr_db={min_snr_db:.1f}, max_gdop={max_gdop:.1f}")
        
        if preload_to_gpu:
            logger.info(f"🚀 PRELOAD MODE: Loading ALL {split} data DIRECTLY TO GPU ({device})...")
            self._load_to_gpu(db_session)
            vram_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"✅ {len(self)} samples in VRAM! Using {vram_used:.2f}GB. GPU GOES BRRRR! 🔥")
        else:
            logger.info(f"📦 NORMAL MODE: Loading {split} data to RAM...")
            self._load_to_ram(db_session)
            logger.info(f"✅ {len(self)} samples in RAM (will copy to GPU per batch)")
    
    def _load_to_gpu(self, session):
        """Load everything directly to GPU memory with quality filtering."""
        # Build WHERE clause with optional filters
        where_clauses = ["dataset_id = ANY(CAST(:dataset_ids AS uuid[]))"]
        
        if self.min_snr_db > -999.0:
            where_clauses.append("mean_snr_db >= :min_snr_db")
        
        if self.max_gdop < 999.0:
            where_clauses.append("gdop <= :max_gdop")
        
        where_sql = " AND ".join(where_clauses)
        
        query = text(f"""
            SELECT 
                tx_latitude, 
                tx_longitude, 
                receiver_features, 
                gdop, 
                num_receivers_detected
            FROM heimdall.measurement_features
            WHERE {where_sql}
            ORDER BY timestamp
        """)
        
        # Build query parameters
        params = {"dataset_ids": self.dataset_ids}
        if self.min_snr_db > -999.0:
            params["min_snr_db"] = self.min_snr_db
        if self.max_gdop < 999.0:
            params["max_gdop"] = self.max_gdop
        
        results = session.execute(query, params).fetchall()
        total_loaded = len(results)
        logger.info(f"Loaded {total_loaded} samples from database (after quality filtering)")
        
        # 🔀 CRITICAL FIX: Shuffle before split to avoid overfitting from sequential data
        # Without shuffling, train/val splits can have completely different distributions
        # (e.g., train gets dataset A, val gets dataset B) leading to poor generalization
        import random
        results = list(results)
        random.seed(42)  # Reproducibility: same split across runs
        random.shuffle(results)
        logger.info(f"✅ Shuffled {len(results)} samples with seed=42 for uniform train/val distribution")
        
        # Split train/val (now properly mixed)
        split_idx = int(len(results) * 0.8)
        if self.split == 'train':
            results = results[:split_idx]
        else:
            results = results[split_idx:]
        
        logger.info(f"Processing {len(results)} samples for {self.split} split...")
        
        # FIRST PASS: Calculate standardization parameters (mean, std) from THIS split's data
        # This ensures no data leakage between train/val
        all_delta_lat_meters = []
        all_delta_lon_meters = []
        
        logger.info("First pass: calculating standardization parameters...")
        for row in results:
            tx_lat, tx_lon, receiver_features_jsonb, _, _ = row
            
            # Calculate centroid for this sample
            receiver_positions = []
            for rx in receiver_features_jsonb[:self.max_receivers]:
                if rx.get('signal_present', True):
                    receiver_positions.append([rx['rx_lat'], rx['rx_lon']])
            
            if len(receiver_positions) > 0:
                centroid_lat = np.mean([pos[0] for pos in receiver_positions])
                centroid_lon = np.mean([pos[1] for pos in receiver_positions])
            else:
                # Fallback: use all receivers
                centroid_lat = np.mean([rx['rx_lat'] for rx in receiver_features_jsonb[:self.max_receivers]])
                centroid_lon = np.mean([rx['rx_lon'] for rx in receiver_features_jsonb[:self.max_receivers]])
            
            # Target delta in meters
            delta_target_lat_deg = tx_lat - centroid_lat
            delta_target_lon_deg = tx_lon - centroid_lon
            delta_target_lat_meters = delta_target_lat_deg * METERS_PER_DEG_LAT
            delta_target_lon_meters = delta_target_lon_deg * METERS_PER_DEG_LON
            
            all_delta_lat_meters.append(delta_target_lat_meters)
            all_delta_lon_meters.append(delta_target_lon_meters)
        
        # Compute standardization parameters
        self.coord_mean_lat_meters = float(np.mean(all_delta_lat_meters))
        self.coord_mean_lon_meters = float(np.mean(all_delta_lon_meters))
        self.coord_std_lat_meters = float(np.std(all_delta_lat_meters))
        self.coord_std_lon_meters = float(np.std(all_delta_lon_meters))
        
        # Prevent division by zero (shouldn't happen with real data)
        if self.coord_std_lat_meters < 1.0:
            self.coord_std_lat_meters = 1.0
        if self.coord_std_lon_meters < 1.0:
            self.coord_std_lon_meters = 1.0
        
        logger.info(
            f"📊 Standardization params for {self.split} split:",
            mean_lat_m=f"{self.coord_mean_lat_meters:.0f}",
            mean_lon_m=f"{self.coord_mean_lon_meters:.0f}",
            std_lat_m=f"{self.coord_std_lat_meters:.0f}",
            std_lon_m=f"{self.coord_std_lon_meters:.0f}",
            std_lat_km=f"{self.coord_std_lat_meters/1000:.1f}",
            std_lon_km=f"{self.coord_std_lon_meters/1000:.1f}"
        )
        
        # Pre-allocate tensors on GPU
        num_samples = len(results)
        self.receiver_features = torch.zeros(
            (num_samples, self.max_receivers, 6), 
            dtype=torch.float32, 
            device=self.device
        )
        self.signal_masks = torch.ones(
            (num_samples, self.max_receivers), 
            dtype=torch.bool, 
            device=self.device
        )
        self.targets = torch.zeros(
            (num_samples, 2), 
            dtype=torch.float32, 
            device=self.device
        )
        self.gdops = torch.zeros(num_samples, dtype=torch.float32, device=self.device)
        self.centroids = torch.zeros((num_samples, 2), dtype=torch.float32, device=self.device)
        
        # Process and load to GPU
        for i, row in enumerate(results):
            tx_lat, tx_lon, receiver_features_jsonb, gdop, _ = row
            
            self.gdops[i] = gdop if gdop else 50.0
            
            # STEP 1: Extract raw receiver features and calculate centroid
            receiver_positions = []
            receiver_data = []
            
            for j, rx in enumerate(receiver_features_jsonb[:self.max_receivers]):
                snr_raw = rx.get('snr_db', {}).get('mean', 0.0) if 'snr_db' in rx else 0.0
                psd_raw = rx.get('psd_dbm_per_hz', {}).get('mean', -100.0) if 'psd_dbm_per_hz' in rx else -100.0
                freq_offset_raw = rx.get('frequency_offset_hz', {}).get('mean', 0.0) if 'frequency_offset_hz' in rx else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                # Normalize signal quality features to [0, 1]
                snr = normalize_feature(snr_raw, NORMALIZATION_PARAMS['snr_min'], NORMALIZATION_PARAMS['snr_max'])
                psd = normalize_feature(psd_raw, NORMALIZATION_PARAMS['psd_min'], NORMALIZATION_PARAMS['psd_max'])
                freq_offset = normalize_feature(freq_offset_raw, NORMALIZATION_PARAMS['freq_offset_min'], NORMALIZATION_PARAMS['freq_offset_max'])
                
                receiver_data.append((snr, psd, freq_offset, rx_lat, rx_lon, signal_present))
                
                # Collect valid positions for centroid calculation
                if signal_present:
                    receiver_positions.append([rx_lat, rx_lon])
                
                self.signal_masks[i, j] = not signal_present
            
            # STEP 2: Calculate centroid from valid receivers
            if len(receiver_positions) > 0:
                centroid_lat = np.mean([pos[0] for pos in receiver_positions])
                centroid_lon = np.mean([pos[1] for pos in receiver_positions])
            else:
                # Fallback: use all receivers
                centroid_lat = np.mean([rd[3] for rd in receiver_data])
                centroid_lon = np.mean([rd[4] for rd in receiver_data])
            
            self.centroids[i, 0] = centroid_lat
            self.centroids[i, 1] = centroid_lon
            
            # STEP 3: Transform coordinates to DELTAS in METERS (relative to centroid)
            for j, (snr, psd, freq_offset, rx_lat, rx_lon, signal_present) in enumerate(receiver_data):
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                
                # Convert delta degrees to meters
                delta_lat_deg = rx_lat - centroid_lat
                delta_lon_deg = rx_lon - centroid_lon
                delta_lat_meters = delta_lat_deg * METERS_PER_DEG_LAT
                delta_lon_meters = delta_lon_deg * METERS_PER_DEG_LON
                
                # Standardize (z-score): (x - mean) / std (NO HARDCODED RANGE!)
                self.receiver_features[i, j, 3] = (delta_lat_meters - self.coord_mean_lat_meters) / self.coord_std_lat_meters
                self.receiver_features[i, j, 4] = (delta_lon_meters - self.coord_mean_lon_meters) / self.coord_std_lon_meters
                self.receiver_features[i, j, 5] = float(signal_present)
            
            # STEP 4: Transform target to DELTA in METERS and standardize
            delta_target_lat_deg = tx_lat - centroid_lat
            delta_target_lon_deg = tx_lon - centroid_lon
            delta_target_lat_meters = delta_target_lat_deg * METERS_PER_DEG_LAT
            delta_target_lon_meters = delta_target_lon_deg * METERS_PER_DEG_LON
            
            # Standardize (z-score): (x - mean) / std (NO HARDCODED RANGE!)
            self.targets[i, 0] = (delta_target_lat_meters - self.coord_mean_lat_meters) / self.coord_std_lat_meters
            self.targets[i, 1] = (delta_target_lon_meters - self.coord_mean_lon_meters) / self.coord_std_lon_meters
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {i+1}/{num_samples} to GPU...")
    
    def _load_to_ram(self, session):
        """Load to RAM (normal mode, will copy to GPU per batch) with quality filtering."""
        # Build WHERE clause with optional filters
        where_clauses = ["dataset_id = ANY(CAST(:dataset_ids AS uuid[]))"]
        
        if self.min_snr_db > -999.0:
            where_clauses.append("mean_snr_db >= :min_snr_db")
        
        if self.max_gdop < 999.0:
            where_clauses.append("gdop <= :max_gdop")
        
        where_sql = " AND ".join(where_clauses)
        
        query = text(f"""
            SELECT 
                tx_latitude, 
                tx_longitude, 
                receiver_features, 
                gdop, 
                num_receivers_detected
            FROM heimdall.measurement_features
            WHERE {where_sql}
            ORDER BY timestamp
        """)
        
        # Build query parameters
        params = {"dataset_ids": self.dataset_ids}
        if self.min_snr_db > -999.0:
            params["min_snr_db"] = self.min_snr_db
        if self.max_gdop < 999.0:
            params["max_gdop"] = self.max_gdop
        
        results = session.execute(query, params).fetchall()
        total_loaded = len(results)
        logger.info(f"Loaded {total_loaded} samples from database (after quality filtering)")
        
        # 🔀 CRITICAL FIX: Shuffle before split to avoid overfitting from sequential data
        # Without shuffling, train/val splits can have completely different distributions
        # (e.g., train gets dataset A, val gets dataset B) leading to poor generalization
        import random
        results = list(results)
        random.seed(42)  # Reproducibility: same split across runs
        random.shuffle(results)
        logger.info(f"✅ Shuffled {len(results)} samples with seed=42 for uniform train/val distribution")
        
        # Split train/val (now properly mixed)
        split_idx = int(len(results) * 0.8)
        if self.split == 'train':
            results = results[:split_idx]
        else:
            results = results[split_idx:]
        
        logger.info(f"Processing {len(results)} samples for {self.split} split...")
        
        # FIRST PASS: Calculate standardization parameters (mean, std) from THIS split's data
        # This ensures no data leakage between train/val
        all_delta_lat_meters = []
        all_delta_lon_meters = []
        
        logger.info("First pass: calculating standardization parameters...")
        for row in results:
            tx_lat, tx_lon, receiver_features_jsonb, _, _ = row
            
            # Calculate centroid for this sample
            receiver_positions = []
            for rx in receiver_features_jsonb[:self.max_receivers]:
                if rx.get('signal_present', True):
                    receiver_positions.append([rx['rx_lat'], rx['rx_lon']])
            
            if len(receiver_positions) > 0:
                centroid_lat = np.mean([pos[0] for pos in receiver_positions])
                centroid_lon = np.mean([pos[1] for pos in receiver_positions])
            else:
                # Fallback: use all receivers
                centroid_lat = np.mean([rx['rx_lat'] for rx in receiver_features_jsonb[:self.max_receivers]])
                centroid_lon = np.mean([rx['rx_lon'] for rx in receiver_features_jsonb[:self.max_receivers]])
            
            # Target delta in meters
            delta_target_lat_deg = tx_lat - centroid_lat
            delta_target_lon_deg = tx_lon - centroid_lon
            delta_target_lat_meters = delta_target_lat_deg * METERS_PER_DEG_LAT
            delta_target_lon_meters = delta_target_lon_deg * METERS_PER_DEG_LON
            
            all_delta_lat_meters.append(delta_target_lat_meters)
            all_delta_lon_meters.append(delta_target_lon_meters)
        
        # Compute standardization parameters
        self.coord_mean_lat_meters = float(np.mean(all_delta_lat_meters))
        self.coord_mean_lon_meters = float(np.mean(all_delta_lon_meters))
        self.coord_std_lat_meters = float(np.std(all_delta_lat_meters))
        self.coord_std_lon_meters = float(np.std(all_delta_lon_meters))
        
        # Prevent division by zero (shouldn't happen with real data)
        if self.coord_std_lat_meters < 1.0:
            self.coord_std_lat_meters = 1.0
        if self.coord_std_lon_meters < 1.0:
            self.coord_std_lon_meters = 1.0
        
        logger.info(
            f"📊 Standardization params for {self.split} split:",
            mean_lat_m=f"{self.coord_mean_lat_meters:.0f}",
            mean_lon_m=f"{self.coord_mean_lon_meters:.0f}",
            std_lat_m=f"{self.coord_std_lat_meters:.0f}",
            std_lon_m=f"{self.coord_std_lon_meters:.0f}",
            std_lat_km=f"{self.coord_std_lat_meters/1000:.1f}",
            std_lon_km=f"{self.coord_std_lon_meters/1000:.1f}"
        )
        
        # Pre-allocate tensors in RAM
        num_samples = len(results)
        self.receiver_features = torch.zeros((num_samples, self.max_receivers, 6), dtype=torch.float32)
        self.signal_masks = torch.ones((num_samples, self.max_receivers), dtype=torch.bool)
        self.targets = torch.zeros((num_samples, 2), dtype=torch.float32)
        self.gdops = torch.zeros(num_samples, dtype=torch.float32)
        self.centroids = torch.zeros((num_samples, 2), dtype=torch.float32)
        
        # Process data
        for i, row in enumerate(results):
            tx_lat, tx_lon, receiver_features_jsonb, gdop, _ = row
            
            self.gdops[i] = gdop if gdop else 50.0
            
            # STEP 1: Extract raw receiver features and calculate centroid
            receiver_positions = []
            receiver_data = []
            
            for j, rx in enumerate(receiver_features_jsonb[:self.max_receivers]):
                snr_raw = rx.get('snr_db', {}).get('mean', 0.0) if 'snr_db' in rx else 0.0
                psd_raw = rx.get('psd_dbm_per_hz', {}).get('mean', -100.0) if 'psd_dbm_per_hz' in rx else -100.0
                freq_offset_raw = rx.get('frequency_offset_hz', {}).get('mean', 0.0) if 'frequency_offset_hz' in rx else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                # Normalize signal quality features to [0, 1]
                snr = normalize_feature(snr_raw, NORMALIZATION_PARAMS['snr_min'], NORMALIZATION_PARAMS['snr_max'])
                psd = normalize_feature(psd_raw, NORMALIZATION_PARAMS['psd_min'], NORMALIZATION_PARAMS['psd_max'])
                freq_offset = normalize_feature(freq_offset_raw, NORMALIZATION_PARAMS['freq_offset_min'], NORMALIZATION_PARAMS['freq_offset_max'])
                
                receiver_data.append((snr, psd, freq_offset, rx_lat, rx_lon, signal_present))
                
                # Collect valid positions for centroid calculation
                if signal_present:
                    receiver_positions.append([rx_lat, rx_lon])
                
                self.signal_masks[i, j] = not signal_present
            
            # STEP 2: Calculate centroid from valid receivers
            if len(receiver_positions) > 0:
                centroid_lat = np.mean([pos[0] for pos in receiver_positions])
                centroid_lon = np.mean([pos[1] for pos in receiver_positions])
            else:
                # Fallback: use all receivers
                centroid_lat = np.mean([rd[3] for rd in receiver_data])
                centroid_lon = np.mean([rd[4] for rd in receiver_data])
            
            self.centroids[i, 0] = centroid_lat
            self.centroids[i, 1] = centroid_lon
            
            # STEP 3: Transform coordinates to DELTAS in METERS (relative to centroid)
            for j, (snr, psd, freq_offset, rx_lat, rx_lon, signal_present) in enumerate(receiver_data):
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                
                # Convert delta degrees to meters
                delta_lat_deg = rx_lat - centroid_lat
                delta_lon_deg = rx_lon - centroid_lon
                delta_lat_meters = delta_lat_deg * METERS_PER_DEG_LAT
                delta_lon_meters = delta_lon_deg * METERS_PER_DEG_LON
                
                # Standardize (z-score): (x - mean) / std (NO HARDCODED RANGE!)
                self.receiver_features[i, j, 3] = (delta_lat_meters - self.coord_mean_lat_meters) / self.coord_std_lat_meters
                self.receiver_features[i, j, 4] = (delta_lon_meters - self.coord_mean_lon_meters) / self.coord_std_lon_meters
                self.receiver_features[i, j, 5] = float(signal_present)
            
            # STEP 4: Transform target to DELTA in METERS and standardize
            delta_target_lat_deg = tx_lat - centroid_lat
            delta_target_lon_deg = tx_lon - centroid_lon
            delta_target_lat_meters = delta_target_lat_deg * METERS_PER_DEG_LAT
            delta_target_lon_meters = delta_target_lon_deg * METERS_PER_DEG_LON
            
            # Standardize (z-score): (x - mean) / std (NO HARDCODED RANGE!)
            self.targets[i, 0] = (delta_target_lat_meters - self.coord_mean_lat_meters) / self.coord_std_lat_meters
            self.targets[i, 1] = (delta_target_lon_meters - self.coord_mean_lon_meters) / self.coord_std_lon_meters
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        """
        Return data.
        - preload_to_gpu=True: Already on GPU, ZERO copy!
        - preload_to_gpu=False: Copy from RAM to GPU
        """
        if self.preload_to_gpu:
            # Already on GPU - instant!
            return {
                "receiver_features": self.receiver_features[idx],
                "signal_mask": self.signal_masks[idx],
                "target_position": self.targets[idx],
                "metadata": {
                    "gdop": self.gdops[idx].item(),
                    "num_receivers": (~self.signal_masks[idx]).sum().item(),
                    "centroid": self.centroids[idx],  # Already on GPU
                    # Z-score standardization parameters (per-split)
                    "coord_mean_lat_meters": self.coord_mean_lat_meters,
                    "coord_mean_lon_meters": self.coord_mean_lon_meters,
                    "coord_std_lat_meters": self.coord_std_lat_meters,
                    "coord_std_lon_meters": self.coord_std_lon_meters
                }
            }
        else:
            # Copy from RAM to GPU
            return {
                "receiver_features": self.receiver_features[idx].to(self.device),
                "signal_mask": self.signal_masks[idx].to(self.device),
                "target_position": self.targets[idx].to(self.device),
                "metadata": {
                    "gdop": self.gdops[idx].item(),
                    "num_receivers": (~self.signal_masks[idx]).sum().item(),
                    "centroid": self.centroids[idx].to(self.device),
                    # Z-score standardization parameters (per-split)
                    "coord_mean_lat_meters": self.coord_mean_lat_meters,
                    "coord_mean_lon_meters": self.coord_mean_lon_meters,
                    "coord_std_lat_meters": self.coord_std_lat_meters,
                    "coord_std_lon_meters": self.coord_std_lon_meters
                }
            }


def collate_gpu_cached(batch: List[Dict]) -> Dict:
    """
    Collate function for GPU-cached dataset.
    
    Matches the structure of collate_fn from triangulation_dataloader.
    Handles centroid metadata for coordinate reconstruction.
    """
    # Stack all tensors (already on GPU if preload_to_gpu=True)
    receiver_features = torch.stack([sample["receiver_features"] for sample in batch])
    signal_mask = torch.stack([sample["signal_mask"] for sample in batch])
    target_position = torch.stack([sample["target_position"] for sample in batch])
    
    # Extract metadata including centroids
    gdop_list = [sample["metadata"]["gdop"] for sample in batch]
    num_receivers_list = [sample["metadata"]["num_receivers"] for sample in batch]
    centroids_list = [sample["metadata"]["centroid"] for sample in batch]
    
    # Get z-score standardization parameters (same for all samples in a split)
    # These are needed for denormalization during training
    first_sample_meta = batch[0]["metadata"]
    coord_mean_lat_meters = first_sample_meta["coord_mean_lat_meters"]
    coord_mean_lon_meters = first_sample_meta["coord_mean_lon_meters"]
    coord_std_lat_meters = first_sample_meta["coord_std_lat_meters"]
    coord_std_lon_meters = first_sample_meta["coord_std_lon_meters"]
    
    # Convert metadata to tensors
    metadata = {
        "gdop": torch.tensor(gdop_list, dtype=torch.float32),
        "num_receivers": torch.tensor(num_receivers_list, dtype=torch.long),
        "centroids": torch.stack(centroids_list),  # [batch, 2] (lat, lon)
        # Z-score standardization parameters (scalars, same for entire split)
        "coord_mean_lat_meters": coord_mean_lat_meters,
        "coord_mean_lon_meters": coord_mean_lon_meters,
        "coord_std_lat_meters": coord_std_lat_meters,
        "coord_std_lon_meters": coord_std_lon_meters
    }
    
    return {
        "receiver_features": receiver_features,
        "signal_mask": signal_mask,
        "target_position": target_position,
        "metadata": metadata
    }
