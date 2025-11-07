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
        max_receivers: int = 7,
        preload_to_gpu: bool = True
    ):
        self.dataset_ids = dataset_ids
        self.split = split
        self.max_receivers = max_receivers
        self.device = device
        self.preload_to_gpu = preload_to_gpu
        
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
        """Load everything directly to GPU memory."""
        query = text("""
            SELECT 
                tx_latitude, 
                tx_longitude, 
                receiver_features, 
                gdop, 
                num_receivers_detected
            FROM heimdall.measurement_features
            WHERE dataset_id = ANY(CAST(:dataset_ids AS uuid[]))
            ORDER BY timestamp
        """)
        
        results = session.execute(query, {"dataset_ids": self.dataset_ids}).fetchall()
        
        # Split train/val
        split_idx = int(len(results) * 0.8)
        if self.split == 'train':
            results = results[:split_idx]
        else:
            results = results[split_idx:]
        
        logger.info(f"Processing {len(results)} samples...")
        
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
            
            # STEP 3: Transform coordinates to DELTAS (relative to centroid)
            for j, (snr, psd, freq_offset, rx_lat, rx_lon, signal_present) in enumerate(receiver_data):
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                self.receiver_features[i, j, 3] = rx_lat - centroid_lat  # DELTA lat
                self.receiver_features[i, j, 4] = rx_lon - centroid_lon  # DELTA lon
                self.receiver_features[i, j, 5] = float(signal_present)
            
            # STEP 4: Transform target to DELTA coordinates
            self.targets[i, 0] = tx_lat - centroid_lat  # DELTA target_lat
            self.targets[i, 1] = tx_lon - centroid_lon  # DELTA target_lon
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {i+1}/{num_samples} to GPU...")
    
    def _load_to_ram(self, session):
        """Load to RAM (normal mode, will copy to GPU per batch)."""
        query = text("""
            SELECT 
                tx_latitude, 
                tx_longitude, 
                receiver_features, 
                gdop, 
                num_receivers_detected
            FROM heimdall.measurement_features
            WHERE dataset_id = ANY(CAST(:dataset_ids AS uuid[]))
            ORDER BY timestamp
        """)
        
        results = session.execute(query, {"dataset_ids": self.dataset_ids}).fetchall()
        
        # Split train/val
        split_idx = int(len(results) * 0.8)
        if self.split == 'train':
            results = results[:split_idx]
        else:
            results = results[split_idx:]
        
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
            
            # STEP 3: Transform coordinates to DELTAS (relative to centroid)
            for j, (snr, psd, freq_offset, rx_lat, rx_lon, signal_present) in enumerate(receiver_data):
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                self.receiver_features[i, j, 3] = rx_lat - centroid_lat  # DELTA lat
                self.receiver_features[i, j, 4] = rx_lon - centroid_lon  # DELTA lon
                self.receiver_features[i, j, 5] = float(signal_present)
            
            # STEP 4: Transform target to DELTA coordinates
            self.targets[i, 0] = tx_lat - centroid_lat  # DELTA target_lat
            self.targets[i, 1] = tx_lon - centroid_lon  # DELTA target_lon
    
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
                    "centroid": self.centroids[idx]  # Already on GPU
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
                    "centroid": self.centroids[idx].to(self.device)
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
    
    # Convert metadata to tensors
    metadata = {
        "gdop": torch.tensor(gdop_list, dtype=torch.float32),
        "num_receivers": torch.tensor(num_receivers_list, dtype=torch.long),
        "centroids": torch.stack(centroids_list)  # [batch, 2] (lat, lon)
    }
    
    return {
        "receiver_features": receiver_features,
        "signal_mask": signal_mask,
        "target_position": target_position,
        "metadata": metadata
    }
