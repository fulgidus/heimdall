"""
GPU-CACHED DATASET - Tutto in VRAM, velocità supersonica
Con 24GB di VRAM possiamo caricare TUTTO sulla GPU!
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)


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
        
        # Process and load to GPU
        for i, row in enumerate(results):
            tx_lat, tx_lon, receiver_features_jsonb, gdop, _ = row
            
            # Target position
            self.targets[i, 0] = tx_lat
            self.targets[i, 1] = tx_lon
            self.gdops[i] = gdop if gdop else 50.0
            
            # Process receiver features
            for j, rx in enumerate(receiver_features_jsonb[:self.max_receivers]):
                snr = rx.get('snr_db', {}).get('mean', 0.0) if 'snr_db' in rx else 0.0
                psd = rx.get('psd_dbm_per_hz', {}).get('mean', -100.0) if 'psd_dbm_per_hz' in rx else -100.0
                freq_offset = rx.get('frequency_offset_hz', {}).get('mean', 0.0) if 'frequency_offset_hz' in rx else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                # Load directly to GPU
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                self.receiver_features[i, j, 3] = rx_lat
                self.receiver_features[i, j, 4] = rx_lon
                self.receiver_features[i, j, 5] = float(signal_present)
                
                self.signal_masks[i, j] = not signal_present
            
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
        
        # Process data
        for i, row in enumerate(results):
            tx_lat, tx_lon, receiver_features_jsonb, gdop, _ = row
            
            self.targets[i, 0] = tx_lat
            self.targets[i, 1] = tx_lon
            self.gdops[i] = gdop if gdop else 50.0
            
            for j, rx in enumerate(receiver_features_jsonb[:self.max_receivers]):
                snr = rx.get('snr_db', {}).get('mean', 0.0) if 'snr_db' in rx else 0.0
                psd = rx.get('psd_dbm_per_hz', {}).get('mean', -100.0) if 'psd_dbm_per_hz' in rx else -100.0
                freq_offset = rx.get('frequency_offset_hz', {}).get('mean', 0.0) if 'frequency_offset_hz' in rx else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                self.receiver_features[i, j, 0] = snr
                self.receiver_features[i, j, 1] = psd
                self.receiver_features[i, j, 2] = freq_offset
                self.receiver_features[i, j, 3] = rx_lat
                self.receiver_features[i, j, 4] = rx_lon
                self.receiver_features[i, j, 5] = float(signal_present)
                
                self.signal_masks[i, j] = not signal_present
    
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
                    "num_receivers": (~self.signal_masks[idx]).sum().item()
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
                    "num_receivers": (~self.signal_masks[idx]).sum().item()
                }
            }
