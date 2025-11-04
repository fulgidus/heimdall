"""
In-memory cached dataset - loads everything once, GPU goes BRRRR
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)


class InMemoryTriangulationDataset(Dataset):
    """
    Loads ALL data into RAM at init.
    GPU will run at 100% because no I/O wait.
    """
    
    def __init__(self, dataset_ids: List[str], split: str, db_session, max_receivers: int = 7):
        self.dataset_ids = dataset_ids
        self.split = split
        self.max_receivers = max_receivers
        
        # LOAD EVERYTHING INTO RAM
        logger.info(f"Loading ALL {split} data into RAM...")
        self.samples = self._load_all_samples(db_session)
        logger.info(f"✅ Loaded {len(self.samples)} samples into RAM. GPU will go BRRRR!")
    
    def _load_all_samples(self, session) -> List[Dict]:
        """Load ALL samples into memory at once."""
        query = text("""
            SELECT 
                recording_session_id,
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
        
        # Process all samples
        samples = []
        for row in results:
            sample_id, tx_lat, tx_lon, receiver_features_jsonb, gdop, num_receivers = row
            
            # Process receiver features
            receiver_features = []
            signal_mask = []
            
            for rx in receiver_features_jsonb:
                snr = rx.get('snr_db', {}).get('mean', 0.0) if 'snr_db' in rx else 0.0
                psd = rx.get('psd_dbm_per_hz', {}).get('mean', -100.0) if 'psd_dbm_per_hz' in rx else -100.0
                freq_offset = rx.get('frequency_offset_hz', {}).get('mean', 0.0) if 'frequency_offset_hz' in rx else 0.0
                rx_lat = rx['rx_lat']
                rx_lon = rx['rx_lon']
                signal_present = rx.get('signal_present', True)
                
                features = [snr, psd, freq_offset, rx_lat, rx_lon, float(signal_present)]
                receiver_features.append(features)
                signal_mask.append(not signal_present)
            
            # Pad to max_receivers
            num_rx = len(receiver_features)
            if num_rx < self.max_receivers:
                padding = [[0.0] * 6 for _ in range(self.max_receivers - num_rx)]
                receiver_features.extend(padding)
                signal_mask.extend([True] * (self.max_receivers - num_rx))
            
            # Convert to tensors
            sample = {
                "receiver_features": torch.tensor(receiver_features, dtype=torch.float32),
                "signal_mask": torch.tensor(signal_mask, dtype=torch.bool),
                "target_position": torch.tensor([tx_lat, tx_lon], dtype=torch.float32),
                "metadata": {
                    "sample_id": str(sample_id),
                    "gdop": gdop if gdop else 50.0,
                    "num_receivers": num_rx
                }
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Just return from RAM - instant!
        return self.samples[idx]
