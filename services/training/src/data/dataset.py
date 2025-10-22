"""
HeimdallDataset: PyTorch Dataset for loading training data.

Loads IQ recordings from MinIO and ground truth labels from PostgreSQL.

Data flow:
1. Query PostgreSQL for recording sessions with known source locations
2. Download IQ data (.npy files) from MinIO
3. Extract features (mel-spectrogram) using features.py
4. Return (features, ground_truth_label) pairs for training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import structlog
from typing import Tuple, Optional, Dict, List
import os
from pathlib import Path
import pickle
from datetime import datetime

logger = structlog.get_logger(__name__)


class HeimdallDataset(Dataset):
    """
    PyTorch Dataset for RF source localization training.
    
    Each sample returns:
    - features: Mel-spectrogram (3, 128, 32) - 3 channels from multi-receiver IQ
    - label: Ground truth position [latitude, longitude]
    - uncertainty: Reference uncertainty (optional) - used for uncertainty-aware loss
    
    Data sources:
    - PostgreSQL: Recording sessions with ground truth coordinates
    - MinIO: IQ data files (.npy format)
    
    Features:
    - Lazy loading (only loads on access to avoid memory bloat)
    - Optional data augmentation
    - Caching of processed features
    - Statistical normalization per sample
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augmentation: bool = False,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
        n_mels: int = 128,
        n_frames: int = 32,
    ):
        """
        Initialize HeimdallDataset.
        
        Args:
            data_dir (str): Directory containing preprocessed training data
            split (str): 'train', 'val', or 'test'
            augmentation (bool): Apply data augmentation
            cache_dir (Optional[str]): Cache processed features to disk
            normalize (bool): Normalize features (zero mean, unit variance)
            n_mels (int): Number of mel frequency bins
            n_frames (int): Number of time frames per spectrogram
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normalize = normalize
        self.n_mels = n_mels
        self.n_frames = n_frames
        
        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        logger.info(
            "heimdall_dataset_initialized",
            split=split,
            num_samples=len(self.samples),
            augmentation=augmentation,
            normalize=normalize,
        )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load sample metadata from disk.
        
        Expected structure:
        data_dir/
        ├── train/
        │   ├── session_001_iq.npy        (IQ data)
        │   ├── session_001_label.npy     (ground truth [lat, lon])
        │   ├── session_001_meta.pkl      (metadata)
        │   └── ...
        ├── val/
        └── test/
        """
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        samples = []
        
        # Find all sessions in this split
        session_files = sorted(split_dir.glob("*_iq.npy"))
        
        for iq_file in session_files:
            session_id = iq_file.stem.replace('_iq', '')
            
            label_file = split_dir / f"{session_id}_label.npy"
            meta_file = split_dir / f"{session_id}_meta.pkl"
            
            if label_file.exists():
                samples.append({
                    'session_id': session_id,
                    'iq_file': str(iq_file),
                    'label_file': str(label_file),
                    'meta_file': str(meta_file) if meta_file.exists() else None,
                })
        
        logger.info(
            "samples_loaded",
            split=self.split,
            count=len(samples),
        )
        
        return samples
    
    def _get_cache_path(self, session_id: str) -> Path:
        """Get cache file path for a session."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{session_id}_features.pt"
    
    def _load_iq_data(self, iq_file: str) -> np.ndarray:
        """Load IQ data from .npy file."""
        iq_data = np.load(iq_file)
        
        # Expected shape: (3, n_samples) for 3-receiver IQ data
        if iq_data.ndim == 1:
            # Single receiver - replicate to 3 channels
            iq_data = np.tile(iq_data[np.newaxis, :], (3, 1))
        elif iq_data.ndim == 2 and iq_data.shape[0] != 3:
            # Wrong number of channels
            logger.warning("unexpected_iq_shape", shape=iq_data.shape)
        
        return iq_data
    
    def _extract_features(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features from IQ data.
        
        Returns:
            np.ndarray: Features of shape (3, n_mels, n_frames)
        """
        from src.data.features import iq_to_mel_spectrogram, normalize_features
        
        # Process each channel
        mel_specs = []
        for ch in range(iq_data.shape[0]):
            mel_spec = iq_to_mel_spectrogram(
                iq_data[ch],
                n_mels=self.n_mels,
            )
            
            # Normalize per-channel
            if self.normalize:
                mel_spec, _ = normalize_features(mel_spec)
            
            # Resize to fixed shape
            mel_spec = self._resize_to_fixed_shape(mel_spec)
            mel_specs.append(mel_spec)
        
        features = np.stack(mel_specs, axis=0)  # (3, n_mels, n_frames)
        return features
    
    def _resize_to_fixed_shape(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Resize mel-spectrogram to fixed shape (n_mels, n_frames).
        
        Uses padding or truncation as needed.
        """
        from scipy import signal as scipy_signal
        
        current_frames = mel_spec.shape[1]
        
        if current_frames == self.n_frames:
            return mel_spec
        elif current_frames > self.n_frames:
            # Truncate to center
            start = (current_frames - self.n_frames) // 2
            return mel_spec[:, start:start + self.n_frames]
        else:
            # Pad with reflection
            pad_total = self.n_frames - current_frames
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(mel_spec, ((0, 0), (pad_left, pad_right)), mode='reflect')
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
        
        Returns:
            Tuple containing:
            - features (torch.Tensor): Mel-spectrogram features (3, 128, 32)
            - label (torch.Tensor): Ground truth [latitude, longitude]
            - metadata (Dict): Additional metadata (session_id, etc.)
        """
        
        sample = self.samples[idx]
        session_id = sample['session_id']
        
        # Check cache first
        cache_path = self._get_cache_path(session_id)
        if cache_path and cache_path.exists():
            data = torch.load(cache_path)
            return data['features'], data['label'], {'session_id': session_id, 'from_cache': True}
        
        # Load IQ data
        iq_data = self._load_iq_data(sample['iq_file'])
        
        # Extract features
        features = self._extract_features(iq_data)
        
        # Load ground truth label
        label = np.load(sample['label_file'])  # [lat, lon]
        
        # Apply augmentation if training
        if self.augmentation and self.split == 'train':
            features = self._augment_features(features)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.from_numpy(label).float()
        
        # Cache if enabled
        if cache_path:
            torch.save({
                'features': features_tensor,
                'label': label_tensor,
            }, cache_path)
        
        metadata = {
            'session_id': session_id,
            'from_cache': False,
        }
        
        if sample['meta_file'] and os.path.exists(sample['meta_file']):
            with open(sample['meta_file'], 'rb') as f:
                metadata.update(pickle.load(f))
        
        return features_tensor, label_tensor, metadata
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation (optional)."""
        
        # Random noise
        if np.random.rand() < 0.3:
            noise = np.random.randn(*features.shape) * 0.1
            features = features + noise
        
        # Random time shift
        if np.random.rand() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                features = np.roll(features, shift, axis=2)
        
        return features
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics (mean, std, min, max across all samples).
        
        Useful for understanding data distribution and debugging.
        """
        all_features = []
        
        for i in range(min(100, len(self))):  # Sample first 100
            features, _, _ = self[i]
            all_features.append(features.numpy())
        
        all_features = np.concatenate([f.flatten() for f in all_features])
        
        stats = {
            'mean': float(np.mean(all_features)),
            'std': float(np.std(all_features)),
            'min': float(np.min(all_features)),
            'max': float(np.max(all_features)),
            'num_samples_analyzed': min(100, len(self)),
        }
        
        logger.info("dataset_statistics_computed", stats=stats)
        
        return stats


def create_dummy_dataset(output_dir: str, n_train: int = 100, n_val: int = 20):
    """
    Create a dummy dataset for testing.
    
    Useful for development and debugging before training on real data.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, n_samples in [('train', n_train), ('val', n_val)]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        for i in range(n_samples):
            session_id = f"session_{i:06d}"
            
            # Create synthetic IQ data
            iq_data = np.random.randn(3, 192000).astype(np.float32)  # 3 channels, 1 sec at 192kHz
            np.save(split_dir / f"{session_id}_iq.npy", iq_data)
            
            # Create synthetic label (lat, lon)
            label = np.array([45.5 + np.random.randn()*0.1, 8.5 + np.random.randn()*0.1]).astype(np.float32)
            np.save(split_dir / f"{session_id}_label.npy", label)
            
            # Create metadata
            meta = {'receiver_ids': ['r1', 'r2', 'r3'], 'timestamp': datetime.now().isoformat()}
            with open(split_dir / f"{session_id}_meta.pkl", 'wb') as f:
                pickle.dump(meta, f)
        
        logger.info("dummy_dataset_created", split=split, count=n_samples)


def verify_dataset():
    """Verification function for dataset."""
    
    logger.info("Starting dataset verification...")
    
    # Create dummy dataset
    dummy_dir = "/tmp/heimdall_dummy_dataset"
    create_dummy_dataset(dummy_dir, n_train=10, n_val=2)
    
    # Create dataset
    dataset = HeimdallDataset(dummy_dir, split='train', augmentation=False)
    
    # Get a sample
    features, label, metadata = dataset[0]
    
    # Verify shapes
    assert features.shape == (3, 128, 32), f"Expected shape (3, 128, 32), got {features.shape}"
    assert label.shape == (2,), f"Expected label shape (2,), got {label.shape}"
    
    logger.info(
        "✅ Dataset verification passed!",
        sample_features_shape=tuple(features.shape),
        sample_label_shape=tuple(label.shape),
        sample_metadata=metadata,
    )
    
    return dataset


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    dataset = verify_dataset()
