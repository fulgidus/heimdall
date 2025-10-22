"""
Feature extraction utilities for RF signal processing.

Transforms raw IQ data from WebSDR receivers into mel-spectrograms for neural network input.

Key functions:
- iq_to_mel_spectrogram(): Convert complex IQ samples to mel-scale spectrogram
- compute_mfcc(): Compute Mel-Frequency Cepstral Coefficients (optional)
- normalize_features(): Zero-mean, unit-variance normalization
"""

import numpy as np
from scipy import signal
from scipy.fftpack import dct
import structlog
from typing import Tuple, Optional

logger = structlog.get_logger(__name__)

def MEL_SPECTROGRAM_SHAPE(n_mels: int = 128, n_frames: int = 2048) -> Tuple[int, int]:
    """Return the shape of the mel-spectrogram feature."""
    return (n_mels, n_frames)

def iq_to_mel_spectrogram(
    iq_data: np.ndarray,
    sample_rate: float = 192000.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    window: str = 'hann',
) -> np.ndarray:
    """
    Convert complex IQ data to mel-scale spectrogram.
    
    This is the primary feature extraction function. It converts raw IQ samples
    from WebSDR receivers into a mel-scale spectrogram suitable for neural network input.
    
    Args:
        iq_data (np.ndarray): Complex IQ samples, shape (n_samples,) or (n_receivers, n_samples)
        sample_rate (float): Sampling rate in Hz (default WebSDR: 192 kHz)
        n_mels (int): Number of mel-scale frequency bins (default: 128)
        n_fft (int): FFT size for STFT (default: 2048)
        hop_length (int): Number of samples between successive frames (default: 512)
        f_min (float): Minimum frequency in Hz
        f_max (Optional[float]): Maximum frequency in Hz (None = Nyquist)
        window (str): Window function ('hann', 'hamming', 'blackman', etc.)
    
    Returns:
        np.ndarray: Mel-spectrogram, shape (n_mels, n_frames) or (n_receivers, n_mels, n_frames)
        
    Example:
        >>> iq_data = np.random.randn(192000) + 1j * np.random.randn(192000)
        >>> mel_spec = iq_to_mel_spectrogram(iq_data)
        >>> print(mel_spec.shape)  # (128, ~375)
    """
    
    if f_max is None:
        f_max = sample_rate / 2  # Nyquist frequency
    
    # Ensure complex dtype
    iq_data = np.asarray(iq_data, dtype=np.complex128)
    
    # Handle both 1D and 2D inputs
    if iq_data.ndim == 1:
        # Single channel: convert to magnitude spectrum
        magnitude = np.abs(iq_data)
        
        # Compute Short-Time Fourier Transform (STFT)
        f, t, stft_result = signal.stft(
            magnitude,
            fs=sample_rate,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
        )
        
        # Convert to power spectrogram
        power_spec = np.abs(stft_result) ** 2
        
        # Create mel-scale filter bank
        mel_filterbank = librosa_compatible_mel_scale(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        
        # Apply mel-scale filter bank
        mel_spec = mel_filterbank @ power_spec
        
        # Convert to log scale (add small epsilon to avoid log(0))
        mel_spec_db = 10 * np.log10(mel_spec + 1e-9)
        
        logger.debug(
            "iq_to_mel_spectrogram_1d",
            input_shape=iq_data.shape,
            output_shape=mel_spec_db.shape,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )
        
        return mel_spec_db
    
    elif iq_data.ndim == 2:
        # Multi-channel: process each channel separately and stack
        mel_specs = []
        for i, channel in enumerate(iq_data):
            mel_spec = iq_to_mel_spectrogram(
                channel,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                window=window,
            )
            mel_specs.append(mel_spec)
        
        result = np.stack(mel_specs, axis=0)
        logger.debug("iq_to_mel_spectrogram_2d", input_shape=iq_data.shape, output_shape=result.shape)
        return result
    
    else:
        raise ValueError(f"Unsupported input shape: {iq_data.shape}")


def librosa_compatible_mel_scale(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create mel-scale filter bank compatible with librosa.
    
    Implements triangular mel-scale filters using standard mel-scale frequency spacing.
    
    Args:
        sr (float): Sample rate
        n_fft (int): FFT size
        n_mels (int): Number of mel bins
        f_min (float): Minimum frequency
        f_max (Optional[float]): Maximum frequency (None = Nyquist)
    
    Returns:
        np.ndarray: Mel-scale filter bank, shape (n_mels, n_fft//2 + 1)
    """
    if f_max is None:
        f_max = sr / 2
    
    # Frequency bins in Hz
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    
    # Convert Hz to mel scale
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Mel-scale points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Create triangular filter bank
    filterbank = np.zeros((n_mels, len(freqs)))
    
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]
        
        # Left side of triangle
        left_slope = 1.0 / (f_center - f_left)
        left_mask = (freqs >= f_left) & (freqs <= f_center)
        filterbank[m, left_mask] = left_slope * (freqs[left_mask] - f_left)
        
        # Right side of triangle
        right_slope = 1.0 / (f_right - f_center)
        right_mask = (freqs > f_center) & (freqs <= f_right)
        filterbank[m, right_mask] = right_slope * (f_right - freqs[right_mask])
    
    return filterbank


def compute_mfcc(
    mel_spec_db: np.ndarray,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs) from mel-spectrogram.
    
    MFCCs are useful for capturing spectral characteristics and are often used
    in audio processing. They can be used as an alternative or complementary
    feature to raw mel-spectrograms.
    
    Args:
        mel_spec_db (np.ndarray): Log mel-spectrogram, shape (n_mels, n_frames)
        n_mfcc (int): Number of MFCC coefficients to extract (default: 13)
    
    Returns:
        np.ndarray: MFCC coefficients, shape (n_mfcc, n_frames)
    
    Example:
        >>> mel_spec = iq_to_mel_spectrogram(iq_data)
        >>> mfcc = compute_mfcc(mel_spec, n_mfcc=13)
        >>> print(mfcc.shape)  # (13, ~375)
    """
    # Discrete Cosine Transform (DCT)
    mfcc = dct(mel_spec_db, axis=0, type=2, norm='ortho')[:n_mfcc]
    
    logger.debug(
        "compute_mfcc",
        input_shape=mel_spec_db.shape,
        output_shape=mfcc.shape,
        n_mfcc=n_mfcc,
    )
    
    return mfcc


def normalize_features(
    features: np.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize features to zero mean and unit variance.
    
    This is typically applied per-sample (per spectrogram) or per-batch,
    and is crucial for neural network training stability.
    
    Args:
        features (np.ndarray): Input features, any shape
        axis (Optional[int]): Axis over which to compute mean/std
                              (None = compute over all elements)
        epsilon (float): Small value to avoid division by zero
    
    Returns:
        Tuple[np.ndarray, dict]:
            - Normalized features (same shape as input)
            - Dict with statistics: {'mean': mean, 'std': std, 'min': min, 'max': max}
    
    Example:
        >>> mel_spec = iq_to_mel_spectrogram(iq_data)
        >>> normalized, stats = normalize_features(mel_spec)
    """
    
    mean = np.mean(features, axis=axis, keepdims=True)
    std = np.std(features, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.maximum(std, epsilon)
    
    normalized = (features - mean) / std
    
    stats = {
        'mean': np.squeeze(mean),
        'std': np.squeeze(std),
        'min': np.min(normalized),
        'max': np.max(normalized),
    }
    
    logger.debug(
        "normalize_features",
        input_shape=features.shape,
        output_shape=normalized.shape,
        stats=stats,
    )
    
    return normalized, stats


def augment_iq_data(
    iq_data: np.ndarray,
    phase_shift_deg: float = 0.0,
    amplitude_noise_factor: float = 0.0,
) -> np.ndarray:
    """
    Apply data augmentation to IQ data (optional enhancement).
    
    This can be useful during training to improve robustness.
    
    Args:
        iq_data (np.ndarray): Complex IQ samples
        phase_shift_deg (float): Random phase shift in degrees (0 = no augmentation)
        amplitude_noise_factor (float): Gaussian noise factor (0 = no augmentation)
    
    Returns:
        np.ndarray: Augmented IQ data
    """
    
    augmented = iq_data.copy()
    
    if phase_shift_deg > 0:
        phase_shift = np.random.uniform(-phase_shift_deg, phase_shift_deg)
        phase_shift_rad = np.radians(phase_shift)
        augmented = augmented * np.exp(1j * phase_shift_rad)
    
    if amplitude_noise_factor > 0:
        noise = np.random.randn(*iq_data.shape) * amplitude_noise_factor
        augmented = augmented + noise * np.exp(1j * np.random.uniform(0, 2 * np.pi))
    
    logger.debug(
        "augment_iq_data",
        phase_shift_deg=phase_shift_deg,
        amplitude_noise_factor=amplitude_noise_factor,
    )
    
    return augmented


def verify_feature_extraction():
    """Verification function for feature extraction pipeline."""
    
    logger.info("Starting feature extraction verification...")
    
    # Create synthetic IQ data (1 second at 192 kHz)
    sample_rate = 192000
    duration_sec = 1.0
    n_samples = int(sample_rate * duration_sec)
    
    # Synthetic signal: 1 kHz sine wave
    t = np.arange(n_samples) / sample_rate
    frequency = 1000  # Hz
    iq_data = np.exp(2j * np.pi * frequency * t)
    
    # Add noise
    iq_data += 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    logger.info("iq_data_created", shape=iq_data.shape, dtype=str(iq_data.dtype))
    
    # Extract features
    mel_spec = iq_to_mel_spectrogram(iq_data, sample_rate=sample_rate)
    logger.info("mel_spectrogram_extracted", shape=mel_spec.shape)
    
    # Normalize
    normalized, stats = normalize_features(mel_spec)
    logger.info("features_normalized", stats=stats)
    
    # Verify output shapes
    assert mel_spec.shape == (128, 375), f"Expected mel_spec shape (128, 375), got {mel_spec.shape}"
    assert normalized.shape == mel_spec.shape, "Normalized shape should match input"
    assert abs(np.mean(normalized)) < 0.1, "Normalized mean should be ~0"
    assert abs(np.std(normalized) - 1.0) < 0.1, "Normalized std should be ~1"
    
    logger.info("✅ Feature extraction verification passed!")
    
    return mel_spec, normalized


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    mel_spec, normalized = verify_feature_extraction()
