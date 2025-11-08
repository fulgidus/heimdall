"""IQ data preprocessing pipeline for Phase 6 Inference Service.

Converts raw IQ data (time-domain) to mel-spectrogram features suitable for
neural network inference. Matches training pipeline from Phase 5.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    # FFT parameters
    n_fft: int = 512
    hop_length: int = 128

    # Mel-spectrogram parameters
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 0.5  # Normalized frequency (Nyquist = 0.5)
    power: float = 2.0

    # Normalization
    normalize: bool = True
    norm_mean: float | None = None
    norm_std: float | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if not (0 <= self.f_min < self.f_max <= 0.5):
            raise ValueError(f"Invalid frequency range: f_min={self.f_min}, f_max={self.f_max}")


class IQPreprocessor:
    """
    Preprocessing pipeline for IQ data.

    Converts:
    - Input: IQ samples (shape: (2, N) or [(I, Q), ...])
    - Output: Mel-spectrogram (shape: (n_mels, time_steps))

    Pipeline:
    1. Convert to complex IQ representation
    2. Compute power spectrogram (magnitude squared)
    3. Convert to mel scale
    4. Apply logarithmic scaling
    5. Normalize (optional)
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """
        Initialize preprocessor.

        Args:
            config: PreprocessingConfig instance. Defaults to standard config.
        """
        self.config = config or PreprocessingConfig()
        self._mel_fb = None  # Cached mel filterbank
        logger.info(f"IQPreprocessor initialized with config: {self.config}")

    def preprocess(self, iq_data: list[list[float]]) -> np.ndarray:
        """
        Preprocess raw IQ data to mel-spectrogram.

        Args:
            iq_data: List of [I, Q] samples. Shape: (N, 2) where N is number of samples.
                    Can also be 2D array: [[I1, Q1], [I2, Q2], ...]

        Returns:
            Mel-spectrogram: np.ndarray of shape (n_mels, time_steps)

        Raises:
            ValueError: If input is invalid
            RuntimeError: If preprocessing fails
        """
        try:
            # Step 1: Convert to complex IQ
            iq_complex = self._to_complex_iq(iq_data)
            logger.debug(f"IQ shape: {iq_complex.shape}, dtype: {iq_complex.dtype}")

            # Step 2: Compute power spectrogram using FFT
            spectrogram = self._compute_spectrogram(iq_complex)
            logger.debug(f"Spectrogram shape: {spectrogram.shape}")

            # Step 3: Convert to mel scale
            mel_spec = self._to_mel_scale(spectrogram)
            logger.debug(f"Mel-spectrogram shape: {mel_spec.shape}")

            # Step 4: Apply log scaling
            mel_spec_log = self._apply_log_scale(mel_spec)

            # Step 5: Normalize if configured
            if self.config.normalize:
                mel_spec_log = self._normalize(mel_spec_log)

            return mel_spec_log

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise RuntimeError(f"IQ preprocessing error: {e}") from e

    def _to_complex_iq(self, iq_data: list[list[float]]) -> np.ndarray:
        """
        Convert IQ samples to complex representation.

        Args:
            iq_data: List of [I, Q] pairs or (N, 2) array

        Returns:
            Complex array: shape (N,)
        """
        # Convert to numpy array
        iq_array = np.array(iq_data, dtype=np.float32)

        # Validate shape
        if len(iq_array.shape) != 2 or iq_array.shape[1] != 2:
            raise ValueError(
                f"Expected (N, 2) array, got shape {iq_array.shape}. "
                f"Input should be list of [I, Q] pairs."
            )

        if iq_array.shape[0] < self.config.n_fft:
            raise ValueError(
                f"Not enough samples: {iq_array.shape[0]} < n_fft={self.config.n_fft}. "
                f"Need at least {self.config.n_fft} samples."
            )

        # Extract I and Q, convert to complex
        I = iq_array[:, 0]  # In-phase
        Q = iq_array[:, 1]  # Quadrature

        # Complex: I + 1j*Q
        iq_complex = I + 1j * Q

        logger.debug(f"Converted {len(iq_complex)} IQ samples to complex")
        return iq_complex

    def _compute_spectrogram(self, iq_complex: np.ndarray) -> np.ndarray:
        """
        Compute power spectrogram via STFT.

        Args:
            iq_complex: Complex IQ signal, shape (N,)

        Returns:
            Power spectrogram: shape (n_fft//2 + 1, time_steps)
        """
        # Compute STFT
        # Window: Hann window by default
        window = np.hanning(self.config.n_fft)

        # Compute STFT manually via sliding windows
        n_frames = (len(iq_complex) - self.config.n_fft) // self.config.hop_length + 1
        spectrogram = np.zeros((self.config.n_fft // 2 + 1, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.config.hop_length
            end = start + self.config.n_fft

            # Extract frame and apply window
            frame = iq_complex[start:end] * window

            # Compute FFT
            fft = np.fft.fft(frame)

            # Compute power (magnitude squared)
            magnitude = np.abs(fft[: self.config.n_fft // 2 + 1])
            power = (magnitude**2) / self.config.n_fft

            spectrogram[:, i] = power

        logger.debug(f"Computed STFT: {n_frames} frames of {self.config.n_fft} samples")
        return spectrogram

    def _to_mel_scale(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Convert power spectrogram to mel scale.

        Args:
            spectrogram: Linear spectrogram, shape (n_fft//2 + 1, time_steps)

        Returns:
            Mel-spectrogram: shape (n_mels, time_steps)
        """
        # Build mel filterbank if not cached
        if self._mel_fb is None:
            self._mel_fb = self._build_mel_filterbank()

        # Apply mel filterbank: (n_mels, n_fft//2+1) @ (n_fft//2+1, time_steps)
        mel_spec = np.dot(self._mel_fb, spectrogram)

        logger.debug(f"Converted to mel scale: {mel_spec.shape}")
        return mel_spec

    def _build_mel_filterbank(self) -> np.ndarray:
        """
        Build mel filterbank matrix.

        Returns:
            Filterbank: shape (n_mels, n_fft//2 + 1)
        """
        # Nyquist frequency

        # Convert frequency range to FFT bins
        n_fft_bins = self.config.n_fft // 2 + 1
        int(np.ceil(self.config.f_min * n_fft_bins))
        int(np.floor(self.config.f_max * n_fft_bins))

        # Create mel-spaced frequencies
        mel_points = np.linspace(
            self._hz_to_mel(self.config.f_min),
            self._hz_to_mel(self.config.f_max),
            self.config.n_mels + 2,
        )
        freq_points = np.array([self._mel_to_hz(m) for m in mel_points])

        # Convert to FFT bins
        bin_points = np.array([int(np.floor(f * n_fft_bins)) for f in freq_points])

        # Build triangular filterbank
        filterbank = np.zeros((self.config.n_mels, n_fft_bins))

        for m in range(self.config.n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]

            # Left slope
            if center > left:
                filterbank[m, left:center] = np.arange(center - left) / (center - left)

            # Right slope
            if right > center:
                filterbank[m, center:right] = np.arange(right - center, 0, -1) / (right - center)

        logger.debug(f"Built mel filterbank: {filterbank.shape}")
        return filterbank

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert mel to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def _apply_log_scale(self, mel_spec: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Apply logarithmic scaling.

        Args:
            mel_spec: Mel-spectrogram
            epsilon: Small value to avoid log(0)

        Returns:
            Log-scaled mel-spectrogram
        """
        mel_spec_log = np.log(mel_spec + epsilon)
        logger.debug(
            f"Applied log scaling: min={mel_spec_log.min():.3f}, max={mel_spec_log.max():.3f}"
        )
        return mel_spec_log

    def _normalize(self, mel_spec_log: np.ndarray) -> np.ndarray:
        """
        Normalize mel-spectrogram.

        Args:
            mel_spec_log: Log-scaled mel-spectrogram

        Returns:
            Normalized mel-spectrogram (zero mean, unit variance)
        """
        # Use provided mean/std or compute from data
        if self.config.norm_mean is not None and self.config.norm_std is not None:
            mean = self.config.norm_mean
            std = self.config.norm_std
        else:
            mean = mel_spec_log.mean()
            std = mel_spec_log.std()

        if std == 0:
            logger.warning("Standard deviation is zero, skipping normalization")
            return mel_spec_log

        normalized = (mel_spec_log - mean) / std
        logger.debug(f"Normalized: mean={normalized.mean():.6f}, std={normalized.std():.6f}")
        return normalized

    def get_config_dict(self) -> dict:
        """Return configuration as dictionary for metadata."""
        return {
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "n_mels": self.config.n_mels,
            "f_min": self.config.f_min,
            "f_max": self.config.f_max,
            "power": self.config.power,
            "normalize": self.config.normalize,
        }


def preprocess_iq_data(iq_data: list[list[float]]) -> tuple[np.ndarray, dict]:
    """
    Convenience function to preprocess IQ data with default configuration.

    Args:
        iq_data: List of [I, Q] samples

    Returns:
        Tuple of (mel_spectrogram, metadata_dict)
    """
    preprocessor = IQPreprocessor()
    mel_spec = preprocessor.preprocess(iq_data)
    metadata = {
        "shape": mel_spec.shape,
        "dtype": str(mel_spec.dtype),
        "min": float(mel_spec.min()),
        "max": float(mel_spec.max()),
        "mean": float(mel_spec.mean()),
        "std": float(mel_spec.std()),
    }
    return mel_spec, metadata
