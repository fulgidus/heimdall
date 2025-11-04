"""
Synthetic IQ sample generator for RF signals.

Generates realistic complex IQ samples with:
- Multipath propagation (2-3 delayed reflections)
- Rayleigh fading (slow fading, coherence time 50-200ms)
- AWGN (additive white Gaussian noise)

GPU Acceleration:
- Automatically uses CuPy if available (10-15x speedup)
- Falls back to NumPy if CuPy not installed or no GPU
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import structlog

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except (ImportError, Exception):
    cp = None
    GPU_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticIQSample:
    """Container for synthetic IQ sample data with additional metadata."""
    samples: np.ndarray  # Complex64 array
    sample_rate_hz: float
    duration_ms: float
    center_frequency_hz: float
    rx_id: str
    rx_lat: float
    rx_lon: float
    timestamp: float

    @property
    def num_samples(self) -> int:
        """Number of complex samples."""
        return len(self.samples)

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis in seconds."""
        return np.arange(self.num_samples) / self.sample_rate_hz

    def to_datetime(self) -> datetime:
        """Convert float timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


class SyntheticIQGenerator:
    """Generate synthetic IQ samples with realistic RF effects."""

    def __init__(
        self,
        sample_rate_hz: float = 200_000,
        duration_ms: float = 1000.0,
        seed: Optional[int] = None,
        use_gpu: bool = True
    ):
        """
        Initialize IQ generator.

        Args:
            sample_rate_hz: Sampling rate (default: 200 kHz for 2x oversampling)
            duration_ms: Signal duration in milliseconds
            seed: Random seed for reproducibility
            use_gpu: Use GPU acceleration if available (default: True)
        """
        self.sample_rate_hz = sample_rate_hz
        self.duration_ms = duration_ms
        
        # GPU acceleration setup
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.xp = cp  # Use CuPy (GPU arrays)
            # Use CuPy's random state for GPU acceleration
            if seed is not None:
                self.xp.random.seed(seed)
            logger.info(f"IQ Generator: GPU acceleration ENABLED (CuPy with GPU RNG)")
        else:
            self.xp = np  # Use NumPy (CPU arrays)
            if use_gpu and not GPU_AVAILABLE:
                logger.warning(f"IQ Generator: GPU requested but not available, using CPU")
        
        # Random number generator (will use CuPy RNG if GPU enabled)
        # NumPy uses modern Generator API, CuPy uses legacy RandomState API
        if self.use_gpu:
            # CuPy uses global random state
            self.rng_normal = lambda loc, scale, size: self.xp.random.normal(loc, scale, size)
            self.rng_uniform = lambda low, high, size=None: self.xp.random.uniform(low, high, size)
            self.rng_integers = lambda low, high: int(self.xp.random.randint(low, high))
        else:
            # NumPy uses Generator API
            cpu_rng = np.random.default_rng(seed)
            self.rng_normal = cpu_rng.normal
            self.rng_uniform = cpu_rng.uniform
            self.rng_integers = cpu_rng.integers

        # Calculate number of samples
        self.num_samples = int(sample_rate_hz * duration_ms / 1000.0)
        self.time_axis = self.xp.arange(self.num_samples) / sample_rate_hz

    def generate_iq_sample(
        self,
        center_frequency_hz: float,
        signal_power_dbm: float,
        noise_floor_dbm: float,
        snr_db: float,
        frequency_offset_hz: float,
        bandwidth_hz: float,
        rx_id: str,
        rx_lat: float,
        rx_lon: float,
        timestamp: float,
        enable_multipath: bool = True,
        enable_fading: bool = True
    ) -> 'SyntheticIQSample':
        """
        Generate a single IQ sample for one receiver.

        Args:
            center_frequency_hz: Center frequency (e.g., 144000000 for 144 MHz)
            signal_power_dbm: Signal power at receiver
            noise_floor_dbm: Noise floor power
            snr_db: Target SNR
            frequency_offset_hz: Doppler/oscillator offset (±50 Hz typical)
            bandwidth_hz: Signal bandwidth (e.g., 12500 Hz for FM)
            rx_id: Receiver identifier
            rx_lat: Receiver latitude
            rx_lon: Receiver longitude
            timestamp: Sample timestamp
            enable_multipath: Add multipath reflections
            enable_fading: Add Rayleigh fading

        Returns:
            SyntheticIQSample with complex samples
        """
        # 1. Generate clean signal (complex exponential with frequency offset)
        signal = self._generate_clean_signal(frequency_offset_hz, bandwidth_hz)

        # 2. Add multipath reflections (if enabled)
        if enable_multipath:
            signal = self._add_multipath(signal, num_paths=self.rng_integers(2, 4))

        # 3. Add Rayleigh fading (if enabled)
        if enable_fading:
            signal = self._add_rayleigh_fading(signal)

        # 4. Normalize signal power
        signal = self._normalize_power(signal, signal_power_dbm)

        # 5. Add AWGN noise
        signal = self._add_awgn(signal, noise_floor_dbm, snr_db)

        # 6. Convert back to NumPy (if on GPU) for storage compatibility
        if self.use_gpu:
            signal = cp.asnumpy(signal)
        
        return SyntheticIQSample(
            samples=signal.astype(np.complex64),
            sample_rate_hz=self.sample_rate_hz,
            duration_ms=self.duration_ms,
            center_frequency_hz=center_frequency_hz,
            rx_id=rx_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=timestamp
        )

    def _generate_clean_signal(
        self,
        frequency_offset_hz: float,
        bandwidth_hz: float
    ) -> np.ndarray:
        """
        Generate clean complex exponential signal.

        Args:
            frequency_offset_hz: Frequency offset from center
            bandwidth_hz: Signal bandwidth (used for phase modulation)

        Returns:
            Complex signal array (NumPy for CPU compatibility)
        """
        # Complex exponential: exp(j * 2π * f * t)
        phase = 2 * self.xp.pi * frequency_offset_hz * self.time_axis

        # Add random phase modulation to simulate FM/PM
        # Modulation bandwidth proportional to signal bandwidth
        phase_mod = self.rng_normal(0, 0.3, self.num_samples)  # Random phase deviation (on GPU if available)
        
        # Smooth the phase modulation (convolve with moving average)
        # Create smoothing kernel
        smooth_kernel = self.xp.ones(10) / 10
        # Use FFT-based convolution for efficiency (especially on GPU)
        phase_mod = self.xp.convolve(phase_mod, smooth_kernel, mode='same')

        phase += phase_mod

        signal = self.xp.exp(1j * phase)

        return signal

    def _add_multipath(
        self,
        signal: np.ndarray,
        num_paths: int = 2
    ) -> np.ndarray:
        """
        Add multipath reflections (delayed, attenuated copies).

        Args:
            signal: Input signal
            num_paths: Number of multipath components (2-3 typical)

        Returns:
            Signal with multipath
        """
        multipath_signal = signal.copy()

        for _ in range(num_paths):
            # Random delay: 0.5-5 microseconds (typical urban/suburban)
            delay_us = float(self.rng_uniform(0.5, 5.0))
            delay_samples = int(delay_us * 1e-6 * self.sample_rate_hz)
            delay_samples = max(1, min(delay_samples, self.num_samples // 4))

            # Attenuation: -10 to -25 dB relative to main signal
            attenuation_db = float(self.rng_uniform(-25, -10))
            attenuation_linear = 10 ** (attenuation_db / 20.0)

            # Random phase shift
            phase_shift = float(self.rng_uniform(0, 2 * float(self.xp.pi)))

            # Create delayed copy
            delayed_signal = self.xp.zeros_like(signal)
            delayed_signal[delay_samples:] = signal[:-delay_samples]
            delayed_signal *= attenuation_linear * self.xp.exp(1j * phase_shift)

            # Add to multipath signal
            multipath_signal += delayed_signal

        return multipath_signal

    def _add_rayleigh_fading(self, signal: np.ndarray) -> np.ndarray:
        """
        Add Rayleigh fading (slow fading for mobile/atmospheric effects).

        Coherence time: 50-200ms (slow fading)

        Args:
            signal: Input signal

        Returns:
            Faded signal
        """
        # Coherence time: 50-200ms
        coherence_time_ms = float(self.rng_uniform(50, 200))
        coherence_samples = int(coherence_time_ms * 1e-3 * self.sample_rate_hz)

        # Generate Rayleigh fading envelope (two Gaussian random processes)
        num_fading_samples = max(1, self.num_samples // coherence_samples)

        # In-phase and quadrature Gaussian processes (GPU-accelerated if available)
        i_fading = self.rng_normal(0, 1, num_fading_samples)
        q_fading = self.rng_normal(0, 1, num_fading_samples)

        # Rayleigh envelope: sqrt(I^2 + Q^2) - GPU accelerated
        fading_envelope = self.xp.sqrt(i_fading**2 + q_fading**2)

        # Upsample to match signal length (repeat each sample)
        fading_envelope = self.xp.repeat(fading_envelope, coherence_samples)

        # Trim or pad to exact length
        if len(fading_envelope) > self.num_samples:
            fading_envelope = fading_envelope[:self.num_samples]
        else:
            # Pad to exact length
            pad_width = self.num_samples - len(fading_envelope)
            if self.use_gpu:
                # CuPy pad
                fading_envelope = self.xp.pad(
                    fading_envelope,
                    (0, pad_width),
                    mode='edge'
                )
            else:
                # NumPy pad
                fading_envelope = np.pad(
                    fading_envelope,
                    (0, pad_width),
                    mode='edge'
                )

        # Apply fading (multiplicative)
        faded_signal = signal * fading_envelope

        return faded_signal

    def _normalize_power(
        self,
        signal: np.ndarray,
        target_power_dbm: float
    ) -> np.ndarray:
        """
        Normalize signal to target power level.

        Args:
            signal: Input signal
            target_power_dbm: Target power in dBm

        Returns:
            Normalized signal
        """
        # Calculate current power (average of |signal|^2)
        current_power_linear = self.xp.mean(self.xp.abs(signal) ** 2)

        # Convert target power from dBm to linear scale
        # P(dBm) = 10 * log10(P_mW)
        # Reference: 1 mW = 0 dBm
        target_power_linear = 10 ** (target_power_dbm / 10.0) * 1e-3  # Convert to watts

        # Scaling factor
        scaling = self.xp.sqrt(target_power_linear / current_power_linear)

        return signal * scaling

    def _add_awgn(
        self,
        signal: np.ndarray,
        noise_floor_dbm: float,
        snr_db: float
    ) -> np.ndarray:
        """
        Add additive white Gaussian noise.

        Args:
            signal: Input signal
            noise_floor_dbm: Noise floor power
            snr_db: Target SNR (signal-to-noise ratio)

        Returns:
            Noisy signal
        """
        # Calculate signal power
        signal_power = self.xp.mean(self.xp.abs(signal) ** 2)

        # Calculate noise power from SNR
        # SNR = P_signal / P_noise (linear)
        # P_noise = P_signal / (10^(SNR_dB / 10))
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate complex Gaussian noise (I and Q components)
        # For complex noise: variance = noise_power / 2 per component
        noise_std = float(self.xp.sqrt(noise_power / 2.0))

        # Generate complex Gaussian noise (GPU-accelerated if available)
        noise_i = self.rng_normal(0, noise_std, self.num_samples)
        noise_q = self.rng_normal(0, noise_std, self.num_samples)
        noise = noise_i + 1j * noise_q

        return signal + noise
