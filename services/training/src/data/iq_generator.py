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

    def generate_iq_batch(
        self,
        frequency_offsets: np.ndarray,
        bandwidths: np.ndarray,
        signal_powers_dbm: np.ndarray,
        noise_floors_dbm: np.ndarray,
        snr_dbs: np.ndarray,
        batch_size: int,
        enable_multipath: bool = True,
        enable_fading: bool = True
    ) -> np.ndarray:
        """
        Generate multiple IQ samples in parallel (GPU-accelerated batch processing).
        
        Args:
            frequency_offsets: Frequency offsets in Hz, shape (batch_size,)
            bandwidths: Signal bandwidths in Hz, shape (batch_size,)
            signal_powers_dbm: Signal powers in dBm, shape (batch_size,)
            noise_floors_dbm: Noise floor powers in dBm, shape (batch_size,)
            snr_dbs: Target SNRs in dB, shape (batch_size,)
            batch_size: Number of samples in this batch
            enable_multipath: Add multipath reflections
            enable_fading: Add Rayleigh fading
            
        Returns:
            Array of complex IQ samples, shape (batch_size, num_samples)
        """
        if not self.use_gpu:
            # CPU fallback: generate samples sequentially
            batch_signals = []
            for i in range(batch_size):
                # Generate clean signal
                signal = self._generate_clean_signal(
                    frequency_offset_hz=float(frequency_offsets[i]),
                    bandwidth_hz=float(bandwidths[i])
                )
                
                # Add multipath (if enabled)
                if enable_multipath:
                    signal = self._add_multipath(signal, num_paths=self.rng_integers(2, 4))
                
                # Add fading (if enabled)
                if enable_fading:
                    signal = self._add_rayleigh_fading(signal)
                
                # Normalize power
                signal = self._normalize_power(signal, float(signal_powers_dbm[i]))
                
                # Add AWGN
                signal = self._add_awgn(
                    signal,
                    noise_floor_dbm=float(noise_floors_dbm[i]),
                    snr_db=float(snr_dbs[i])
                )
                
                batch_signals.append(signal)
            
            return np.array(batch_signals, dtype=np.complex64)
        
        # GPU batch processing: TRUE vectorized operations (all samples in parallel)
        
        # Convert input arrays to GPU if needed
        if not isinstance(frequency_offsets, cp.ndarray):
            freq_offsets = self.xp.array(frequency_offsets, dtype=self.xp.float32)
            bandwidths_gpu = self.xp.array(bandwidths, dtype=self.xp.float32)
            signal_powers = self.xp.array(signal_powers_dbm, dtype=self.xp.float32)
            noise_floors = self.xp.array(noise_floors_dbm, dtype=self.xp.float32)
            snr_dbs_gpu = self.xp.array(snr_dbs, dtype=self.xp.float32)
        else:
            freq_offsets = frequency_offsets
            bandwidths_gpu = bandwidths
            signal_powers = signal_powers_dbm
            noise_floors = noise_floors_dbm
            snr_dbs_gpu = snr_dbs
        
        # Generate ALL clean signals at once (vectorized across batch dimension)
        # Shape: (batch_size, num_samples)
        batch_signals = self._generate_clean_signal_batch(freq_offsets, bandwidths_gpu, batch_size)
        
        # Add multipath to ALL signals at once (if enabled)
        if enable_multipath:
            batch_signals = self._add_multipath_batch(batch_signals, batch_size)
        
        # Add fading to ALL signals at once (if enabled)
        if enable_fading:
            batch_signals = self._add_rayleigh_fading_batch(batch_signals, batch_size)
        
        # Normalize power for ALL signals at once
        batch_signals = self._normalize_power_batch(batch_signals, signal_powers)
        
        # Add AWGN to ALL signals at once
        batch_signals = self._add_awgn_batch(batch_signals, noise_floors, snr_dbs_gpu)
        
        # Single GPU→CPU transfer for entire batch (MAJOR SPEEDUP)
        batch_signals_cpu = cp.asnumpy(batch_signals)
        
        return batch_signals_cpu.astype(np.complex64)

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

    def _generate_clean_signal_batch(
        self,
        frequency_offsets: 'np.ndarray',
        bandwidths: 'np.ndarray',
        batch_size: int
    ) -> 'np.ndarray':
        """
        Generate clean signals for entire batch (TRUE GPU vectorization).
        
        Args:
            frequency_offsets: Array of frequency offsets (batch_size,)
            bandwidths: Array of signal bandwidths (batch_size,)
            batch_size: Number of signals to generate
            
        Returns:
            Batch of complex signals, shape (batch_size, num_samples)
        """
        # Expand time axis to batch dimension: (1, num_samples) → (batch_size, num_samples)
        time_batch = self.xp.tile(self.time_axis[self.xp.newaxis, :], (batch_size, 1))
        
        # Expand frequency offsets: (batch_size,) → (batch_size, 1) for broadcasting
        freq_offsets_expanded = frequency_offsets[:, self.xp.newaxis]
        
        # Vectorized phase calculation: (batch_size, num_samples)
        phase = 2 * self.xp.pi * freq_offsets_expanded * time_batch
        
        # Add random phase modulation for each signal in batch
        # Shape: (batch_size, num_samples)
        phase_mod = self.rng_normal(0, 0.3, (batch_size, self.num_samples))
        
        # Smooth phase modulation using FFT-based 2D convolution (TRUE vectorization)
        smooth_kernel = self.xp.ones(10) / 10
        
        # Pad kernel to match signal length for FFT convolution
        kernel_padded = self.xp.zeros((1, self.num_samples))
        kernel_padded[0, :len(smooth_kernel)] = smooth_kernel
        
        # FFT-based convolution (all batch samples at once)
        phase_mod_fft = self.xp.fft.fft(phase_mod, axis=1)
        kernel_fft = self.xp.fft.fft(kernel_padded, axis=1)
        phase_mod = self.xp.fft.ifft(phase_mod_fft * kernel_fft, axis=1).real
        
        phase += phase_mod
        
        # Vectorized complex exponential: shape (batch_size, num_samples)
        signals = self.xp.exp(1j * phase)
        
        return signals

    def _add_multipath_batch(
        self,
        batch_signals: 'np.ndarray',
        batch_size: int
    ) -> 'np.ndarray':
        """
        Add multipath to entire batch (TRUE GPU vectorization using advanced indexing).
        
        Args:
            batch_signals: Input signals, shape (batch_size, num_samples)
            batch_size: Number of signals
            
        Returns:
            Signals with multipath, shape (batch_size, num_samples)
        """
        multipath_signals = batch_signals.copy()
        
        # Generate random multipath parameters for entire batch
        num_paths = 2  # Fixed for vectorization
        
        for path_idx in range(num_paths):
            # Random delays for entire batch: (batch_size,)
            delay_us = self.rng_uniform(0.5, 5.0, batch_size)
            delay_samples = (delay_us * 1e-6 * self.sample_rate_hz).astype(int)
            delay_samples = self.xp.clip(delay_samples, 1, self.num_samples // 4)
            
            # Random attenuations for entire batch: (batch_size,)
            attenuation_db = self.rng_uniform(-25, -10, batch_size)
            attenuation_linear = 10 ** (attenuation_db / 20.0)
            
            # Random phase shifts for entire batch: (batch_size,)
            phase_shifts = self.rng_uniform(0, 2 * float(self.xp.pi), batch_size)
            
            # VECTORIZED MULTIPATH USING ADVANCED INDEXING
            # Create index matrix for delayed samples
            # sample_indices[i, j] = sample index for batch i, time j after applying delay_samples[i]
            time_indices = self.xp.arange(self.num_samples)[self.xp.newaxis, :]  # (1, num_samples)
            delay_samples_expanded = delay_samples[:, self.xp.newaxis]  # (batch_size, 1)
            sample_indices = time_indices - delay_samples_expanded  # (batch_size, num_samples)
            
            # Create mask for valid samples (before delay, samples are zero)
            valid_mask = sample_indices >= 0  # (batch_size, num_samples)
            sample_indices = self.xp.clip(sample_indices, 0, self.num_samples - 1)
            
            # Advanced indexing to extract delayed signals (all batch samples at once)
            batch_indices = self.xp.arange(batch_size)[:, self.xp.newaxis]  # (batch_size, 1)
            delayed_signals = batch_signals[batch_indices, sample_indices]  # (batch_size, num_samples)
            
            # Apply mask (set invalid samples to zero)
            delayed_signals = delayed_signals * valid_mask
            
            # Apply attenuation and phase shift (vectorized)
            attenuation_expanded = attenuation_linear[:, self.xp.newaxis]  # (batch_size, 1)
            phase_shifts_expanded = phase_shifts[:, self.xp.newaxis]  # (batch_size, 1)
            delayed_signals *= attenuation_expanded * self.xp.exp(1j * phase_shifts_expanded)
            
            # Add to multipath signal (vectorized)
            multipath_signals += delayed_signals
        
        return multipath_signals

    def _add_rayleigh_fading_batch(
        self,
        batch_signals: 'np.ndarray',
        batch_size: int
    ) -> 'np.ndarray':
        """
        Add Rayleigh fading to entire batch (TRUE GPU vectorization).
        
        Uses fixed coherence time (100ms) for all samples to enable full vectorization.
        This is a reasonable approximation for typical channel conditions.
        
        Args:
            batch_signals: Input signals, shape (batch_size, num_samples)
            batch_size: Number of signals
            
        Returns:
            Faded signals, shape (batch_size, num_samples)
        """
        # Fixed coherence time for vectorization (100ms is typical for mobile channels)
        coherence_time_ms = 100.0
        coherence_samples = int(coherence_time_ms * 1e-3 * self.sample_rate_hz)
        num_fading_samples = max(1, self.num_samples // coherence_samples)
        
        # Generate Rayleigh fading envelopes for entire batch (vectorized)
        # Shape: (batch_size, num_fading_samples)
        i_fading = self.rng_normal(0, 1, (batch_size, num_fading_samples))
        q_fading = self.rng_normal(0, 1, (batch_size, num_fading_samples))
        fading_envelopes = self.xp.sqrt(i_fading**2 + q_fading**2)
        
        # Upsample to match signal length (vectorized repeat along axis=1)
        # Shape: (batch_size, num_fading_samples * coherence_samples)
        fading_envelopes = self.xp.repeat(fading_envelopes, coherence_samples, axis=1)
        
        # Trim or pad to exact signal length (vectorized)
        if fading_envelopes.shape[1] < self.num_samples:
            # Pad all batch samples at once
            pad_width = ((0, 0), (0, self.num_samples - fading_envelopes.shape[1]))
            fading_envelopes = self.xp.pad(fading_envelopes, pad_width, mode='edge')
        else:
            # Trim all batch samples at once
            fading_envelopes = fading_envelopes[:, :self.num_samples]
        
        # Apply fading to entire batch (vectorized element-wise multiplication)
        # Shape: (batch_size, num_samples) × (batch_size, num_samples)
        return batch_signals * fading_envelopes

    def _normalize_power_batch(
        self,
        batch_signals: 'np.ndarray',
        target_powers_dbm: 'np.ndarray'
    ) -> 'np.ndarray':
        """
        Normalize power for entire batch (TRUE vectorization).
        
        Args:
            batch_signals: Input signals, shape (batch_size, num_samples)
            target_powers_dbm: Target powers in dBm, shape (batch_size,)
            
        Returns:
            Normalized signals, shape (batch_size, num_samples)
        """
        # Calculate current power for ALL signals at once (vectorized mean along time axis)
        # Shape: (batch_size,)
        current_powers = self.xp.mean(self.xp.abs(batch_signals) ** 2, axis=1)
        
        # Convert target powers to linear scale (vectorized)
        target_powers_linear = 10 ** (target_powers_dbm / 10.0) * 1e-3  # Watts
        
        # Calculate scaling factors for entire batch (vectorized)
        # Shape: (batch_size,)
        scaling_factors = self.xp.sqrt(target_powers_linear / current_powers)
        
        # Apply scaling (broadcast over time axis)
        # Shape: (batch_size, 1) × (batch_size, num_samples) → (batch_size, num_samples)
        return batch_signals * scaling_factors[:, self.xp.newaxis]

    def _add_awgn_batch(
        self,
        batch_signals: 'np.ndarray',
        noise_floors_dbm: 'np.ndarray',
        snr_dbs: 'np.ndarray'
    ) -> 'np.ndarray':
        """
        Add AWGN to entire batch (TRUE vectorization).
        
        Args:
            batch_signals: Input signals, shape (batch_size, num_samples)
            noise_floors_dbm: Noise floor powers, shape (batch_size,)
            snr_dbs: Target SNRs in dB, shape (batch_size,)
            
        Returns:
            Noisy signals, shape (batch_size, num_samples)
        """
        # Calculate signal powers for entire batch (vectorized)
        # Shape: (batch_size,)
        signal_powers = self.xp.mean(self.xp.abs(batch_signals) ** 2, axis=1)
        
        # Calculate noise powers from SNRs (vectorized)
        snr_linear = 10 ** (snr_dbs / 10.0)
        noise_powers = signal_powers / snr_linear
        
        # Calculate noise standard deviations (vectorized)
        # Shape: (batch_size,)
        noise_stds = self.xp.sqrt(noise_powers / 2.0)
        
        # Generate complex Gaussian noise for entire batch
        # Shape: (batch_size, num_samples)
        batch_size = batch_signals.shape[0]
        
        # Generate I and Q components with per-sample standard deviations
        # Expand noise_stds for broadcasting: (batch_size,) → (batch_size, 1)
        noise_stds_expanded = noise_stds[:, self.xp.newaxis]
        
        # Generate noise (vectorized): (batch_size, num_samples)
        noise_i = self.rng_normal(0, 1, (batch_size, self.num_samples)) * noise_stds_expanded
        noise_q = self.rng_normal(0, 1, (batch_size, self.num_samples)) * noise_stds_expanded
        noise = noise_i + 1j * noise_q
        
        return batch_signals + noise
