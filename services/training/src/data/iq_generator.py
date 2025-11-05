"""
Synthetic IQ sample generator for RF signals.

Generates realistic complex IQ samples with:
- Multipath propagation (2-3 delayed reflections)
- Rayleigh fading (slow fading, coherence time 50-200ms)
- AWGN (additive white Gaussian noise)

GPU Acceleration:
- Automatically uses CuPy if available (10-15x speedup)
- Falls back to NumPy if CuPy not installed or no GPU

Spatial Coherence for ML Training:
- When generating batches (multiple receivers), all receivers MUST receive
  the SAME audio content to maintain spatial coherence
- Only RF propagation effects (multipath, fading, SNR, frequency offset)
  should differ between receivers
- This is critical for ML training: the model learns spatial localization
  from time-of-arrival and signal strength differences, NOT from different
  audio content at each receiver
- Both GPU and CPU paths implement this by calling batch audio generation
  once, then applying per-receiver effects
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import structlog

# Audio library integration
from .audio_library import get_audio_loader, AudioLibraryEmptyError

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
        use_gpu: bool = True,
        use_audio_library: bool = False,
        audio_library_fallback: bool = True
    ):
        """
        Initialize IQ generator.

        Args:
            sample_rate_hz: Sampling rate (default: 200 kHz for 2x oversampling)
            duration_ms: Signal duration in milliseconds
            seed: Random seed for reproducibility
            use_gpu: Use GPU acceleration if available (default: True)
            use_audio_library: Use audio library for voice samples (default: False)
            audio_library_fallback: Fallback to formant synthesis if library fails (default: True)
        """
        self.sample_rate_hz = sample_rate_hz
        self.duration_ms = duration_ms
        self.use_audio_library = use_audio_library
        self.audio_library_fallback = audio_library_fallback
        
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
            # Store the RNG object for audio library (GPU mode uses global state)
            self.rng = None  # GPU doesn't have a dedicated RNG object
        else:
            # NumPy uses Generator API
            cpu_rng = np.random.default_rng(seed)
            self.rng_normal = cpu_rng.normal
            self.rng_uniform = cpu_rng.uniform
            self.rng_integers = cpu_rng.integers
            # Store the RNG object for audio library
            self.rng = cpu_rng

        # Calculate number of samples
        self.num_samples = int(sample_rate_hz * duration_ms / 1000.0)
        self.time_axis = self.xp.arange(self.num_samples) / sample_rate_hz
        
        # Initialize audio loader ONCE (reuse same instance for all batches)
        # This ensures audio consistency: all batches use the same loader with
        # predictable RNG state progression
        self.audio_loader = None  # Lazy initialization on first use
        if self.use_audio_library:
            # Create audio loader with seeded RNG for reproducibility
            # Use CPU RNG even in GPU mode (audio loading happens on CPU)
            audio_rng = np.random.default_rng(seed) if seed is not None else None
            self.audio_loader = get_audio_loader(
                target_sample_rate=int(self.sample_rate_hz),
                rng=audio_rng,
                force_new=True,  # Don't use singleton (each generator has its own loader)
            )
            logger.info(
                "audio_loader_initialized",
                use_library=True,
                rng_seeded=seed is not None,
            )

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
        
        All receivers in the batch receive the SAME audio content, with only
        RF propagation effects (multipath, fading, SNR) differing. This ensures
        spatial coherence required for ML training.
        
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
            # CPU fallback: Use batch function to ensure audio consistency
            # Convert to numpy arrays if needed
            freq_offsets = np.array(frequency_offsets, dtype=np.float32)
            bandwidths_cpu = np.array(bandwidths, dtype=np.float32)
            signal_powers = np.array(signal_powers_dbm, dtype=np.float32)
            noise_floors = np.array(noise_floors_dbm, dtype=np.float32)
            snr_dbs_cpu = np.array(snr_dbs, dtype=np.float32)
            
            # Generate ALL clean signals at once (same audio for all receivers)
            batch_signals = self._generate_clean_signal_batch(freq_offsets, bandwidths_cpu, batch_size)
            
            # Add effects per-receiver (sequential, but same audio content)
            for i in range(batch_size):
                # Add multipath (if enabled)
                if enable_multipath:
                    batch_signals[i] = self._add_multipath(batch_signals[i], num_paths=self.rng_integers(2, 4))
                
                # Add fading (if enabled)
                if enable_fading:
                    batch_signals[i] = self._add_rayleigh_fading(batch_signals[i])
                
                # Normalize power
                batch_signals[i] = self._normalize_power(batch_signals[i], float(signal_powers[i]))
                
                # Add AWGN
                batch_signals[i] = self._add_awgn(
                    batch_signals[i],
                    noise_floor_dbm=float(noise_floors[i]),
                    snr_db=float(snr_dbs_cpu[i])
                )
            
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

    def _generate_voice_audio(self) -> np.ndarray:
        """
        Generate voice audio using audio library or formant synthesis fallback.
        
        Uses the pre-initialized audio loader (created in __init__) to ensure
        consistent audio selection across batch generations. The loader's RNG
        state advances predictably, ensuring reproducibility.
        
        Returns:
            Audio signal array normalized to [-1, 1]
        """
        # If audio library disabled, use formant synthesis
        if not self.use_audio_library or self.audio_loader is None:
            return self._generate_formant_voice_audio()
        
        # Load from audio library using the pre-initialized loader
        try:
            audio_samples, sample_rate = self.audio_loader.get_random_sample(
                category=None,  # Use weighted random selection based on category weights
                duration_ms=self.duration_ms
            )
            
            # Convert to GPU array if using GPU
            if self.use_gpu:
                audio_samples = self.xp.array(audio_samples)
            
            logger.info(
                "voice_audio_generated_from_library",
                duration_ms=self.duration_ms,
                sample_rate=sample_rate,
                audio_shape=audio_samples.shape if hasattr(audio_samples, 'shape') else len(audio_samples),
            )
            
            return audio_samples
            
        except (AudioLibraryEmptyError, Exception) as e:
            # Fallback to formant synthesis if configured
            if self.audio_library_fallback:
                logger.warning(
                    "audio_library_fallback",
                    reason=str(e),
                    fallback_to="formant_synthesis",
                )
                return self._generate_formant_voice_audio()
            else:
                # Re-raise if no fallback configured
                raise

    def _generate_formant_voice_audio(self) -> np.ndarray:
        """
        Generate realistic human voice audio using formant synthesis.
        
        Simulates amateur radio voice transmission with:
        - 3 formants (F1: 500-800 Hz, F2: 1000-2000 Hz, F3: 2500-3000 Hz)
        - Speech pauses (70% duty cycle)
        - Smooth attack/decay envelopes
        - 300-3000 Hz voice band
        
        Returns:
            Audio signal array normalized to [-1, 1]
        """
        # Generate 3 formant frequencies (fundamental voice components)
        f1 = self.rng_uniform(500, 800)    # First formant (vowel quality)
        f2 = self.rng_uniform(1000, 2000)  # Second formant (vowel quality)
        f3 = self.rng_uniform(2500, 3000)  # Third formant (naturalness)
        
        # Generate formant signals
        formant1 = self.xp.sin(2 * self.xp.pi * f1 * self.time_axis)
        formant2 = self.xp.sin(2 * self.xp.pi * f2 * self.time_axis)
        formant3 = self.xp.sin(2 * self.xp.pi * f3 * self.time_axis)
        
        # Mix formants with realistic amplitudes (F1 strongest, F3 weakest)
        audio = 0.6 * formant1 + 0.3 * formant2 + 0.1 * formant3
        
        # Add speech pauses (70% duty cycle for realistic speech)
        pause_duration = int(self.num_samples * 0.3)  # 30% pauses
        speech_duration = self.num_samples - pause_duration
        
        # Create envelope with attack/decay
        envelope = self.xp.ones(self.num_samples)
        
        # Random pause in the middle
        pause_start = self.rng_integers(speech_duration // 4, 3 * speech_duration // 4)
        pause_end = pause_start + pause_duration
        
        # Smooth attack (10ms)
        attack_samples = int(0.01 * self.sample_rate_hz)
        envelope[:attack_samples] = self.xp.linspace(0, 1, attack_samples)
        
        # Smooth decay (10ms)
        decay_samples = int(0.01 * self.sample_rate_hz)
        envelope[-decay_samples:] = self.xp.linspace(1, 0, decay_samples)
        
        # Apply pause
        if pause_end < self.num_samples:
            # Decay before pause
            envelope[pause_start:pause_start + decay_samples] = self.xp.linspace(1, 0, decay_samples)
            # Zero during pause
            envelope[pause_start + decay_samples:pause_end - attack_samples] = 0
            # Attack after pause
            envelope[pause_end - attack_samples:pause_end] = self.xp.linspace(0, 1, attack_samples)
        
        # Apply envelope to audio
        audio = audio * envelope
        
        # Normalize to [-1, 1]
        audio_max = self.xp.max(self.xp.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        
        return audio

    def _generate_clean_signal(
        self,
        frequency_offset_hz: float,
        bandwidth_hz: float
    ) -> np.ndarray:
        """
        Generate FM-modulated signal with realistic audio content.
        
        Amateur radio FM characteristics:
        - Narrow FM: ±5 kHz deviation, 12.5 kHz bandwidth
        - Wide FM: ±15 kHz deviation, 25 kHz bandwidth
        - Signal mix: 75% voice, 10% single tone, 10% dual-tone (DTMF), 5% carrier
        - Pre-emphasis: 6 dB/octave

        Args:
            frequency_offset_hz: Frequency offset from center
            bandwidth_hz: Signal bandwidth (12.5 kHz or 25 kHz)

        Returns:
            Complex FM-modulated signal array
        """
        # Determine FM deviation based on bandwidth
        # Narrow FM: 12.5 kHz → ±5 kHz deviation
        # Wide FM: 25 kHz → ±15 kHz deviation
        if bandwidth_hz <= 15000:
            freq_deviation_hz = 5000.0  # Narrow FM
        else:
            freq_deviation_hz = 15000.0  # Wide FM
        
        # Generate audio content based on signal type distribution
        signal_type = self.rng_uniform(0, 1)
        
        if signal_type < 0.75:
            # 75% - Voice modulation
            audio = self._generate_voice_audio()
            
            # Apply pre-emphasis filter (6 dB/octave) for voice
            # Pre-emphasis = HPF with time constant τ = 75 μs (Europe) or 50 μs (US)
            # Transfer function: H(s) = 1 + sτ
            # Discrete approximation: y[n] = x[n] - 0.95*x[n-1]
            pre_emphasized = self.xp.copy(audio)
            pre_emphasized[1:] = audio[1:] - 0.95 * audio[:-1]
            audio = pre_emphasized
            
        elif signal_type < 0.85:
            # 10% - Single tone (e.g., test tone, beacon)
            tone_freq = self.rng_uniform(800, 1200)  # Voice range single tone
            audio = self.xp.sin(2 * self.xp.pi * tone_freq * self.time_axis)
            
        elif signal_type < 0.95:
            # 10% - Dual-tone (DTMF-like)
            tone1_freq = self.rng_uniform(697, 941)   # Low DTMF group
            tone2_freq = self.rng_uniform(1209, 1633) # High DTMF group
            audio = 0.5 * self.xp.sin(2 * self.xp.pi * tone1_freq * self.time_axis) + \
                    0.5 * self.xp.sin(2 * self.xp.pi * tone2_freq * self.time_axis)
        else:
            # 5% - Unmodulated carrier (CW-like)
            audio = self.xp.zeros(self.num_samples)
        
        # FM modulation: phase(t) = 2π * fc * t + 2π * Δf * ∫audio(t)dt
        # Discrete integration: cumulative sum with time step
        dt = 1.0 / self.sample_rate_hz
        audio_integral = self.xp.cumsum(audio) * dt
        
        # Carrier phase with frequency offset
        carrier_phase = 2 * self.xp.pi * frequency_offset_hz * self.time_axis
        
        # FM modulation phase
        modulation_phase = 2 * self.xp.pi * freq_deviation_hz * audio_integral
        
        # Total phase
        phase = carrier_phase + modulation_phase
        
        # Generate complex FM signal
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
        Add additive white Gaussian noise with enhanced realism.
        
        For waterfall visualization:
        - Increases noise for SNR < 10 dB (30% boost)
        - Adds 10% extra noise floor for visible "carpet" in waterfall
        - Ensures realistic noise characteristics for amateur radio

        Args:
            signal: Input signal
            noise_floor_dbm: Noise floor power
            snr_db: Target SNR (signal-to-noise ratio)

        Returns:
            Noisy signal with realistic noise floor
        """
        # Calculate signal power
        signal_power = self.xp.mean(self.xp.abs(signal) ** 2)

        # Calculate base noise power from SNR
        # SNR = P_signal / P_noise (linear)
        # P_noise = P_signal / (10^(SNR_dB / 10))
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Increase noise for low SNR signals (< 10 dB)
        # This creates a more visible waterfall for weak signals
        if snr_db < 10.0:
            noise_boost_factor = 1.3  # 30% more noise
            noise_power *= noise_boost_factor
        
        # Always add 10% extra noise for realistic background
        # This creates the visible "noise carpet" in waterfall
        noise_power *= 1.1

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
        Generate FM-modulated signals for entire batch (GPU vectorization).
        
        CRITICAL FOR ML TRAINING: This method generates the SAME audio content
        for all receivers in a batch to maintain spatial coherence. Only the
        frequency offsets differ per receiver. This ensures the model learns
        spatial localization from propagation effects (time-of-arrival, signal
        strength), NOT from different audio content at each receiver.
        
        Generates realistic FM voice signals with:
        - 75% voice, 10% single tone, 10% dual-tone, 5% carrier
        - ±5 kHz deviation (narrow FM) or ±15 kHz (wide FM)
        - Pre-emphasis for voice
        
        Args:
            frequency_offsets: Array of frequency offsets (batch_size,)
            bandwidths: Array of signal bandwidths (batch_size,)
            batch_size: Number of signals to generate
            
        Returns:
            Batch of complex FM signals, shape (batch_size, num_samples)
            All signals have identical audio content, different frequency offsets
        """
        # Expand time axis to batch dimension: (1, num_samples) → (batch_size, num_samples)
        time_batch = self.xp.tile(self.time_axis[self.xp.newaxis, :], (batch_size, 1))
        
        # Determine FM deviation per signal based on bandwidth
        # Shape: (batch_size,)
        freq_deviations = self.xp.where(
            bandwidths <= 15000,
            5000.0,   # Narrow FM
            15000.0   # Wide FM
        )
        
        # Generate audio content for each signal in batch
        # Initialize audio batch: (batch_size, num_samples)
        audio_batch = self.xp.zeros((batch_size, self.num_samples))
        
        # Determine signal type ONCE for the entire batch (all receivers get same signal type)
        # This ensures spatial coherence - all receivers "hear" the same content
        signal_type = self.rng_uniform(0, 1)
        
        logger.info(
            "batch_signal_type_determined",
            signal_type=float(signal_type),
            batch_size=batch_size,
        )
        
        if signal_type < 0.75:
            # 75% - Voice modulation
            # Generate audio ONCE and broadcast to all receivers
            logger.info("generating_voice_audio_for_batch", batch_size=batch_size)
            audio = self._generate_voice_audio()
            
            # Apply pre-emphasis filter (6 dB/octave)
            pre_emphasized = self.xp.copy(audio)
            pre_emphasized[1:] = audio[1:] - 0.95 * audio[:-1]
            
            # Broadcast same audio to ALL receivers in batch
            logger.info("broadcasting_audio_to_batch", batch_size=batch_size)
            for i in range(batch_size):
                audio_batch[i] = pre_emphasized
                
        elif signal_type < 0.85:
            # 10% - Single tone
            # Generate ONCE and broadcast to all receivers
            tone_freq = self.rng_uniform(800, 1200)
            audio = self.xp.sin(2 * self.xp.pi * tone_freq * self.time_axis)
            for i in range(batch_size):
                audio_batch[i] = audio
                
        elif signal_type < 0.95:
            # 10% - Dual-tone (DTMF-like)
            # Generate ONCE and broadcast to all receivers
            tone1_freq = self.rng_uniform(697, 941)
            tone2_freq = self.rng_uniform(1209, 1633)
            audio = 0.5 * self.xp.sin(2 * self.xp.pi * tone1_freq * self.time_axis) + \
                    0.5 * self.xp.sin(2 * self.xp.pi * tone2_freq * self.time_axis)
            for i in range(batch_size):
                audio_batch[i] = audio
        # else: 5% - Carrier (audio_batch remains zeros for all receivers)
        
        # FM modulation for entire batch
        # Discrete integration: cumulative sum along time axis
        dt = 1.0 / self.sample_rate_hz
        audio_integral = self.xp.cumsum(audio_batch, axis=1) * dt
        
        # Expand frequency offsets and deviations for broadcasting
        freq_offsets_expanded = frequency_offsets[:, self.xp.newaxis]
        freq_deviations_expanded = freq_deviations[:, self.xp.newaxis]
        
        # Carrier phase with frequency offset: (batch_size, num_samples)
        carrier_phase = 2 * self.xp.pi * freq_offsets_expanded * time_batch
        
        # FM modulation phase: (batch_size, num_samples)
        modulation_phase = 2 * self.xp.pi * freq_deviations_expanded * audio_integral
        
        # Total phase
        phase = carrier_phase + modulation_phase
        
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
