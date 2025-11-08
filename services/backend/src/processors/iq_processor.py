"""IQ data processing and signal metrics computation."""

import logging

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from ..models.websdrs import SignalMetrics

logger = logging.getLogger(__name__)


class IQProcessor:
    """Process IQ data and compute signal metrics."""

    @staticmethod
    def compute_metrics(
        iq_data: np.ndarray,
        sample_rate_hz: int,
        target_frequency_hz: int,
        noise_bandwidth_hz: int = 10000,
    ) -> SignalMetrics:
        """
        Compute signal metrics from IQ data.

        Args:
            iq_data: Complex64 IQ data array
            sample_rate_hz: Sample rate in Hz
            target_frequency_hz: Target frequency in Hz
            noise_bandwidth_hz: Noise measurement bandwidth in Hz

        Returns:
            SignalMetrics object with computed metrics
        """
        if len(iq_data) == 0:
            raise ValueError("Empty IQ data")

        # Normalize IQ data
        iq_normalized = iq_data / (np.max(np.abs(iq_data)) + 1e-10)

        # Compute power spectrum
        psd_db, freqs = IQProcessor._compute_psd(iq_normalized, sample_rate_hz)

        # Find frequency offset
        frequency_offset_hz = IQProcessor._estimate_frequency_offset(
            iq_normalized, sample_rate_hz, target_frequency_hz
        )

        # Compute SNR
        signal_power_db, noise_power_db = IQProcessor._compute_snr(
            psd_db, freqs, target_frequency_hz, noise_bandwidth_hz
        )

        snr_db = signal_power_db - noise_power_db

        # Average PSD at center frequency
        center_idx = np.argmin(np.abs(freqs - 0))  # Baseband center
        psd_dbm = psd_db[center_idx]

        logger.debug(
            "Computed metrics - SNR: %.2f dB, PSD: %.2f dBm, Freq Offset: %.2f Hz",
            snr_db,
            psd_dbm,
            frequency_offset_hz,
        )

        return SignalMetrics(
            snr_db=float(snr_db),
            psd_dbm=float(psd_dbm),
            frequency_offset_hz=float(frequency_offset_hz),
            signal_power_dbm=float(signal_power_db),
            noise_power_dbm=float(noise_power_db),
        )

    @staticmethod
    def _compute_psd(
        iq_data: np.ndarray,
        sample_rate_hz: int,
        nperseg: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.

        Returns:
            Tuple of (PSD in dB, frequencies)
        """
        # Use Welch's method for stable PSD estimate
        freqs, psd = signal.welch(
            iq_data,
            fs=sample_rate_hz,
            nperseg=min(nperseg, len(iq_data)),
            scaling="density",
            window="hann",
        )

        # Convert to dB (reference 1 Watt)
        psd_db = 10 * np.log10(psd + 1e-12)

        return psd_db, freqs

    @staticmethod
    def _estimate_frequency_offset(
        iq_data: np.ndarray,
        sample_rate_hz: int,
        target_frequency_hz: int,
    ) -> float:
        """
        Estimate frequency offset using Phase Locked Loop (PLL) technique.

        Returns:
            Frequency offset in Hz
        """
        # Simple frequency offset estimation using FFT
        # Find peak in power spectrum
        n = len(iq_data)
        fft_result = fft(iq_data, n=2 ** int(np.ceil(np.log2(n))))
        freqs = fftfreq(len(fft_result), 1 / sample_rate_hz)

        # Only consider frequencies near baseband (within ±sample_rate/4)
        valid_range = sample_rate_hz / 4
        mask = np.abs(freqs) <= valid_range

        power_spectrum = np.abs(fft_result[mask]) ** 2
        freqs_masked = freqs[mask]

        if np.max(power_spectrum) == 0:
            return 0.0

        # Find peak frequency
        peak_idx = np.argmax(power_spectrum)
        estimated_offset = freqs_masked[peak_idx]

        return float(estimated_offset)

    @staticmethod
    def _compute_snr(
        psd_db: np.ndarray,
        freqs: np.ndarray,
        target_frequency_hz: int,
        noise_bandwidth_hz: int = 10000,
    ) -> tuple[float, float]:
        """
        Compute Signal and Noise power from PSD.

        Returns:
            Tuple of (signal_power_db, noise_power_db)
        """
        # Signal region: center ± noise_bandwidth / 2
        signal_mask = np.abs(freqs - 0) <= (noise_bandwidth_hz / 2)
        signal_power_db = float(np.mean(psd_db[signal_mask]))

        # Noise region: use edges of spectrum
        edge_width = int(0.1 * len(psd_db))
        noise_region = np.concatenate([psd_db[:edge_width], psd_db[-edge_width:]])
        noise_power_db = float(np.mean(noise_region))

        return signal_power_db, noise_power_db

    @staticmethod
    def save_iq_data_hdf5(
        iq_data: np.ndarray,
        filename: str,
        metadata: dict = None,
    ):
        """
        Save IQ data to HDF5 file with metadata.

        Args:
            iq_data: Complex64 IQ data array
            filename: Output HDF5 filename
            metadata: Optional metadata dictionary
        """
        try:
            import h5py
        except ImportError:
            logger.warning("h5py not installed, skipping HDF5 save")
            return

        with h5py.File(filename, "w") as f:
            # Save IQ data as separate I and Q arrays
            f.create_dataset("I", data=np.real(iq_data), dtype=np.float32)
            f.create_dataset("Q", data=np.imag(iq_data), dtype=np.float32)

            # Save metadata
            if metadata:
                meta_group = f.create_group("metadata")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value

        logger.info("Saved IQ data to %s", filename)

    @staticmethod
    def save_iq_data_npy(
        iq_data: np.ndarray,
        filename: str,
        metadata: dict = None,
    ):
        """
        Save IQ data to NPY file with optional metadata JSON.

        Args:
            iq_data: Complex64 IQ data array
            filename: Output NPY filename (without extension)
            metadata: Optional metadata dictionary
        """
        import json

        # Save IQ data
        np.save(f"{filename}.npy", iq_data.astype(np.complex64))

        # Save metadata as JSON
        if metadata:
            with open(f"{filename}_meta.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info("Saved IQ data to %s.npy", filename)
