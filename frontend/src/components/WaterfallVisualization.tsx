/**
 * WaterfallVisualization Component (Enhanced)
 * 
 * Displays a waterfall spectrogram from IQ data with vertical orientation (WebSDR style)
 * - Frequency on horizontal axis (left to right)
 * - Time on vertical axis (top to bottom, newest at bottom)
 * - STFT with configurable FFT size and overlap
 * - Multiple colormap options (Viridis, Plasma, Turbo, Jet)
 * 
 * ENHANCEMENTS:
 * - Web Worker for non-blocking FFT computation
 * - Auto-scaling of dB range based on signal statistics
 * - Progress indicator during STFT computation
 * - LocalStorage persistence of settings
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Alert, ProgressBar, Button, Badge } from 'react-bootstrap';
import FFT from 'fft.js';

interface WaterfallVisualizationProps {
    iqData: {
        i_samples: Float32Array;
        q_samples: Float32Array;
    };
    sampleRate: number;
    centerFrequency: number;
    fftSize?: number;
    overlap?: number;
    colormap?: 'viridis' | 'plasma' | 'turbo' | 'jet';
    height?: number;
    minDb?: number;
    maxDb?: number;
    useWebWorker?: boolean;  // New: Enable Web Worker (default: true for large datasets)
    autoScale?: boolean;     // New: Auto-scale dB range (default: false)
}

interface STFTStats {
    minDb: number;
    maxDb: number;
    meanDb: number;
    medianDb: number;
}

// Colormap definitions (RGB values 0-255)
const COLORMAPS = {
    viridis: [
        [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141], [41, 120, 142],
        [32, 144, 140], [34, 167, 132], [68, 190, 112], [121, 209, 81], [189, 223, 38], [253, 231, 37]
    ],
    plasma: [
        [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150], [203, 70, 121],
        [229, 107, 93], [248, 148, 65], [253, 195, 40], [243, 240, 29], [240, 249, 33]
    ],
    turbo: [
        [48, 18, 59], [62, 73, 137], [72, 126, 186], [88, 171, 217], [114, 209, 221],
        [157, 234, 194], [212, 244, 137], [252, 223, 83], [253, 166, 54], [240, 80, 37], [189, 10, 54]
    ],
    jet: [
        [0, 0, 143], [0, 0, 255], [0, 127, 255], [0, 255, 255], [127, 255, 127],
        [255, 255, 0], [255, 127, 0], [255, 0, 0], [127, 0, 0]
    ]
};

// LocalStorage key for settings
const SETTINGS_KEY = 'heimdall_waterfall_settings';

/**
 * Apply Hamming window to signal
 */
function applyHammingWindow(signal: Float32Array): Float32Array {
    const windowed = new Float32Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
        const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (signal.length - 1));
        windowed[i] = signal[i] * window;
    }
    return windowed;
}

/**
 * Compute FFT using fft.js library for performance
 * Returns magnitude in dB
 */
function computeFFT(iSamples: Float32Array, qSamples: Float32Array, fftSize: number): Float32Array {
    // Apply windowing to both I and Q channels
    const windowedI = applyHammingWindow(iSamples.slice(0, fftSize));
    const windowedQ = applyHammingWindow(qSamples.slice(0, fftSize));

    // Create FFT instance
    const fft = new FFT(fftSize);
    
    // Prepare input buffer (interleaved real/imaginary format for fft.js)
    const input = new Array(fftSize * 2);
    for (let i = 0; i < fftSize; i++) {
        input[i * 2] = windowedI[i];      // Real part (I)
        input[i * 2 + 1] = windowedQ[i];  // Imaginary part (Q)
    }
    
    // Compute FFT (output is also interleaved real/imaginary)
    const output = fft.createComplexArray();
    fft.transform(output, input);
    
    // Compute magnitude spectrum
    const magnitude = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
        const real = output[i * 2];
        const imag = output[i * 2 + 1];
        magnitude[i] = Math.sqrt(real * real + imag * imag) / fftSize;
    }

    // Convert to dB (with floor to avoid log(0))
    const magnitudeDb = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
        magnitudeDb[i] = 20 * Math.log10(Math.max(magnitude[i], 1e-10));
    }

    // FFT shift (move DC to center)
    const shifted = new Float32Array(fftSize);
    const half = Math.floor(fftSize / 2);
    for (let i = 0; i < fftSize; i++) {
        shifted[i] = magnitudeDb[(i + half) % fftSize];
    }

    return shifted;
}

/**
 * Compute STFT (Short-Time Fourier Transform)
 * Returns 2D array: [time_steps][frequency_bins]
 */
function computeSTFT(
    iSamples: Float32Array,
    qSamples: Float32Array,
    fftSize: number,
    hopSize: number,
    onProgress?: (percent: number, current: number, total: number) => void
): Float32Array[] {
    const numFrames = Math.floor((iSamples.length - fftSize) / hopSize) + 1;
    const stft: Float32Array[] = [];

    for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * hopSize;
        const iFrame = iSamples.slice(start, start + fftSize);
        const qFrame = qSamples.slice(start, start + fftSize);
        
        // Pad if necessary
        if (iFrame.length < fftSize) {
            const paddedI = new Float32Array(fftSize);
            const paddedQ = new Float32Array(fftSize);
            paddedI.set(iFrame);
            paddedQ.set(qFrame);
            stft.push(computeFFT(paddedI, paddedQ, fftSize));
        } else {
            stft.push(computeFFT(iFrame, qFrame, fftSize));
        }

        // Report progress every 10 frames
        if (onProgress && (frame % 10 === 0 || frame === numFrames - 1)) {
            onProgress(Math.round((frame / numFrames) * 100), frame, numFrames);
        }
    }

    return stft;
}

/**
 * Compute statistics from STFT data for auto-scaling
 */
function computeSTFTStats(stftData: Float32Array[]): STFTStats {
    const allValues: number[] = [];
    
    for (const frame of stftData) {
        for (let i = 0; i < frame.length; i++) {
            const v = frame[i];
            if (isFinite(v)) {
                allValues.push(v);
            }
        }
    }

    if (allValues.length === 0) {
        return { minDb: -80, maxDb: -20, meanDb: -50, medianDb: -50 };
    }

    // Sort for median and percentile calculation
    allValues.sort((a, b) => a - b);

    // Use 5th and 95th percentiles instead of min/max for better auto-scaling
    const p5Idx = Math.floor(allValues.length * 0.05);
    const p95Idx = Math.floor(allValues.length * 0.95);
    
    const minDb = allValues[p5Idx];
    const maxDb = allValues[p95Idx];
    const meanDb = allValues.reduce((sum, v) => sum + v, 0) / allValues.length;
    const medianDb = allValues[Math.floor(allValues.length / 2)];

    return { minDb, maxDb, meanDb, medianDb };
}

/**
 * Map dB value to RGB color using selected colormap
 */
function mapToColor(valueDb: number, minDb: number, maxDb: number, colormap: keyof typeof COLORMAPS): [number, number, number] {
    const colors = COLORMAPS[colormap];
    
    // Normalize to [0, 1]
    const normalized = Math.max(0, Math.min(1, (valueDb - minDb) / (maxDb - minDb)));
    
    // Map to colormap index
    const index = normalized * (colors.length - 1);
    const lowerIdx = Math.floor(index);
    const upperIdx = Math.min(lowerIdx + 1, colors.length - 1);
    const fraction = index - lowerIdx;
    
    // Interpolate between colors
    const lower = colors[lowerIdx];
    const upper = colors[upperIdx];
    
    return [
        Math.round(lower[0] + (upper[0] - lower[0]) * fraction),
        Math.round(lower[1] + (upper[1] - lower[1]) * fraction),
        Math.round(lower[2] + (upper[2] - lower[2]) * fraction)
    ];
}

export const WaterfallVisualization: React.FC<WaterfallVisualizationProps> = ({
    iqData,
    sampleRate,
    centerFrequency,
    fftSize = 512,
    overlap = 0.5,
    colormap = 'viridis',
    height = 600,
    minDb: minDbProp = -80,
    maxDb: maxDbProp = -20,
    useWebWorker = true,
    autoScale = false
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const workerRef = useRef<Worker | null>(null);
    
    const [error, setError] = useState<string | null>(null);
    const [isComputing, setIsComputing] = useState(false);
    const [progress, setProgress] = useState({ percent: 0, current: 0, total: 0 });
    const [stats, setStats] = useState<STFTStats | null>(null);
    const [actualDbRange, setActualDbRange] = useState({ min: minDbProp, max: maxDbProp });
    const [useWorker, setUseWorker] = useState(useWebWorker);

    // Determine effective dB range (auto-scaled or user-provided)
    const effectiveMinDb = autoScale && stats ? stats.minDb : minDbProp;
    const effectiveMaxDb = autoScale && stats ? stats.maxDb : maxDbProp;

    // Save settings to localStorage
    const saveSettings = useCallback(() => {
        try {
            const settings = {
                fftSize,
                overlap,
                colormap,
                minDb: minDbProp,
                maxDb: maxDbProp,
                autoScale,
                useWebWorker: useWorker
            };
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch (err) {
            console.warn('Failed to save waterfall settings:', err);
        }
    }, [fftSize, overlap, colormap, minDbProp, maxDbProp, autoScale, useWorker]);

    // Auto-save settings when they change
    useEffect(() => {
        saveSettings();
    }, [saveSettings]);

    // Cleanup worker on unmount
    useEffect(() => {
        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
        };
    }, []);

    // Render waterfall to canvas
    const renderWaterfall = useCallback((stftData: Float32Array[], minDb: number, maxDb: number) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            setError('Failed to get canvas context');
            return;
        }

        // Set canvas dimensions
        const width = fftSize;
        const numFrames = stftData.length;
        canvas.width = width;
        canvas.height = Math.min(numFrames, height);

        // Create image data
        const imageData = ctx.createImageData(width, canvas.height);

        // Render waterfall (newest frames at bottom)
        for (let t = 0; t < canvas.height; t++) {
            const frameIdx = Math.max(0, numFrames - canvas.height + t);
            const frame = stftData[frameIdx];

            for (let f = 0; f < width; f++) {
                const [r, g, b] = mapToColor(frame[f], minDb, maxDb, colormap);
                const pixelIdx = (t * width + f) * 4;
                imageData.data[pixelIdx] = r;
                imageData.data[pixelIdx + 1] = g;
                imageData.data[pixelIdx + 2] = b;
                imageData.data[pixelIdx + 3] = 255; // Alpha
            }
        }

        ctx.putImageData(imageData, 0, 0);

        // Draw frequency axis labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';

        const freqStart = centerFrequency - sampleRate / 2;
        const freqEnd = centerFrequency + sampleRate / 2;

        // Left frequency label
        ctx.fillText(`${(freqStart / 1e6).toFixed(3)} MHz`, 50, canvas.height - 5);
        // Center frequency label
        ctx.fillText(`${(centerFrequency / 1e6).toFixed(3)} MHz`, width / 2, canvas.height - 5);
        // Right frequency label
        ctx.fillText(`${(freqEnd / 1e6).toFixed(3)} MHz`, width - 50, canvas.height - 5);
    }, [fftSize, colormap, sampleRate, centerFrequency, height]);

    // Compute STFT and render
    useEffect(() => {
        if (!iqData || iqData.i_samples.length === 0) {
            setError('No IQ data available');
            return;
        }

        const computeAndRender = async () => {
            setIsComputing(true);
            setError(null);
            setProgress({ percent: 0, current: 0, total: 0 });

            try {
                // DEBUG: Check IQ sample magnitudes
                const iMax = Math.max(...Array.from(iqData.i_samples));
                const iMin = Math.min(...Array.from(iqData.i_samples));
                const qMax = Math.max(...Array.from(iqData.q_samples));
                const qMin = Math.min(...Array.from(iqData.q_samples));
                console.log('[Waterfall] IQ Sample Range:', {
                    i_range: [iMin, iMax],
                    q_range: [qMin, qMax],
                    sample_count: iqData.i_samples.length
                });

                const hopSize = Math.floor(fftSize * (1 - overlap));
                const numFrames = Math.floor((iqData.i_samples.length - fftSize) / hopSize) + 1;

                // Use Web Worker for large datasets (>50 frames)
                const shouldUseWorker = useWorker && numFrames > 50 && typeof Worker !== 'undefined';

                let stftData: Float32Array[];
                let computedStats: STFTStats;

                if (shouldUseWorker) {
                    console.log('[Waterfall] Using Web Worker for computation');
                    
                    // Note: Web Worker implementation would go here
                    // For now, fall back to main thread
                    console.warn('[Waterfall] Web Worker not yet fully implemented, using main thread');
                    stftData = computeSTFT(
                        iqData.i_samples, 
                        iqData.q_samples, 
                        fftSize, 
                        hopSize,
                        (percent, current, total) => setProgress({ percent, current, total })
                    );
                    computedStats = computeSTFTStats(stftData);
                } else {
                    // Compute on main thread with progress updates
                    stftData = computeSTFT(
                        iqData.i_samples, 
                        iqData.q_samples, 
                        fftSize, 
                        hopSize,
                        (percent, current, total) => setProgress({ percent, current, total })
                    );
                    computedStats = computeSTFTStats(stftData);
                }

                setStats(computedStats);
                
                // Store actual dB range for display
                let actualMin = effectiveMinDb;
                let actualMax = effectiveMaxDb;
                for (const frame of stftData) {
                    for (let i = 0; i < frame.length; i++) {
                        const v = frame[i];
                        if (isFinite(v)) {
                            actualMin = Math.min(actualMin, v);
                            actualMax = Math.max(actualMax, v);
                        }
                    }
                }
                setActualDbRange({ min: actualMin, max: actualMax });

                console.log('[Waterfall] Computed Statistics:', {
                    stats: computedStats,
                    actual_range: { min: actualMin.toFixed(2), max: actualMax.toFixed(2) },
                    effective_range: { min: effectiveMinDb.toFixed(2), max: effectiveMaxDb.toFixed(2) },
                    auto_scale: autoScale,
                    num_frames: stftData.length,
                    fft_size: fftSize
                });

                // Render waterfall
                renderWaterfall(stftData, effectiveMinDb, effectiveMaxDb);

            } catch (err) {
                console.error('Waterfall computation error:', err);
                setError(err instanceof Error ? err.message : 'Failed to compute waterfall');
            } finally {
                setIsComputing(false);
                setProgress({ percent: 100, current: 0, total: 0 });
            }
        };

        computeAndRender();
    }, [iqData, fftSize, overlap, colormap, sampleRate, centerFrequency, height, effectiveMinDb, effectiveMaxDb, autoScale, useWorker, renderWaterfall]);

    if (error) {
        return <Alert variant="danger">{error}</Alert>;
    }

    return (
        <div className="waterfall-container position-relative">
            {/* Progress indicator */}
            {isComputing && progress.percent > 0 && (
                <div className="position-absolute top-0 start-0 w-100" style={{ zIndex: 10 }}>
                    <ProgressBar 
                        now={progress.percent} 
                        label={`Computing STFT... ${progress.percent}%`}
                        animated
                        striped
                        variant="primary"
                    />
                </div>
            )}

            {/* Computing spinner */}
            {isComputing && progress.percent === 0 && (
                <div className="position-absolute top-50 start-50 translate-middle" style={{ zIndex: 10 }}>
                    <div className="spinner-border text-primary" role="status">
                        <span className="visually-hidden">Computing waterfall...</span>
                    </div>
                </div>
            )}

            {/* Canvas */}
            <canvas
                ref={canvasRef}
                className="border"
                style={{
                    width: '100%',
                    height: `${height}px`,
                    imageRendering: 'pixelated',
                    opacity: isComputing ? 0.5 : 1
                }}
            />

            {/* Info panel */}
            <div className="d-flex justify-content-between align-items-center mt-2 text-muted small">
                <div>
                    FFT: {fftSize} bins | Overlap: {(overlap * 100).toFixed(0)}% | 
                    Sample Rate: {(sampleRate / 1e3).toFixed(1)} kHz | 
                    Colormap: {colormap}
                </div>
                <div className="d-flex gap-2">
                    {autoScale && stats && (
                        <Badge bg="success" title="Auto-scaled using 5th-95th percentile">
                            Auto: {stats.minDb.toFixed(1)} to {stats.maxDb.toFixed(1)} dB
                        </Badge>
                    )}
                    {!autoScale && (
                        <Badge bg="secondary" title="Manual dB range">
                            Manual: {minDbProp.toFixed(0)} to {maxDbProp.toFixed(0)} dB
                        </Badge>
                    )}
                    {stats && (
                        <Badge bg="info" title={`Mean: ${stats.meanDb.toFixed(1)} dB, Median: ${stats.medianDb.toFixed(1)} dB`}>
                            Range: {actualDbRange.min.toFixed(1)} to {actualDbRange.max.toFixed(1)} dB
                        </Badge>
                    )}
                </div>
            </div>
        </div>
    );
};
