/**
 * Web Worker for Waterfall FFT Computation
 * 
 * Offloads expensive STFT computation to a background thread
 * to prevent UI blocking on large IQ datasets.
 * 
 * Messages:
 * - Input: { type: 'compute', iSamples, qSamples, fftSize, hopSize }
 * - Output: { type: 'result', stftData, stats } | { type: 'progress', percent } | { type: 'error', message }
 */

// Import FFT library (note: may need special handling for workers)
// @ts-ignore - FFT import in worker context
import FFT from 'fft.js';

interface ComputeMessage {
    type: 'compute';
    iSamples: Float32Array;
    qSamples: Float32Array;
    fftSize: number;
    hopSize: number;
}

interface ResultMessage {
    type: 'result';
    stftData: Float32Array[];
    stats: {
        minDb: number;
        maxDb: number;
        meanDb: number;
        medianDb: number;
    };
}

interface ProgressMessage {
    type: 'progress';
    percent: number;
    currentFrame: number;
    totalFrames: number;
}

interface ErrorMessage {
    type: 'error';
    message: string;
}

type WorkerMessage = ResultMessage | ProgressMessage | ErrorMessage;

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
 * Compute FFT using fft.js library
 * Returns magnitude in dB
 */
function computeFFT(iSamples: Float32Array, qSamples: Float32Array, fftSize: number): Float32Array {
    // Apply windowing
    const windowedI = applyHammingWindow(iSamples.slice(0, fftSize));
    const windowedQ = applyHammingWindow(qSamples.slice(0, fftSize));

    // Create FFT instance
    const fft = new FFT(fftSize);
    
    // Prepare input buffer
    const input = new Array(fftSize * 2);
    for (let i = 0; i < fftSize; i++) {
        input[i * 2] = windowedI[i];
        input[i * 2 + 1] = windowedQ[i];
    }
    
    // Compute FFT
    const output = fft.createComplexArray();
    fft.transform(output, input);
    
    // Compute magnitude spectrum
    const magnitude = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
        const real = output[i * 2];
        const imag = output[i * 2 + 1];
        magnitude[i] = Math.sqrt(real * real + imag * imag) / fftSize;
    }

    // Convert to dB
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
 * Compute STFT with progress updates
 */
function computeSTFT(
    iSamples: Float32Array,
    qSamples: Float32Array,
    fftSize: number,
    hopSize: number
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

        // Send progress update every 10 frames
        if (frame % 10 === 0 || frame === numFrames - 1) {
            const progressMsg: ProgressMessage = {
                type: 'progress',
                percent: Math.round((frame / numFrames) * 100),
                currentFrame: frame,
                totalFrames: numFrames
            };
            self.postMessage(progressMsg);
        }
    }

    return stft;
}

/**
 * Compute statistics from STFT data
 */
function computeStats(stftData: Float32Array[]): {
    minDb: number;
    maxDb: number;
    meanDb: number;
    medianDb: number;
} {
    // Collect all finite dB values
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

    // Sort for median calculation
    allValues.sort((a, b) => a - b);

    const minDb = allValues[0];
    const maxDb = allValues[allValues.length - 1];
    const meanDb = allValues.reduce((sum, v) => sum + v, 0) / allValues.length;
    const medianDb = allValues[Math.floor(allValues.length / 2)];

    return { minDb, maxDb, meanDb, medianDb };
}

// Worker message handler
self.onmessage = (event: MessageEvent<ComputeMessage>) => {
    const { type, iSamples, qSamples, fftSize, hopSize } = event.data;

    if (type !== 'compute') {
        const errorMsg: ErrorMessage = {
            type: 'error',
            message: `Unknown message type: ${type}`
        };
        self.postMessage(errorMsg);
        return;
    }

    try {
        // Compute STFT (with progress updates)
        const stftData = computeSTFT(iSamples, qSamples, fftSize, hopSize);

        // Compute statistics
        const stats = computeStats(stftData);

        // Send result
        const resultMsg: ResultMessage = {
            type: 'result',
            stftData,
            stats
        };
        self.postMessage(resultMsg);
    } catch (error) {
        const errorMsg: ErrorMessage = {
            type: 'error',
            message: error instanceof Error ? error.message : 'Unknown error during STFT computation'
        };
        self.postMessage(errorMsg);
    }
};
