/**
 * AudioPlayer Component
 * 
 * Real-time IQ signal demodulator and audio player with SDR-style controls
 * 
 * Features:
 * - FM/AM/USB/LSB demodulation
 * - AGC (Automatic Gain Control)
 * - RF Gain control (pre-demodulation)
 * - AF Gain control (post-demodulation audio)
 * - Real-time audio playback via Web Audio API
 * - Waveform visualization (optional)
 * 
 * Architecture:
 * 1. IQ samples → Demodulator → Audio samples
 * 2. Audio samples → AGC → RF Gain → AF Gain → Web Audio API
 * 3. Visualization: Real-time waveform display
 * 
 * MIGRATION NOTE (2025-11-09):
 * - Audio chunks migrated from 1000ms to 200ms for real-time localization
 * - IQ samples: 10,000 samples @ 50kHz = 200ms duration
 * - Resampling (50kHz → 48kHz) preserves time duration correctly
 * - Example: 10,000 @ 50kHz → 9,600 @ 48kHz (both = 200ms)
 * - Audio playback timing is handled correctly by Web Audio API
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Card, Form, Button, ButtonGroup, Row, Col, Badge, Alert } from 'react-bootstrap';

type DemodulationMode = 'FM' | 'AM' | 'USB' | 'LSB';

interface AudioPlayerProps {
    iqData: {
        i_samples: Float32Array;
        q_samples: Float32Array;
    };
    sampleRate: number;
    centerFrequency: number;
}

/**
 * AGC (Automatic Gain Control) implementation
 * Maintains constant audio output level regardless of signal strength
 */
class AGC {
    private gain: number = 1.0;
    private targetLevel: number = 0.5;
    private attackTime: number = 0.001; // 1ms
    private decayTime: number = 0.1;    // 100ms
    private maxGain: number = 100.0;
    private minGain: number = 0.01;

    process(samples: Float32Array): Float32Array {
        const output = new Float32Array(samples.length);
        
        for (let i = 0; i < samples.length; i++) {
            // Apply current gain
            output[i] = samples[i] * this.gain;
            
            // Measure output level
            const level = Math.abs(output[i]);
            
            // Adjust gain based on level
            if (level > this.targetLevel) {
                // Attack: reduce gain quickly when signal is too loud
                this.gain *= (1.0 - this.attackTime);
            } else {
                // Decay: increase gain slowly when signal is too quiet
                this.gain *= (1.0 + this.decayTime);
            }
            
            // Clamp gain to reasonable range
            this.gain = Math.max(this.minGain, Math.min(this.maxGain, this.gain));
            
            // Hard limiter to prevent clipping
            output[i] = Math.max(-1.0, Math.min(1.0, output[i]));
        }
        
        return output;
    }

    reset() {
        this.gain = 1.0;
    }

    setAttackTime(seconds: number) {
        this.attackTime = seconds;
    }

    setDecayTime(seconds: number) {
        this.decayTime = seconds;
    }
}

/**
 * FM Demodulator (Frequency Modulation)
 * Used for: FM voice, FM broadcast, NBFM
 */
function demodulateFM(iSamples: Float32Array, qSamples: Float32Array, sampleRate: number): Float32Array {
    const output = new Float32Array(iSamples.length);
    
    // FM demodulation using phase difference
    let prevPhase = Math.atan2(qSamples[0], iSamples[0]);
    
    for (let i = 1; i < iSamples.length; i++) {
        const currentPhase = Math.atan2(qSamples[i], iSamples[i]);
        
        // Calculate phase difference
        let phaseDiff = currentPhase - prevPhase;
        
        // Unwrap phase (handle 2π discontinuities)
        if (phaseDiff > Math.PI) phaseDiff -= 2 * Math.PI;
        if (phaseDiff < -Math.PI) phaseDiff += 2 * Math.PI;
        
        // Phase difference is proportional to instantaneous frequency
        output[i] = phaseDiff / Math.PI; // Normalize to [-1, 1]
        
        prevPhase = currentPhase;
    }
    
    // De-emphasis filter (6 dB/octave, tau = 75μs for wideband FM)
    const tau = 75e-6;
    const alpha = 1.0 / (1.0 + sampleRate * tau);
    
    for (let i = 1; i < output.length; i++) {
        output[i] = output[i] * alpha + output[i - 1] * (1.0 - alpha);
    }
    
    return output;
}

/**
 * AM Demodulator (Amplitude Modulation)
 * Used for: AM broadcast, aircraft band
 */
function demodulateAM(iSamples: Float32Array, qSamples: Float32Array): Float32Array {
    const output = new Float32Array(iSamples.length);
    
    // AM demodulation: extract envelope (magnitude)
    for (let i = 0; i < iSamples.length; i++) {
        const magnitude = Math.sqrt(iSamples[i] * iSamples[i] + qSamples[i] * qSamples[i]);
        output[i] = magnitude;
    }
    
    // Remove DC bias (carrier)
    const dcLevel = output.reduce((sum, val) => sum + val, 0) / output.length;
    for (let i = 0; i < output.length; i++) {
        output[i] -= dcLevel;
    }
    
    return output;
}

/**
 * USB Demodulator (Upper Sideband)
 * Used for: SSB voice, amateur radio
 */
function demodulateUSB(iSamples: Float32Array, qSamples: Float32Array): Float32Array {
    const output = new Float32Array(iSamples.length);
    
    // USB: I + Q (simple phasing method)
    for (let i = 0; i < iSamples.length; i++) {
        output[i] = iSamples[i] + qSamples[i];
    }
    
    return output;
}

/**
 * LSB Demodulator (Lower Sideband)
 * Used for: SSB voice, amateur radio
 */
function demodulateLSB(iSamples: Float32Array, qSamples: Float32Array): Float32Array {
    const output = new Float32Array(iSamples.length);
    
    // LSB: I - Q (simple phasing method)
    for (let i = 0; i < iSamples.length; i++) {
        output[i] = iSamples[i] - qSamples[i];
    }
    
    return output;
}

/**
 * Audio resampler (simple linear interpolation)
 * Converts arbitrary sample rate to audio rate (48 kHz)
 */
function resampleAudio(input: Float32Array, inputRate: number, outputRate: number = 48000): Float32Array {
    const ratio = inputRate / outputRate;
    const outputLength = Math.floor(input.length / ratio);
    const output = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
        const srcIndex = i * ratio;
        const srcIndexFloor = Math.floor(srcIndex);
        const srcIndexCeil = Math.min(srcIndexFloor + 1, input.length - 1);
        const fraction = srcIndex - srcIndexFloor;
        
        // Linear interpolation
        output[i] = input[srcIndexFloor] * (1 - fraction) + input[srcIndexCeil] * fraction;
    }
    
    return output;
}

/**
 * Normalize audio to target level (prevent too quiet audio)
 */
function normalizeAudio(samples: Float32Array, targetLevel: number = 0.5): Float32Array {
    const output = new Float32Array(samples.length);
    
    // Find peak level
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
        const abs = Math.abs(samples[i]);
        if (abs > peak) peak = abs;
    }
    
    // Normalize to target level
    const gain = peak > 0 ? targetLevel / peak : 1.0;
    console.log('[normalizeAudio] Peak:', peak.toFixed(4), 'Gain:', gain.toFixed(4));
    
    for (let i = 0; i < samples.length; i++) {
        output[i] = samples[i] * gain;
    }
    
    return output;
}

/**
 * Band-pass filter for audio (300 Hz - 3000 Hz for voice)
 */
function bandpassFilter(samples: Float32Array, sampleRate: number, lowFreq: number = 300, highFreq: number = 3000): Float32Array {
    // Simple IIR filter (biquad)
    // For production, use a proper DSP library
    const output = new Float32Array(samples.length);
    
    // High-pass filter (remove low frequencies)
    const alphaHP = 1.0 / (1.0 + sampleRate / (2 * Math.PI * lowFreq));
    let prevInputHP = 0, prevOutputHP = 0;
    
    for (let i = 0; i < samples.length; i++) {
        output[i] = alphaHP * (prevOutputHP + samples[i] - prevInputHP);
        prevInputHP = samples[i];
        prevOutputHP = output[i];
    }
    
    // Low-pass filter (remove high frequencies)
    const alphaLP = 1.0 / (1.0 + sampleRate / (2 * Math.PI * highFreq));
    let prevOutputLP = 0;
    
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] * alphaLP + prevOutputLP * (1.0 - alphaLP);
        prevOutputLP = output[i];
    }
    
    return output;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ iqData, sampleRate, centerFrequency }) => {
    // Audio playback state
    const [isPlaying, setIsPlaying] = useState(false);
    const [mode, setMode] = useState<DemodulationMode>('FM');
    const [agcEnabled, setAgcEnabled] = useState(true);
    const [loopEnabled, setLoopEnabled] = useState(true);  // Loop enabled by default
    const [rfGain, setRfGain] = useState(1.0);    // 0.0 - 2.0
    const [afGain, setAfGain] = useState(1.0);    // 0.0 - 1.0 (default to max)
    const [error, setError] = useState<string | null>(null);
    
    // Audio context and buffers
    const audioContextRef = useRef<AudioContext | null>(null);
    const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
    const gainNodeRef = useRef<GainNode | null>(null);
    const agcRef = useRef<AGC>(new AGC());
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    // Initialize Web Audio API
    useEffect(() => {
        try {
            // @ts-ignore - AudioContext may not be in all browsers
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioContextRef.current = new AudioContextClass();
        } catch (err) {
            setError('Web Audio API not supported in this browser');
            console.error('AudioContext creation failed:', err);
        }
        
        return () => {
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
        };
    }, []);
    
    /**
     * Demodulate IQ samples based on selected mode
     */
    const demodulate = useCallback((mode: DemodulationMode): Float32Array => {
        let audioSamples: Float32Array;
        
        switch (mode) {
            case 'FM':
                audioSamples = demodulateFM(iqData.i_samples, iqData.q_samples, sampleRate);
                break;
            case 'AM':
                audioSamples = demodulateAM(iqData.i_samples, iqData.q_samples);
                break;
            case 'USB':
                audioSamples = demodulateUSB(iqData.i_samples, iqData.q_samples);
                break;
            case 'LSB':
                audioSamples = demodulateLSB(iqData.i_samples, iqData.q_samples);
                break;
            default:
                audioSamples = demodulateFM(iqData.i_samples, iqData.q_samples, sampleRate);
        }
        
        return audioSamples;
    }, [iqData, sampleRate]);
    
    /**
     * Start audio playback
     */
    const handlePlay = useCallback(async () => {
        if (!audioContextRef.current) {
            setError('Audio context not initialized');
            return;
        }
        
        try {
            setError(null);
            
            // Resume audio context if suspended (browser autoplay policy)
            if (audioContextRef.current.state === 'suspended') {
                console.log('[AudioPlayer] Resuming suspended AudioContext');
                await audioContextRef.current.resume();
            }
            
            // Step 1: Demodulate IQ samples
            console.log('[AudioPlayer] Demodulating with mode:', mode);
            let audioSamples = demodulate(mode);
            
            // Step 2: Apply RF gain (pre-demodulation gain)
            for (let i = 0; i < audioSamples.length; i++) {
                audioSamples[i] *= rfGain;
            }
            
            // Step 3: Apply AGC if enabled
            if (agcEnabled) {
                console.log('[AudioPlayer] Applying AGC');
                agcRef.current.reset();
                audioSamples = agcRef.current.process(audioSamples);
            }
            
            // Step 4: Band-pass filter for voice
            const audioRate = 48000;
            audioSamples = bandpassFilter(audioSamples, sampleRate);
            
            // Step 5: Resample to audio rate (48 kHz)
            // NOTE: This preserves time duration (e.g., 10,000 samples @ 50kHz = 200ms → 9,600 samples @ 48kHz = 200ms)
            console.log('[AudioPlayer] Resampling from', sampleRate, 'Hz to', audioRate, 'Hz');
            audioSamples = resampleAudio(audioSamples, sampleRate, audioRate);
            
            // Step 5.5: Normalize audio to prevent too quiet output
            console.log('[AudioPlayer] Normalizing audio');
            audioSamples = normalizeAudio(audioSamples, 0.7);  // Target 70% of max
            
            // Step 6: Create audio buffer
            const audioBuffer = audioContextRef.current.createBuffer(
                1, // mono
                audioSamples.length,
                audioRate
            );
            audioBuffer.getChannelData(0).set(audioSamples);
            
            // Debug: Check audio levels
            const maxLevel = Math.max(...Array.from(audioSamples).map(Math.abs));
            const avgLevel = Array.from(audioSamples).reduce((sum, s) => sum + Math.abs(s), 0) / audioSamples.length;
            console.log('[AudioPlayer] Audio levels:', {
                maxLevel: maxLevel.toFixed(4),
                avgLevel: avgLevel.toFixed(4),
                sampleCount: audioSamples.length,
                duration: (audioSamples.length / audioRate).toFixed(2) + 's'
            });
            
            // Step 7: Create audio graph
            const sourceNode = audioContextRef.current.createBufferSource();
            const gainNode = audioContextRef.current.createGain();
            
            sourceNode.buffer = audioBuffer;
            sourceNode.loop = loopEnabled;  // Enable/disable loop
            gainNode.gain.value = afGain; // AF Gain
            
            sourceNode.connect(gainNode);
            gainNode.connect(audioContextRef.current.destination);
            
            // Store references
            sourceNodeRef.current = sourceNode;
            gainNodeRef.current = gainNode;
            
            // Handle playback end (only if not looping)
            sourceNode.onended = () => {
                if (!loopEnabled) {
                    setIsPlaying(false);
                    console.log('[AudioPlayer] Playback ended');
                }
            };
            
            // Step 8: Start playback
            sourceNode.start();
            setIsPlaying(true);
            
            console.log('[AudioPlayer] Playback started', {
                mode,
                sampleRate,
                audioRate,
                duration: audioSamples.length / audioRate,
                agcEnabled,
                loopEnabled,
                rfGain,
                afGain,
                audioContextState: audioContextRef.current.state
            });
            
            // Draw waveform
            drawWaveform(audioSamples);
            
        } catch (err) {
            console.error('[AudioPlayer] Playback error:', err);
            setError(err instanceof Error ? err.message : 'Failed to play audio');
            setIsPlaying(false);
        }
    }, [mode, agcEnabled, loopEnabled, rfGain, afGain, demodulate, sampleRate]);
    
    /**
     * Stop audio playback
     */
    const handleStop = useCallback(() => {
        if (sourceNodeRef.current) {
            try {
                sourceNodeRef.current.stop();
                sourceNodeRef.current.disconnect();
            } catch (err) {
                console.warn('[AudioPlayer] Stop error:', err);
            }
            sourceNodeRef.current = null;
        }
        setIsPlaying(false);
    }, []);
    
    /**
     * Update AF gain in real-time
     */
    useEffect(() => {
        if (gainNodeRef.current && isPlaying) {
            gainNodeRef.current.gain.value = afGain;
        }
    }, [afGain, isPlaying]);
    
    /**
     * Draw waveform visualization
     */
    const drawWaveform = useCallback((samples: Float32Array) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw waveform
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        const step = Math.ceil(samples.length / width);
        const halfHeight = height / 2;
        
        for (let i = 0; i < width; i++) {
            const sampleIndex = i * step;
            const sample = samples[sampleIndex] || 0;
            const y = halfHeight - sample * halfHeight;
            
            if (i === 0) {
                ctx.moveTo(i, y);
            } else {
                ctx.lineTo(i, y);
            }
        }
        
        ctx.stroke();
        
        // Draw center line
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, halfHeight);
        ctx.lineTo(width, halfHeight);
        ctx.stroke();
    }, []);
    
    return (
        <Card className="audio-player">
            <Card.Body>
                <Row className="mb-3">
                    <Col>
                        <h5 className="d-flex align-items-center gap-2">
                            🔊 Audio Player
                            <Badge bg={isPlaying ? 'success' : 'secondary'}>
                                {isPlaying ? 'Playing' : 'Stopped'}
                            </Badge>
                            <Badge bg="info">{mode}</Badge>
                        </h5>
                    </Col>
                </Row>
                
                {error && (
                    <Alert variant="danger" dismissible onClose={() => setError(null)}>
                        {error}
                    </Alert>
                )}
                
                {/* Playback Controls */}
                <Row className="mb-3">
                    <Col md={6}>
                        <ButtonGroup className="w-100">
                            <Button
                                variant={isPlaying ? 'outline-primary' : 'primary'}
                                onClick={handlePlay}
                                disabled={isPlaying}
                            >
                                ▶ Play
                            </Button>
                            <Button
                                variant="danger"
                                onClick={handleStop}
                                disabled={!isPlaying}
                            >
                                ⏹ Stop
                            </Button>
                        </ButtonGroup>
                    </Col>
                    
                    {/* Demodulation Mode */}
                    <Col md={6}>
                        <ButtonGroup className="w-100">
                            {(['FM', 'AM', 'USB', 'LSB'] as DemodulationMode[]).map((m) => (
                                <Button
                                    key={m}
                                    variant={mode === m ? 'primary' : 'outline-primary'}
                                    onClick={() => setMode(m)}
                                    disabled={isPlaying}
                                    size="sm"
                                >
                                    {m}
                                </Button>
                            ))}
                        </ButtonGroup>
                    </Col>
                </Row>
                
                {/* AGC and Loop Toggles */}
                <Row className="mb-3">
                    <Col md={6}>
                        <Form.Check
                            type="switch"
                            id="agc-switch"
                            label="AGC (Automatic Gain Control)"
                            checked={agcEnabled}
                            onChange={(e) => setAgcEnabled(e.target.checked)}
                            disabled={isPlaying}
                        />
                    </Col>
                    <Col md={6}>
                        <Form.Check
                            type="switch"
                            id="loop-switch"
                            label="🔁 Repeat/Loop"
                            checked={loopEnabled}
                            onChange={(e) => setLoopEnabled(e.target.checked)}
                            disabled={isPlaying}
                        />
                    </Col>
                </Row>
                
                {/* RF Gain */}
                <Row className="mb-3">
                    <Col>
                        <Form.Group>
                            <Form.Label>
                                RF Gain: <strong>{(rfGain * 100).toFixed(0)}%</strong>
                            </Form.Label>
                            <Form.Range
                                min={0}
                                max={2}
                                step={0.1}
                                value={rfGain}
                                onChange={(e) => setRfGain(parseFloat(e.target.value))}
                                disabled={isPlaying}
                            />
                            <Form.Text className="text-muted">
                                Pre-demodulation signal gain
                            </Form.Text>
                        </Form.Group>
                    </Col>
                </Row>
                
                {/* AF Gain */}
                <Row className="mb-3">
                    <Col>
                        <Form.Group>
                            <Form.Label>
                                AF Gain (Volume): <strong>{(afGain * 100).toFixed(0)}%</strong>
                            </Form.Label>
                            <Form.Range
                                min={0}
                                max={1}
                                step={0.05}
                                value={afGain}
                                onChange={(e) => setAfGain(parseFloat(e.target.value))}
                            />
                            <Form.Text className="text-muted">
                                Post-demodulation audio gain
                            </Form.Text>
                        </Form.Group>
                    </Col>
                </Row>
                
                {/* Waveform Visualization */}
                <Row>
                    <Col>
                        <div className="border rounded bg-dark p-2">
                            <canvas
                                ref={canvasRef}
                                width={600}
                                height={100}
                                style={{ width: '100%', height: '100px' }}
                            />
                        </div>
                        <Form.Text className="text-muted">
                            Audio waveform (after demodulation)
                        </Form.Text>
                    </Col>
                </Row>
                
                {/* Info */}
                <Row className="mt-3">
                    <Col>
                        <div className="text-muted small">
                            <strong>Sample Rate:</strong> {(sampleRate / 1e3).toFixed(1)} kHz | 
                            <strong> Center Freq:</strong> {(centerFrequency / 1e6).toFixed(3)} MHz | 
                            <strong> Samples:</strong> {iqData.i_samples.length.toLocaleString()}
                        </div>
                    </Col>
                </Row>
            </Card.Body>
        </Card>
    );
};
