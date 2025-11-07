/**
 * WaterfallViewTab Component
 * 
 * Tab view for displaying waterfall spectrograms from IQ data
 * - Receiver selector (dropdown or buttons)
 * - Waterfall visualization with configurable parameters
 * - FFT size, overlap, and colormap controls
 */

import React, { useState, useEffect } from 'react';
import { Row, Col, Form, Button, ButtonGroup, Spinner, Alert } from 'react-bootstrap';
import { useTrainingStore } from '../../../../store/trainingStore';
import { WaterfallVisualization } from '../../../../components/WaterfallVisualization';
import { AudioPlayer } from '../../../../components/AudioPlayer';
import type { SyntheticSample, ReceiverMetadata, IQDataResponse } from '../../types';

interface WaterfallViewTabProps {
    sample: SyntheticSample;
    datasetId: string;
}

export const WaterfallViewTab: React.FC<WaterfallViewTabProps> = ({ sample, datasetId }) => {
    const fetchSampleIQData = useTrainingStore(state => state.fetchSampleIQData);
    
    const [selectedRxId, setSelectedRxId] = useState<string | null>(null);
    const [iqDataResponse, setIqDataResponse] = useState<IQDataResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Waterfall parameters
    const [fftSize, setFftSize] = useState(512);
    const [overlap, setOverlap] = useState(0.5);
    const [colormap, setColormap] = useState<'viridis' | 'plasma' | 'turbo' | 'jet'>('viridis');
    const [minDb, setMinDb] = useState(-80);  // Fixed: was -160 (too low for normalized IQ)
    const [maxDb, setMaxDb] = useState(-20);  // Fixed: was -140 (too low for normalized IQ)
    const [autoScale, setAutoScale] = useState(true);  // Auto-scale dB range by default
    const [useWebWorker, setUseWebWorker] = useState(true);  // Use Web Worker by default

    const receivers = sample.receivers as ReceiverMetadata[];

    // Calculate min/max SNR for normalization
    const snrValues = receivers.map(rx => rx.snr_db);
    const minSnr = Math.min(...snrValues);
    const maxSnr = Math.max(...snrValues);
    const snrRange = maxSnr - minSnr;

    /**
     * Normalize SNR value to 0-1 range
     */
    const normalizeSnr = (snr: number): number => {
        if (snrRange === 0) return 1; // All receivers have same SNR
        return (snr - minSnr) / snrRange;
    };

    /**
     * Convert normalized SNR (0-1) to color gradient: red → yellow → green
     * 0.0 = red (#dc3545)
     * 0.5 = yellow (#ffc107)
     * 1.0 = green (#28a745)
     */
    const getSnrColor = (normalizedSnr: number): string => {
        // Clamp to 0-1 range
        const value = Math.max(0, Math.min(1, normalizedSnr));
        
        if (value < 0.5) {
            // Interpolate red → yellow (0.0 to 0.5)
            const t = value * 2; // Scale to 0-1
            const r = 220; // Red component stays high
            const g = Math.round(60 + (252 - 60) * t); // 60 → 252
            const b = Math.round(69 - 69 * t); // 69 → 0
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Interpolate yellow → green (0.5 to 1.0)
            const t = (value - 0.5) * 2; // Scale to 0-1
            const r = Math.round(255 - (255 - 40) * t); // 255 → 40
            const g = Math.round(193 + (167 - 193) * t); // 193 → 167
            const b = Math.round(7 + (69 - 7) * t); // 7 → 69
            return `rgb(${r}, ${g}, ${b})`;
        }
    };

    // Auto-select first receiver if available
    useEffect(() => {
        if (receivers.length > 0 && !selectedRxId) {
            setSelectedRxId(receivers[0].rx_id);
        }
    }, [receivers, selectedRxId]);

    // Fetch IQ data when receiver is selected
    useEffect(() => {
        if (!selectedRxId || sample.sample_idx === undefined) return;

        const loadIQData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const data = await fetchSampleIQData(datasetId, sample.sample_idx!, selectedRxId);
                setIqDataResponse(data);
            } catch (err) {
                console.error('Failed to load IQ data:', err);
                setError(err instanceof Error ? err.message : 'Failed to load IQ data');
            } finally {
                setIsLoading(false);
            }
        };

        loadIQData();
    }, [datasetId, sample.sample_idx, selectedRxId, fetchSampleIQData]);

    // Check if IQ data is available
    if (!sample.iq_available) {
        return (
            <Alert variant="warning">
                IQ data is not available for this sample. Only feature-based samples support waterfall visualization.
            </Alert>
        );
    }

    return (
        <div className="waterfall-view-tab">
            {/* Receiver Selector */}
            <Row className="mb-3">
                <Col md={6}>
                    <Form.Label className="text-dark"><strong>Select Receiver:</strong></Form.Label>
                    <ButtonGroup className="d-flex flex-wrap">
                        {receivers.map((rx) => {
                            const normalizedSnr = normalizeSnr(rx.snr_db);
                            const snrColor = getSnrColor(normalizedSnr);
                            const isSelected = selectedRxId === rx.rx_id;

                            return (
                                <Button
                                    key={rx.rx_id}
                                    variant={isSelected ? 'primary' : undefined}
                                    onClick={() => setSelectedRxId(rx.rx_id)}
                                    size="sm"
                                    style={!isSelected ? {
                                        backgroundColor: snrColor,
                                        borderColor: snrColor,
                                        color: 'white',
                                        fontWeight: '500'
                                    } : undefined}
                                >
                                    {rx.rx_id} ({rx.snr_db.toFixed(1)} dB)
                                </Button>
                            );
                        })}
                    </ButtonGroup>
                </Col>
            </Row>

            {/* Waterfall Controls */}
            <Row className="mb-3">
                <Col md={2}>
                    <Form.Group>
                        <Form.Label className="text-dark">FFT Size</Form.Label>
                        <Form.Select
                            value={fftSize}
                            onChange={(e) => setFftSize(Number(e.target.value))}
                            size="sm"
                        >
                            <option value={256}>256</option>
                            <option value={512}>512</option>
                            <option value={1024}>1024</option>
                            <option value={2048}>2048</option>
                        </Form.Select>
                    </Form.Group>
                </Col>
                <Col md={2}>
                    <Form.Group>
                        <Form.Label className="text-dark">Overlap</Form.Label>
                        <Form.Select
                            value={overlap}
                            onChange={(e) => setOverlap(Number(e.target.value))}
                            size="sm"
                        >
                            <option value={0.25}>25%</option>
                            <option value={0.5}>50%</option>
                            <option value={0.75}>75%</option>
                        </Form.Select>
                    </Form.Group>
                </Col>
                <Col md={2}>
                    <Form.Group>
                        <Form.Label className="text-dark">Colormap</Form.Label>
                        <Form.Select
                            value={colormap}
                            onChange={(e) => setColormap(e.target.value as any)}
                            size="sm"
                        >
                            <option value="viridis">Viridis</option>
                            <option value="plasma">Plasma</option>
                            <option value="turbo">Turbo</option>
                            <option value="jet">Jet</option>
                        </Form.Select>
                    </Form.Group>
                </Col>
                <Col md={3}>
                    <Form.Group>
                        <Form.Label className="text-dark">dB Range</Form.Label>
                        <div className="d-flex gap-2">
                            <Form.Control
                                type="number"
                                value={minDb}
                                onChange={(e) => setMinDb(Number(e.target.value))}
                                size="sm"
                                placeholder="Min"
                                disabled={autoScale}
                            />
                            <Form.Control
                                type="number"
                                value={maxDb}
                                onChange={(e) => setMaxDb(Number(e.target.value))}
                                size="sm"
                                placeholder="Max"
                                disabled={autoScale}
                            />
                        </div>
                    </Form.Group>
                </Col>
                <Col md={3}>
                    <Form.Group>
                        <Form.Label className="text-dark">Options</Form.Label>
                        <div className="d-flex flex-column gap-1">
                            <Form.Check
                                type="switch"
                                id="auto-scale-switch"
                                label="Auto-scale dB"
                                checked={autoScale}
                                onChange={(e) => setAutoScale(e.target.checked)}
                                className="text-dark"
                            />
                            <Form.Check
                                type="switch"
                                id="web-worker-switch"
                                label="Use Web Worker"
                                checked={useWebWorker}
                                onChange={(e) => setUseWebWorker(e.target.checked)}
                                className="text-dark"
                            />
                        </div>
                    </Form.Group>
                </Col>
            </Row>
            
            {/* Reset Button */}
            <Row className="mb-3">
                <Col>
                    <Button 
                        variant="outline-secondary" 
                        size="sm"
                        onClick={() => {
                            setFftSize(512);
                            setOverlap(0.5);
                            setColormap('viridis');
                            setMinDb(-80);
                            setMaxDb(-20);
                            setAutoScale(true);
                            setUseWebWorker(true);
                            localStorage.removeItem('heimdall_waterfall_settings');
                        }}
                    >
                        Reset to Defaults
                    </Button>
                </Col>
            </Row>

            {/* Loading/Error States */}
            {isLoading && (
                <div className="text-center py-5">
                    <Spinner animation="border" variant="primary" />
                    <p className="mt-2 text-muted">Loading IQ data...</p>
                </div>
            )}

            {error && (
                <Alert variant="danger">
                    <strong>Error:</strong> {error}
                </Alert>
            )}

            {/* Waterfall Display */}
            {!isLoading && !error && iqDataResponse && (
                <>
                    {/* Validate Float32Array conversion */}
                    {!(iqDataResponse.iq_data.i_samples instanceof Float32Array) || 
                     !(iqDataResponse.iq_data.q_samples instanceof Float32Array) ? (
                        <Alert variant="danger">
                            <strong>Error:</strong> IQ data not properly decoded to Float32Array. 
                            Please check the API response format.
                        </Alert>
                    ) : (
                        <>
                            <Row>
                                <Col>
                                    <WaterfallVisualization
                                        iqData={{
                                            i_samples: iqDataResponse.iq_data.i_samples,
                                            q_samples: iqDataResponse.iq_data.q_samples
                                        }}
                                        sampleRate={iqDataResponse.iq_metadata.sample_rate_hz}
                                        centerFrequency={iqDataResponse.iq_metadata.center_frequency_hz}
                                        fftSize={fftSize}
                                        overlap={overlap}
                                        colormap={colormap}
                                        minDb={minDb}
                                        maxDb={maxDb}
                                        height={600}
                                        autoScale={autoScale}
                                        useWebWorker={useWebWorker}
                                    />

                                    {/* Metadata Display */}
                                    <div className="mt-3 p-3 bg-dark text-white border rounded">
                                        <Row>
                                            <Col md={4}>
                                                <strong>Receiver:</strong> {iqDataResponse.rx_metadata.rx_id}<br/>
                                                <strong>SNR:</strong> {iqDataResponse.rx_metadata.snr_db.toFixed(1)} dB<br/>
                                                <strong>Distance:</strong> {iqDataResponse.rx_metadata.distance_km.toFixed(1)} km
                                            </Col>
                                            <Col md={4}>
                                                <strong>Sample Rate:</strong> {(iqDataResponse.iq_metadata.sample_rate_hz / 1e3).toFixed(1)} kHz<br/>
                                                <strong>Duration:</strong> {iqDataResponse.iq_metadata.duration_ms.toFixed(1)} ms<br/>
                                                <strong>Samples:</strong> {iqDataResponse.iq_data.length.toLocaleString()}
                                            </Col>
                                            <Col md={4}>
                                                <strong>Frequency:</strong> {(iqDataResponse.iq_metadata.center_frequency_hz / 1e6).toFixed(3)} MHz<br/>
                                                <strong>TX Power:</strong> {iqDataResponse.tx_metadata.power_dbm.toFixed(1)} dBm<br/>
                                                {iqDataResponse.rx_metadata.rx_power_dbm !== undefined && (
                                                    <><strong>RX Power:</strong> {iqDataResponse.rx_metadata.rx_power_dbm.toFixed(1)} dBm</>
                                                )}
                                            </Col>
                                        </Row>
                                    </div>
                                </Col>
                            </Row>

                            {/* Audio Player */}
                            <Row className="mt-3">
                                <Col>
                                    <AudioPlayer
                                        iqData={{
                                            i_samples: iqDataResponse.iq_data.i_samples,
                                            q_samples: iqDataResponse.iq_data.q_samples
                                        }}
                                        sampleRate={iqDataResponse.iq_metadata.sample_rate_hz}
                                        centerFrequency={iqDataResponse.iq_metadata.center_frequency_hz}
                                    />
                                </Col>
                            </Row>
                        </>
                    )}
                </>
            )}
        </div>
    );
};
