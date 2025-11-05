/**
 * SampleDetailsPanel Component
 * 
 * Displays detailed information about a synthetic sample including:
 * - Transmitter information
 * - Receiver table with SNR, distance, power
 * - Propagation breakdown for each receiver
 * - Antenna configuration details
 */

import React from 'react';
import { Card, Table, Badge } from 'react-bootstrap';
import type { SyntheticSample, ReceiverMetadata } from '../../types';

interface SampleDetailsPanelProps {
    sample: SyntheticSample;
    selectedRxId?: string | null;
    onRxSelect?: (rxId: string) => void;
}

export const SampleDetailsPanel: React.FC<SampleDetailsPanelProps> = ({
    sample,
    selectedRxId,
    onRxSelect
}) => {
    const receivers = sample.receivers as ReceiverMetadata[];

    // Sort receivers by SNR (descending)
    const sortedReceivers = [...receivers].sort((a, b) => b.snr_db - a.snr_db);

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

    return (
        <div className="sample-details-panel">
            {/* Transmitter Info */}
            <Card className="mb-3">
                <Card.Header className="bg-danger text-white">
                    <strong>Transmitter</strong>
                </Card.Header>
                <Card.Body>
                    <div className="row">
                        <div className="col-md-6">
                            <p className="mb-1">
                                <strong>Position:</strong> {sample.tx_lat != null ? sample.tx_lat.toFixed(6) : 'N/A'}, {sample.tx_lon != null ? sample.tx_lon.toFixed(6) : 'N/A'}
                            </p>
                            <p className="mb-1">
                                <strong>Power:</strong> {sample.tx_power_dbm != null ? sample.tx_power_dbm.toFixed(1) : 'N/A'} dBm
                            </p>
                        </div>
                        <div className="col-md-6">
                            <p className="mb-1">
                                <strong>Frequency:</strong> {sample.frequency_hz != null ? (sample.frequency_hz / 1e6).toFixed(3) : 'N/A'} MHz
                            </p>
                            <p className="mb-1">
                                <strong>GDOP:</strong> {sample.gdop != null ? sample.gdop.toFixed(2) : 'N/A'}
                            </p>
                        </div>
                    </div>
                </Card.Body>
            </Card>

            {/* Receivers Table */}
            <Card>
                <Card.Header className="bg-primary text-white">
                    <strong>Receivers ({receivers.length})</strong>
                </Card.Header>
                <Card.Body className="p-0">
                    <Table hover responsive className="mb-0">
                        <thead>
                            <tr>
                                <th>RX ID</th>
                                <th>Distance</th>
                                <th>SNR</th>
                                <th>Signal</th>
                                {sortedReceivers.some(rx => rx.rx_power_dbm !== undefined) && <th>RX Power</th>}
                                {sortedReceivers.some(rx => rx.tx_antenna_type || rx.rx_antenna_type) && <th>Antenna</th>}
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedReceivers.map((rx) => {
                                const isSelected = selectedRxId === rx.rx_id;
                                const hasSignal = rx.signal_present !== undefined ? rx.signal_present : (rx.snr_db > -20);
                                
                                return (
                                    <tr
                                        key={rx.rx_id}
                                        className={isSelected ? 'table-warning' : ''}
                                        style={{ cursor: onRxSelect ? 'pointer' : 'default' }}
                                        onClick={() => onRxSelect && onRxSelect(rx.rx_id)}
                                    >
                                        <td>
                                            <Badge bg={isSelected ? 'warning' : 'primary'}>
                                                {rx.rx_id}
                                            </Badge>
                                        </td>
                                        <td>{rx.distance_km != null ? rx.distance_km.toFixed(1) : 'N/A'} km</td>
                                        <td>
                                            <span className={rx.snr_db != null && rx.snr_db > 10 ? 'text-success' : rx.snr_db != null && rx.snr_db > 0 ? 'text-warning' : 'text-danger'}>
                                                {rx.snr_db != null ? rx.snr_db.toFixed(1) : 'N/A'} dB
                                            </span>
                                        </td>
                                        <td>
                                            <Badge bg={hasSignal ? 'success' : 'secondary'}>
                                                {hasSignal ? 'Present' : 'Absent'}
                                            </Badge>
                                        </td>
                                        {sortedReceivers.some(r => r.rx_power_dbm !== undefined) && (
                                            <td>{rx.rx_power_dbm !== undefined ? `${rx.rx_power_dbm.toFixed(1)} dBm` : 'N/A'}</td>
                                        )}
                                        {sortedReceivers.some(r => r.tx_antenna_type || r.rx_antenna_type) && (
                                            <td>
                                                <small className="text-muted">
                                                    {rx.tx_antenna_type || 'N/A'} → {rx.rx_antenna_type || 'N/A'}
                                                </small>
                                            </td>
                                        )}
                                        <td>
                                            <button
                                                className="btn btn-sm btn-outline-primary"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (onRxSelect) {
                                                        onRxSelect(rx.rx_id);
                                                    }
                                                }}
                                            >
                                                Details
                                            </button>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </Table>
                </Card.Body>
            </Card>

            {/* Selected Receiver Propagation Details */}
            {selectedRxId && (
                <Card className="mt-3">
                    <Card.Header className="bg-warning">
                        <strong>Receiver {selectedRxId} - Propagation Breakdown</strong>
                    </Card.Header>
                    <Card.Body>
                        {(() => {
                            const rx = receivers.find(r => r.rx_id === selectedRxId);
                            if (!rx) return <p>Receiver not found</p>;

                            return (
                                <div>
                                    {/* Position & Basic Info */}
                                    <div className="row mb-3">
                                        <div className="col-md-6">
                                            <p className="mb-1">
                                                <strong>Position:</strong> {rx.lat != null ? rx.lat.toFixed(6) : 'N/A'}, {rx.lon != null ? rx.lon.toFixed(6) : 'N/A'}
                                            </p>
                                            <p className="mb-1">
                                                <strong>Altitude:</strong> {rx.alt != null ? rx.alt.toFixed(0) : 'N/A'} m
                                            </p>
                                        </div>
                                        <div className="col-md-6">
                                            <p className="mb-1">
                                                <strong>Distance:</strong> {rx.distance_km != null ? rx.distance_km.toFixed(2) : 'N/A'} km
                                            </p>
                                            <p className="mb-1">
                                                <strong>SNR:</strong> {rx.snr_db != null ? rx.snr_db.toFixed(2) : 'N/A'} dB
                                            </p>
                                        </div>
                                    </div>

                                    <hr />

                                    {/* Propagation Details */}
                                    <h6 className="text-muted mb-2">Propagation Path Loss Components:</h6>
                                    <Table size="sm" bordered>
                                        <tbody>
                                            {rx.fspl_db != null && (
                                                <tr>
                                                    <td><strong>Free Space Path Loss (FSPL)</strong></td>
                                                    <td className="text-end">{rx.fspl_db.toFixed(2)} dB</td>
                                                </tr>
                                            )}
                                            {rx.terrain_loss_db != null && (
                                                <tr>
                                                    <td><strong>Terrain Obstruction Loss</strong></td>
                                                    <td className="text-end">{rx.terrain_loss_db.toFixed(2)} dB</td>
                                                </tr>
                                            )}
                                            {rx.knife_edge_loss_db != null && (
                                                <tr>
                                                    <td><strong>Knife-Edge Diffraction</strong></td>
                                                    <td className="text-end">{rx.knife_edge_loss_db.toFixed(2)} dB</td>
                                                </tr>
                                            )}
                                            {rx.atmospheric_absorption_db != null && (
                                                <tr>
                                                    <td><strong>Atmospheric Absorption</strong></td>
                                                    <td className="text-end">{rx.atmospheric_absorption_db.toFixed(2)} dB</td>
                                                </tr>
                                            )}
                                            {rx.polarization_loss_db != null && (
                                                <tr>
                                                    <td><strong>Polarization Mismatch</strong></td>
                                                    <td className="text-end">{rx.polarization_loss_db.toFixed(2)} dB</td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </Table>

                                    {/* Enhancements */}
                                    {(rx.tropospheric_effect_db != null || rx.sporadic_e_enhancement_db != null) && (
                                        <>
                                            <h6 className="text-muted mb-2 mt-3">Propagation Enhancements:</h6>
                                            <Table size="sm" bordered>
                                                <tbody>
                                                    {rx.tropospheric_effect_db != null && (
                                                        <tr>
                                                            <td><strong>Tropospheric Effect</strong></td>
                                                            <td className="text-end text-success">
                                                                {rx.tropospheric_effect_db > 0 ? '+' : ''}
                                                                {rx.tropospheric_effect_db.toFixed(2)} dB
                                                            </td>
                                                        </tr>
                                                    )}
                                                    {rx.sporadic_e_enhancement_db != null && (
                                                        <tr>
                                                            <td><strong>Sporadic-E Enhancement</strong></td>
                                                            <td className="text-end text-success">
                                                                {rx.sporadic_e_enhancement_db > 0 ? '+' : ''}
                                                                {rx.sporadic_e_enhancement_db.toFixed(2)} dB
                                                            </td>
                                                        </tr>
                                                    )}
                                                </tbody>
                                            </Table>
                                        </>
                                    )}

                                    {/* Antenna Details */}
                                    <hr />
                                    <h6 className="text-muted mb-2">Antenna Configuration:</h6>
                                    <Table size="sm" bordered>
                                        <tbody>
                                            <tr>
                                                <td><strong>TX Antenna</strong></td>
                                                <td>
                                                    {rx.tx_antenna_type || 'N/A'}
                                                    {rx.tx_antenna_gain_db != null && (
                                                        <span className="text-muted"> ({rx.tx_antenna_gain_db.toFixed(1)} dBi)</span>
                                                    )}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td><strong>RX Antenna</strong></td>
                                                <td>
                                                    {rx.rx_antenna_type || 'N/A'}
                                                    {rx.rx_antenna_gain_db != null && (
                                                        <span className="text-muted"> ({rx.rx_antenna_gain_db.toFixed(1)} dBi)</span>
                                                    )}
                                                </td>
                                            </tr>
                                            {rx.tx_polarization && rx.rx_polarization && (
                                                <tr>
                                                    <td><strong>Polarization</strong></td>
                                                    <td>
                                                        {rx.tx_polarization} → {rx.rx_polarization}
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </Table>
                                </div>
                            );
                        })()}
                    </Card.Body>
                </Card>
            )}
        </div>
    );
};
