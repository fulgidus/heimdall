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
import { Card, Table, Badge, Alert, OverlayTrigger, Tooltip } from 'react-bootstrap';
import type { SyntheticSample, ReceiverMetadata } from '../../types';
import {
    checkMonotonicity,
    getPhysicsAnomalies,
    getPhysicsValidationSummary,
    getAnomalyBadgeColor,
    type MonotonicityWarning,
    type PhysicsAnomaly
} from '../../utils/physicsValidation';

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

    // Physics validation
    const validationSummary = getPhysicsValidationSummary(receivers);
    const monotonicityWarnings = validationSummary.monotonicity_warnings;
    
    // Create map of receiver ID to anomalies for quick lookup
    const receiverAnomalies = new Map<string, PhysicsAnomaly[]>();
    receivers.forEach(rx => {
        const anomalies = getPhysicsAnomalies(rx);
        if (anomalies.length > 0) {
            receiverAnomalies.set(rx.rx_id, anomalies);
        }
    });
    
    // Check if a receiver is involved in monotonicity violation
    const getMonotonicityWarningForReceiver = (rxId: string): MonotonicityWarning | undefined => {
        return monotonicityWarnings.find(w => w.rx_closer === rxId || w.rx_farther === rxId);
    };

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
            {/* Physics Validation Alerts */}
            {monotonicityWarnings.length > 0 && (
                <Alert variant="warning" className="mb-3">
                    <Alert.Heading className="h6">
                        ⚠️ Physics Anomaly Detected
                    </Alert.Heading>
                    <p className="mb-2">
                        The following receivers show unexpected distance-SNR relationships:
                    </p>
                    <ul className="mb-0">
                        {monotonicityWarnings.map((warning, idx) => (
                            <li key={idx}>
                                <strong>{warning.rx_closer}</strong> ({warning.distance_closer_km.toFixed(1)} km, {warning.snr_closer_db.toFixed(1)} dB) 
                                has worse SNR than <strong>{warning.rx_farther}</strong> ({warning.distance_farther_km.toFixed(1)} km, {warning.snr_farther_db.toFixed(1)} dB)
                                <br />
                                <small className="text-muted">
                                    SNR difference: {warning.snr_difference_db.toFixed(1)} dB (exceeds 12 dB tolerance)
                                </small>
                            </li>
                        ))}
                    </ul>
                    <hr />
                    <small className="text-muted">
                        <strong>Possible causes:</strong> Antenna pattern nulls/lobes, polarization mismatch, 
                        extreme fading, terrain obstruction, or sporadic-E propagation. Check receiver details below.
                    </small>
                </Alert>
            )}

            {/* Physics Statistics Summary */}
            {(validationSummary.sporadic_e_count > 0 || validationSummary.cross_pol_count > 0) && (
                <Alert variant="info" className="mb-3">
                    <div className="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>Sample Statistics:</strong>
                            {validationSummary.sporadic_e_count > 0 && (
                                <span className="ms-2">
                                    <Badge bg="info">Sporadic-E: {validationSummary.sporadic_e_count}</Badge>
                                </span>
                            )}
                            {validationSummary.cross_pol_count > 0 && (
                                <span className="ms-2">
                                    <Badge bg="warning">Cross-Pol: {validationSummary.cross_pol_count}</Badge>
                                </span>
                            )}
                            {validationSummary.antenna_null_count > 0 && (
                                <span className="ms-2">
                                    <Badge bg="secondary">Antenna Nulls: {validationSummary.antenna_null_count}</Badge>
                                </span>
                            )}
                        </div>
                    </div>
                </Alert>
            )}

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
                                <th>Anomalies</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedReceivers.map((rx) => {
                                const isSelected = selectedRxId === rx.rx_id;
                                const hasSignal = rx.signal_present !== undefined ? rx.signal_present : (rx.snr_db > -20);
                                const anomalies = receiverAnomalies.get(rx.rx_id) || [];
                                const monotonicityWarning = getMonotonicityWarningForReceiver(rx.rx_id);
                                
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
                                            {monotonicityWarning && (
                                                <OverlayTrigger
                                                    placement="top"
                                                    overlay={
                                                        <Tooltip>
                                                            Physics anomaly: This receiver has unexpected SNR compared to distance. 
                                                            Check propagation breakdown for explanation.
                                                        </Tooltip>
                                                    }
                                                >
                                                    <Badge bg="warning" className="ms-1" style={{ cursor: 'help' }}>⚠️</Badge>
                                                </OverlayTrigger>
                                            )}
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
                                            {anomalies.length > 0 ? (
                                                <div className="d-flex gap-1">
                                                    {anomalies.map((anomaly, idx) => (
                                                        <OverlayTrigger
                                                            key={idx}
                                                            placement="top"
                                                            overlay={<Tooltip>{anomaly.tooltip}</Tooltip>}
                                                        >
                                                            <Badge 
                                                                bg={getAnomalyBadgeColor(anomaly.type)}
                                                                style={{ cursor: 'help', fontSize: '0.7rem' }}
                                                            >
                                                                {anomaly.label}
                                                            </Badge>
                                                        </OverlayTrigger>
                                                    ))}
                                                </div>
                                            ) : (
                                                <span className="text-muted">-</span>
                                            )}
                                        </td>
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
