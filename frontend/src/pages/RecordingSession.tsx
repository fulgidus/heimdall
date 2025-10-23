import React, { useEffect, useState } from 'react';
import { useSessionStore } from '../store/sessionStore';
import { useWebSDRStore } from '../store/websdrStore';
import { acquisitionService } from '../services/api';
import type { AcquisitionStatusResponse } from '../services/api/types';

const RecordingSession: React.FC = () => {
    const {
        knownSources,
        fetchKnownSources,
        createSession,
        isLoading: sessionLoading,
        error: sessionError,
    } = useSessionStore();

    const { websdrs, healthStatus, fetchWebSDRs } = useWebSDRStore();

    const [formData, setFormData] = useState({
        knownSourceId: '',
        sessionName: '',
        frequency: '',
        duration: 60,
        notes: '',
    });

    const [acquisitionStatus, setAcquisitionStatus] = useState<AcquisitionStatusResponse | null>(null);
    const [isAcquiring, setIsAcquiring] = useState(false);
    const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

    useEffect(() => {
        fetchKnownSources();
        fetchWebSDRs();
    }, [fetchKnownSources, fetchWebSDRs]);

    const selectedSource = knownSources.find(s => s.id === formData.knownSourceId);
    const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;

    const handleStartAcquisition = async () => {
        if (!formData.knownSourceId || !formData.sessionName) {
            alert('Please select a source and enter a session name');
            return;
        }

        try {
            setIsAcquiring(true);

            // Create session in database
            const session = await createSession({
                known_source_id: formData.knownSourceId,
                session_name: formData.sessionName,
                frequency_hz: parseFloat(formData.frequency) * 1e6,
                duration_seconds: formData.duration,
                notes: formData.notes,
            });

            // Trigger acquisition
            const acquisitionResponse = await acquisitionService.triggerAcquisition({
                frequency_mhz: parseFloat(formData.frequency),
                duration_seconds: formData.duration,
                session_id: session.id,
            });

            setCurrentTaskId(acquisitionResponse.task_id);

            // Poll status
            const finalStatus = await acquisitionService.pollAcquisitionStatus(
                acquisitionResponse.task_id,
                (status) => {
                    setAcquisitionStatus(status);
                }
            );

            setAcquisitionStatus(finalStatus);
        } catch (error) {
            console.error('Acquisition failed:', error);
            alert('Failed to start acquisition');
        } finally {
            setIsAcquiring(false);
        }
    };

    const handleReset = () => {
        setFormData({
            knownSourceId: '',
            sessionName: '',
            frequency: '',
            duration: 60,
            notes: '',
        });
        setAcquisitionStatus(null);
        setCurrentTaskId(null);
    };

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item"><a href="#">RF Operations</a></li>
                                <li className="breadcrumb-item" aria-current="page">Recording Session</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Create Recording Session</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {sessionError && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {sessionError}
                    <button type="button" className="btn-close" data-bs-dismiss="alert"></button>
                </div>
            )}

            <div className="row">
                {/* Session Configuration */}
                <div className="col-lg-8">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Session Configuration</h5>
                        </div>
                        <div className="card-body">
                            <div className="row g-3">
                                <div className="col-12">
                                    <label className="form-label">Known Source *</label>
                                    <select
                                        className="form-select"
                                        value={formData.knownSourceId}
                                        onChange={(e) => {
                                            const source = knownSources.find(s => s.id === e.target.value);
                                            setFormData({
                                                ...formData,
                                                knownSourceId: e.target.value,
                                                frequency: source ? (source.frequency_hz / 1e6).toString() : '',
                                            });
                                        }}
                                        disabled={isAcquiring}
                                    >
                                        <option value="">Select a source...</option>
                                        {knownSources.map((source) => (
                                            <option key={source.id} value={source.id}>
                                                {source.name} - {(source.frequency_hz / 1e6).toFixed(3)} MHz
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div className="col-md-6">
                                    <label className="form-label">Session Name *</label>
                                    <input
                                        type="text"
                                        className="form-control"
                                        value={formData.sessionName}
                                        onChange={(e) => setFormData({ ...formData, sessionName: e.target.value })}
                                        placeholder="e.g., Beacon Recording 2024-10-23"
                                        disabled={isAcquiring}
                                    />
                                </div>

                                <div className="col-md-6">
                                    <label className="form-label">Frequency (MHz) *</label>
                                    <input
                                        type="number"
                                        className="form-control"
                                        value={formData.frequency}
                                        onChange={(e) => setFormData({ ...formData, frequency: e.target.value })}
                                        step="0.001"
                                        min="144"
                                        max="148"
                                        disabled={isAcquiring}
                                    />
                                </div>

                                <div className="col-12">
                                    <label className="form-label">Duration (seconds)</label>
                                    <input
                                        type="range"
                                        className="form-range"
                                        min="10"
                                        max="300"
                                        step="10"
                                        value={formData.duration}
                                        onChange={(e) => setFormData({ ...formData, duration: parseInt(e.target.value) })}
                                        disabled={isAcquiring}
                                    />
                                    <div className="d-flex justify-content-between">
                                        <span className="f-12 text-muted">10s</span>
                                        <span className="badge bg-primary">{formData.duration}s</span>
                                        <span className="f-12 text-muted">300s (5min)</span>
                                    </div>
                                </div>

                                <div className="col-12">
                                    <label className="form-label">Notes (Optional)</label>
                                    <textarea
                                        className="form-control"
                                        rows={3}
                                        value={formData.notes}
                                        onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                                        placeholder="Additional notes about this recording session..."
                                        disabled={isAcquiring}
                                    ></textarea>
                                </div>

                                {selectedSource && (
                                    <div className="col-12">
                                        <div className="alert alert-info">
                                            <h6 className="mb-2">Source Information</h6>
                                            <table className="table table-sm table-borderless mb-0">
                                                <tbody>
                                                    <tr>
                                                        <td className="text-muted">Name:</td>
                                                        <td>{selectedSource.name}</td>
                                                    </tr>
                                                    <tr>
                                                        <td className="text-muted">Location:</td>
                                                        <td>
                                                            {selectedSource.latitude.toFixed(4)}, {selectedSource.longitude.toFixed(4)}
                                                        </td>
                                                    </tr>
                                                    <tr>
                                                        <td className="text-muted">Power:</td>
                                                        <td>{selectedSource.power_dbm ? `${selectedSource.power_dbm} dBm` : 'N/A'}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="card-footer">
                            <button
                                className="btn btn-primary"
                                onClick={handleStartAcquisition}
                                disabled={isAcquiring || !formData.knownSourceId || !formData.sessionName}
                            >
                                {isAcquiring ? (
                                    <>
                                        <span className="spinner-border spinner-border-sm me-2"></span>
                                        Acquiring...
                                    </>
                                ) : (
                                    <>
                                        <i className="ph ph-record me-2"></i>
                                        Start Acquisition
                                    </>
                                )}
                            </button>
                            <button
                                className="btn btn-outline-secondary ms-2"
                                onClick={handleReset}
                                disabled={isAcquiring}
                            >
                                <i className="ph ph-arrow-counter-clockwise me-2"></i>
                                Reset
                            </button>
                        </div>
                    </div>
                </div>

                {/* Status Panel */}
                <div className="col-lg-4">
                    {/* System Status */}
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">System Status</h5>
                        </div>
                        <div className="card-body">
                            <div className="d-flex justify-content-between align-items-center mb-3">
                                <span>WebSDR Receivers</span>
                                <span className="badge bg-light-primary">{onlineWebSDRs}/7 Online</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                                <span>Known Sources</span>
                                <span className="badge bg-light-info">{knownSources.length}</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center">
                                <span>System Ready</span>
                                <span className={`badge ${onlineWebSDRs > 0 ? 'bg-light-success' : 'bg-light-danger'}`}>
                                    {onlineWebSDRs > 0 ? 'Yes' : 'No'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Acquisition Status */}
                    {acquisitionStatus && (
                        <div className="card">
                            <div className="card-header">
                                <h5 className="mb-0">Acquisition Status</h5>
                            </div>
                            <div className="card-body">
                                <div className="d-flex align-items-center mb-3">
                                    <div className={`avtar avtar-s ${acquisitionStatus.status === 'SUCCESS'
                                            ? 'bg-light-success'
                                            : acquisitionStatus.status === 'FAILURE'
                                                ? 'bg-light-danger'
                                                : 'bg-light-primary'
                                        }`}>
                                        <i className={`ph ${acquisitionStatus.status === 'SUCCESS'
                                                ? 'ph-check-circle'
                                                : acquisitionStatus.status === 'FAILURE'
                                                    ? 'ph-x-circle'
                                                    : 'ph-hourglass'
                                            }`}></i>
                                    </div>
                                    <div className="ms-3 flex-grow-1">
                                        <h6 className="mb-0">
                                            {acquisitionStatus.status === 'SUCCESS'
                                                ? 'Completed'
                                                : acquisitionStatus.status === 'FAILURE'
                                                    ? 'Failed'
                                                    : 'In Progress'}
                                        </h6>
                                        <p className="text-muted f-12 mb-0">
                                            {acquisitionStatus.state || 'Processing...'}
                                        </p>
                                    </div>
                                </div>

                                {acquisitionStatus.progress !== undefined && (
                                    <div className="mb-3">
                                        <div className="d-flex justify-content-between mb-1">
                                            <span className="f-12">Progress</span>
                                            <span className="f-12">{Math.round(acquisitionStatus.progress * 100)}%</span>
                                        </div>
                                        <div className="progress" style={{ height: '6px' }}>
                                            <div
                                                className="progress-bar bg-primary"
                                                style={{ width: `${acquisitionStatus.progress * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                )}

                                {currentTaskId && (
                                    <div className="f-12 text-muted">
                                        Task ID: <code>{currentTaskId}</code>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

export default RecordingSession;
