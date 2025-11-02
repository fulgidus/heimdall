import React, { useCallback, useEffect, useState } from 'react';
import { useSessionStore } from '../store/sessionStore';
import { useWebSDRStore } from '../store/websdrStore';
import { useWebSocket } from '../contexts/WebSocketContext';
import { getDetailedHealth } from '../services/api/system';
import type { DetailedHealthResponse } from '../services/api/schemas';

type SessionState = 'idle' | 'ready_to_assign' | 'acquiring' | 'complete' | 'error';

const RecordingSession: React.FC = () => {
  const { knownSources, fetchKnownSources, error: sessionError } = useSessionStore();

  const { healthStatus, fetchWebSDRs } = useWebSDRStore();

  const [formData, setFormData] = useState({
    sessionName: '',
    frequency: '',
    duration: 15,
    notes: '',
  });

  const [sessionState, setSessionState] = useState<SessionState>('idle');
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [measurementsCount, setMeasurementsCount] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [systemHealth, setSystemHealth] = useState<DetailedHealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  
  const { manager: ws, isConnected } = useWebSocket();

  // WebSocket event handlers (defined before useEffect)
  const handleSessionStarted = useCallback((data: any) => {
    console.log('Session configuration validated:', data);
    setSessionState('ready_to_assign');
    setStatusMessage('Configuration validated. Now select a source.');
  }, []);

  const handleSessionCompleted = useCallback((data: any) => {
    console.log('Session created and acquisition started:', data);
    setCurrentSessionId(data.session_id);
    setSessionState('acquiring');
    setTotalChunks(data.chunks || 0);
    setStatusMessage(`Session created! Acquiring ${data.chunks} 1-second samples...`);
  }, []);

  const handleSessionProgress = useCallback((data: any) => {
    console.log('Session progress:', data);
    setCurrentChunk(data.chunk || 0);
    setTotalChunks(data.total_chunks || 0);
    setProgress(data.progress || 0);
    setMeasurementsCount(data.measurements_count || 0);
    setStatusMessage(`Chunk ${data.chunk}/${data.total_chunks} acquired`);

    if (data.chunk === data.total_chunks) {
      setSessionState('complete');
      setStatusMessage('Acquisition complete!');
    }
  }, []);

  const handleSessionError = useCallback((data: any) => {
    console.error('Session error:', data);
    setSessionState('error');
    setStatusMessage(`Error: ${data.error}`);
  }, []);

  // Load system health on mount
  useEffect(() => {
    const loadSystemHealth = async () => {
      try {
        setHealthLoading(true);
        const health = await getDetailedHealth();
        setSystemHealth(health);
      } catch (error) {
        console.error('Failed to load system health:', error);
      } finally {
        setHealthLoading(false);
      }
    };
    
    loadSystemHealth();
    // Refresh every 30 seconds
    const interval = setInterval(loadSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchKnownSources();
    fetchWebSDRs();

    if (!ws || !isConnected) {
      console.warn('WebSocket not connected, waiting for connection...');
      return;
    }

    // Subscribe to session events
    ws.subscribe('session:started', handleSessionStarted);
    ws.subscribe('session:completed', handleSessionCompleted);
    ws.subscribe('session:progress', handleSessionProgress);
    ws.subscribe('session:error', handleSessionError);

    return () => {
      ws.unsubscribe('session:started', handleSessionStarted);
      ws.unsubscribe('session:completed', handleSessionCompleted);
      ws.unsubscribe('session:progress', handleSessionProgress);
      ws.unsubscribe('session:error', handleSessionError);
    };
  }, [fetchKnownSources, fetchWebSDRs, ws, isConnected, handleSessionStarted, handleSessionCompleted, handleSessionProgress, handleSessionError]);

  const hasHealthData = Object.keys(healthStatus).length > 0;
  const totalWebSDRs = Object.keys(healthStatus).length;
  const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;
  const onlineWebSDRsDisplay = hasHealthData ? `${onlineWebSDRs}/${totalWebSDRs}` : '?/?';

  // Step 1: Validate configuration
  const handleValidateConfig = () => {
    if (!formData.sessionName || !formData.frequency) {
      alert('Please enter a session name and specify frequency');
      return;
    }

    if (!ws || !isConnected) {
      alert('WebSocket not connected. Please wait and try again.');
      return;
    }

    // Send start command via WebSocket (now just validates)
    ws.send('session:start', {
      session_name: formData.sessionName,
      frequency_mhz: parseFloat(formData.frequency),
      duration_seconds: formData.duration,
      notes: formData.notes,
    });
  };

  // Step 2: Complete and start acquisition (combined - creates session + starts acquisition)
  const handleCompleteAndAcquire = () => {
    if (!selectedSourceId) {
      alert('Please select a source');
      return;
    }

    if (!formData.sessionName || !formData.frequency) {
      alert('Please enter a session name and specify frequency');
      return;
    }

    if (!ws || !isConnected) {
      alert('WebSocket not connected. Please wait and try again.');
      return;
    }

    // Send complete command via WebSocket (now creates session + starts acquisition)
    ws.send('session:complete', {
      session_name: formData.sessionName,
      frequency_hz: Math.round(parseFloat(formData.frequency) * 1e6), // Convert MHz to Hz
      duration_seconds: formData.duration,
      source_id: selectedSourceId,
      notes: formData.notes,
    });
  };

  const handleReset = () => {
    setFormData({
      sessionName: '',
      frequency: '',
      duration: 15,
      notes: '',
    });
    setSessionState('idle');
    setCurrentSessionId(null);
    setSelectedSourceId('');
    setProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
    setMeasurementsCount(0);
    setStatusMessage('');
  };

  return (
    <>
      {/* Breadcrumb */}
      <div className="page-header">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ul className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/dashboard">Home</a>
                </li>
                <li className="breadcrumb-item">
                  <a href="#">RF Operations</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Recording Session
                </li>
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
                {/* Step 1: Basic Info */}
                <div className="col-12">
                  <div className="alert alert-info">
                    <strong>Step 1:</strong> Configure session and start recording
                  </div>
                </div>

                <div className="col-md-6">
                  <label className="form-label">Session Name *</label>
                  <input
                    type="text"
                    className="form-control"
                    value={formData.sessionName}
                    onChange={e => setFormData({ ...formData, sessionName: e.target.value })}
                    placeholder="e.g., Beacon Recording 2024-10-30"
                    disabled={sessionState !== 'idle'}
                  />
                </div>

                <div className="col-md-6">
                  <label className="form-label">Frequency (MHz) *</label>
                  <input
                    type="number"
                    className="form-control"
                    value={formData.frequency}
                    onChange={e => setFormData({ ...formData, frequency: e.target.value })}
                    step="0.001"
                    min="144"
                    max="148"
                    disabled={sessionState !== 'idle'}
                  />
                </div>

                <div className="col-12">
                  <label className="form-label">
                    Duration (seconds) - {formData.duration} samples
                  </label>
                  <input
                    type="range"
                    className="form-range"
                    min="1"
                    max="30"
                    step="1"
                    value={formData.duration}
                    onChange={e => setFormData({ ...formData, duration: parseInt(e.target.value) })}
                    disabled={sessionState !== 'idle'}
                  />
                  <div className="d-flex justify-content-between">
                    <span className="f-12 text-muted">1s (1 sample)</span>
                    <span className="badge bg-primary">
                      {formData.duration}s = {formData.duration} samples
                    </span>
                    <span className="f-12 text-muted">30s (30 samples)</span>
                  </div>
                  <div className="f-12 text-muted mt-1">
                    <i className="ph ph-info me-1"></i>
                    Each second is recorded as a separate 1s sample for better training granularity
                  </div>
                </div>

                <div className="col-12">
                  <label className="form-label">Notes (Optional)</label>
                  <textarea
                    className="form-control"
                    rows={2}
                    value={formData.notes}
                    onChange={e => setFormData({ ...formData, notes: e.target.value })}
                    placeholder="Additional notes about this recording session..."
                    disabled={sessionState !== 'idle'}
                  ></textarea>
                </div>

                {/* Step 2: Source Assignment */}
                {sessionState === 'ready_to_assign' && (
                  <>
                    <div className="col-12">
                      <div className="alert alert-info">
                        <strong>Step 2:</strong> Select a source for this recording
                      </div>
                    </div>

                    <div className="col-12">
                      <label className="form-label">Select Source</label>
                      <select
                        className="form-select"
                        value={selectedSourceId}
                        onChange={e => setSelectedSourceId(e.target.value)}
                      >
                        <option value="">-- Select Source --</option>
                        <option value="unknown">Unknown (Generic)</option>
                        {knownSources.map(source => (
                          <option key={source.id} value={source.id}>
                            {source.name}
                            {source.frequency_hz
                              ? ` - ${(source.frequency_hz / 1e6).toFixed(3)} MHz`
                              : ''}
                          </option>
                        ))}
                      </select>
                    </div>
                  </>
                )}
              </div>
            </div>
            <div className="card-footer">
              {sessionState === 'idle' && (
                <button
                  className="btn btn-primary"
                  onClick={handleValidateConfig}
                  disabled={!formData.sessionName || !formData.frequency}
                >
                  <i className="ph ph-play me-2"></i>
                  Step 1: Validate Configuration
                </button>
              )}

              {sessionState === 'ready_to_assign' && (
                <button
                  className="btn btn-success"
                  onClick={handleCompleteAndAcquire}
                  disabled={!selectedSourceId}
                >
                  <i className="ph ph-rocket-launch me-2"></i>
                  Step 2: Start Acquisition
                </button>
              )}

              {(sessionState === 'acquiring' || sessionState === 'complete') && (
                <button
                  className="btn btn-primary"
                  onClick={handleReset}
                  disabled={sessionState === 'acquiring'}
                >
                  <i className="ph ph-plus me-2"></i>
                  New Session
                </button>
              )}

              {sessionState !== 'idle' && sessionState !== 'acquiring' && (
                <button className="btn btn-outline-secondary ms-2" onClick={handleReset}>
                  <i className="ph ph-x me-2"></i>
                  Cancel
                </button>
              )}
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
              {healthLoading ? (
                <div className="text-center py-3">
                  <div className="spinner-border spinner-border-sm text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <p className="text-muted f-12 mt-2 mb-0">Checking system status...</p>
                </div>
              ) : systemHealth ? (
                <>
                  {/* Component Status */}
                  {Object.entries(systemHealth.components).map(([name, component]) => {
                    const statusColors = {
                      healthy: 'bg-light-success',
                      unhealthy: 'bg-light-danger',
                      warning: 'bg-light-warning',
                      unknown: 'bg-light-secondary',
                    };
                    const statusIcons = {
                      healthy: 'ph-check-circle',
                      unhealthy: 'ph-x-circle',
                      warning: 'ph-warning-circle',
                      unknown: 'ph-question',
                    };
                    const statusColor = statusColors[component.status] || 'bg-light-secondary';
                    const statusIcon = statusIcons[component.status] || 'ph-question';

                    return (
                      <div
                        key={name}
                        className="d-flex justify-content-between align-items-center mb-2"
                        title={component.message}
                      >
                        <div className="d-flex align-items-center">
                          <i className={`ph ${statusIcon} me-2 f-14`}></i>
                          <span className="f-14 text-capitalize">{name.replace('_', ' ')}</span>
                        </div>
                        <span className={`badge ${statusColor} f-11`}>
                          {component.status.toUpperCase()}
                        </span>
                      </div>
                    );
                  })}

                  <hr className="my-3" />

                  {/* Summary */}
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="f-12 text-muted">WebSocket</span>
                    <span className={`badge ${isConnected ? 'bg-light-success' : 'bg-light-danger'} f-11`}>
                      {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                    </span>
                  </div>

                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="f-12 text-muted">WebSDR Receivers</span>
                    <span className="badge bg-light-primary f-11">{onlineWebSDRsDisplay} Online</span>
                  </div>

                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="f-12 text-muted">Known Sources</span>
                    <span className="badge bg-light-info f-11">{knownSources.length}</span>
                  </div>

                  {/* Overall System Ready */}
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="fw-bold">System Ready</span>
                    <span className={`badge ${systemHealth.ready ? 'bg-success' : 'bg-danger'}`}>
                      {systemHealth.ready ? 'YES' : 'NO'}
                    </span>
                  </div>

                  {!systemHealth.ready && (
                    <div className="alert alert-warning mt-3 mb-0 f-12" role="alert">
                      <i className="ph ph-warning me-2"></i>
                      <strong>{systemHealth.summary.unhealthy}</strong> component(s) unhealthy.
                      Check statuses above.
                    </div>
                  )}
                </>
              ) : (
                <div className="alert alert-danger f-12" role="alert">
                  <i className="ph ph-warning me-2"></i>
                  Unable to check system status
                </div>
              )}
            </div>
          </div>

          {/* Session Status */}
          {sessionState !== 'idle' && (
            <div className="card">
              <div className="card-header">
                <h5 className="mb-0">Session Status</h5>
              </div>
              <div className="card-body">
                <div className="d-flex align-items-center mb-3">
                  <div
                    className={`avtar avtar-s ${
                      sessionState === 'complete'
                        ? 'bg-light-success'
                        : sessionState === 'error'
                          ? 'bg-light-danger'
                          : 'bg-light-primary'
                    }`}
                  >
                    <i
                      className={`ph ${
                        sessionState === 'complete'
                          ? 'ph-check-circle'
                          : sessionState === 'error'
                            ? 'ph-x-circle'
                            : sessionState === 'acquiring'
                              ? 'ph-spinner'
                              : 'ph-record'
                      }`}
                    ></i>
                  </div>
                  <div className="ms-3 flex-grow-1">
                    <h6 className="mb-0">
                      {sessionState === 'ready_to_assign' && 'Ready to Assign Source'}
                      {sessionState === 'acquiring' && 'Acquiring Data'}
                      {sessionState === 'complete' && 'Complete'}
                      {sessionState === 'error' && 'Error'}
                    </h6>
                    <p className="text-muted f-12 mb-0">{statusMessage || 'Processing...'}</p>
                  </div>
                </div>

                {/* Progress for acquisition */}
                {sessionState === 'acquiring' && totalChunks > 0 && (
                  <>
                    <div className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="f-12">Samples Acquired</span>
                        <span className="f-12">
                          {currentChunk}/{totalChunks}
                        </span>
                      </div>
                      <div className="progress" style={{ height: '8px' }}>
                        <div
                          className="progress-bar bg-primary progress-bar-striped progress-bar-animated"
                          style={{ width: `${progress}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="d-flex justify-content-between mb-2">
                      <span className="f-12 text-muted">Total Measurements</span>
                      <span className="badge bg-light-info">{measurementsCount}</span>
                    </div>
                  </>
                )}

                {/* Summary for completed */}
                {sessionState === 'complete' && (
                  <div className="alert alert-success mb-0">
                    <i className="ph ph-check-circle me-2"></i>
                    Acquired {totalChunks} samples ({measurementsCount} total measurements)
                  </div>
                )}

                {currentSessionId && (
                  <div className="f-12 text-muted mt-2">
                    Session ID: <code>{currentSessionId.substring(0, 8)}...</code>
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
