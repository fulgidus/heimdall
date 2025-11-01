import React, { useEffect, useState } from 'react';
import { useSystemStore } from '../store';
import { useWebSDRStore } from '../store';
import { inferenceService } from '../services/api';

const SystemStatus: React.FC = () => {
  const { servicesHealth, isLoading, checkAllServices, fetchModelPerformance, modelPerformance } = useSystemStore();
  const { websdrs, healthStatus, fetchWebSDRs, checkHealth } = useWebSDRStore();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [modelInfo, setModelInfo] = useState<any>(null);

  useEffect(() => {
    // Initial fetch
    checkAllServices();
    fetchModelPerformance();
    fetchWebSDRs();
    checkHealth();
    loadModelInfo();

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      checkAllServices();
      fetchModelPerformance();
      checkHealth();
      loadModelInfo();
    }, 30000);

    return () => clearInterval(interval);
  }, [checkAllServices, fetchModelPerformance, fetchWebSDRs, checkHealth]);

  const loadModelInfo = async () => {
    try {
      const info = await inferenceService.getModelInfo();
      setModelInfo(info);
    } catch (error) {
      console.error('Failed to fetch model info:', error);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([
      checkAllServices(),
      fetchModelPerformance(),
      checkHealth(),
      loadModelInfo(),
    ]);
    setIsRefreshing(false);
  };

  const services = Object.entries(servicesHealth).map(([name, health]) => ({
    name: name.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    status: health.status,
    rawName: name,
  }));

  const safeWebsdrs = Array.isArray(websdrs) ? websdrs : [];
  const onlineWebSDRCount = Object.values(healthStatus).filter(h => h?.status === 'online').length;
  const totalWebSDRCount = safeWebsdrs.length;

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
                <li className="breadcrumb-item" aria-current="page">
                  System Status
                </li>
              </ul>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h2 className="mb-0 text-white">System Status</h2>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="row">
        {/* System Overview */}
        <div className="col-12">
          <div className="card">
            <div className="card-header d-flex align-items-center justify-content-between">
              <h5 className="mb-0">System Overview</h5>
              <button
                className="btn btn-sm btn-primary"
                onClick={handleRefresh}
                disabled={isRefreshing}
              >
                <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                {isRefreshing ? ' Refreshing...' : ' Refresh'}
              </button>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-3">
                  <div className="d-grid">
                    <div className="bg-light-primary p-3 rounded text-center">
                      <i className="ph ph-cpu f-40 text-primary mb-2"></i>
                      <h6 className="mb-0">Microservices</h6>
                      <h3 className="mb-0 mt-2">
                        {services.filter(s => s.status === 'healthy').length}/{services.length}
                      </h3>
                      <p className="text-muted f-12 mb-0">Healthy</p>
                    </div>
                  </div>
                </div>
                <div className="col-md-3">
                  <div className="d-grid">
                    <div className="bg-light-success p-3 rounded text-center">
                      <i className="ph ph-radio-button f-40 text-success mb-2"></i>
                      <h6 className="mb-0">WebSDR Receivers</h6>
                      <h3 className="mb-0 mt-2">
                        {onlineWebSDRCount}/{totalWebSDRCount}
                      </h3>
                      <p className="text-muted f-12 mb-0">Online</p>
                    </div>
                  </div>
                </div>
                <div className="col-md-3">
                  <div className="d-grid">
                    <div className="bg-light-warning p-3 rounded text-center">
                      <i className="ph ph-brain f-40 text-warning mb-2"></i>
                      <h6 className="mb-0">ML Model</h6>
                      <h3 className="mb-0 mt-2">
                        {modelInfo?.is_ready ? 'Ready' : 'Not Ready'}
                      </h3>
                      <p className="text-muted f-12 mb-0">Status</p>
                    </div>
                  </div>
                </div>
                <div className="col-md-3">
                  <div className="d-grid">
                    <div className="bg-light-info p-3 rounded text-center">
                      <i className="ph ph-activity f-40 text-info mb-2"></i>
                      <h6 className="mb-0">System Health</h6>
                      <h3 className="mb-0 mt-2">
                        {services.length > 0 && services.filter(s => s.status === 'healthy').length === services.length
                          ? 'Good'
                          : 'Degraded'}
                      </h3>
                      <p className="text-muted f-12 mb-0">Overall</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Services Details */}
        <div className="col-lg-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Microservices Health</h5>
            </div>
            <div className="card-body">
              {isLoading ? (
                <div className="text-center py-5">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <p className="text-muted mt-2">Checking services...</p>
                </div>
              ) : services.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-hover mb-0">
                    <thead>
                      <tr>
                        <th>Service</th>
                        <th>Status</th>
                        <th>Health</th>
                      </tr>
                    </thead>
                    <tbody>
                      {services.map(service => (
                        <tr key={service.rawName}>
                          <td>
                            <div className="d-flex align-items-center">
                              <div className="flex-shrink-0">
                                <div
                                  className={`avtar avtar-s ${
                                    service.status === 'healthy'
                                      ? 'bg-light-success'
                                      : service.status === 'degraded'
                                        ? 'bg-light-warning'
                                        : 'bg-light-danger'
                                  }`}
                                >
                                  <i
                                    className={`ph ${
                                      service.status === 'healthy'
                                        ? 'ph-check-circle'
                                        : service.status === 'degraded'
                                          ? 'ph-warning-circle'
                                          : 'ph-x-circle'
                                    }`}
                                  ></i>
                                </div>
                              </div>
                              <div className="flex-grow-1 ms-3">
                                <h6 className="mb-0">{service.name}</h6>
                              </div>
                            </div>
                          </td>
                          <td>
                            <span
                              className={`badge ${
                                service.status === 'healthy'
                                  ? 'bg-light-success'
                                  : service.status === 'degraded'
                                    ? 'bg-light-warning'
                                    : 'bg-light-danger'
                              }`}
                            >
                              {service.status}
                            </span>
                          </td>
                          <td>
                            <div className="progress" style={{ height: '6px' }}>
                              <div
                                className={`progress-bar ${
                                  service.status === 'healthy'
                                    ? 'bg-success'
                                    : service.status === 'degraded'
                                      ? 'bg-warning'
                                      : 'bg-danger'
                                }`}
                                role="progressbar"
                                style={{
                                  width:
                                    service.status === 'healthy'
                                      ? '100%'
                                      : service.status === 'degraded'
                                        ? '50%'
                                        : '25%',
                                }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-5">
                  <i className="ph ph-warning-circle f-40 text-warning mb-3"></i>
                  <p className="text-muted mb-0">No service data available</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* WebSDR Status */}
        <div className="col-lg-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">WebSDR Receivers</h5>
            </div>
            <div className="card-body">
              {safeWebsdrs.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-hover mb-0">
                    <thead>
                      <tr>
                        <th>Location</th>
                        <th>Status</th>
                        <th>Response Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {safeWebsdrs.slice(0, 7).map(sdr => {
                        const health = healthStatus[sdr.id];
                        const isOnline = health?.status === 'online';
                        
                        return (
                          <tr key={sdr.id}>
                            <td>
                              <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                  <div
                                    className={`avtar avtar-s ${
                                      isOnline
                                        ? 'bg-light-success'
                                        : 'bg-light-danger'
                                    }`}
                                  >
                                    <i
                                      className={`ph ${
                                        isOnline
                                          ? 'ph-radio-button'
                                          : 'ph-x-circle'
                                      }`}
                                    ></i>
                                  </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                  <h6 className="mb-0">
                                    {sdr.location_description?.split(',')[0] || sdr.name}
                                  </h6>
                                  <small className="text-muted">{sdr.country}</small>
                                </div>
                              </div>
                            </td>
                            <td>
                              <span
                                className={`badge ${
                                  isOnline
                                    ? 'bg-light-success'
                                    : 'bg-light-danger'
                                }`}
                              >
                                {isOnline ? 'Online' : 'Offline'}
                              </span>
                            </td>
                            <td>
                              {health?.response_time_ms
                                ? `${health.response_time_ms.toFixed(0)}ms`
                                : 'N/A'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-5">
                  <i className="ph ph-radio-button f-40 text-muted mb-3"></i>
                  <p className="text-muted mb-0">No WebSDR receivers configured</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ML Model Status */}
        <div className="col-lg-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">ML Model Status</h5>
            </div>
            <div className="card-body">
              {modelInfo ? (
                <>
                  <div className="row mb-3">
                    <div className="col-6">
                      <p className="text-muted mb-1">Version</p>
                      <h6 className="mb-0">{modelInfo.active_version || 'N/A'}</h6>
                    </div>
                    <div className="col-6">
                      <p className="text-muted mb-1">Health Status</p>
                      <h6 className="mb-0">
                        <span
                          className={`badge ${
                            modelInfo.health_status === 'healthy'
                              ? 'bg-light-success'
                              : 'bg-light-warning'
                          }`}
                        >
                          {modelInfo.health_status}
                        </span>
                      </h6>
                    </div>
                  </div>
                  <hr />
                  <div className="row mb-3">
                    <div className="col-6">
                      <p className="text-muted mb-1">Accuracy</p>
                      <h6 className="mb-0">
                        {modelInfo.accuracy
                          ? `${(modelInfo.accuracy * 100).toFixed(2)}%`
                          : 'N/A'}
                      </h6>
                    </div>
                    <div className="col-6">
                      <p className="text-muted mb-1">Loaded At</p>
                      <h6 className="mb-0">
                        {modelInfo.loaded_at
                          ? new Date(modelInfo.loaded_at).toLocaleString()
                          : 'N/A'}
                      </h6>
                    </div>
                  </div>
                  <hr />
                  <div className="row">
                    <div className="col-4">
                      <p className="text-muted mb-1">Total Predictions</p>
                      <h6 className="mb-0">{modelInfo.predictions_total ?? 0}</h6>
                    </div>
                    <div className="col-4">
                      <p className="text-muted mb-1">Successful</p>
                      <h6 className="mb-0 text-success">
                        {modelInfo.predictions_successful ?? 0}
                      </h6>
                    </div>
                    <div className="col-4">
                      <p className="text-muted mb-1">Failed</p>
                      <h6 className="mb-0 text-danger">
                        {modelInfo.predictions_failed ?? 0}
                      </h6>
                    </div>
                  </div>
                  {modelInfo.last_prediction_at && (
                    <>
                      <hr />
                      <div className="row">
                        <div className="col-12">
                          <p className="text-muted mb-1">Last Prediction</p>
                          <h6 className="mb-0">
                            {new Date(modelInfo.last_prediction_at).toLocaleString()}
                          </h6>
                        </div>
                      </div>
                    </>
                  )}
                </>
              ) : (
                <div className="text-center py-5">
                  <i className="ph ph-brain f-40 text-muted mb-3"></i>
                  <p className="text-muted mb-0">Model information not available</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default SystemStatus;
