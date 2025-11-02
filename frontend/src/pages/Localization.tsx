import React, { useEffect, useState } from 'react';
import { useDashboardStore, useWebSDRStore, useLocalizationStore } from '../store';
import { MapContainer } from '../components/Map';

const Localization: React.FC = () => {
  const { data, fetchDashboardData } = useDashboardStore();
  const { websdrs, healthStatus } = useWebSDRStore();
  const { recentLocalizations, fetchRecentLocalizations } = useLocalizationStore();
  const [selectedResult, setSelectedResultState] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    fetchRecentLocalizations();
  }, [fetchDashboardData, fetchRecentLocalizations]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchDashboardData();
    setIsRefreshing(false);
  };

  const totalWebSDRs = Object.keys(healthStatus).length;
  const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;
  const avgAccuracy = data.modelInfo?.accuracy ? (data.modelInfo.accuracy * 100).toFixed(1) : 'N/A';

  // Calculate average confidence from recent localizations
  const avgConfidence =
    recentLocalizations.length > 0
      ? (
          (recentLocalizations.reduce((sum, loc) => sum + loc.confidence, 0) /
            recentLocalizations.length) *
          100
        ).toFixed(0)
      : 'N/A';

  // Calculate average SNR and map to quality label
  const avgSnr =
    recentLocalizations.length > 0
      ? recentLocalizations.reduce((sum, loc) => sum + loc.snr_avg_db, 0) /
        recentLocalizations.length
      : 0;

  const signalQuality =
    recentLocalizations.length === 0
      ? 'N/A'
      : avgSnr > 20
        ? 'Excellent'
        : avgSnr > 10
          ? 'Good'
          : 'Poor';

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
                  Localization
                </li>
              </ul>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h2 className="mb-0">RF Source Localization</h2>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics Row */}
      <div className="row">
        <div className="col-md-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="shrink-0">
                  <div className="avtar avtar-s bg-light-success rounded">
                    <i className="ph ph-crosshair f-20"></i>
                  </div>
                </div>
                <div className="grow ms-3">
                  <h6 className="mb-0">Active Receivers</h6>
                  <h4 className="mb-0">{onlineWebSDRs}/{totalWebSDRs}</h4>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="shrink-0">
                  <div className="avtar avtar-s bg-light-primary rounded">
                    <i className="ph ph-target f-20"></i>
                  </div>
                </div>
                <div className="grow ms-3">
                  <h6 className="mb-0">Avg Accuracy</h6>
                  <h4 className="mb-0">±{avgAccuracy}m</h4>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="shrink-0">
                  <div className="avtar avtar-s bg-light-warning rounded">
                    <i className="ph ph-chart-line f-20"></i>
                  </div>
                </div>
                <div className="grow ms-3">
                  <h6 className="mb-0">Confidence</h6>
                  <h4 className="mb-0">{avgConfidence}%</h4>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="shrink-0">
                  <div className="avtar avtar-s bg-light-info rounded">
                    <i className="ph ph-activity f-20"></i>
                  </div>
                </div>
                <div className="grow ms-3">
                  <h6 className="mb-0">Signal Quality</h6>
                  <h4 className="mb-0 f-14">{signalQuality}</h4>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Map and Results */}
      <div className="row">
        {/* Map Area */}
        <div className="col-lg-8">
          <div className="card">
            <div className="card-header d-flex align-items-center justify-content-between">
              <h5 className="mb-0">Localization Map</h5>
              <div className="btn-group">
                <button
                  className="btn btn-sm btn-outline-primary"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                </button>
                <button className="btn btn-sm btn-outline-secondary">
                  <i className="ph ph-arrows-out-simple"></i>
                </button>
              </div>
            </div>
            <div className="card-body p-0">
              {/* Mapbox GL JS Map */}
              <MapContainer
                websdrs={websdrs}
                healthStatus={healthStatus}
                localizations={recentLocalizations}
                onLocalizationClick={loc => setSelectedResultState(loc.id)}
                style={{ height: '500px' }}
                mapStyle="mapbox://styles/mapbox/dark-v11"
                fitBoundsOnLoad={true}
              />
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="col-lg-4">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Recent Localizations</h5>
            </div>
            <div className="card-body">
              {recentLocalizations.length > 0 ? (
                <div className="d-flex flex-column gap-3">
                  {recentLocalizations.map(result => (
                    <div
                      key={result.id}
                      className={`card cursor-pointer ${selectedResult === result.id ? 'border-primary' : ''}`}
                      onClick={() => setSelectedResultState(result.id)}
                    >
                      <div className="card-body p-3">
                        <div className="d-flex justify-content-between align-items-start mb-2">
                          <span className="badge bg-light-success">
                            {(result.confidence * 100).toFixed(0)}% Confidence
                          </span>
                          <small className="text-muted">
                            {new Date(result.timestamp).toLocaleTimeString()}
                          </small>
                        </div>
                        <div className="mb-2">
                          <h6 className="mb-1">
                            <i className="ph ph-map-pin text-primary me-1"></i>
                            {result.latitude.toFixed(4)}, {result.longitude.toFixed(4)}
                          </h6>
                          <p className="f-12 text-muted mb-0">
                            Uncertainty: ±{result.uncertainty_m.toFixed(1)}m
                          </p>
                        </div>
                        <div className="row">
                          <div className="col-6">
                            <p className="f-12 text-muted mb-0">Confidence</p>
                            <h6 className="mb-0">{(result.confidence * 100).toFixed(0)}%</h6>
                          </div>
                          <div className="col-6">
                            <p className="f-12 text-muted mb-0">Receivers</p>
                            <h6 className="mb-0">{result.websdr_count}/{totalWebSDRs}</h6>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-5">
                  <i className="ph ph-crosshair-simple f-40 text-muted mb-3"></i>
                  <p className="text-muted mb-0">No localization results yet</p>
                  <button className="btn btn-primary btn-sm mt-3">
                    <i className="ph ph-play-circle me-1"></i>
                    Start Localization
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Selected Result Details */}
          {selectedResult && (
            <div className="card mt-3">
              <div className="card-header">
                <h5 className="mb-0">Result Details</h5>
              </div>
              <div className="card-body">
                {(() => {
                  const result = recentLocalizations.find(r => r.id === selectedResult);
                  if (!result) return null;

                  return (
                    <div>
                      <table className="table table-sm">
                        <tbody>
                          <tr>
                            <td className="text-muted">Timestamp</td>
                            <td>{new Date(result.timestamp).toLocaleString()}</td>
                          </tr>
                          <tr>
                            <td className="text-muted">Latitude</td>
                            <td>{result.latitude.toFixed(6)}</td>
                          </tr>
                          <tr>
                            <td className="text-muted">Longitude</td>
                            <td>{result.longitude.toFixed(6)}</td>
                          </tr>
                          <tr>
                            <td className="text-muted">Uncertainty</td>
                            <td>±{result.uncertainty_m.toFixed(1)}m</td>
                          </tr>
                          <tr>
                            <td className="text-muted">Confidence</td>
                            <td>
                              <div className="d-flex align-items-center">
                                <div className="progress grow me-2" style={{ height: '6px' }}>
                                  <div
                                    className="progress-bar bg-success"
                                    style={{ width: `${result.confidence * 100}%` }}
                                  ></div>
                                </div>
                                <span>{(result.confidence * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                          </tr>
                          <tr>
                            <td className="text-muted">Signal Quality</td>
                            <td>
                              <span className="badge bg-light-success">
                                {result.snr_avg_db > 20
                                  ? 'Excellent'
                                  : result.snr_avg_db > 10
                                    ? 'Good'
                                    : 'Poor'}
                              </span>
                            </td>
                          </tr>
                          <tr>
                            <td className="text-muted">Active Receivers</td>
                            <td>{result.websdr_count}/{totalWebSDRs}</td>
                          </tr>
                        </tbody>
                      </table>
                      <button className="btn btn-primary btn-sm w-100 mt-2">
                        <i className="ph ph-download-simple me-1"></i>
                        Export Results
                      </button>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default Localization;
