import React, { useEffect, useState } from 'react';
import { useDashboardStore, useWebSDRStore } from '../store';

const Dashboard: React.FC = () => {
    const {
        metrics,
        data,
        isLoading,
        error,
        fetchDashboardData,
        lastUpdate,
    } = useDashboardStore();
    const { websdrs, healthStatus } = useWebSDRStore();
    const [isRefreshing, setIsRefreshing] = useState(false);

    useEffect(() => {
        // Fetch data on component mount
        fetchDashboardData();

        // Setup auto-refresh every 30 seconds
        const interval = setInterval(() => {
            fetchDashboardData();
        }, 30000);

        return () => clearInterval(interval);
    }, [fetchDashboardData]);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await fetchDashboardData();
        setIsRefreshing(false);
    };

    // Calculate online WebSDRs from health status
    const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;
    const totalWebSDRs = websdrs.length || 7;

    // WebSDR status for display
    const webSDRStatuses = websdrs.slice(0, 7).map((sdr) => {
        const health = healthStatus[sdr.id];
        return {
            id: sdr.id,
            city: sdr.location_name.split(',')[0],
            status: health?.status === 'online' ? 'online' : 'offline',
            signal: health?.response_time_ms
                ? Math.max(0, 100 - health.response_time_ms / 10)
                : 0,
            frequency: '144.2 MHz',
        };
    });

    // Fill remaining slots if less than 7 WebSDRs
    const defaultCities = ['Turin', 'Milan', 'Genoa', 'Alessandria', 'Asti', 'La Spezia', 'Piacenza'];
    while (webSDRStatuses.length < 7) {
        webSDRStatuses.push({
            id: webSDRStatuses.length + 1,
            city: defaultCities[webSDRStatuses.length] || `Location ${webSDRStatuses.length + 1}`,
            status: 'offline' as const,
            signal: 0,
            frequency: '144.2 MHz',
        });
    }

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/">Home</a></li>
                                <li className="breadcrumb-item" aria-current="page">Dashboard</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Dashboard</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
            )}

            {/* Stats Cards Row */}
            <div className="row">
                {/* Active WebSDR Card */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <h6 className="mb-4">Active WebSDR</h6>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className={`ph ${onlineWebSDRs === totalWebSDRs ? 'ph-arrow-up text-success' : 'ph-arrow-down text-danger'} f-30 m-r-10`}></i>
                                        {onlineWebSDRs}/{totalWebSDRs}
                                    </h3>
                                </div>
                                <div className="col-3 text-end">
                                    <p className="m-b-0">{Math.round((onlineWebSDRs / totalWebSDRs) * 100)}%</p>
                                </div>
                            </div>
                            <div className="progress m-t-30" style={{ height: '7px' }}>
                                <div
                                    className="progress-bar bg-brand-color-1"
                                    role="progressbar"
                                    style={{ width: `${(onlineWebSDRs / totalWebSDRs) * 100}%` }}
                                    aria-valuenow={(onlineWebSDRs / totalWebSDRs) * 100}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Signal Detection Card */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <h6 className="mb-4">Signal Detections</h6>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-chart-line-up text-warning f-30 m-r-10"></i>
                                        {metrics.signalDetections || 0}
                                    </h3>
                                </div>
                                <div className="col-3 text-end">
                                    <p className="m-b-0">24h</p>
                                </div>
                            </div>
                            <div className="progress m-t-30" style={{ height: '7px' }}>
                                <div
                                    className="progress-bar bg-brand-color-2"
                                    role="progressbar"
                                    style={{ width: '75%' }}
                                    aria-valuenow={75}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Uptime Card */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <h6 className="mb-4">System Uptime</h6>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-activity text-success f-30 m-r-10"></i>
                                        {metrics.systemUptime > 0
                                            ? `${(metrics.systemUptime / 3600).toFixed(1)}h`
                                            : '0h'}
                                    </h3>
                                </div>
                                <div className="col-3 text-end">
                                    <p className="m-b-0">Live</p>
                                </div>
                            </div>
                            <div className="progress m-t-30" style={{ height: '7px' }}>
                                <div
                                    className="progress-bar bg-success"
                                    role="progressbar"
                                    style={{ width: '90%' }}
                                    aria-valuenow={90}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Model Accuracy Card */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <h6 className="mb-4">Model Accuracy</h6>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-target text-primary f-30 m-r-10"></i>
                                        {data.modelInfo?.accuracy
                                            ? `${(data.modelInfo.accuracy * 100).toFixed(1)}%`
                                            : 'N/A'}
                                    </h3>
                                </div>
                                <div className="col-3 text-end">
                                    <p className="m-b-0">ML</p>
                                </div>
                            </div>
                            <div className="progress m-t-30" style={{ height: '7px' }}>
                                <div
                                    className="progress-bar bg-primary"
                                    role="progressbar"
                                    style={{ width: `${data.modelInfo?.accuracy ? data.modelInfo.accuracy * 100 : 0}%` }}
                                    aria-valuenow={data.modelInfo?.accuracy ? data.modelInfo.accuracy * 100 : 0}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Row */}
            <div className="row">
                {/* System Activity */}
                <div className="col-lg-8">
                    <div className="card table-card">
                        <div className="card-header d-flex align-items-center justify-content-between">
                            <h5 className="mb-0">System Activity</h5>
                            <button
                                className="btn btn-sm btn-link-primary"
                                onClick={handleRefresh}
                                disabled={isRefreshing}
                            >
                                <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                                {isRefreshing ? ' Refreshing...' : ' Refresh'}
                            </button>
                        </div>
                        <div className="card-body">
                            <div className="table-responsive">
                                <table className="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>Status</th>
                                            <th>Activity</th>
                                            <th>Details</th>
                                            <th>Timestamp</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-success">
                                                    <i className="ph ph-check-circle"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h6 className="mb-0">System Status</h6>
                                                <p className="text-muted f-12 mb-0">WebSDR Network</p>
                                            </td>
                                            <td>
                                                {onlineWebSDRs} of {totalWebSDRs} receivers online
                                            </td>
                                            <td className="text-muted">
                                                {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Just now'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span className={`badge ${data.modelInfo ? 'bg-light-primary' : 'bg-light-warning'}`}>
                                                    <i className={`ph ${data.modelInfo ? 'ph-brain' : 'ph-warning-circle'}`}></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h6 className="mb-0">ML Model</h6>
                                                <p className="text-muted f-12 mb-0">Inference Engine</p>
                                            </td>
                                            <td>
                                                {data.modelInfo
                                                    ? `Version ${data.modelInfo.active_version} - ${data.modelInfo.health_status}`
                                                    : 'Initializing...'}
                                            </td>
                                            <td className="text-muted">
                                                {data.modelInfo?.loaded_at
                                                    ? new Date(data.modelInfo.loaded_at).toLocaleTimeString()
                                                    : '-'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-success">
                                                    <i className="ph ph-cpu"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h6 className="mb-0">Services Health</h6>
                                                <p className="text-muted f-12 mb-0">Microservices</p>
                                            </td>
                                            <td>
                                                {Object.values(data.servicesHealth).filter(s => s.status === 'healthy').length} of{' '}
                                                {Object.keys(data.servicesHealth).length} services healthy
                                            </td>
                                            <td className="text-muted">
                                                {lastUpdate ? 'Updated' : 'Checking...'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-info">
                                                    <i className="ph ph-chart-bar"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h6 className="mb-0">Predictions</h6>
                                                <p className="text-muted f-12 mb-0">Total Count</p>
                                            </td>
                                            <td>
                                                {data.modelInfo
                                                    ? `${data.modelInfo.predictions_total} total (${data.modelInfo.predictions_successful} successful)`
                                                    : 'No predictions yet'}
                                            </td>
                                            <td className="text-muted">
                                                {data.modelInfo?.last_prediction_at || '-'}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Services Status */}
                <div className="col-lg-4">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Services Status</h5>
                        </div>
                        <div className="card-body">
                            {Object.entries(data.servicesHealth).length > 0 ? (
                                <ul className="list-group list-group-flush">
                                    {Object.entries(data.servicesHealth).map(([name, health]) => (
                                        <li key={name} className="list-group-item px-0">
                                            <div className="d-flex align-items-center justify-content-between">
                                                <div className="flex-grow-1">
                                                    <h6 className="mb-0 text-capitalize">
                                                        {name.replace('-', ' ')}
                                                    </h6>
                                                </div>
                                                <div className="flex-shrink-0">
                                                    <span
                                                        className={`badge ${health.status === 'healthy'
                                                                ? 'bg-light-success'
                                                                : health.status === 'degraded'
                                                                    ? 'bg-light-warning'
                                                                    : 'bg-light-danger'
                                                            }`}
                                                    >
                                                        {health.status}
                                                    </span>
                                                </div>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <div className="text-center py-4">
                                    <i className="ph ph-warning-circle f-40 text-muted mb-3"></i>
                                    <p className="text-muted mb-0">
                                        {isLoading ? 'Checking services...' : 'No service data available'}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* WebSDR Network Status */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">WebSDR Network Status</h5>
                        </div>
                        <div className="card-body">
                            <div className="row">
                                {webSDRStatuses.map((sdr) => (
                                    <div key={sdr.id} className="col-lg-3 col-md-4 col-sm-6">
                                        <div className="card bg-light border-0 mb-3">
                                            <div className="card-body">
                                                <div className="d-flex align-items-center justify-content-between mb-2">
                                                    <h6 className="mb-0">{sdr.city}</h6>
                                                    <div
                                                        className={`avtar avtar-xs ${sdr.status === 'online'
                                                                ? 'bg-light-success'
                                                                : 'bg-light-danger'
                                                            }`}
                                                    >
                                                        <i
                                                            className={`ph ${sdr.status === 'online'
                                                                    ? 'ph-radio-button'
                                                                    : 'ph-radio-button'
                                                                } f-18`}
                                                        ></i>
                                                    </div>
                                                </div>
                                                <p className="text-muted f-12 mb-2">{sdr.frequency}</p>
                                                <div className="d-flex align-items-center">
                                                    <div className="flex-grow-1 me-2">
                                                        <div className="progress" style={{ height: '5px' }}>
                                                            <div
                                                                className={`progress-bar ${sdr.status === 'online'
                                                                        ? 'bg-success'
                                                                        : 'bg-danger'
                                                                    }`}
                                                                role="progressbar"
                                                                style={{ width: `${sdr.signal}%` }}
                                                                aria-valuenow={sdr.signal}
                                                                aria-valuemin={0}
                                                                aria-valuemax={100}
                                                            ></div>
                                                        </div>
                                                    </div>
                                                    <span className="f-12 text-muted">{Math.round(sdr.signal)}%</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Dashboard;
