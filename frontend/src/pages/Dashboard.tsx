import React, { useEffect, useState } from 'react';
import { useDashboardStore, useWebSDRStore } from '../store';
import { ConnectionState } from '../lib/websocket';
import { ServiceHealthSkeleton, WebSDRCardSkeleton } from '../components';

const Dashboard: React.FC = () => {
    const {
        metrics,
        data,
        isLoading,
        error,
        fetchDashboardData,
        lastUpdate,
        wsConnectionState,
        wsEnabled,
        connectWebSocket,
        disconnectWebSocket,
    } = useDashboardStore();
    const { websdrs, healthStatus } = useWebSDRStore();
    const [isRefreshing, setIsRefreshing] = useState(false);

    useEffect(() => {
        // Fetch initial data
        fetchDashboardData();

        // Try to connect WebSocket for real-time updates
        connectWebSocket();

        // Setup polling fallback (only if WebSocket disabled)
        const interval = setInterval(() => {
            if (!wsEnabled || wsConnectionState !== ConnectionState.CONNECTED) {
                fetchDashboardData();
            }
        }, 30000); // Poll every 30 seconds as fallback

        return () => {
            clearInterval(interval);
            disconnectWebSocket();
        };
    }, [fetchDashboardData, connectWebSocket, disconnectWebSocket, wsEnabled, wsConnectionState]);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await fetchDashboardData();
        setIsRefreshing(false);
    };

    const handleReconnect = async () => {
        await connectWebSocket();
    };

    // Get connection status display
    const getConnectionStatus = () => {
        switch (wsConnectionState) {
            case ConnectionState.CONNECTED:
                return { text: 'Connected', color: 'success', icon: 'ph-check-circle' };
            case ConnectionState.CONNECTING:
                return { text: 'Connecting...', color: 'warning', icon: 'ph-circle-notch' };
            case ConnectionState.RECONNECTING:
                return { text: 'Reconnecting...', color: 'warning', icon: 'ph-arrows-clockwise' };
            case ConnectionState.DISCONNECTED:
            default:
                return { text: wsEnabled ? 'Disconnected' : 'Polling Mode', color: 'danger', icon: 'ph-x-circle' };
        }
    };

    const connectionStatus = getConnectionStatus();

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
            <nav className="page-header" aria-label="Breadcrumb">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ol className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/">Home</a></li>
                                <li className="breadcrumb-item" aria-current="page">Dashboard</li>
                            </ol>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title d-flex align-items-center justify-content-between">
                                <h2 className="mb-0">Dashboard</h2>
                                {/* Connection Status Indicator */}
                                <div className="d-flex align-items-center gap-2">
                                    <span className={`badge bg-light-${connectionStatus.color} d-flex align-items-center gap-1`}>
                                        <i className={`ph ${connectionStatus.icon} ${wsConnectionState === ConnectionState.CONNECTING || wsConnectionState === ConnectionState.RECONNECTING ? 'spin' : ''}`}></i>
                                        {connectionStatus.text}
                                    </span>
                                    {wsConnectionState === ConnectionState.DISCONNECTED && wsEnabled && (
                                        <button
                                            className="btn btn-sm btn-outline-primary"
                                            onClick={handleReconnect}
                                            title="Reconnect WebSocket"
                                        >
                                            <i className="ph ph-arrows-clockwise"></i>
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Error Display */}
            {error && (
                <div 
                    className="alert alert-danger alert-dismissible fade show" 
                    role="alert"
                    aria-live="assertive"
                    aria-atomic="true"
                >
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close error message"></button>
                </div>
            )}

            {/* Connection Status Indicator */}
            {isLoading && (
                <div className="alert alert-info d-flex align-items-center" role="status">
                    <div className="spinner-border spinner-border-sm me-2" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                    <span>Connecting to services...</span>
                </div>
            )}

            {/* Stats Cards Row */}
            <section aria-label="System metrics" className="row">
                {/* Active WebSDR Card */}
                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <h2 className="mb-4 h6">Active WebSDR</h2>
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
                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <h2 className="mb-4 h6">Signal Detections</h2>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-chart-line-up text-warning f-30 m-r-10" aria-hidden="true"></i>
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
                                    aria-label="Signal detection progress: 75%"
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Uptime Card */}
                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <h2 className="mb-4 h6">System Uptime</h2>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-activity text-success f-30 m-r-10" aria-hidden="true"></i>
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
                                    aria-label="System uptime: 90%"
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Model Accuracy Card */}
                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <h2 className="mb-4 h6">Model Accuracy</h2>
                            <div className="row d-flex align-items-center">
                                <div className="col-9">
                                    <h3 className="f-w-300 d-flex align-items-center m-b-0">
                                        <i className="ph ph-target text-primary f-30 m-r-10" aria-hidden="true"></i>
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
                                    aria-label={`Model accuracy: ${data.modelInfo?.accuracy ? (data.modelInfo.accuracy * 100).toFixed(1) : 0}%`}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Main Content Row */}
            <div className="row">
                {/* System Activity */}
                <section className="col-12 col-lg-8 mb-3" aria-labelledby="system-activity-heading">
                    <div className="card table-card">
                        <div className="card-header d-flex align-items-center justify-content-between">
                            <h2 id="system-activity-heading" className="mb-0 h5">System Activity</h2>
                            <button
                                className="btn btn-sm btn-link-primary touch-target"
                                onClick={handleRefresh}
                                disabled={isRefreshing}
                                aria-label={isRefreshing ? 'Refreshing system activity' : 'Refresh system activity'}
                            >
                                <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`} aria-hidden="true"></i>
                                <span className="d-none d-sm-inline">{isRefreshing ? ' Refreshing...' : ' Refresh'}</span>
                            </button>
                        </div>
                        <div className="card-body">
                            <div className="table-responsive">
                                <table className="table table-hover">
                                    <thead>
                                        <tr>
                                            <th scope="col">Status</th>
                                            <th scope="col">Activity</th>
                                            <th scope="col" className="d-none d-md-table-cell">Details</th>
                                            <th scope="col" className="d-none d-lg-table-cell">Timestamp</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-success" role="img" aria-label="Success">
                                                    <i className="ph ph-check-circle" aria-hidden="true"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h3 className="mb-0 h6">System Status</h3>
                                                <p className="text-muted f-12 mb-0">WebSDR Network</p>
                                            </td>
                                            <td className="d-none d-md-table-cell">
                                                {onlineWebSDRs} of {totalWebSDRs} receivers online
                                            </td>
                                            <td className="text-muted d-none d-lg-table-cell">
                                                <time dateTime={lastUpdate ? new Date(lastUpdate).toISOString() : ''}>
                                                    {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Just now'}
                                                </time>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span 
                                                    className={`badge ${data.modelInfo ? 'bg-light-primary' : 'bg-light-warning'}`}
                                                    role="img" 
                                                    aria-label={data.modelInfo ? 'Active' : 'Warning'}
                                                >
                                                    <i className={`ph ${data.modelInfo ? 'ph-brain' : 'ph-warning-circle'}`} aria-hidden="true"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h3 className="mb-0 h6">ML Model</h3>
                                                <p className="text-muted f-12 mb-0">Inference Engine</p>
                                            </td>
                                            <td className="d-none d-md-table-cell">
                                                {data.modelInfo
                                                    ? `Version ${data.modelInfo.active_version} - ${data.modelInfo.health_status}`
                                                    : 'Initializing...'}
                                            </td>
                                            <td className="text-muted d-none d-lg-table-cell">
                                                {data.modelInfo?.loaded_at
                                                    ? new Date(data.modelInfo.loaded_at).toLocaleTimeString()
                                                    : '-'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-success" role="img" aria-label="Success">
                                                    <i className="ph ph-cpu" aria-hidden="true"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h3 className="mb-0 h6">Services Health</h3>
                                                <p className="text-muted f-12 mb-0">Microservices</p>
                                            </td>
                                            <td className="d-none d-md-table-cell">
                                                {Object.values(data.servicesHealth).filter(s => s.status === 'healthy').length} of{' '}
                                                {Object.keys(data.servicesHealth).length} services healthy
                                            </td>
                                            <td className="text-muted d-none d-lg-table-cell">
                                                {lastUpdate ? 'Updated' : 'Checking...'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>
                                                <span className="badge bg-light-info" role="img" aria-label="Info">
                                                    <i className="ph ph-chart-bar" aria-hidden="true"></i>
                                                </span>
                                            </td>
                                            <td>
                                                <h3 className="mb-0 h6">Predictions</h3>
                                                <p className="text-muted f-12 mb-0">Total Count</p>
                                            </td>
                                            <td className="d-none d-md-table-cell">
                                                {data.modelInfo
                                                    ? `${data.modelInfo.predictions_total} total (${data.modelInfo.predictions_successful} successful)`
                                                    : 'No predictions yet'}
                                            </td>
                                            <td className="text-muted d-none d-lg-table-cell">
                                                {data.modelInfo?.last_prediction_at || '-'}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Services Status */}
                <section className="col-12 col-lg-4 mb-3" aria-labelledby="services-status-heading">
                    <div 
                        className="card"
                        aria-live="polite"
                        aria-atomic="true"
                    >
                        <div className="card-header">
                            <h2 id="services-status-heading" className="mb-0 h5">Services Status</h2>
                        </div>
                        <div className="card-body">
                            {isLoading && Object.entries(data.servicesHealth).length === 0 ? (
                                <ServiceHealthSkeleton />
                            ) : Object.entries(data.servicesHealth).length > 0 ? (
                                <ul className="list-group list-group-flush">
                                    {Object.entries(data.servicesHealth).map(([name, health]) => (
                                        <li key={name} className="list-group-item px-0" role="listitem">
                                            <div className="d-flex align-items-center justify-content-between">
                                                <div className="flex-grow-1">
                                                    <h6 className="mb-0 text-capitalize">
                                                        {name.replace(/-/g, ' ')}
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
                                                        role="status"
                                                        aria-label={`Service ${name} is ${health.status}`}
                                                    >
                                                        {health.status}
                                                    </span>
                                                </div>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <div className="text-center py-4" role="status">
                                    <i className="ph ph-warning-circle f-40 text-muted mb-3" aria-hidden="true"></i>
                                    <p className="text-muted mb-0">
                                        {error ? 'Failed to load services' : 'No service data available'}
                                    </p>
                                    {error && (
                                        <button
                                            className="btn btn-sm btn-link-primary mt-2"
                                            onClick={handleRefresh}
                                        >
                                            <i className="ph ph-arrow-clockwise"></i> Retry
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </section>
            </div>

            {/* WebSDR Network Status */}
            <section className="row" aria-labelledby="websdr-network-heading">
                <div className="col-12">
                    <div 
                        className="card"
                        aria-live="polite"
                        aria-atomic="false"
                    >
                        <div className="card-header">
                            <h2 id="websdr-network-heading" className="mb-0 h5">WebSDR Network Status</h2>
                        </div>
                        <div className="card-body">
                            <div className="row">
                                {isLoading && webSDRStatuses.length === 0 ? (
                                    <WebSDRCardSkeleton />
                                ) : (
                                    webSDRStatuses.map((sdr) => (
                                        <div key={sdr.id} className="col-12 col-sm-6 col-md-4 col-lg-3 mb-3">
                                            <div className="card bg-light border-0">
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
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </>
    );
};

export default Dashboard;
