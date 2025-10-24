import React, { useEffect, useState } from 'react';
import { useWebSDRStore } from '../store/websdrStore';

const WebSDRManagement: React.FC = () => {
    const {
        websdrs,
        healthStatus,
        isLoading,
        error,
        fetchWebSDRs,
        checkHealth,
        refreshAll,
        lastHealthCheck,
    } = useWebSDRStore();

    const [isRefreshing, setIsRefreshing] = useState(false);
    const [selectedWebSDR, setSelectedWebSDR] = useState<number | null>(null);

    useEffect(() => {
        // Initial data load
        const loadData = async () => {
            await fetchWebSDRs();
            await checkHealth();
        };
        loadData();

        // Auto-refresh health every 30 seconds
        const interval = setInterval(() => {
            checkHealth();
        }, 30000);

        return () => clearInterval(interval);
    }, [fetchWebSDRs, checkHealth]);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await refreshAll();
        setIsRefreshing(false);
    };

    // Calculate statistics
    const onlineCount = Object.values(healthStatus).filter(h => h.status === 'online').length;
    const totalCount = websdrs.length;
    const activeCount = websdrs.filter(w => w.is_active).length;
    const avgResponseTime = Object.values(healthStatus).reduce((sum, h) => sum + (h.response_time_ms || 0), 0) / (Object.keys(healthStatus).length || 1);

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
                                <li className="breadcrumb-item" aria-current="page">WebSDR Management</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">WebSDR Management</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" data-bs-dismiss="alert"></button>
                </div>
            )}

            {/* Statistics Cards */}
            <div className="row">
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-success rounded">
                                        <i className="ph ph-radio-button f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Online Receivers</h6>
                                    <h4 className="mb-0">{onlineCount}/{totalCount}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Availability</p>
                                    <span className="badge bg-light-success">
                                        {totalCount > 0 ? Math.round((onlineCount / totalCount) * 100) : 0}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-primary rounded">
                                        <i className="ph ph-check-circle f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Active Receivers</h6>
                                    <h4 className="mb-0">{activeCount}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Total Configured</p>
                                    <span className="badge bg-light-primary">{totalCount}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-warning rounded">
                                        <i className="ph ph-timer f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Avg Response</h6>
                                    <h4 className="mb-0">{Math.round(avgResponseTime)}ms</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Performance</p>
                                    <span className={`badge ${avgResponseTime < 200 ? 'bg-light-success' : avgResponseTime < 500 ? 'bg-light-warning' : 'bg-light-danger'}`}>
                                        {avgResponseTime < 200 ? 'Excellent' : avgResponseTime < 500 ? 'Good' : 'Slow'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-info rounded">
                                        <i className="ph ph-clock f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Last Check</h6>
                                    <h4 className="mb-0 f-14">
                                        {lastHealthCheck 
                                            ? new Date(lastHealthCheck).toLocaleTimeString()
                                            : 'Never'
                                        }
                                    </h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <button
                                    className="btn btn-sm btn-link-primary w-100"
                                    onClick={handleRefresh}
                                    disabled={isRefreshing}
                                >
                                    <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                                    {isRefreshing ? ' Checking...' : ' Refresh Now'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* WebSDR List */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header d-flex align-items-center justify-content-between">
                            <h5 className="mb-0">Configured WebSDR Receivers</h5>
                            <div className="btn-group">
                                <button className="btn btn-sm btn-outline-primary" onClick={handleRefresh} disabled={isRefreshing}>
                                    <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                                    Refresh
                                </button>
                            </div>
                        </div>
                        <div className="card-body">
                            {isLoading ? (
                                <div className="text-center py-5">
                                    <div className="spinner-border text-primary" role="status">
                                        <span className="visually-hidden">Loading...</span>
                                    </div>
                                    <p className="text-muted mt-2">Loading WebSDR receivers...</p>
                                </div>
                            ) : websdrs.length > 0 ? (
                                <div className="table-responsive">
                                    <table className="table table-hover mb-0">
                                        <thead>
                                            <tr>
                                                <th>Status</th>
                                                <th>Name</th>
                                                <th>Location</th>
                                                <th>Coordinates</th>
                                                <th>URL</th>
                                                <th>Response Time</th>
                                                <th>Active</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {websdrs.map((websdr) => {
                                                const health = healthStatus[websdr.id];
                                                const isOnline = health?.status === 'online';
                                                const responseTime = health?.response_time_ms || 0;

                                                return (
                                                    <tr key={websdr.id}>
                                                        <td>
                                                            <span className={`badge ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                <i className={`ph ${isOnline ? 'ph-check-circle' : 'ph-x-circle'}`}></i>
                                                                {isOnline ? ' Online' : ' Offline'}
                                                            </span>
                                                        </td>
                                                        <td>
                                                            <div className="d-flex align-items-center">
                                                                <div className={`avtar avtar-xs ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                    <i className="ph ph-radio-button"></i>
                                                                </div>
                                                                <span className="ms-2">{websdr.name}</span>
                                                            </div>
                                                        </td>
                                                        <td>{websdr.location_name}</td>
                                                        <td className="f-12">
                                                            {websdr.latitude.toFixed(4)}, {websdr.longitude.toFixed(4)}
                                                        </td>
                                                        <td>
                                                            <a 
                                                                href={websdr.url} 
                                                                target="_blank" 
                                                                rel="noopener noreferrer"
                                                                className="link-primary"
                                                            >
                                                                <i className="ph ph-arrow-square-out"></i>
                                                            </a>
                                                        </td>
                                                        <td>
                                                            {responseTime > 0 ? (
                                                                <span className={`badge ${responseTime < 200 ? 'bg-light-success' : responseTime < 500 ? 'bg-light-warning' : 'bg-light-danger'}`}>
                                                                    {responseTime}ms
                                                                </span>
                                                            ) : (
                                                                <span className="text-muted">-</span>
                                                            )}
                                                        </td>
                                                        <td>
                                                            <span className={`badge ${websdr.is_active ? 'bg-light-primary' : 'bg-light-secondary'}`}>
                                                                {websdr.is_active ? 'Yes' : 'No'}
                                                            </span>
                                                        </td>
                                                        <td>
                                                            <div className="btn-group">
                                                                <button
                                                                    className="btn btn-sm btn-link-secondary"
                                                                    onClick={() => setSelectedWebSDR(websdr.id)}
                                                                    title="View Details"
                                                                >
                                                                    <i className="ph ph-eye"></i>
                                                                </button>
                                                                <button
                                                                    className="btn btn-sm btn-link-primary"
                                                                    title="Edit"
                                                                >
                                                                    <i className="ph ph-pencil-simple"></i>
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <div className="text-center py-5">
                                    <i className="ph ph-warning-circle f-40 text-warning mb-3"></i>
                                    <p className="text-muted mb-0">No WebSDR receivers configured</p>
                                    <button className="btn btn-primary mt-3">
                                        <i className="ph ph-plus-circle"></i>
                                        Add WebSDR Receiver
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Selected WebSDR Details */}
            {selectedWebSDR && websdrs.find(w => w.id === selectedWebSDR) && (
                <div className="row">
                    <div className="col-12">
                        <div className="card">
                            <div className="card-header d-flex align-items-center justify-content-between">
                                <h5 className="mb-0">WebSDR Details</h5>
                                <button 
                                    className="btn btn-sm btn-link-secondary"
                                    onClick={() => setSelectedWebSDR(null)}
                                >
                                    <i className="ph ph-x"></i>
                                </button>
                            </div>
                            <div className="card-body">
                                {(() => {
                                    const websdr = websdrs.find(w => w.id === selectedWebSDR);
                                    const health = healthStatus[selectedWebSDR];
                                    if (!websdr) return null;

                                    return (
                                        <div className="row">
                                            <div className="col-md-6">
                                                <h6>General Information</h6>
                                                <table className="table table-sm">
                                                    <tbody>
                                                        <tr>
                                                            <td className="text-muted">ID</td>
                                                            <td>{websdr.id}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Name</td>
                                                            <td>{websdr.name}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Location</td>
                                                            <td>{websdr.location_name}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Latitude</td>
                                                            <td>{websdr.latitude.toFixed(6)}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Longitude</td>
                                                            <td>{websdr.longitude.toFixed(6)}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">URL</td>
                                                            <td>
                                                                <a href={websdr.url} target="_blank" rel="noopener noreferrer">
                                                                    {websdr.url}
                                                                </a>
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Active</td>
                                                            <td>
                                                                <span className={`badge ${websdr.is_active ? 'bg-light-success' : 'bg-light-secondary'}`}>
                                                                    {websdr.is_active ? 'Yes' : 'No'}
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                            <div className="col-md-6">
                                                <h6>Health Status</h6>
                                                {health ? (
                                                    <table className="table table-sm">
                                                        <tbody>
                                                            <tr>
                                                                <td className="text-muted">Status</td>
                                                                <td>
                                                                    <span className={`badge ${health.status === 'online' ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                        {health.status}
                                                                    </span>
                                                                </td>
                                                            </tr>
                                                            <tr>
                                                                <td className="text-muted">Response Time</td>
                                                                <td>{health.response_time_ms || 0}ms</td>
                                                            </tr>
                                                            <tr>
                                                                <td className="text-muted">Last Check</td>
                                                                <td>{health.last_check || 'Never'}</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                ) : (
                                                    <p className="text-muted">No health data available</p>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })()}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default WebSDRManagement;
