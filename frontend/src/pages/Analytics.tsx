import React, { useEffect, useState } from 'react';
import { useDashboardStore, useWebSDRStore } from '../store';

const Analytics: React.FC = () => {
    const { data, metrics, fetchDashboardData } = useDashboardStore();
    const { websdrs, healthStatus } = useWebSDRStore();
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [timeRange, setTimeRange] = useState('7d');

    useEffect(() => {
        fetchDashboardData();
        const interval = setInterval(fetchDashboardData, 30000);
        return () => clearInterval(interval);
    }, [fetchDashboardData]);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await fetchDashboardData();
        setIsRefreshing(false);
    };

    // Calculate metrics from real data
    const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;
    const totalWebSDRs = websdrs.length || 7;
    const avgAccuracy = data.modelInfo?.accuracy ? `±${(data.modelInfo.accuracy * 100).toFixed(1)}m` : 'N/A';
    const totalPredictions = data.modelInfo?.predictions_total || 0;
    const successfulPredictions = data.modelInfo?.predictions_successful || 0;
    const successRate = totalPredictions > 0 ? ((successfulPredictions / totalPredictions) * 100).toFixed(1) : '0';

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item" aria-current="page">Analytics</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Analytics</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Time Range Selector */}
            <div className="row mb-3">
                <div className="col-12">
                    <div className="btn-group" role="group">
                        <button
                            type="button"
                            className={`btn ${timeRange === '24h' ? 'btn-primary' : 'btn-outline-primary'}`}
                            onClick={() => setTimeRange('24h')}
                        >
                            24 Hours
                        </button>
                        <button
                            type="button"
                            className={`btn ${timeRange === '7d' ? 'btn-primary' : 'btn-outline-primary'}`}
                            onClick={() => setTimeRange('7d')}
                        >
                            7 Days
                        </button>
                        <button
                            type="button"
                            className={`btn ${timeRange === '30d' ? 'btn-primary' : 'btn-outline-primary'}`}
                            onClick={() => setTimeRange('30d')}
                        >
                            30 Days
                        </button>
                        <button
                            type="button"
                            className="btn btn-outline-secondary ms-auto"
                            onClick={handleRefresh}
                            disabled={isRefreshing}
                        >
                            <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                            {isRefreshing ? ' Refreshing...' : ' Refresh'}
                        </button>
                    </div>
                </div>
            </div>

            {/* Key Metrics Cards */}
            <div className="row">
                {/* Total Localizations */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-primary rounded">
                                        <i className="ph ph-chart-line f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Total Predictions</h6>
                                    <h4 className="mb-0">{totalPredictions.toLocaleString()}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Success Rate</p>
                                    <span className="badge bg-light-success">{successRate}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Model Accuracy */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-success rounded">
                                        <i className="ph ph-target f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Model Accuracy</h6>
                                    <h4 className="mb-0">{avgAccuracy}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Confidence</p>
                                    <span className="badge bg-light-success">High</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Active Receivers */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-warning rounded">
                                        <i className="ph ph-radio-button f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Active Receivers</h6>
                                    <h4 className="mb-0">{onlineWebSDRs}/{totalWebSDRs}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Uptime</p>
                                    <span className="badge bg-light-warning">
                                        {Math.round((onlineWebSDRs / totalWebSDRs) * 100)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Uptime */}
                <div className="col-md-6 col-xl-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-info rounded">
                                        <i className="ph ph-activity f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">System Uptime</h6>
                                    <h4 className="mb-0">
                                        {metrics.systemUptime > 0
                                            ? `${(metrics.systemUptime / 3600).toFixed(1)}h`
                                            : '0h'}
                                    </h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Status</p>
                                    <span className="badge bg-light-success">Online</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="row">
                {/* Prediction Trends Chart */}
                <div className="col-lg-8">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Prediction Trends</h5>
                        </div>
                        <div className="card-body">
                            <div className="text-center py-5">
                                <i className="ph ph-chart-line-up f-40 text-primary mb-3"></i>
                                <p className="text-muted">
                                    Prediction trends chart will be displayed here.
                                    <br />
                                    Chart data: {totalPredictions} total predictions | {successfulPredictions} successful
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Accuracy Distribution */}
                <div className="col-lg-4">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">Accuracy Distribution</h5>
                        </div>
                        <div className="card-body">
                            <div className="text-center py-4">
                                <div className="mb-3">
                                    <i className="ph ph-chart-pie-slice f-40 text-success"></i>
                                </div>
                                <h3 className="mb-0">{avgAccuracy}</h3>
                                <p className="text-muted f-12 mb-0 mt-2">Average Accuracy</p>
                            </div>
                            <hr />
                            <div className="row text-center">
                                <div className="col-6">
                                    <h6 className="mb-0">{successfulPredictions}</h6>
                                    <p className="text-success f-12 mb-0">Successful</p>
                                </div>
                                <div className="col-6">
                                    <h6 className="mb-0">{totalPredictions - successfulPredictions}</h6>
                                    <p className="text-danger f-12 mb-0">Failed</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* WebSDR Performance */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="mb-0">WebSDR Performance</h5>
                        </div>
                        <div className="card-body">
                            <div className="table-responsive">
                                <table className="table table-hover mb-0">
                                    <thead>
                                        <tr>
                                            <th>Receiver</th>
                                            <th>Location</th>
                                            <th>Status</th>
                                            <th>Response Time</th>
                                            <th>Reliability</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {websdrs.map((sdr) => {
                                            const health = healthStatus[sdr.id];
                                            const isOnline = health?.status === 'online';
                                            const responseTime = health?.response_time_ms || 0;
                                            const reliability = isOnline ? Math.max(80, 100 - responseTime / 10) : 0;

                                            return (
                                                <tr key={sdr.id}>
                                                    <td>
                                                        <div className="d-flex align-items-center">
                                                            <div className={`avtar avtar-xs ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                <i className="ph ph-radio-button"></i>
                                                            </div>
                                                            <span className="ms-2">{sdr.name}</span>
                                                        </div>
                                                    </td>
                                                    <td>{sdr.location_name}</td>
                                                    <td>
                                                        <span className={`badge ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                            {isOnline ? 'Online' : 'Offline'}
                                                        </span>
                                                    </td>
                                                    <td>{responseTime > 0 ? `${responseTime}ms` : '-'}</td>
                                                    <td>
                                                        <div className="d-flex align-items-center">
                                                            <div className="progress flex-grow-1 me-2" style={{ height: '6px' }}>
                                                                <div
                                                                    className={`progress-bar ${isOnline ? 'bg-success' : 'bg-danger'}`}
                                                                    role="progressbar"
                                                                    style={{ width: `${reliability}%` }}
                                                                ></div>
                                                            </div>
                                                            <span className="f-12">{Math.round(reliability)}%</span>
                                                        </div>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                        {websdrs.length === 0 && (
                                            <tr>
                                                <td colSpan={5} className="text-center py-4">
                                                    <i className="ph ph-warning-circle f-40 text-muted mb-2"></i>
                                                    <p className="text-muted mb-0">No WebSDR data available</p>
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Analytics;
