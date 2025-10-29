import React, { useEffect, useState } from 'react';
import { useSessionStore } from '../store/sessionStore';

const DataIngestion: React.FC = () => {
    const {
        knownSources,
        sessions,
        analytics,
        isLoading,
        error,
        fetchKnownSources,
        fetchSessions,
        fetchAnalytics,
        clearError,
    } = useSessionStore();

    const [activeTab, setActiveTab] = useState<'sources' | 'sessions'>('sources');

    useEffect(() => {
        // Load data on mount
        const loadData = async () => {
            await Promise.all([
                fetchKnownSources(),
                fetchSessions(),
                fetchAnalytics(),
            ]);
        };
        loadData();

        // Auto-refresh every minute
        const interval = setInterval(() => {
            fetchSessions();
            fetchAnalytics();
        }, 60000);

        return () => clearInterval(interval);
    }, [fetchKnownSources, fetchSessions, fetchAnalytics]);

    const handleRefresh = async () => {
        await Promise.all([
            fetchKnownSources(),
            fetchSessions(),
            fetchAnalytics(),
        ]);
    };

    // Calculate statistics
    const totalSources = knownSources.length;
    const validatedSources = knownSources.filter(s => s.is_validated).length;
    const pendingSessions = sessions.filter(s => s.status === 'pending').length;
    const completedSessions = sessions.filter(s => s.status === 'completed').length;
    const successRate = analytics?.success_rate || 0;

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
                                <li className="breadcrumb-item" aria-current="page">Data Ingestion</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Data Ingestion</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" onClick={clearError}></button>
                </div>
            )}

            {/* Statistics Cards */}
            <div className="row">
                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-primary rounded">
                                        <i className="ph ph-database f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Known Sources</h6>
                                    <h4 className="mb-0">{totalSources}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Validated</p>
                                    <span className="badge bg-light-success">{validatedSources}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-warning rounded">
                                        <i className="ph ph-clock f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Pending Sessions</h6>
                                    <h4 className="mb-0">{pendingSessions}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Status</p>
                                    <span className="badge bg-light-warning">Awaiting</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-success rounded">
                                        <i className="ph ph-check-circle f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Completed</h6>
                                    <h4 className="mb-0">{completedSessions}</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Total Sessions</p>
                                    <span className="badge bg-light-primary">{sessions.length}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-12 col-sm-6 col-md-6 col-xl-3 mb-3">
                    <div className="card">
                        <div className="card-body">
                            <div className="d-flex align-items-center">
                                <div className="flex-shrink-0">
                                    <div className="avtar avtar-s bg-light-info rounded">
                                        <i className="ph ph-chart-line f-20"></i>
                                    </div>
                                </div>
                                <div className="flex-grow-1 ms-3">
                                    <h6 className="mb-0">Success Rate</h6>
                                    <h4 className="mb-0">{successRate.toFixed(1)}%</h4>
                                </div>
                            </div>
                            <div className="bg-body p-2 mt-2 rounded">
                                <div className="d-flex align-items-center justify-content-between">
                                    <p className="text-muted mb-0 f-12">Quality</p>
                                    <span className={`badge ${successRate > 80 ? 'bg-light-success' : successRate > 60 ? 'bg-light-warning' : 'bg-light-danger'}`}>
                                        {successRate > 80 ? 'Excellent' : successRate > 60 ? 'Good' : 'Poor'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Tabs Navigation */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header">
                            <ul className="nav nav-tabs card-header-tabs" role="tablist">
                                <li className="nav-item">
                                    <a
                                        className={`nav-link ${activeTab === 'sources' ? 'active' : ''}`}
                                        href="#!"
                                        onClick={(e) => {
                                            e.preventDefault();
                                            setActiveTab('sources');
                                        }}
                                    >
                                        <i className="ph ph-radio-button me-2"></i>
                                        Known Sources ({totalSources})
                                    </a>
                                </li>
                                <li className="nav-item">
                                    <a
                                        className={`nav-link ${activeTab === 'sessions' ? 'active' : ''}`}
                                        href="#!"
                                        onClick={(e) => {
                                            e.preventDefault();
                                            setActiveTab('sessions');
                                        }}
                                    >
                                        <i className="ph ph-record me-2"></i>
                                        Recording Sessions ({sessions.length})
                                    </a>
                                </li>
                            </ul>
                        </div>
                        <div className="card-body">
                            {/* Known Sources Tab */}
                            {activeTab === 'sources' && (
                                <>
                                    <div className="d-flex justify-content-between align-items-center mb-3">
                                        <h5 className="mb-0">Known RF Sources</h5>
                                        <div className="btn-group">
                                            <button
                                                className="btn btn-sm btn-outline-secondary"
                                                onClick={handleRefresh}
                                            >
                                                <i className="ph ph-arrows-clockwise"></i>
                                            </button>
                                        </div>
                                    </div>

                                    {isLoading ? (
                                        <div className="text-center py-5">
                                            <div className="spinner-border text-primary" role="status">
                                                <span className="visually-hidden">Loading...</span>
                                            </div>
                                            <p className="text-muted mt-2">Loading sources...</p>
                                        </div>
                                    ) : knownSources.length > 0 ? (
                                        <div className="table-responsive">
                                            <table className="table table-hover mb-0">
                                                <thead>
                                                    <tr>
                                                        <th>Name</th>
                                                        <th>Frequency</th>
                                                        <th className="d-none d-md-table-cell">Location</th>
                                                        <th className="d-none d-lg-table-cell">Coordinates</th>
                                                        <th className="d-none d-lg-table-cell">Power</th>
                                                        <th>Validated</th>
                                                        <th>Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {knownSources.map((source) => (
                                                        <tr key={source.id}>
                                                            <td>
                                                                <div className="d-flex align-items-center">
                                                                    <div className={`avtar avtar-xs ${source.is_validated ? 'bg-light-success' : 'bg-light-warning'}`}>
                                                                        <i className="ph ph-radio-button"></i>
                                                                    </div>
                                                                    <div className="ms-2">
                                                                        <h6 className="mb-0">{source.name}</h6>
                                                                        {source.description && (
                                                                            <p className="text-muted f-12 mb-0">{source.description}</p>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            </td>
                                                            <td>{(source.frequency_hz / 1e6).toFixed(3)} MHz</td>
                                                            <td className="d-none d-md-table-cell">{source.source_type || '-'}</td>
                                                            <td className="f-12 d-none d-lg-table-cell">
                                                                {source.latitude.toFixed(4)}, {source.longitude.toFixed(4)}
                                                            </td>
                                                            <td className="d-none d-lg-table-cell">{source.power_dbm ? `${source.power_dbm} dBm` : '-'}</td>
                                                            <td>
                                                                <span className={`badge ${source.is_validated ? 'bg-light-success' : 'bg-light-warning'}`}>
                                                                    {source.is_validated ? 'Yes' : 'Pending'}
                                                                </span>
                                                            </td>
                                                            <td>
                                                                <div className="btn-group">
                                                                    <button className="btn btn-sm btn-link-primary touch-target" title="View">
                                                                        <i className="ph ph-eye"></i>
                                                                    </button>
                                                                    <button className="btn btn-sm btn-link-secondary touch-target" title="Edit">
                                                                        <i className="ph ph-pencil-simple"></i>
                                                                    </button>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    ) : (
                                        <div className="text-center py-5">
                                            <i className="ph ph-radio-button f-40 text-muted mb-3"></i>
                                            <p className="text-muted mb-0">No known sources configured</p>
                                        </div>
                                    )}
                                </>
                            )}

                            {/* Recording Sessions Tab */}
                            {activeTab === 'sessions' && (
                                <>
                                    <div className="d-flex justify-content-between align-items-center mb-3">
                                        <h5 className="mb-0">Recording Sessions</h5>
                                        <button className="btn btn-sm btn-outline-secondary" onClick={handleRefresh}>
                                            <i className="ph ph-arrows-clockwise"></i>
                                        </button>
                                    </div>

                                    {isLoading ? (
                                        <div className="text-center py-5">
                                            <div className="spinner-border text-primary" role="status">
                                                <span className="visually-hidden">Loading...</span>
                                            </div>
                                            <p className="text-muted mt-2">Loading sessions...</p>
                                        </div>
                                    ) : sessions.length > 0 ? (
                                        <div className="table-responsive">
                                            <table className="table table-hover mb-0">
                                                <thead>
                                                    <tr>
                                                        <th>Session</th>
                                                        <th className="d-none d-sm-table-cell">Source</th>
                                                        <th className="d-none d-md-table-cell">Started</th>
                                                        <th className="d-none d-lg-table-cell">Duration</th>
                                                        <th>Status</th>
                                                        <th className="d-none d-md-table-cell">Approval</th>
                                                        <th className="d-none d-lg-table-cell">Measurements</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {sessions.map((session) => (
                                                        <tr key={session.id}>
                                                            <td>
                                                                <h6 className="mb-0">{session.session_name}</h6>
                                                                {session.notes && (
                                                                    <p className="text-muted f-12 mb-0">{session.notes}</p>
                                                                )}
                                                            </td>
                                                            <td className="d-none d-sm-table-cell">
                                                                <div>
                                                                    <div className="mb-1">{session.source_name}</div>
                                                                    <span className="f-12 text-muted">
                                                                        {session.source_frequency ? (session.source_frequency / 1e6).toFixed(3) + ' MHz' : 'N/A'}
                                                                    </span>
                                                                </div>
                                                            </td>
                                                            <td className="f-12 d-none d-md-table-cell">
                                                                {session.started_at ? new Date(session.started_at).toLocaleString() : 'Not started'}
                                                            </td>
                                                            <td className="d-none d-lg-table-cell">
                                                                {session.duration_seconds
                                                                    ? `${Math.round(session.duration_seconds / 60)}min`
                                                                    : '-'}
                                                            </td>
                                                            <td>
                                                                <span
                                                                    className={`badge ${session.status === 'completed'
                                                                        ? 'bg-light-success'
                                                                        : session.status === 'in_progress'
                                                                            ? 'bg-light-primary'
                                                                            : session.status === 'failed'
                                                                                ? 'bg-light-danger'
                                                                                : 'bg-light-warning'
                                                                        }`}
                                                                >
                                                                    {session.status}
                                                                </span>
                                                            </td>
                                                            <td className="d-none d-md-table-cell">
                                                                <span
                                                                    className={`badge ${session.approval_status === 'approved'
                                                                        ? 'bg-light-success'
                                                                        : session.approval_status === 'rejected'
                                                                            ? 'bg-light-danger'
                                                                            : 'bg-light-warning'
                                                                        }`}
                                                                >
                                                                    {session.approval_status}
                                                                </span>
                                                            </td>
                                                            <td className="d-none d-lg-table-cell">{session.measurements_count}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    ) : (
                                        <div className="text-center py-5">
                                            <i className="ph ph-record f-40 text-muted mb-3"></i>
                                            <p className="text-muted mb-0">No recording sessions yet</p>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default DataIngestion;
