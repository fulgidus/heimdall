// @ts-nocheck
import React, { useEffect, useState } from 'react';
import { useSessionStore } from '../store/sessionStore';
import SessionEditModal from '../components/SessionEditModal';

const SessionHistory: React.FC = () => {
    const {
        sessions,
        analytics,
        currentPage,
        totalSessions,
        perPage,
        statusFilter,
        isLoading,
        error,
        fetchSessions,
        fetchAnalytics,
        setStatusFilter,
        clearError,
        updateSession,
    } = useSessionStore();

    const [selectedSession, setSelectedSession] = useState<number | null>(null);
    const [editingSession, setEditingSession] = useState<number | null>(null);

    useEffect(() => {
        const loadData = async () => {
            await fetchSessions({
                page: currentPage,
                per_page: perPage,
                status: statusFilter || undefined,
            });
            await fetchAnalytics();
        };
        loadData();

        // Auto-refresh every minute
        const interval = setInterval(() => {
            fetchSessions({
                page: currentPage,
                per_page: perPage,
                status: statusFilter || undefined,
            });
        }, 60000);

        return () => clearInterval(interval);
    }, [currentPage, perPage, statusFilter, fetchSessions, fetchAnalytics]);

    const handlePageChange = (newPage: number) => {
        fetchSessions({
            page: newPage,
            per_page: perPage,
            status: statusFilter || undefined,
        });
    };

    const handleSaveSession = async (
        sessionId: number,
        updates: {
            session_name?: string;
            notes?: string;
            approval_status?: 'pending' | 'approved' | 'rejected';
        }
    ) => {
        await updateSession(sessionId, updates);
    };

    const totalPages = Math.ceil(totalSessions / perPage);

    const selectedSessionData = sessions.find(s => s.id === selectedSession);

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
                                <li className="breadcrumb-item" aria-current="page">Session History</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">Session History</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" data-bs-dismiss="alert" onClick={clearError}></button>
                </div>
            )}

            {/* Analytics Cards */}
            {analytics && (
                <div className="row">
                    <div className="col-md-6 col-xl-3">
                        <div className="card">
                            <div className="card-body">
                                <div className="d-flex align-items-center">
                                    <div className="flex-shrink-0">
                                        <div className="avtar avtar-s bg-light-primary rounded">
                                            <i className="ph ph-database f-20"></i>
                                        </div>
                                    </div>
                                    <div className="flex-grow-1 ms-3">
                                        <h6 className="mb-0">Total Sessions</h6>
                                        <h4 className="mb-0">{analytics.total_sessions}</h4>
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
                                        <div className="avtar avtar-s bg-light-success rounded">
                                            <i className="ph ph-check-circle f-20"></i>
                                        </div>
                                    </div>
                                    <div className="flex-grow-1 ms-3">
                                        <h6 className="mb-0">Completed</h6>
                                        <h4 className="mb-0">{analytics.completed_sessions}</h4>
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
                                            <i className="ph ph-clock f-20"></i>
                                        </div>
                                    </div>
                                    <div className="flex-grow-1 ms-3">
                                        <h6 className="mb-0">Pending</h6>
                                        <h4 className="mb-0">{analytics.pending_sessions}</h4>
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
                                            <i className="ph ph-chart-line f-20"></i>
                                        </div>
                                    </div>
                                    <div className="flex-grow-1 ms-3">
                                        <h6 className="mb-0">Success Rate</h6>
                                        <h4 className="mb-0">{analytics.success_rate.toFixed(1)}%</h4>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Sessions Table */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header">
                            <div className="d-flex align-items-center justify-content-between">
                                <h5 className="mb-0">Recording Sessions</h5>
                                <div className="btn-group">
                                    <button
                                        className={`btn btn-sm ${statusFilter === null ? 'btn-primary' : 'btn-outline-primary'}`}
                                        onClick={() => setStatusFilter(null)}
                                    >
                                        All
                                    </button>
                                    <button
                                        className={`btn btn-sm ${statusFilter === 'completed' ? 'btn-primary' : 'btn-outline-primary'}`}
                                        onClick={() => setStatusFilter('completed')}
                                    >
                                        Completed
                                    </button>
                                    <button
                                        className={`btn btn-sm ${statusFilter === 'pending' ? 'btn-primary' : 'btn-outline-primary'}`}
                                        onClick={() => setStatusFilter('pending')}
                                    >
                                        Pending
                                    </button>
                                    <button
                                        className={`btn btn-sm ${statusFilter === 'failed' ? 'btn-primary' : 'btn-outline-primary'}`}
                                        onClick={() => setStatusFilter('failed')}
                                    >
                                        Failed
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="card-body">
                            {isLoading ? (
                                <div className="text-center py-5">
                                    <div className="spinner-border text-primary" role="status">
                                        <span className="visually-hidden">Loading...</span>
                                    </div>
                                    <p className="text-muted mt-2">Loading sessions...</p>
                                </div>
                            ) : sessions.length > 0 ? (
                                <>
                                    <div className="table-responsive">
                                        <table className="table table-hover mb-0">
                                            <thead>
                                                <tr>
                                                    <th>Session</th>
                                                    <th>Source</th>
                                                    <th>Started</th>
                                                    <th>Duration</th>
                                                    <th>Status</th>
                                                    <th>Approval</th>
                                                    <th>Measurements</th>
                                                    <th>Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sessions.map((session) => (
                                                    <tr
                                                        key={session.id}
                                                        className={selectedSession === session.id ? 'table-active' : ''}
                                                    >
                                                        <td>
                                                            <h6 className="mb-0">{session.session_name}</h6>
                                                            {session.notes && (
                                                                <p className="text-muted f-12 mb-0 text-truncate" style={{ maxWidth: '200px' }}>
                                                                    {session.notes}
                                                                </p>
                                                            )}
                                                        </td>
                                                        <td>
                                                            <div>
                                                                <div className="mb-1">{session.source_name}</div>
                                                                <span className="f-12 text-muted">
                                                                    {session.source_frequency ? (session.source_frequency / 1e6).toFixed(3) + ' MHz' : 'N/A'}
                                                                </span>
                                                            </div>
                                                        </td>
                                                        <td className="f-12">
                                                            {session.started_at ? new Date(session.started_at).toLocaleString() : 'Not started'}
                                                        </td>
                                                        <td>
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
                                                        <td>
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
                                                        <td>{session.measurements_count}</td>
                                                        <td>
                                                            <div className="btn-group">
                                                                <button
                                                                    className="btn btn-sm btn-link-primary"
                                                                    onClick={() => setSelectedSession(session.id)}
                                                                    title="View Details"
                                                                >
                                                                    <i className="ph ph-eye"></i>
                                                                </button>
                                                                <button
                                                                    className="btn btn-sm btn-link-warning"
                                                                    onClick={() => setEditingSession(session.id)}
                                                                    title="Edit Session"
                                                                >
                                                                    <i className="ph ph-pencil-simple"></i>
                                                                </button>
                                                                <button
                                                                    className="btn btn-sm btn-link-secondary"
                                                                    title="Download"
                                                                >
                                                                    <i className="ph ph-download-simple"></i>
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* Pagination */}
                                    {totalPages > 1 && (
                                        <div className="d-flex justify-content-between align-items-center mt-3">
                                            <span className="text-muted f-12">
                                                Showing {sessions.length} of {totalSessions} sessions
                                            </span>
                                            <nav>
                                                <ul className="pagination pagination-sm mb-0">
                                                    <li className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}>
                                                        <a
                                                            className="page-link"
                                                            href="#!"
                                                            onClick={(e) => {
                                                                e.preventDefault();
                                                                if (currentPage > 1) handlePageChange(currentPage - 1);
                                                            }}
                                                        >
                                                            Previous
                                                        </a>
                                                    </li>
                                                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => i + 1).map((page) => (
                                                        <li key={page} className={`page-item ${currentPage === page ? 'active' : ''}`}>
                                                            <a
                                                                className="page-link"
                                                                href="#!"
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    handlePageChange(page);
                                                                }}
                                                            >
                                                                {page}
                                                            </a>
                                                        </li>
                                                    ))}
                                                    <li className={`page-item ${currentPage === totalPages ? 'disabled' : ''}`}>
                                                        <a
                                                            className="page-link"
                                                            href="#!"
                                                            onClick={(e) => {
                                                                e.preventDefault();
                                                                if (currentPage < totalPages) handlePageChange(currentPage + 1);
                                                            }}
                                                        >
                                                            Next
                                                        </a>
                                                    </li>
                                                </ul>
                                            </nav>
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="text-center py-5">
                                    <i className="ph ph-folder-open f-40 text-muted mb-3"></i>
                                    <p className="text-muted mb-0">No recording sessions found</p>
                                    <a href="/recording" className="btn btn-primary mt-3">
                                        <i className="ph ph-plus-circle me-1"></i>
                                        Create New Session
                                    </a>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Session Details Modal/Panel */}
            {selectedSessionData && (
                <div className="row">
                    <div className="col-12">
                        <div className="card">
                            <div className="card-header d-flex align-items-center justify-content-between">
                                <h5 className="mb-0">Session Details</h5>
                                <div className="d-flex gap-2">
                                    <button
                                        className="btn btn-sm btn-warning"
                                        onClick={() => {
                                            setEditingSession(selectedSessionData.id);
                                            setSelectedSession(null);
                                        }}
                                    >
                                        <i className="ph ph-pencil-simple me-1"></i>
                                        Edit
                                    </button>
                                    <button
                                        className="btn btn-sm btn-link-secondary"
                                        onClick={() => setSelectedSession(null)}
                                    >
                                        <i className="ph ph-x"></i>
                                    </button>
                                </div>
                            </div>
                            <div className="card-body">
                                <div className="row">
                                    <div className="col-md-6">
                                        <h6>Session Information</h6>
                                        <table className="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <td className="text-muted">ID</td>
                                                    <td>{selectedSessionData.id}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Name</td>
                                                    <td>{selectedSessionData.session_name}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Source</td>
                                                    <td>{selectedSessionData.source_name}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Frequency</td>
                                                    <td>{selectedSessionData.source_frequency ? (selectedSessionData.source_frequency / 1e6).toFixed(3) + ' MHz' : 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Started</td>
                                                    <td>{selectedSessionData.started_at ? new Date(selectedSessionData.started_at).toLocaleString() : 'Not started'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Duration</td>
                                                    <td>
                                                        {selectedSessionData.duration_seconds
                                                            ? `${Math.round(selectedSessionData.duration_seconds / 60)} minutes`
                                                            : 'N/A'}
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                    <div className="col-md-6">
                                        <h6>Status & Results</h6>
                                        <table className="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <td className="text-muted">Status</td>
                                                    <td>
                                                        <span
                                                            className={`badge ${selectedSessionData.status === 'completed'
                                                                ? 'bg-light-success'
                                                                : selectedSessionData.status === 'failed'
                                                                    ? 'bg-light-danger'
                                                                    : 'bg-light-warning'
                                                                }`}
                                                        >
                                                            {selectedSessionData.status}
                                                        </span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Approval</td>
                                                    <td>
                                                        <span
                                                            className={`badge ${selectedSessionData.approval_status === 'approved'
                                                                ? 'bg-light-success'
                                                                : selectedSessionData.approval_status === 'rejected'
                                                                    ? 'bg-light-danger'
                                                                    : 'bg-light-warning'
                                                                }`}
                                                        >
                                                            {selectedSessionData.approval_status}
                                                        </span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Measurements</td>
                                                    <td>{selectedSessionData.measurements_count}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-muted">Created</td>
                                                    <td>{new Date(selectedSessionData.created_at).toLocaleString()}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                    {selectedSessionData.notes && (
                                        <div className="col-12">
                                            <h6>Notes</h6>
                                            <p className="text-muted">{selectedSessionData.notes}</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Edit Modal */}
            {editingSession && sessions.find(s => s.id === editingSession) && (
                <SessionEditModal
                    session={sessions.find(s => s.id === editingSession)!}
                    onSave={handleSaveSession}
                    onClose={() => setEditingSession(null)}
                />
            )}
        </>
    );
};

export default SessionHistory;
