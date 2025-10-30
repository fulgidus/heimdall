import React, { useEffect } from 'react';
import { useSessionStore } from '@/store/sessionStore';
import { Link } from 'react-router-dom';

interface RecentActivityWidgetProps {
    widgetId: string;
}

export const RecentActivityWidget: React.FC<RecentActivityWidgetProps> = () => {
    const { sessions, isLoading, fetchSessions } = useSessionStore();

    useEffect(() => {
        fetchSessions({ page: 1, per_page: 5 });
    }, [fetchSessions]);

    const recentSessions = sessions.slice(0, 5);

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'success';
            case 'recording': return 'primary';
            case 'processing': return 'info';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed': return 'ph-check-circle';
            case 'recording': return 'ph-record';
            case 'processing': return 'ph-spinner';
            case 'failed': return 'ph-x-circle';
            default: return 'ph-clock';
        }
    };

    return (
        <div className="widget-content">
            {isLoading ? (
                <div className="text-center py-4">
                    <div className="spinner-border spinner-border-sm" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                </div>
            ) : recentSessions.length > 0 ? (
                <>
                    <div className="list-group list-group-flush">
                        {recentSessions.map((session) => (
                            <div key={session.id} className="list-group-item px-0 py-3">
                                <div className="d-flex align-items-start justify-content-between">
                                    <div className="flex-grow-1">
                                        <div className="d-flex align-items-center gap-2 mb-1">
                                            <i className={`ph ${getStatusIcon(session.status)}`} />
                                            <Link 
                                                to={`/sessions/${session.id}`}
                                                className="fw-medium text-decoration-none"
                                            >
                                                {session.session_name}
                                            </Link>
                                        </div>
                                        <div className="small text-muted">
                                            {session.source_frequency ? (session.source_frequency / 1e6).toFixed(2) : '0.00'} MHz • {session.duration_seconds}s
                                        </div>
                                        <div className="small text-muted">
                                            {new Date(session.created_at).toLocaleString()}
                                        </div>
                                    </div>
                                    <span className={`badge bg-light-${getStatusColor(session.status)}`}>
                                        {session.status}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="text-center mt-3">
                        <Link to="/sessions" className="btn btn-sm btn-link-primary">
                            View All Sessions <i className="ph ph-arrow-right" />
                        </Link>
                    </div>
                </>
            ) : (
                <div className="text-center py-5 text-muted">
                    <i className="ph ph-clock-countdown fs-1 mb-2" />
                    <p className="mb-0">No recent activity</p>
                </div>
            )}
        </div>
    );
};
