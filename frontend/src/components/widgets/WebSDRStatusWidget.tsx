import React, { useEffect } from 'react';
import { useWebSDRStore } from '@/store';

interface WebSDRStatusWidgetProps {
    widgetId: string;
}

export const WebSDRStatusWidget: React.FC<WebSDRStatusWidgetProps> = () => {
    const { websdrs, healthStatus, fetchWebSDRs, checkHealth, isLoading } = useWebSDRStore();

    useEffect(() => {
        // Initial fetch
        fetchWebSDRs();
        checkHealth();

        // Refresh every 30 seconds
        const interval = setInterval(() => {
            checkHealth();
        }, 30000);

        return () => clearInterval(interval);
    }, [fetchWebSDRs, checkHealth]);

    // Defensive: ensure websdrs is an array
    const safeWebsdrs = Array.isArray(websdrs) ? websdrs : [];
    const onlineCount = Object.values(healthStatus).filter(h => h?.status === 'online').length;
    const totalCount = safeWebsdrs.length;

    return (
        <div className="widget-content">
            {isLoading ? (
                <div className="text-center py-4">
                    <div className="spinner-border spinner-border-sm" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                </div>
            ) : (
                <>
                    <div className="d-flex align-items-center justify-content-between mb-3">
                        <div>
                            <h3 className="h4 mb-0">{onlineCount}/{totalCount}</h3>
                            <p className="text-muted small mb-0">Receivers Online</p>
                        </div>
                        <div className={`badge ${onlineCount === totalCount ? 'bg-success' : 'bg-warning'} fs-6`}>
                            {totalCount > 0 ? Math.round((onlineCount / totalCount) * 100) : 0}%
                        </div>
                    </div>

                    <div className="list-group list-group-flush">
                        {safeWebsdrs.slice(0, 7).map((sdr) => {
                            const health = healthStatus[sdr.id];
                            const isOnline = health?.status === 'online';

                            return (
                                <div key={sdr.id} className="list-group-item px-0 py-2">
                                    <div className="d-flex align-items-center justify-content-between">
                                        <div className="d-flex align-items-center gap-2">
                                            <i
                                                className={`ph ph-radio-button ${isOnline ? 'text-success' : 'text-danger'}`}
                                                aria-hidden="true"
                                            />
                                            <div>
                                                <div className="fw-medium">
                                                    {sdr.location_description?.split(',')[0] || sdr.name}
                                                </div>
                                                {health?.response_time_ms && (
                                                    <small className="text-muted">
                                                        {health.response_time_ms.toFixed(0)}ms
                                                    </small>
                                                )}
                                            </div>
                                        </div>
                                        <span className={`badge bg-light-${isOnline ? 'success' : 'danger'}`}>
                                            {isOnline ? 'Online' : 'Offline'}
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
};
