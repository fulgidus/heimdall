import React from 'react';
import { useDashboardStore } from '@/store';

interface SystemHealthWidgetProps {
  widgetId: string;
  selectedConstellationId?: string | null;
}

export const SystemHealthWidget: React.FC<SystemHealthWidgetProps> = () => {
  const { data } = useDashboardStore();
  const services = data.servicesHealth || {};

  const servicesList = Object.entries(services);
  const healthyCount = servicesList.filter(([_, s]) => s?.status === 'healthy').length;
  const totalCount = servicesList.length;
  const isLoading = totalCount === 0;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      default:
        return 'danger';
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
      ) : (
        <>
          <div className="d-flex align-items-center justify-content-between mb-3">
            <div>
              <h3 className="h4 mb-0">
                {healthyCount}/{totalCount}
              </h3>
              <p className="text-muted small mb-0">Services Healthy</p>
            </div>
            <i
              className={`ph ph-cpu fs-1 text-${healthyCount === totalCount ? 'success' : 'warning'}`}
            />
          </div>

          <div className="list-group list-group-flush">
            {servicesList.map(([name, health]) => (
              <div key={name} className="list-group-item px-0 py-2">
                <div className="d-flex align-items-center justify-content-between">
                  <span className="text-capitalize">{name.replace(/-/g, ' ')}</span>
                  <span className={`badge bg-light-${getStatusColor(health.status)}`}>
                    {health.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};
