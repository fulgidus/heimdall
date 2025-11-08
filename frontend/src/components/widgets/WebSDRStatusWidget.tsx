import React, { useEffect, useState } from 'react';
import { useDashboardStore } from '@/store';
import { useWebSDRStore } from '@/store';
import { getConstellation, type Constellation } from '@/services/api/constellations';

interface WebSDRStatusWidgetProps {
  widgetId: string;
  selectedConstellationId?: string | null;
}

export const WebSDRStatusWidget: React.FC<WebSDRStatusWidgetProps> = ({ selectedConstellationId }) => {
  const { websdrs, fetchWebSDRs, isLoading } = useWebSDRStore();
  const { data } = useDashboardStore();
  const healthStatus = data.websdrsHealth || {};
  const [constellationDetails, setConstellationDetails] = useState<Constellation | null>(null);

  useEffect(() => {
    // Initial fetch of WebSDR configurations (once)
    fetchWebSDRs();
    // Health status updates come via WebSocket (no polling needed)
  }, [fetchWebSDRs]);
  
  // Fetch constellation details when selected
  useEffect(() => {
    if (selectedConstellationId) {
      getConstellation(selectedConstellationId)
        .then(setConstellationDetails)
        .catch(err => {
          console.error('Failed to load constellation details:', err);
          setConstellationDetails(null);
        });
    } else {
      setConstellationDetails(null);
    }
  }, [selectedConstellationId]);

  // Defensive: ensure websdrs is an array
  const allWebsdrs = Array.isArray(websdrs) ? websdrs : [];
  
  // Filter websdrs by constellation if selected
  const safeWebsdrs = selectedConstellationId && constellationDetails
    ? allWebsdrs.filter(sdr => 
        constellationDetails.members?.some(m => m.websdr_station_id === sdr.id)
      )
    : allWebsdrs;
  
  // Calculate counts based on filtered websdrs
  const filteredWebsdrIds = new Set(safeWebsdrs.map(w => w.id));
  const onlineCount = Object.entries(healthStatus)
    .filter(([id, h]) => filteredWebsdrIds.has(id) && h?.status === 'online')
    .length;
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
              <h3 className="h4 mb-0">
                {onlineCount}/{totalCount}
              </h3>
              <p className="text-muted small mb-0">Receivers Online</p>
            </div>
            <div
              className={`badge ${onlineCount === totalCount ? 'bg-success' : 'bg-warning'} fs-6`}
            >
              {totalCount > 0 ? Math.round((onlineCount / totalCount) * 100) : 0}%
            </div>
          </div>

          <div className="list-group list-group-flush">
            {safeWebsdrs.slice(0, 7).map(sdr => {
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
                        {health?.response_time_ms != null && (
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
