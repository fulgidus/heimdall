import React, { useEffect, useState, useRef } from 'react';
import { useDashboardStore } from '../store';
import { useWidgetStore } from '../store/widgetStore';
import { ConnectionState } from '../lib/websocket';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../hooks/useAuth';
import {
  WidgetContainer,
  WebSDRStatusWidget,
  SystemHealthWidget,
  RecentActivityWidget,
  SignalChartWidget,
  ModelPerformanceWidget,
  QuickActionsWidget,
} from '../components/widgets';
import { WidgetPicker } from '../components/widgets/WidgetPicker';
import { getConstellations, type Constellation } from '../services/api/constellations';
import type { WidgetType } from '@/types/widgets';

const Dashboard: React.FC = () => {
  const { user, isOperator } = useAuth();
  const { widgets, resetToDefault } = useWidgetStore();
  const { connectionState, connect, subscribe } = useWebSocket();
  const [showWidgetPicker, setShowWidgetPicker] = useState(false);
  const isMountedRef = useRef(true);
  
  // Constellation filtering state
  const [constellations, setConstellations] = useState<Constellation[]>([]);
  const [selectedConstellationId, setSelectedConstellationId] = useState<string | null>(null);
  const [isLoadingConstellations, setIsLoadingConstellations] = useState(false);


  // Load constellations on mount
  useEffect(() => {
    const loadConstellations = async () => {
      setIsLoadingConstellations(true);
      try {
        const data = await getConstellations();
        setConstellations(data);
        
        // Auto-select first constellation if available
        if (data.length > 0 && !selectedConstellationId) {
          setSelectedConstellationId(data[0].id);
        }
      } catch (error) {
        console.error('Failed to load constellations:', error);
      } finally {
        setIsLoadingConstellations(false);
      }
    };

    loadConstellations();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Subscribe to WebSocket events
  useEffect(() => {
    if (!isMountedRef.current) return;

    const handleServiceHealth = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received service health update:', data);
      
      // Individual service health event - aggregate into servicesHealth map
      const { service, status, ...rest } = data;
      if (!service) {
        console.warn('[Dashboard] Service health event missing service name', data);
        return;
      }
      
      useDashboardStore.setState(state => ({
        data: {
          ...state.data,
          servicesHealth: {
            ...state.data.servicesHealth,
            [service]: {
              service,
              status,
              ...rest,
            },
          },
        },
        lastUpdate: new Date(),
      }));
    };

    const handleWebSDRUpdate = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received WebSDR status update:', data);
      useDashboardStore.setState(state => ({
        data: {
          ...state.data,
          websdrsHealth: data,
        },
        lastUpdate: new Date(),
      }));
    };

    const handleSignalDetected = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received signal detection:', data);
      useDashboardStore.setState(state => ({
        metrics: {
          ...state.metrics,
          signalDetections: (state.metrics.signalDetections || 0) + 1,
        },
        lastUpdate: new Date(),
      }));
    };

    const handleLocalizationUpdate = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received localization update:', data);
      useDashboardStore.setState({ lastUpdate: new Date() });
    };

    const handleSessionUpdate = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received session status update:', data);
      // Trigger last update to notify components that data may have changed
      useDashboardStore.setState({ lastUpdate: new Date() });
    };

    // Subscribe to events and store unsubscribe functions
    const unsubscribeServiceHealth = subscribe('service:health', handleServiceHealth);
    const unsubscribeWebSDRUpdate = subscribe('websdrs_update', handleWebSDRUpdate);
    const unsubscribeSignalDetected = subscribe('signals:detected', handleSignalDetected);
    const unsubscribeLocalizationUpdate = subscribe(
      'localizations:updated',
      handleLocalizationUpdate
    );
    const unsubscribeSessionUpdate = subscribe('session:status_update', handleSessionUpdate);

    // Cleanup: unsubscribe from all events
    return () => {
      isMountedRef.current = false;
      unsubscribeServiceHealth();
      unsubscribeWebSDRUpdate();
      unsubscribeSignalDetected();
      unsubscribeLocalizationUpdate();
      unsubscribeSessionUpdate();
    };
  }, [subscribe]);

  const handleReconnect = async () => {
    await connect();
  };

  // Get connection status display
  const getConnectionStatus = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return { text: 'Connected', color: 'success', icon: 'ph-check-circle' };
      case ConnectionState.CONNECTING:
        return { text: 'Connecting...', color: 'warning', icon: 'ph-circle-notch' };
      case ConnectionState.RECONNECTING:
        return { text: 'Reconnecting...', color: 'warning', icon: 'ph-arrows-clockwise' };
      case ConnectionState.DISCONNECTED:
      default:
        return { text: 'Disconnected', color: 'danger', icon: 'ph-x-circle' };
    }
  };

  const connectionStatus = getConnectionStatus();

  // Render widget based on type
  const renderWidget = (widgetConfig: (typeof widgets)[0]) => {
    let WidgetComponent;

    switch (widgetConfig.type) {
      case 'websdr-status':
        WidgetComponent = WebSDRStatusWidget;
        break;
      case 'system-health':
        WidgetComponent = SystemHealthWidget;
        break;
      case 'recent-activity':
        WidgetComponent = RecentActivityWidget;
        break;
      case 'signal-chart':
        WidgetComponent = SignalChartWidget;
        break;
      case 'model-performance':
        WidgetComponent = ModelPerformanceWidget;
        break;
      case 'quick-actions':
        WidgetComponent = QuickActionsWidget;
        break;
      default:
        return null;
    }

    return (
      <WidgetContainer key={widgetConfig.id} widget={widgetConfig}>
        <WidgetComponent 
          widgetId={widgetConfig.id}
          selectedConstellationId={selectedConstellationId}
        />
      </WidgetContainer>
    );
  };

  // Sort widgets by position
  const sortedWidgets = Array.isArray(widgets)
    ? [...widgets].sort((a, b) => a.position - b.position)
    : [];

  return (
    <>
      {/* Breadcrumb */}
      <nav className="page-header" aria-label="Breadcrumb">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ol className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/">Home</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Dashboard
                </li>
              </ol>
            </div>
            <div className="col-md-12">
              <div className="page-header-title d-flex align-items-center justify-content-between">
                <h1 className="mb-0">Dashboard</h1>
                {/* Connection Status & Actions */}
                <div className="d-flex align-items-center gap-2">
                  <span
                    className={`badge bg-light-${connectionStatus.color} d-flex align-items-center gap-1`}
                  >
                    <i
                      className={`ph ${connectionStatus.icon} ${connectionState === ConnectionState.CONNECTING || connectionState === ConnectionState.RECONNECTING ? 'spin' : ''}`}
                    ></i>
                    {connectionStatus.text}
                  </span>
                  {connectionState === ConnectionState.DISCONNECTED && (
                    <button
                      className="btn btn-sm btn-outline-primary"
                      onClick={handleReconnect}
                      title="Reconnect WebSocket"
                    >
                      <i className="ph ph-arrows-clockwise"></i>
                    </button>
                  )}
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() => setShowWidgetPicker(true)}
                    title="Add widget"
                  >
                    <i className="ph ph-plus"></i> Add Widget
                  </button>
                  <button
                    className="btn btn-sm btn-outline-secondary"
                    onClick={resetToDefault}
                    title="Reset to default layout"
                  >
                    <i className="ph ph-arrow-clockwise"></i>
                  </button>
                </div>
              </div>
            </div>
            
            {/* Constellation Filter Row */}
            {constellations.length > 0 && (
              <div className="col-md-12 mt-3">
                <div className="d-flex align-items-center gap-3">
                  <label htmlFor="constellation-filter" className="form-label mb-0 fw-semibold">
                    <i className="ph ph-circles-three me-2"></i>
                    Filter by Constellation:
                  </label>
                  <div className="flex-grow-1" style={{ maxWidth: '400px' }}>
                    <select
                      id="constellation-filter"
                      className="form-select"
                      value={selectedConstellationId || ''}
                      onChange={(e) => setSelectedConstellationId(e.target.value || null)}
                      disabled={isLoadingConstellations}
                    >
                      <option value="">All WebSDRs (No Filter)</option>
                      {constellations.map(constellation => (
                        <option key={constellation.id} value={constellation.id}>
                          {constellation.name} ({constellation.member_count} receivers)
                        </option>
                      ))}
                    </select>
                  </div>
                  {isOperator && (
                    <a href="/constellations" className="btn btn-sm btn-outline-primary">
                      <i className="ph ph-gear me-1"></i>
                      Manage Constellations
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>



      {/* Widgets Grid */}
      <section aria-label="Dashboard Widgets">
        <div className="row g-3">
          {sortedWidgets.length > 0 ? (
            sortedWidgets.map(widget => renderWidget(widget))
          ) : (
            <div className="col-12">
              <div className="card">
                <div className="card-body text-center py-5">
                  <i className="ph ph-grid-four fs-1 text-muted mb-3"></i>
                  <h3 className="h5 mb-2">No Widgets Added</h3>
                  <p className="text-muted mb-3">Add widgets to customize your dashboard</p>
                  <button className="btn btn-primary" onClick={() => setShowWidgetPicker(true)}>
                    <i className="ph ph-plus me-2"></i>
                    Add Your First Widget
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Widget Picker Modal */}
      <WidgetPicker show={showWidgetPicker} onClose={() => setShowWidgetPicker(false)} />
    </>
  );
};

export default Dashboard;
