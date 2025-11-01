import React, { useEffect, useState, useRef } from 'react';
import { useDashboardStore } from '../store';
import { useWidgetStore } from '../store/widgetStore';
import { ConnectionState } from '../lib/websocket';
import { useWebSocket } from '../contexts/WebSocketContext';
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
import type { WidgetType } from '@/types/widgets';

const Dashboard: React.FC = () => {
  const { metrics, error, fetchDashboardData, lastUpdate } = useDashboardStore();

  const { widgets, resetToDefault } = useWidgetStore();
  const { connectionState, isConnected, connect, subscribe } = useWebSocket();
  const [showWidgetPicker, setShowWidgetPicker] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;

    // Fetch initial data
    fetchDashboardData();

    // Setup polling fallback (only if WebSocket not connected)
    const interval = setInterval(() => {
      if (!isConnected && isMountedRef.current) {
        fetchDashboardData();
      }
    }, 30000); // Poll every 30 seconds as fallback

    return () => {
      isMountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchDashboardData, isConnected]);

  // Subscribe to WebSocket events
  useEffect(() => {
    if (!isMountedRef.current) return;

    const handleServicesHealth = (data: any) => {
      if (!isMountedRef.current) return;
      console.log('[Dashboard] Received services health update:', data);
      useDashboardStore.setState(state => ({
        data: {
          ...state.data,
          servicesHealth: data,
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

    // Subscribe to events and store unsubscribe functions
    const unsubscribeServicesHealth = subscribe('services:health', handleServicesHealth);
    const unsubscribeWebSDRUpdate = subscribe('websdrs_update', handleWebSDRUpdate);
    const unsubscribeSignalDetected = subscribe('signals:detected', handleSignalDetected);
    const unsubscribeLocalizationUpdate = subscribe(
      'localizations:updated',
      handleLocalizationUpdate
    );

    // Cleanup: unsubscribe from all events
    return () => {
      isMountedRef.current = false;
      unsubscribeServicesHealth();
      unsubscribeWebSDRUpdate();
      unsubscribeSignalDetected();
      unsubscribeLocalizationUpdate();
    };
  }, [subscribe]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchDashboardData();
    setIsRefreshing(false);
  };

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

    switch (widgetConfig.type as WidgetType) {
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
        <WidgetComponent widgetId={widgetConfig.id} />
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
                    className="btn btn-sm btn-outline-primary"
                    onClick={handleRefresh}
                    disabled={isRefreshing}
                    title="Refresh data"
                  >
                    <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                  </button>
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
          </div>
        </div>
      </nav>

      {/* Error Alert */}
      {error && (
        <div
          className="alert alert-danger alert-dismissible fade show"
          role="alert"
          aria-live="assertive"
          aria-atomic="true"
        >
          <strong>Error!</strong> {error}
          <button
            type="button"
            className="btn-close"
            data-bs-dismiss="alert"
            aria-label="Close error message"
          ></button>
        </div>
      )}

      {/* Stats Overview Cards */}
      <section aria-label="Quick Stats" className="row mb-3">
        {/* Active WebSDR Card */}
        <div className="col-6 col-md-3 mb-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <div>
                  <p className="text-muted mb-1 small">Active WebSDRs</p>
                  <h3 className="mb-0">
                    {metrics.activeWebSDRs}/{metrics.totalWebSDRs}
                  </h3>
                </div>
                <i className="ph ph-radio-button fs-1 text-primary"></i>
              </div>
            </div>
          </div>
        </div>

        {/* Signal Detections Card */}
        <div className="col-6 col-md-3 mb-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <div>
                  <p className="text-muted mb-1 small">Detections (24h)</p>
                  <h3 className="mb-0">{metrics.signalDetections || 0}</h3>
                </div>
                <i className="ph ph-chart-line fs-1 text-warning"></i>
              </div>
            </div>
          </div>
        </div>

        {/* System Uptime Card */}
        <div className="col-6 col-md-3 mb-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <div>
                  <p className="text-muted mb-1 small">System Uptime</p>
                  <h3 className="mb-0">
                    {metrics.systemUptime > 0
                      ? `${(metrics.systemUptime / 3600).toFixed(1)}h`
                      : '0h'}
                  </h3>
                </div>
                <i className="ph ph-activity fs-1 text-success"></i>
              </div>
            </div>
          </div>
        </div>

        {/* Model Accuracy Card */}
        <div className="col-6 col-md-3 mb-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <div>
                  <p className="text-muted mb-1 small">Model Accuracy</p>
                  <h3 className="mb-0">
                    {metrics.averageAccuracy > 0 ? `${metrics.averageAccuracy.toFixed(1)}%` : 'N/A'}
                  </h3>
                </div>
                <i className="ph ph-brain fs-1 text-info"></i>
              </div>
            </div>
          </div>
        </div>
      </section>

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

      {/* Last Updated */}
      {lastUpdate && (
        <div className="text-center text-muted small mt-3">
          Last updated: {new Date(lastUpdate).toLocaleTimeString()}
        </div>
      )}

      {/* Widget Picker Modal */}
      <WidgetPicker show={showWidgetPicker} onClose={() => setShowWidgetPicker(false)} />
    </>
  );
};

export default Dashboard;
