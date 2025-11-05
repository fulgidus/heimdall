import React from 'react';
import { useDashboardStore } from '@/store';
import type { ModelInfo } from '@/services/api/types';

interface ModelPerformanceWidgetProps {
  widgetId: string;
}

export const ModelPerformanceWidget: React.FC<ModelPerformanceWidgetProps> = () => {
  const { data } = useDashboardStore();
  
  // Extract model info from inference service health data (WebSocket-driven)
  const inferenceHealth = data.servicesHealth?.inference;
  // Model info is in the details field - safely cast after null check
  const modelInfo = (inferenceHealth?.details?.model_info ?? null) as ModelInfo | null;
  const isLoading = !inferenceHealth;

  if (isLoading) {
    return (
      <div className="widget-content">
        <div className="text-center py-4">
          <div className="spinner-border spinner-border-sm" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="widget-content">
        <div className="text-center py-5 text-muted">
          <i className="ph ph-brain fs-1 mb-2" />
          <p className="mb-0">Model not loaded</p>
        </div>
      </div>
    );
  }

  const accuracy = modelInfo.accuracy ? (modelInfo.accuracy * 100).toFixed(1) : 'N/A';
  const successRate =
    modelInfo.predictions_total &&
    modelInfo.predictions_total > 0 &&
    modelInfo.predictions_successful
      ? (((modelInfo.predictions_successful ?? 0) / modelInfo.predictions_total) * 100).toFixed(1)
      : '0';

  return (
    <div className="widget-content">
      <div className="row g-3 mb-3">
        <div className="col-6">
          <div className="text-center p-3 bg-light text-dark rounded">
            <h4 className="h2 mb-1 text-dark">{accuracy}%</h4>
            <p className="text-muted small mb-0">Accuracy</p>
          </div>
        </div>
        <div className="col-6">
          <div className="text-center p-3 bg-light text-dark rounded">
            <h4 className="h2 mb-1 text-dark">{successRate}%</h4>
            <p className="text-muted small mb-0">Success Rate</p>
          </div>
        </div>
      </div>

      <div className="list-group list-group-flush">
        <div className="list-group-item px-0 py-2">
          <div className="d-flex justify-content-between">
            <span className="text-muted">Version</span>
            <span className="fw-medium">{modelInfo.active_version}</span>
          </div>
        </div>
        <div className="list-group-item px-0 py-2">
          <div className="d-flex justify-content-between">
            <span className="text-muted">Status</span>
            <span
              className={`badge bg-light-${
                modelInfo.health_status === 'healthy'
                  ? 'success'
                  : modelInfo.health_status === 'degraded'
                    ? 'warning'
                    : 'danger'
              }`}
            >
              {modelInfo.health_status}
            </span>
          </div>
        </div>
        <div className="list-group-item px-0 py-2">
          <div className="d-flex justify-content-between">
            <span className="text-muted">Predictions</span>
            <span className="fw-medium">
              {modelInfo.predictions_total?.toLocaleString() ?? '0'}
            </span>
          </div>
        </div>
        {modelInfo.latency_p95_ms != null && (
          <div className="list-group-item px-0 py-2">
            <div className="d-flex justify-content-between">
              <span className="text-muted">P95 Latency</span>
              <span className="fw-medium">{modelInfo.latency_p95_ms.toFixed(0)}ms</span>
            </div>
          </div>
        )}
        {modelInfo.cache_hit_rate != null && (
          <div className="list-group-item px-0 py-2">
            <div className="d-flex justify-content-between">
              <span className="text-muted">Cache Hit Rate</span>
              <span className="fw-medium">{(modelInfo.cache_hit_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
