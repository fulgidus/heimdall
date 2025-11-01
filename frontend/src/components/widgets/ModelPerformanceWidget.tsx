import React, { useEffect, useState } from 'react';
import { inferenceService } from '@/services/api';
import type { ModelInfo } from '@/services/api/types';

interface ModelPerformanceWidgetProps {
  widgetId: string;
}

export const ModelPerformanceWidget: React.FC<ModelPerformanceWidgetProps> = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const fetchModelInfo = async () => {
      try {
        const info = await inferenceService.getModelInfo();
        if (isMounted) {
          setModelInfo(info);
        }
      } catch (error) {
        console.error('Failed to fetch model info:', error);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchModelInfo();
    const interval = setInterval(fetchModelInfo, 30000); // Refresh every 30s

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

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
          <div className="text-center p-3 bg-light rounded">
            <h4 className="h2 mb-1">{accuracy}%</h4>
            <p className="text-muted small mb-0">Accuracy</p>
          </div>
        </div>
        <div className="col-6">
          <div className="text-center p-3 bg-light rounded">
            <h4 className="h2 mb-1">{successRate}%</h4>
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
        {modelInfo.latency_p95_ms && (
          <div className="list-group-item px-0 py-2">
            <div className="d-flex justify-content-between">
              <span className="text-muted">P95 Latency</span>
              <span className="fw-medium">{modelInfo.latency_p95_ms.toFixed(0)}ms</span>
            </div>
          </div>
        )}
        {modelInfo.cache_hit_rate !== undefined && (
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
