/**
 * DatasetCard Component
 * 
 * Displays a synthetic dataset with quality metrics and actions
 */

import React, { useState, useEffect } from 'react';
import type { SyntheticDataset } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { DatasetDetailsDialog } from './DatasetDetailsDialog';
import { ExpandDatasetDialog } from './ExpandDatasetDialog';
import { InlineEditText } from '../../../../components/InlineEditText';
import { validateDataset, repairDataset } from '../../../../services/api/training';
import { useWebSocket } from '../../../../contexts/WebSocketContext';

interface DatasetCardProps {
  dataset: SyntheticDataset;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({ dataset }) => {
  const { deleteDataset, updateDatasetName, updateDatasetHealth } = useTrainingStore();
  const { subscribe } = useWebSocket();
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isRepairing, setIsRepairing] = useState(false);
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false);
  const [isExpandDialogOpen, setIsExpandDialogOpen] = useState(false);

  // Subscribe to WebSocket events for real-time health updates
  // This catches updates triggered by other users or background processes
  useEffect(() => {
    const unsubscribeValidated = subscribe('dataset_validated', (data: any) => {
      console.log('[DatasetCard] Received dataset_validated event:', data);
      if (data.dataset_id === dataset.id) {
        // Backend sends: { dataset_id, health_status, orphan_percentage, num_samples }
        // Update health status badge immediately (details will be in validation_issues if user re-validates)
        updateDatasetHealth(dataset.id, {
          health_status: data.health_status,
          last_validated_at: new Date().toISOString(),
          num_samples: data.num_samples,
          // Keep existing validation_issues if present, or clear if now healthy
          validation_issues: data.health_status === 'healthy' ? undefined : dataset.validation_issues,
        });
      }
    });

    const unsubscribeRepaired = subscribe('dataset_repaired', (data: any) => {
      console.log('[DatasetCard] Received dataset_repaired event:', data);
      if (data.dataset_id === dataset.id) {
        // Backend sends: { dataset_id, dataset_name, health_status, deleted_iq_files, deleted_features, num_samples }
        updateDatasetHealth(dataset.id, {
          health_status: data.health_status,
          last_validated_at: new Date().toISOString(),
          num_samples: data.num_samples,
          validation_issues: undefined, // Cleared after repair
        });
      }
    });

    return () => {
      unsubscribeValidated();
      unsubscribeRepaired();
    };
  }, [dataset.id, dataset.validation_issues, subscribe, updateDatasetHealth]);

  const formatNumber = (num: number | undefined) => {
    if (num === undefined) return 'N/A';
    if (num > 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (num > 1_000) return `${(num / 1_000).toFixed(2)}K`;
    return num.toFixed(0);
  };

  const formatMetric = (value: number | undefined, decimals = 2, suffix = '') => {
    if (value === undefined) return 'N/A';
    return `${value.toFixed(decimals)}${suffix}`;
  };

  const formatStorageSize = (bytes: number | undefined) => {
    if (bytes === undefined || bytes === null) return 'N/A';
    if (bytes === 0) return '0 B';
    
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const value = bytes / Math.pow(k, i);
    
    return `${value.toFixed(2)} ${units[i]}`;
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      await deleteDataset(dataset.id);
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      // API call validates, saves to database, and returns complete updated data
      const validationReport = await validateDataset(dataset.id);
      
      // Update store immediately with the validation response
      // The backend returns the complete structure that was saved to DB
      updateDatasetHealth(dataset.id, {
        health_status: validationReport.health_status,
        last_validated_at: validationReport.validated_at,
        num_samples: validationReport.num_samples,
        storage_size_bytes: validationReport.storage_size_bytes,
        validation_issues: validationReport.validation_issues,
      });
      
      console.log('[DatasetCard] Validation complete, store updated with response');
    } catch (error) {
      console.error('Failed to validate dataset:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleRepair = async () => {
    setIsRepairing(true);
    try {
      // API call returns repair result immediately
      const result = await repairDataset(dataset.id, 'delete_orphans');
      
      // Update store immediately with API response
      updateDatasetHealth(dataset.id, {
        health_status: result.new_health_status as 'unknown' | 'healthy' | 'warning' | 'critical',
        last_validated_at: new Date().toISOString(),
        num_samples: result.num_samples,
        validation_issues: undefined, // Cleared after successful repair
      });
      
      console.log('[DatasetCard] Repair complete, health status updated');
    } catch (error) {
      console.error('Failed to repair dataset:', error);
    } finally {
      setIsRepairing(false);
    }
  };

  const getHealthStatusBadge = () => {
    const status = dataset.health_status || 'unknown';
    const badges = {
      healthy: { color: 'bg-success', icon: 'ph-check-circle', text: 'Healthy' },
      warning: { color: 'bg-warning', icon: 'ph-warning', text: 'Warning' },
      critical: { color: 'bg-danger', icon: 'ph-x-circle', text: 'Critical' },
      unknown: { color: 'bg-secondary', icon: 'ph-question', text: 'Unknown' },
    };
    
    const badge = badges[status];
    return (
      <span className={`badge ${badge.color} d-flex align-items-center gap-1`}>
        <i className={`ph ${badge.icon}`}></i>
        {badge.text}
      </span>
    );
  };

  const metrics = dataset.quality_metrics;

  return (
    <>
      <div className="card h-100">
        <div className="card-body">
          {/* Header */}
          <div className="d-flex justify-content-between align-items-start mb-3">
            <div className="flex-grow-1">
              <h5 className="mb-1">
                <InlineEditText
                  value={dataset.name}
                  onSave={(newName) => updateDatasetName(dataset.id, newName)}
                  className="d-inline-block"
                  placeholder="Dataset Name"
                  maxLength={200}
                />
              </h5>
              {dataset.description && (
                <p className="text-muted small mb-0">{dataset.description}</p>
              )}
            </div>
            <div className="d-flex flex-column gap-1 align-items-end">
              <span className="badge bg-light-success">
                READY
              </span>
              {dataset.dataset_type && (
                <span className={`badge ${dataset.dataset_type === 'iq_raw' ? 'bg-info' : 'bg-secondary'}`}>
                  {dataset.dataset_type === 'iq_raw' ? 'IQ Raw' : 'Features'}
                </span>
              )}
              {getHealthStatusBadge()}
            </div>
          </div>

          {/* Dataset Info */}
          <div className="mb-3">
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Samples:</span>
              <span className="fw-medium small">{formatNumber(dataset.num_samples)}</span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Storage Size:</span>
              <span className="fw-medium small">
                {formatStorageSize(dataset.storage_size_bytes)}
              </span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Created:</span>
              <span className="fw-medium small">
                {new Date(dataset.created_at).toLocaleDateString()}
              </span>
            </div>
            {dataset.created_by_job_id && (
              <div className="d-flex justify-content-between">
                <span className="text-muted small">Job ID:</span>
                <code className="small">
                  {dataset.created_by_job_id.slice(0, 8)}
                </code>
              </div>
            )}
          </div>

          {/* Validation Issues */}
          {dataset.validation_issues && dataset.validation_issues.total_issues > 0 && (
            <div className="mb-3 p-2 bg-danger bg-opacity-10 border border-danger rounded">
              <h6 className="small fw-semibold mb-2 text-danger">Data Integrity Issues</h6>
              <div className="d-flex flex-column gap-1">
                {dataset.validation_issues.orphaned_iq_files > 0 && (
                  <div className="small">
                    <i className="ph ph-warning text-warning me-1"></i>
                    <span className="text-dark">
                      {dataset.validation_issues.orphaned_iq_files} orphaned IQ files
                    </span>
                  </div>
                )}
                {dataset.validation_issues.orphaned_features > 0 && (
                  <div className="small">
                    <i className="ph ph-warning text-warning me-1"></i>
                    <span className="text-dark">
                      {dataset.validation_issues.orphaned_features} orphaned features
                    </span>
                  </div>
                )}
              </div>
              {dataset.last_validated_at && (
                <div className="text-muted small mt-1">
                  Last validated: {new Date(dataset.last_validated_at).toLocaleString()}
                </div>
              )}
            </div>
          )}

          {/* Quality Metrics */}
          {metrics && (
            <div className="mb-3 p-2 bg-light text-dark border rounded">
              <h6 className="small fw-semibold mb-2">Quality Metrics</h6>
              <div className="row g-2">
                {metrics.mean_snr_db !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Mean SNR:</span>
                    <span className="ms-2 fw-medium small">{formatMetric(metrics.mean_snr_db, 1, ' dB')}</span>
                  </div>
                )}
                {metrics.mean_gdop !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Mean GDOP:</span>
                    <span className="ms-2 fw-medium small">{formatMetric(metrics.mean_gdop, 2)}</span>
                  </div>
                )}
                {metrics.mean_receivers !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Avg RX:</span>
                    <span className="ms-2 fw-medium small">{formatMetric(metrics.mean_receivers, 1)}</span>
                  </div>
                )}
                {metrics.mean_distance_km !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Avg Dist:</span>
                    <span className="ms-2 fw-medium small">{formatMetric(metrics.mean_distance_km, 1, ' km')}</span>
                  </div>
                )}
                {metrics.inside_ratio !== undefined && (
                  <div className="col-12">
                    <span className="text-muted small">Inside Ratio:</span>
                    <span className="ms-2 fw-medium small">{formatMetric(metrics.inside_ratio * 100, 1, '%')}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="d-grid gap-2">
            <button
              onClick={() => setIsDetailsDialogOpen(true)}
              className="btn btn-outline-primary d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-info"></i>
              Details
            </button>
            <button
              onClick={handleValidate}
              disabled={isValidating}
              className="btn btn-outline-info d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-detective"></i>
              {isValidating ? 'Validating...' : 'Validate'}
            </button>
            {dataset.health_status && dataset.health_status !== 'healthy' && dataset.health_status !== 'unknown' && (
              <button
                onClick={handleRepair}
                disabled={isRepairing}
                className="btn btn-warning d-flex align-items-center justify-content-center gap-2"
              >
                <i className="ph ph-wrench"></i>
                {isRepairing ? 'Repairing...' : 'Repair Dataset'}
              </button>
            )}
            <button
              onClick={() => setIsExpandDialogOpen(true)}
              className="btn btn-primary d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-plus-circle"></i>
              Expand Dataset
            </button>
            <button
              onClick={handleDelete}
              disabled={isLoading}
              className="btn btn-outline-danger d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-trash"></i>
              {isLoading ? 'Deleting...' : 'Delete Dataset'}
            </button>
          </div>
        </div>
      </div>

      {/* Dataset Details Dialog */}
      <DatasetDetailsDialog
        dataset={dataset}
        isOpen={isDetailsDialogOpen}
        onClose={() => setIsDetailsDialogOpen(false)}
      />

      {/* Expand Dataset Dialog */}
      <ExpandDatasetDialog
        dataset={dataset}
        isOpen={isExpandDialogOpen}
        onClose={() => setIsExpandDialogOpen(false)}
      />
    </>
  );
};
