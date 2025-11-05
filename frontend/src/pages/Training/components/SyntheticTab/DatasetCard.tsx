/**
 * DatasetCard Component
 * 
 * Displays a synthetic dataset with quality metrics and actions
 */

import React, { useState } from 'react';
import type { SyntheticDataset } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { DatasetDetailsDialog } from './DatasetDetailsDialog';
import { ExpandDatasetDialog } from './ExpandDatasetDialog';
import { InlineEditText } from '../../../../components/InlineEditText';

interface DatasetCardProps {
  dataset: SyntheticDataset;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({ dataset }) => {
  const { deleteDataset, updateDatasetName } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false);
  const [isExpandDialogOpen, setIsExpandDialogOpen] = useState(false);

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
