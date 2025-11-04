/**
 * DatasetDetailsDialog Component
 * 
 * Modal for viewing synthetic dataset samples
 */

import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import type { SyntheticDataset, SyntheticSample } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { usePortal } from '@/hooks/usePortal';

interface DatasetDetailsDialogProps {
  dataset: SyntheticDataset;
  isOpen: boolean;
  onClose: () => void;
}

export const DatasetDetailsDialog: React.FC<DatasetDetailsDialogProps> = ({
  dataset,
  isOpen,
  onClose,
}) => {
  const { fetchDatasetSamples } = useTrainingStore();
  const [samples, setSamples] = useState<SyntheticSample[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(isOpen);

  useEffect(() => {
    if (isOpen) {
      loadSamples();
    }
  }, [isOpen, dataset.id]);

  const loadSamples = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetchDatasetSamples(dataset.id, 10);
      setSamples(response.samples);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load samples');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen || !portalTarget) return null;

  return createPortal(
    <>
      {/* Bootstrap Modal Backdrop */}
      <div 
        className="modal-backdrop fade show"
        onClick={onClose}
      ></div>

      {/* Bootstrap Modal */}
      <div 
        className="modal fade show d-block" 
        tabIndex={-1}
        style={{ display: 'block' }}
      >
        <div className="modal-dialog modal-lg modal-dialog-scrollable">
          <div className="modal-content">
            {/* Modal Header */}
            <div className="modal-header">
              <h5 className="modal-title">Dataset Samples: {dataset.name}</h5>
              <button
                type="button"
                className="btn-close"
                onClick={onClose}
                aria-label="Close"
              ></button>
            </div>

            {/* Modal Body */}
            <div className="modal-body">
              {/* Loading State */}
              {isLoading && (
                <div className="text-center py-4">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading samples...</span>
                  </div>
                </div>
              )}

              {/* Error State */}
              {error && (
                <div className="alert alert-danger" role="alert">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                </div>
              )}

              {/* Samples Table */}
              {!isLoading && !error && samples.length > 0 && (
                <div className="table-responsive">
                  <table className="table table-sm table-hover">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Lat</th>
                        <th>Lon</th>
                        <th>Power (dBm)</th>
                        <th>Receivers</th>
                        <th>GDOP</th>
                        <th>Split</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samples.map((sample) => (
                        <tr key={sample.id}>
                          <td><code className="small">{sample.id}</code></td>
                          <td className="small">{sample.tx_lat.toFixed(4)}</td>
                          <td className="small">{sample.tx_lon.toFixed(4)}</td>
                          <td className="small">{sample.tx_power_dbm.toFixed(1)}</td>
                          <td className="small">{sample.num_receivers}</td>
                          <td className="small">{sample.gdop.toFixed(2)}</td>
                          <td>
                            <span className={`badge bg-${
                              sample.split === 'train' ? 'primary' : 
                              sample.split === 'val' ? 'info' : 
                              'secondary'
                            }`}>
                              {sample.split}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-muted small mb-0">
                    Showing {samples.length} of {dataset.num_samples} total samples
                  </p>
                </div>
              )}

              {/* Empty State */}
              {!isLoading && !error && samples.length === 0 && (
                <div className="text-center py-4">
                  <i className="ph ph-file-search" style={{ fontSize: '2rem', color: 'var(--bs-gray-400)' }}></i>
                  <p className="text-muted mt-2">No samples found</p>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </>,
    portalTarget
  );
};
