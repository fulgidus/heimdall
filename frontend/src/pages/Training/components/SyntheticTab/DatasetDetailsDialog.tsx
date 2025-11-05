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

  // Convert dBm to Watts
  const dbmToWatts = (dbm: number): number => {
    return Math.pow(10, dbm / 10) / 1000;
  };

  const formatPower = (dbm: number): string => {
    const watts = dbmToWatts(dbm);
    return `${watts.toFixed(2)}W (${dbm.toFixed(2)}dBm)`;
  };

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
              <h5 className="modal-title">Dataset Details: {dataset.name}</h5>
              <button
                type="button"
                className="btn-close"
                onClick={onClose}
                aria-label="Close"
              ></button>
            </div>

            {/* Modal Body */}
            <div className="modal-body">
              {/* Dataset Configuration Info */}
              <div className="card bg-light text-dark border-0 mb-3">
                <div className="card-body">
                  <h6 className="card-title mb-3">
                    <i className="ph ph-gear me-2"></i>
                    Dataset Configuration
                  </h6>
                  <div className="row g-3">
                    {/* Frequency */}
                    {dataset.config?.frequency_mhz !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Frequency:</span>
                          <span className="fw-medium">{dataset.config.frequency_mhz} MHz</span>
                        </div>
                      </div>
                    )}

                    {/* TX Power */}
                    {dataset.config?.tx_power_dbm !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">TX Power:</span>
                          <span className="fw-medium">{formatPower(dataset.config.tx_power_dbm)}</span>
                        </div>
                      </div>
                    )}

                    {/* Min SNR */}
                    {dataset.config?.min_snr_db !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Min SNR:</span>
                          <span className="fw-medium">{dataset.config.min_snr_db} dB</span>
                        </div>
                      </div>
                    )}

                    {/* Min Receivers */}
                    {dataset.config?.min_receivers !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Min Receivers:</span>
                          <span className="fw-medium">{dataset.config.min_receivers}</span>
                        </div>
                      </div>
                    )}

                    {/* Max GDOP */}
                    {dataset.config?.max_gdop !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Max GDOP:</span>
                          <span className="fw-medium">{dataset.config.max_gdop}</span>
                        </div>
                      </div>
                    )}

                    {/* Inside Ratio */}
                    {dataset.config?.inside_ratio !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Inside Ratio:</span>
                          <span className="fw-medium">{(dataset.config.inside_ratio * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    )}

                    {/* Use SRTM */}
                    {dataset.config?.use_srtm !== undefined && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Use SRTM:</span>
                          <span className="fw-medium">
                            {dataset.config.use_srtm ? (
                              <span className="badge bg-success">Yes</span>
                            ) : (
                              <span className="badge bg-secondary">No</span>
                            )}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Dataset Type */}
                    {dataset.dataset_type && (
                      <div className="col-md-6">
                        <div className="d-flex justify-content-between">
                          <span className="text-muted small">Type:</span>
                          <span className="fw-medium">
                            {dataset.dataset_type === 'iq_raw' ? 'IQ Raw' : 'Feature-based'}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Random Receivers (for IQ raw) */}
                    {dataset.config?.use_random_receivers && (
                      <div className="col-12">
                        <div className="alert alert-info small mb-0 py-2">
                          <i className="ph ph-info me-2"></i>
                          Random Receivers: {dataset.config.min_receivers_count || 'N/A'} - {dataset.config.max_receivers_count || 'N/A'}
                          {dataset.config.receiver_seed && ` (seed: ${dataset.config.receiver_seed})`}
                        </div>
                      </div>
                    )}

                    {/* Geographic Area (if configured) */}
                    {(dataset.config?.area_lat_min !== undefined || dataset.config?.area_lon_min !== undefined) && (
                      <div className="col-12">
                        <hr className="my-2" />
                        <div className="small">
                          <strong className="text-muted">Geographic Area:</strong>
                          <div className="mt-1">
                            Lat: {dataset.config.area_lat_min?.toFixed(4)} - {dataset.config.area_lat_max?.toFixed(4)}, 
                            Lon: {dataset.config.area_lon_min?.toFixed(4)} - {dataset.config.area_lon_max?.toFixed(4)}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

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
