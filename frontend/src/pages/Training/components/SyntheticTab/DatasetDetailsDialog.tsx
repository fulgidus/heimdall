/**
 * DatasetDetailsDialog Component
 * 
 * Modal for viewing synthetic dataset samples with 3-tab layout:
 * 1. Overview - Dataset configuration and sample count
 * 2. Sample Explorer - Geographic map + propagation details
 * 3. Waterfall - IQ data visualization
 */

import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Nav } from 'react-bootstrap';
import type { SyntheticDataset, SyntheticSample } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { usePortal } from '@/hooks/usePortal';
import { SampleExplorerTab } from './SampleExplorerTab';
import { WaterfallViewTab } from './WaterfallViewTab';

interface DatasetDetailsDialogProps {
  dataset: SyntheticDataset;
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'overview' | 'explorer' | 'waterfall';

export const DatasetDetailsDialog: React.FC<DatasetDetailsDialogProps> = ({
  dataset,
  isOpen,
  onClose,
}) => {
  const { fetchDatasetSamples } = useTrainingStore();
  const [samples, setSamples] = useState<SyntheticSample[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number>(0);
  
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
      setActiveTab('overview'); // Reset to overview when dialog opens
      setSelectedSampleIdx(0); // Reset to first sample
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

  // Get currently selected sample
  const selectedSample = samples[selectedSampleIdx];
  
  // Check if any sample has IQ data available
  const hasIQData = samples.some(s => s.iq_available);

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
        <div className="modal-dialog modal-xl modal-dialog-scrollable">
          <div className="modal-content">
            {/* Modal Header */}
            <div className="modal-header">
              <div className="d-flex flex-column flex-grow-1">
                <h5 className="modal-title mb-2">Dataset Details: {dataset.name}</h5>
                
                {/* Sample Selector */}
                {!isLoading && samples.length > 0 && (
                  <div className="d-flex align-items-center gap-2">
                    <label className="small text-muted mb-0">Sample:</label>
                    <select 
                      className="form-select form-select-sm" 
                      style={{ width: 'auto', minWidth: '200px' }}
                      value={selectedSampleIdx}
                      onChange={(e) => setSelectedSampleIdx(Number(e.target.value))}
                    >
                      {samples.map((sample, idx) => (
                        <option key={sample.id} value={idx}>
                          #{idx + 1} - {sample.tx_lat.toFixed(4)}, {sample.tx_lon.toFixed(4)} ({sample.num_receivers} RX)
                        </option>
                      ))}
                    </select>
                    <span className="badge bg-secondary">
                      {samples.length} of {dataset.num_samples} loaded
                    </span>
                  </div>
                )}
              </div>
              <button
                type="button"
                className="btn-close"
                onClick={onClose}
                aria-label="Close"
              ></button>
            </div>

            {/* Modal Body */}
            <div className="modal-body" style={{ minHeight: '500px' }}>
              {/* Tab Navigation */}
              <Nav variant="tabs" className="mb-3">
                <Nav.Item>
                  <Nav.Link 
                    active={activeTab === 'overview'} 
                    onClick={() => setActiveTab('overview')}
                    style={{ cursor: 'pointer' }}
                  >
                    <i className="ph ph-info me-2"></i>
                    Overview
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link 
                    active={activeTab === 'explorer'} 
                    onClick={() => setActiveTab('explorer')}
                    disabled={!selectedSample}
                    style={{ cursor: selectedSample ? 'pointer' : 'not-allowed' }}
                  >
                    <i className="ph ph-map-trifold me-2"></i>
                    Sample Explorer
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link 
                    active={activeTab === 'waterfall'} 
                    onClick={() => setActiveTab('waterfall')}
                    disabled={!selectedSample || !hasIQData}
                    style={{ cursor: (selectedSample && hasIQData) ? 'pointer' : 'not-allowed' }}
                  >
                    <i className="ph ph-waveform me-2"></i>
                    Waterfall
                    {!hasIQData && (
                      <span className="badge bg-secondary ms-2 small">No IQ Data</span>
                    )}
                  </Nav.Link>
                </Nav.Item>
              </Nav>

              {/* Tab Content */}
              {isLoading && (
                <div className="text-center py-5">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading samples...</span>
                  </div>
                  <p className="text-muted mt-3">Loading dataset samples...</p>
                </div>
              )}

              {error && (
                <div className="alert alert-danger" role="alert">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                </div>
              )}

              {!isLoading && !error && (
                <>
                  {/* Overview Tab */}
                  {activeTab === 'overview' && (
                    <div>
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

                      {/* Sample Summary */}
                      {samples.length > 0 && (
                        <div className="card border-0">
                          <div className="card-body">
                            <h6 className="card-title mb-3">
                              <i className="ph ph-list-bullets me-2"></i>
                              Sample Summary
                            </h6>
                            <div className="d-flex gap-3 justify-content-around text-center">
                              <div>
                                <div className="h4 mb-1">{dataset.num_samples}</div>
                                <div className="small text-muted">Total Samples</div>
                              </div>
                              <div>
                                <div className="h4 mb-1">{samples.filter(s => s.split === 'train').length}</div>
                                <div className="small text-muted">Training</div>
                              </div>
                              <div>
                                <div className="h4 mb-1">{samples.filter(s => s.split === 'val').length}</div>
                                <div className="small text-muted">Validation</div>
                              </div>
                              <div>
                                <div className="h4 mb-1">{samples.filter(s => s.split === 'test').length}</div>
                                <div className="small text-muted">Test</div>
                              </div>
                              {hasIQData && (
                                <div>
                                  <div className="h4 mb-1 text-success">
                                    <i className="ph ph-check-circle"></i>
                                  </div>
                                  <div className="small text-muted">IQ Data Available</div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Empty State */}
                      {samples.length === 0 && (
                        <div className="text-center py-5">
                          <i className="ph ph-file-search" style={{ fontSize: '3rem', color: 'var(--bs-gray-400)' }}></i>
                          <p className="text-muted mt-3">No samples found in this dataset</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Sample Explorer Tab */}
                  {activeTab === 'explorer' && selectedSample && (
                    <SampleExplorerTab sample={selectedSample} />
                  )}

                  {/* Waterfall Tab */}
                  {activeTab === 'waterfall' && selectedSample && (
                    <WaterfallViewTab 
                      datasetId={dataset.id} 
                      sample={selectedSample} 
                    />
                  )}
                </>
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
