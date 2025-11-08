/**
 * ExportDialog Component
 * 
 * Modal for configuring .heimdall model export options
 */

import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import type { TrainedModel, ExportOptions } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { usePortal } from '@/hooks/usePortal';

interface ExportDialogProps {
  model: TrainedModel;
  isOpen: boolean;
  onClose: () => void;
}

export const ExportDialog: React.FC<ExportDialogProps> = ({ model, isOpen, onClose }) => {
  const { downloadModel } = useTrainingStore();
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(isOpen);

  const [options, setOptions] = useState<ExportOptions>({
    include_config: true,
    include_metrics: true,
    include_normalization: true,
    include_samples: false,
    num_samples: 100,
    description: '',
  });

  const handleCheckboxChange = (key: keyof ExportOptions) => {
    setOptions(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleExport = async () => {
    setIsExporting(true);
    setError(null);

    try {
      await downloadModel(model.id, options);
      onClose();
      // Reset options to defaults
      setOptions({
        include_config: true,
        include_metrics: true,
        include_normalization: true,
        include_samples: false,
        num_samples: 100,
        description: '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  if (!isOpen || !portalTarget) return null;

  return createPortal(
    <>
      {/* Modal Backdrop */}
      <div className="modal-backdrop fade show" onClick={onClose}></div>

      {/* Modal */}
      <div className="modal fade show" style={{ display: 'block' }} tabIndex={-1}>
        <div className="modal-dialog modal-dialog-scrollable">
          <div className="modal-content">
            {/* Header */}
            <div className="modal-header">
              <h5 className="modal-title">
                <i className="ph ph-download-simple me-2"></i>
                Export Model
              </h5>
              <button 
                type="button" 
                className="btn-close" 
                onClick={onClose}
                disabled={isExporting}
              ></button>
            </div>

            {/* Body */}
            <div className="modal-body">
              <p className="text-muted small mb-4">
                {model.name} v{model.version}
              </p>

              {/* Error Message */}
              {error && (
                <div className="alert alert-danger">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                </div>
              )}

              {/* Export Options */}
              <div className="mb-4">
                <h6 className="mb-3">
                  <i className="ph ph-gear me-2"></i>
                  Include in Export
                </h6>

                {/* Training Configuration */}
                <div className="form-check mb-3">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    id="include_config"
                    checked={options.include_config}
                    onChange={() => handleCheckboxChange('include_config')}
                    disabled={isExporting}
                  />
                  <label className="form-check-label" htmlFor="include_config">
                    <div className="fw-medium">Training Configuration</div>
                    <small className="text-muted">Hyperparameters, architecture, dataset info</small>
                  </label>
                </div>

                {/* Metrics History */}
                <div className="form-check mb-3">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    id="include_metrics"
                    checked={options.include_metrics}
                    onChange={() => handleCheckboxChange('include_metrics')}
                    disabled={isExporting}
                  />
                  <label className="form-check-label" htmlFor="include_metrics">
                    <div className="fw-medium">Metrics History</div>
                    <small className="text-muted">Loss, accuracy, learning rate per epoch</small>
                  </label>
                </div>

                {/* Normalization Parameters */}
                <div className="form-check mb-3">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    id="include_normalization"
                    checked={options.include_normalization}
                    onChange={() => handleCheckboxChange('include_normalization')}
                    disabled={isExporting}
                  />
                  <label className="form-check-label" htmlFor="include_normalization">
                    <div className="fw-medium">Normalization Parameters</div>
                    <small className="text-muted">Feature scaling coefficients (mean, std)</small>
                  </label>
                </div>

                {/* Sample Data */}
                <div className="form-check mb-3">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    id="include_samples"
                    checked={options.include_samples}
                    onChange={() => handleCheckboxChange('include_samples')}
                    disabled={isExporting}
                  />
                  <label className="form-check-label" htmlFor="include_samples">
                    <div className="fw-medium">Sample Data</div>
                    <small className="text-muted">Test examples for verification</small>
                  </label>
                </div>

                {/* Number of Samples (conditional) */}
                {options.include_samples && (
                  <div className="ms-4 mb-3">
                    <label htmlFor="num_samples" className="form-label small fw-medium">
                      Number of Samples
                    </label>
                    <input
                      type="number"
                      className="form-control"
                      id="num_samples"
                      min="1"
                      max="1000"
                      value={options.num_samples}
                      onChange={(e) => setOptions(prev => ({ ...prev, num_samples: parseInt(e.target.value) }))}
                      disabled={isExporting}
                    />
                    <small className="text-muted">Max: 1000 samples</small>
                  </div>
                )}
              </div>

              {/* Description (optional) */}
              <div className="mb-3">
                <label htmlFor="description" className="form-label">
                  Description (Optional)
                </label>
                <textarea
                  className="form-control"
                  id="description"
                  rows={3}
                  value={options.description}
                  onChange={(e) => setOptions(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Add notes about this model export..."
                  disabled={isExporting}
                />
              </div>

              {/* File Format Info */}
              <div className="alert alert-info mb-0">
                <i className="ph ph-info me-2"></i>
                <strong>File Format:</strong> .heimdall (JSON bundle with base64-encoded ONNX model)
              </div>
            </div>

            {/* Footer */}
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={onClose}
                disabled={isExporting}
              >
                <i className="ph ph-x me-2"></i>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleExport}
                disabled={isExporting}
              >
                {isExporting ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2"></span>
                    Exporting...
                  </>
                ) : (
                  <>
                    <i className="ph ph-download-simple me-2"></i>
                    Export
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>,
    portalTarget
  );
};
