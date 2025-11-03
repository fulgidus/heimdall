/**
 * ExpandDatasetDialog Component
 * 
 * Dialog for expanding an existing synthetic dataset with additional samples
 */

import React, { useState } from 'react';
import type { SyntheticDataset } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';

interface ExpandDatasetDialogProps {
  dataset: SyntheticDataset;
  isOpen: boolean;
  onClose: () => void;
}

export const ExpandDatasetDialog: React.FC<ExpandDatasetDialogProps> = ({
  dataset,
  isOpen,
  onClose,
}) => {
  const { expandDataset } = useTrainingStore();
  const [numSamples, setNumSamples] = useState(10000);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      await expandDataset({
        dataset_id: dataset.id,
        num_additional_samples: numSamples,
      });
      
      // Success - close dialog
      onClose();
      setNumSamples(10000); // Reset
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to expand dataset');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setError(null);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">
              <i className="ph ph-plus-circle me-2"></i>
              Expand Dataset
            </h5>
            <button
              type="button"
              className="btn-close"
              onClick={handleClose}
              disabled={isSubmitting}
              aria-label="Close"
            ></button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="modal-body">
              {/* Dataset Info */}
              <div className="alert alert-info mb-3">
                <strong>Dataset:</strong> {dataset.name}
                <br />
                <strong>Current samples:</strong> {dataset.num_samples.toLocaleString()}
                <br />
                <small className="text-muted">
                  New samples will use the same configuration as the original dataset
                </small>
              </div>

              {/* Error Alert */}
              {error && (
                <div className="alert alert-danger" role="alert">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                </div>
              )}

              {/* Number of Samples */}
              <div className="mb-3">
                <label htmlFor="numSamples" className="form-label">
                  Additional Samples <span className="text-danger">*</span>
                </label>
                <input
                  type="number"
                  className="form-control"
                  id="numSamples"
                  value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value) || 1000)}
                  min={1000}
                  max={5000000}
                  step={1000}
                  required
                  disabled={isSubmitting}
                />
                <div className="form-text">
                  Min: 1,000 - Max: 5,000,000
                </div>
              </div>

              {/* Preview */}
              <div className="bg-light p-3 rounded">
                <div className="d-flex justify-content-between mb-2">
                  <span className="text-muted">Current samples:</span>
                  <strong>{dataset.num_samples.toLocaleString()}</strong>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span className="text-muted">Additional samples:</span>
                  <strong className="text-primary">+{numSamples.toLocaleString()}</strong>
                </div>
                <hr className="my-2" />
                <div className="d-flex justify-content-between">
                  <span className="fw-semibold">Total after expansion:</span>
                  <strong className="text-success">
                    {(dataset.num_samples + numSamples).toLocaleString()}
                  </strong>
                </div>
              </div>

              <div className="alert alert-warning mt-3 mb-0">
                <small>
                  <i className="ph ph-warning me-1"></i>
                  <strong>Note:</strong> This will create a new generation job. 
                  You can monitor progress and control the job in the "Active Generation Jobs" section.
                </small>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClose}
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Creating Job...
                  </>
                ) : (
                  <>
                    <i className="ph ph-play me-2"></i>
                    Start Expansion
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
