/**
 * EvolveTrainingModal Component
 * 
 * Modal for evolving an existing model by training additional epochs
 * while preserving the parent model's learned weights
 */

import React, { useState } from 'react';
import type { TrainedModel } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';

interface EvolveTrainingModalProps {
  model: TrainedModel;
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (jobId: string) => void;
}

export const EvolveTrainingModal: React.FC<EvolveTrainingModalProps> = ({
  model,
  isOpen,
  onClose,
  onSuccess,
}) => {
  const { evolveTraining } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);
  const [additionalEpochs, setAdditionalEpochs] = useState(50);
  const [earlyStopPatience, setEarlyStopPatience] = useState(20);

  // Smart step calculation for better slider UX
  const getEpochStep = (value: number): number => {
    if (value <= 100) return 1;
    if (value <= 1000) return 10;
    return 50;
  };

  // Handle slider change with smart stepping
  const handleEpochSliderChange = (rawValue: number) => {
    const step = getEpochStep(rawValue);
    const steppedValue = Math.round(rawValue / step) * step;
    setAdditionalEpochs(Math.max(1, Math.min(5000, steppedValue)));
  };

  // Preset epoch values for quick selection
  const epochPresets = [10, 50, 100, 500, 1000, 5000];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (additionalEpochs < 1) {
      alert('Additional epochs must be at least 1');
      return;
    }

    setIsLoading(true);
    try {
      const jobId = await evolveTraining(model.id, additionalEpochs, earlyStopPatience);
      
      if (onSuccess) {
        onSuccess(jobId);
      }
      
      onClose();
      
      // Show success message
      alert(`Evolution training job created successfully! Job ID: ${jobId}\n\nThe new model will be trained for ${additionalEpochs} additional epochs starting from ${model.model_name} v${model.version} weights.`);
    } catch (error) {
      console.error('Failed to create evolution training job:', error);
      alert('Failed to create evolution training job. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    if (!isLoading) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.7)', zIndex: 1060 }}>
      <div className="modal-dialog modal-lg" style={{ zIndex: 1061 }}>
        <div className="modal-content">
          {/* Modal Header */}
          <div className="modal-header">
            <div>
              <h5 className="modal-title mb-1">
                <i className="ph ph-arrow-up-right me-2"></i>
                Evolve Model
              </h5>
              <p className="text-muted small mb-0">
                Train additional epochs starting from existing weights
              </p>
            </div>
            <button
              type="button"
              className="btn-close"
              onClick={handleCancel}
              disabled={isLoading}
            ></button>
          </div>

          <form onSubmit={handleSubmit}>
            {/* Modal Body */}
            <div className="modal-body">
              {/* Parent Model Info */}
              <div className="alert alert-info d-flex align-items-start gap-3 mb-4">
                <i className="ph ph-info fs-4 mt-1"></i>
                <div className="flex-grow-1">
                  <h6 className="mb-2">Parent Model</h6>
                  <div className="row g-2 small">
                    <div className="col-md-6">
                      <strong>Name:</strong> {model.model_name} v{model.version}
                    </div>
                    <div className="col-md-6">
                      <strong>Accuracy (RMSE):</strong>{' '}
                      {model.accuracy_meters ? `${model.accuracy_meters.toFixed(2)} m` : 'N/A'}
                    </div>
                    <div className="col-md-6">
                      <strong>Architecture:</strong>{' '}
                      {model.hyperparameters?.model_architecture || 'N/A'}
                    </div>
                    <div className="col-md-6">
                      <strong>Epochs Trained:</strong>{' '}
                      {model.epoch || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Configuration */}
              <div className="card mb-3">
                <div className="card-header">
                  <h6 className="mb-0">Evolution Configuration</h6>
                </div>
                <div className="card-body">
                  {/* Additional Epochs */}
                  <div className="mb-4">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                      <label htmlFor="additionalEpochs" className="form-label mb-0">
                        Additional Epochs
                        <span className="text-danger ms-1">*</span>
                      </label>
                      <span className="badge bg-primary">{additionalEpochs} epochs</span>
                    </div>
                    
                    {/* Preset Buttons */}
                    <div className="d-flex gap-2 mb-3 flex-wrap">
                      {epochPresets.map((preset) => (
                        <button
                          key={preset}
                          type="button"
                          className={`btn btn-sm ${additionalEpochs === preset ? 'btn-primary' : 'btn-outline-secondary'}`}
                          onClick={() => setAdditionalEpochs(preset)}
                          disabled={isLoading}
                        >
                          {preset}
                        </button>
                      ))}
                    </div>
                    
                    <input
                      type="range"
                      className="form-range mb-2"
                      id="additionalEpochs"
                      min="1"
                      max="5000"
                      value={additionalEpochs}
                      onChange={(e) => handleEpochSliderChange(parseInt(e.target.value))}
                      disabled={isLoading}
                    />
                    <div className="d-flex justify-content-between align-items-center">
                      <span className="text-muted small">
                        Smart stepping: 1-100 (±1), 101-1000 (±10), 1001-5000 (±50)
                      </span>
                      <div className="input-group" style={{ width: '120px' }}>
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          min="1"
                          max="5000"
                          value={additionalEpochs}
                          onChange={(e) => setAdditionalEpochs(Math.max(1, Math.min(5000, parseInt(e.target.value) || 1)))}
                          disabled={isLoading}
                        />
                        <span className="input-group-text">epochs</span>
                      </div>
                    </div>
                  </div>

                  {/* Early Stop Patience */}
                  <div className="mb-3">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                      <label htmlFor="earlyStopPatience" className="form-label mb-0">
                        Early Stop Patience
                        <span className="text-danger ms-1">*</span>
                      </label>
                      <span className="badge bg-primary">
                        {earlyStopPatience === 0 ? 'Disabled' : `${earlyStopPatience} epochs`}
                      </span>
                    </div>
                    <input
                      type="range"
                      className="form-range mb-2"
                      id="earlyStopPatience"
                      min="0"
                      max="100"
                      value={earlyStopPatience}
                      onChange={(e) => setEarlyStopPatience(parseInt(e.target.value))}
                      disabled={isLoading}
                    />
                    <div className="d-flex justify-content-between align-items-center">
                      <span className="text-muted small">Epochs to wait before stopping early (0 = disabled)</span>
                      <div className="input-group" style={{ width: '120px' }}>
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          min="0"
                          max="100"
                          value={earlyStopPatience}
                          onChange={(e) => setEarlyStopPatience(Math.max(0, parseInt(e.target.value) || 0))}
                          disabled={isLoading}
                        />
                        <span className="input-group-text">epochs</span>
                      </div>
                    </div>
                  </div>

                  {/* Inherited Configuration Notice */}
                  <div className="alert alert-warning d-flex align-items-start gap-2 mb-0">
                    <i className="ph ph-warning mt-1"></i>
                    <div className="small">
                      <strong>Note:</strong> All other hyperparameters (learning rate, batch size, architecture, etc.)
                      will be inherited from the parent model and cannot be changed.
                    </div>
                  </div>
                </div>
              </div>

              {/* What Happens */}
              <div className="card">
                <div className="card-header">
                  <h6 className="mb-0">What Happens During Evolution</h6>
                </div>
                <div className="card-body">
                  <ol className="small mb-0 ps-3">
                    <li className="mb-2">
                      A new training job will be created with the parent model's hyperparameters
                    </li>
                    <li className="mb-2">
                      The parent model's weights will be loaded as the starting point
                    </li>
                    <li className="mb-2">
                      Training will continue for {additionalEpochs} additional epoch{additionalEpochs !== 1 ? 's' : ''}
                    </li>
                    <li className="mb-2">
                      The model version will automatically increment (v{model.version} → v{model.version + 1})
                    </li>
                    <li className="mb-0">
                      The new model will be saved separately without affecting the parent model
                    </li>
                  </ol>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleCancel}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary d-flex align-items-center gap-2"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                    Creating Job...
                  </>
                ) : (
                  <>
                    <i className="ph ph-rocket-launch"></i>
                    Start Evolution Training
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
