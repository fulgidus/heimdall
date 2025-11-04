/**
 * ModelCard Component
 * 
 * Displays a trained model with export action
 */

import React, { useState } from 'react';
import type { TrainedModel } from '../../types';
import { ExportDialog } from './ExportDialog';
import { ModelDetailsModal } from './ModelDetailsModal';
import { useTrainingStore } from '../../../../store/trainingStore';

interface ModelCardProps {
  model: TrainedModel;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [isDetailsModalOpen, setIsDetailsModalOpen] = useState(false);
  const { deleteModel } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);

  const formatNumber = (num: number | undefined) => {
    if (num === undefined) return 'N/A';
    if (num > 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (num > 1_000) return `${(num / 1_000).toFixed(2)}K`;
    return num.toFixed(0);
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      await deleteModel(model.id);
    } catch (error) {
      console.error('Failed to delete model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="card h-100">
        <div className="card-body">
          {/* Header */}
          <div className="d-flex justify-content-between align-items-start mb-3">
            <div className="flex-grow-1">
              <h5 className="mb-1">{model.model_name}</h5>
              <p className="text-muted small mb-0">
                Version: {model.version} | Type: {model.model_type || 'N/A'}
              </p>
            </div>
            <div className="d-flex gap-2">
              {model.is_active && (
                <span className="badge bg-success">
                  ACTIVE
                </span>
              )}
              {model.is_production && (
                <span className="badge bg-primary">
                  PRODUCTION
                </span>
              )}
              {!model.is_active && !model.is_production && (
                <span className="badge bg-light-success">
                  TRAINED
                </span>
              )}
            </div>
          </div>

          {/* Model Details */}
          <div className="mb-3">
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Architecture:</span>
              <span className="fw-medium small">
                {model.hyperparameters?.model_architecture || 'N/A'}
              </span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Accuracy (RMSE):</span>
              <span className="fw-medium small">
                {model.accuracy_meters ? `${model.accuracy_meters.toFixed(2)}m` : 'N/A'}
              </span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Final Loss:</span>
              <span className="fw-medium small">
                {model.loss_value ? model.loss_value.toFixed(4) : 'N/A'}
              </span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Epochs Trained:</span>
              <span className="fw-medium small">
                {model.epoch || 'N/A'} / {model.hyperparameters?.epochs || 'N/A'}
              </span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Created:</span>
              <span className="fw-medium small">
                {new Date(model.created_at).toLocaleDateString()}
              </span>
            </div>
            {model.trained_by_job_id && (
              <div className="d-flex justify-content-between">
                <span className="text-muted small">Job ID:</span>
                <code className="small">
                  {model.trained_by_job_id.slice(0, 8)}
                </code>
              </div>
            )}
          </div>

          {/* Training Metrics */}
          {model.training_metrics && (
            <div className="mb-3 p-2 bg-light border rounded">
              <h6 className="small fw-semibold mb-2">Training Metrics</h6>
              <div className="row g-2">
                {model.training_metrics.best_val_loss !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Best Val Loss:</span>
                    <span className="ms-2 fw-medium small">
                      {model.training_metrics.best_val_loss.toFixed(4)}
                    </span>
                  </div>
                )}
                {model.training_metrics.best_epoch !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Best Epoch:</span>
                    <span className="ms-2 fw-medium small">
                      {model.training_metrics.best_epoch}
                    </span>
                  </div>
                )}
                {model.training_metrics.final_val_rmse !== undefined && (
                  <div className="col-12">
                    <span className="text-muted small">Final Val RMSE:</span>
                    <span className="ms-2 fw-medium small">
                      {model.training_metrics.final_val_rmse.toFixed(2)}m
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Model Files */}
          <div className="mb-3">
            <div className="d-flex justify-content-between mb-1">
              <span className="text-muted small fw-medium">ONNX:</span>
              <span className={`small ${model.onnx_model_location ? 'text-success' : 'text-danger'}`}>
                {model.onnx_model_location ? '✓ Available' : '✗ Not exported'}
              </span>
            </div>
            <div className="d-flex justify-content-between">
              <span className="text-muted small fw-medium">PyTorch:</span>
              <span className={`small ${model.pytorch_model_location ? 'text-success' : 'text-danger'}`}>
                {model.pytorch_model_location ? '✓ Available' : '✗ Not saved'}
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="d-grid gap-2">
            <button
              onClick={() => setIsDetailsModalOpen(true)}
              className="btn btn-outline-primary d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-info"></i>
              View Details
            </button>
            <button
              onClick={() => setIsExportDialogOpen(true)}
              className="btn btn-primary d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-download-simple"></i>
              Export Model
            </button>
            <button
              onClick={handleDelete}
              disabled={isLoading}
              className="btn btn-outline-danger d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-trash"></i>
              {isLoading ? 'Deleting...' : 'Delete Model'}
            </button>
          </div>
        </div>
      </div>

      {/* Export Dialog */}
      <ExportDialog
        model={model}
        isOpen={isExportDialogOpen}
        onClose={() => setIsExportDialogOpen(false)}
      />

      {/* Details Modal */}
      <ModelDetailsModal
        model={model}
        isOpen={isDetailsModalOpen}
        onClose={() => setIsDetailsModalOpen(false)}
      />
    </>
  );
};
