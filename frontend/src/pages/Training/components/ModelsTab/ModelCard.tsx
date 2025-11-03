/**
 * ModelCard Component
 * 
 * Displays a trained model with export action
 */

import React, { useState } from 'react';
import type { TrainedModel } from '../../types';
import { ExportDialog } from './ExportDialog';
import { useTrainingStore } from '../../../../store/trainingStore';

interface ModelCardProps {
  model: TrainedModel;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
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
              <h5 className="mb-1">{model.name}</h5>
              <p className="text-muted small mb-0">Version: {model.version}</p>
            </div>
            <span className="badge bg-light-success">
              TRAINED
            </span>
          </div>

          {/* Model Details */}
          <div className="mb-3">
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Architecture:</span>
              <span className="fw-medium small">{model.architecture}</span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Parameters:</span>
              <span className="fw-medium small">{formatNumber(model.parameters_count)}</span>
            </div>
            <div className="d-flex justify-content-between mb-2">
              <span className="text-muted small">Created:</span>
              <span className="fw-medium small">
                {new Date(model.created_at).toLocaleDateString()}
              </span>
            </div>
            {model.training_job_id && (
              <div className="d-flex justify-content-between">
                <span className="text-muted small">Job ID:</span>
                <code className="small">
                  {model.training_job_id.slice(0, 8)}
                </code>
              </div>
            )}
          </div>

          {/* Final Metrics */}
          {model.final_metrics && (
            <div className="mb-3 p-2 bg-light border rounded">
              <h6 className="small fw-semibold mb-2">Final Metrics</h6>
              <div className="row g-2">
                <div className="col-6">
                  <span className="text-muted small">Train Loss:</span>
                  <span className="ms-2 fw-medium small">{model.final_metrics.train_loss.toFixed(4)}</span>
                </div>
                <div className="col-6">
                  <span className="text-muted small">Val Loss:</span>
                  <span className="ms-2 fw-medium small">{model.final_metrics.val_loss.toFixed(4)}</span>
                </div>
                {model.final_metrics.train_accuracy !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Train Acc:</span>
                    <span className="ms-2 fw-medium small">
                      {(model.final_metrics.train_accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
                {model.final_metrics.val_accuracy !== undefined && (
                  <div className="col-6">
                    <span className="text-muted small">Val Acc:</span>
                    <span className="ms-2 fw-medium small">
                      {(model.final_metrics.val_accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ONNX Path */}
          <div className="mb-3">
            <span className="text-muted small fw-medium">ONNX:</span>
            <code className="ms-2 small text-break">{model.onnx_path}</code>
          </div>

          {/* Action Buttons */}
          <div className="d-grid gap-2">
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
    </>
  );
};
