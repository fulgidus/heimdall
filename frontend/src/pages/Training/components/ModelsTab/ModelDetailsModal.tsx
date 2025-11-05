/**
 * ModelDetailsModal Component
 * 
 * Comprehensive view of all model information including:
 * - Overview (accuracy, loss, epochs, dates)
 * - Hyperparameters (full config)
 * - Training metrics
 * - File information (ONNX/PyTorch paths, sizes)
 * - Dataset information
 * - Actions (export, delete, set active/production, evolve)
 */

import React, { useState, useEffect } from 'react';
import type { TrainedModel } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';
import { EvolveTrainingModal } from './EvolveTrainingModal';
import { InlineEditText } from '../../../../components/InlineEditText';

interface ModelDetailsModalProps {
  model: TrainedModel;
  isOpen: boolean;
  onClose: () => void;
}

interface FileInfo {
  path: string;
  size_bytes?: number;
  size_display?: string;
  exists: boolean;
}

export const ModelDetailsModal: React.FC<ModelDetailsModalProps> = ({ model, isOpen, onClose }) => {
  const { deleteModel, setModelActive, setModelProduction, updateModelName } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);
  const [activeSection, setActiveSection] = useState<string>('overview');
  const [fileInfo, setFileInfo] = useState<{ onnx?: FileInfo; pytorch?: FileInfo }>({});
  const [showEvolveModal, setShowEvolveModal] = useState(false);

  useEffect(() => {
    if (isOpen) {
      // Fetch file sizes if available
      fetchFileInfo();
    }
  }, [isOpen, model.id]);

  const fetchFileInfo = async () => {
    const info: { onnx?: FileInfo; pytorch?: FileInfo } = {};

    if (model.onnx_model_location) {
      info.onnx = {
        path: model.onnx_model_location,
        exists: true,
        // TODO: Fetch actual size from MinIO if needed
        size_display: 'Available'
      };
    }

    if (model.pytorch_model_location) {
      info.pytorch = {
        path: model.pytorch_model_location,
        exists: true,
        size_display: 'Available'
      };
    }

    setFileInfo(info);
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete model "${model.model_name}" (v${model.version})? This action cannot be undone.`)) {
      return;
    }

    setIsLoading(true);
    try {
      await deleteModel(model.id);
      onClose();
    } catch (error) {
      console.error('Failed to delete model:', error);
      alert('Failed to delete model. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSetActive = async () => {
    setIsLoading(true);
    try {
      await setModelActive(model.id);
    } catch (error) {
      console.error('Failed to set model as active:', error);
      alert('Failed to set model as active. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSetProduction = async () => {
    if (!confirm(`Are you sure you want to set model "${model.model_name}" as production? This will affect live inference.`)) {
      return;
    }

    setIsLoading(true);
    try {
      await setModelProduction(model.id);
    } catch (error) {
      console.error('Failed to set model as production:', error);
      alert('Failed to set model as production. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderOverviewSection = () => (
    <div className="row g-3">
      {/* Basic Information */}
      <div className="col-12">
        <div className="card">
          <div className="card-header">
            <h6 className="mb-0">Basic Information</h6>
          </div>
          <div className="card-body">
            <div className="row g-3">
              <div className="col-md-6">
                <label className="form-label text-muted small">Model Name</label>
                <p className="mb-0 fw-medium">
                  <InlineEditText
                    value={model.model_name}
                    onSave={(newName) => updateModelName(model.id, newName)}
                    placeholder="Model Name"
                  />
                </p>
              </div>
              <div className="col-md-6">
                <label className="form-label text-muted small">Version</label>
                <p className="mb-0 fw-medium">{model.version}</p>
              </div>
              <div className="col-md-6">
                <label className="form-label text-muted small">Type</label>
                <p className="mb-0 fw-medium">{model.model_type || 'N/A'}</p>
              </div>
              <div className="col-md-6">
                <label className="form-label text-muted small">Architecture</label>
                <p className="mb-0 fw-medium">{model.hyperparameters?.model_architecture || 'N/A'}</p>
              </div>
              <div className="col-md-6">
                <label className="form-label text-muted small">Created At</label>
                <p className="mb-0 fw-medium">{formatDate(model.created_at)}</p>
              </div>
              <div className="col-md-6">
                <label className="form-label text-muted small">Status</label>
                <div className="d-flex gap-2">
                  {model.is_active && <span className="badge bg-success">ACTIVE</span>}
                  {model.is_production && <span className="badge bg-primary">PRODUCTION</span>}
                  {!model.is_active && !model.is_production && <span className="badge bg-secondary">TRAINED</span>}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="col-12">
        <div className="card">
          <div className="card-header">
            <h6 className="mb-0">Performance Metrics</h6>
          </div>
          <div className="card-body">
            <div className="row g-3">
              <div className="col-md-4">
                <label className="form-label text-muted small">Accuracy (RMSE)</label>
                <p className="mb-0 h5 text-primary">
                  {model.accuracy_meters ? `${model.accuracy_meters.toFixed(2)} m` : 'N/A'}
                </p>
              </div>
              <div className="col-md-4">
                <label className="form-label text-muted small">Accuracy (σ)</label>
                <p className="mb-0 h5 text-info">
                  {model.accuracy_sigma_meters ? `${model.accuracy_sigma_meters.toFixed(2)} m` : 'N/A'}
                </p>
              </div>
              <div className="col-md-4">
                <label className="form-label text-muted small">Final Loss</label>
                <p className="mb-0 h5 text-secondary">
                  {model.loss_value ? model.loss_value.toFixed(4) : 'N/A'}
                </p>
              </div>
              <div className="col-md-4">
                <label className="form-label text-muted small">Epochs Trained</label>
                <p className="mb-0 fw-medium">
                  {model.epoch || 'N/A'} / {model.hyperparameters?.epochs || 'N/A'}
                </p>
              </div>
              {model.training_metrics?.best_epoch !== undefined && (
                <div className="col-md-4">
                  <label className="form-label text-muted small">Best Epoch</label>
                  <p className="mb-0 fw-medium">{model.training_metrics.best_epoch}</p>
                </div>
              )}
              {model.training_metrics?.best_val_loss !== undefined && (
                <div className="col-md-4">
                  <label className="form-label text-muted small">Best Val Loss</label>
                  <p className="mb-0 fw-medium">{model.training_metrics.best_val_loss.toFixed(4)}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Training Job Link */}
      {model.trained_by_job_id && (
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h6 className="mb-0">Training Job</h6>
            </div>
            <div className="card-body">
              <div className="d-flex align-items-center justify-content-between">
                <div>
                  <label className="form-label text-muted small mb-1">Job ID</label>
                  <code className="d-block">{model.trained_by_job_id}</code>
                </div>
                <button className="btn btn-sm btn-outline-primary">
                  View Job Details
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderHyperparametersSection = () => (
    <div className="card">
      <div className="card-header">
        <h6 className="mb-0">Hyperparameters</h6>
      </div>
      <div className="card-body">
        {model.hyperparameters ? (
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(model.hyperparameters).map(([key, value]) => (
                  <tr key={key}>
                    <td className="text-muted">{key}</td>
                    <td>
                      <code>{typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted mb-0">No hyperparameters available</p>
        )}
      </div>
    </div>
  );

  const renderMetricsSection = () => (
    <div className="row g-3">
      {/* Training Metrics */}
      {model.training_metrics && (
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h6 className="mb-0">Training Metrics</h6>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(model.training_metrics).map(([key, value]) => (
                      <tr key={key}>
                        <td className="text-muted">{key}</td>
                        <td>
                          {typeof value === 'number' ? (
                            <span className="fw-medium">{value.toFixed(4)}</span>
                          ) : (
                            <code>{typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}</code>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Metrics */}
      {model.test_metrics && (
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h6 className="mb-0">Test Metrics</h6>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(model.test_metrics).map(([key, value]) => (
                      <tr key={key}>
                        <td className="text-muted">{key}</td>
                        <td>
                          {typeof value === 'number' ? (
                            <span className="fw-medium">{value.toFixed(4)}</span>
                          ) : (
                            <code>{typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}</code>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {!model.training_metrics && !model.test_metrics && (
        <div className="col-12">
          <div className="alert alert-info mb-0">
            <i className="ph ph-info me-2"></i>
            No detailed metrics available for this model
          </div>
        </div>
      )}
    </div>
  );

  const renderFilesSection = () => (
    <div className="row g-3">
      {/* ONNX Model */}
      <div className="col-12">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <h6 className="mb-0">ONNX Model</h6>
            {fileInfo.onnx?.exists && (
              <span className="badge bg-success">Available</span>
            )}
            {!fileInfo.onnx?.exists && (
              <span className="badge bg-danger">Not Exported</span>
            )}
          </div>
          <div className="card-body">
            {fileInfo.onnx?.exists ? (
              <>
                <div className="mb-2">
                  <label className="form-label text-muted small">Path</label>
                  <code className="d-block small bg-light text-dark p-2 rounded">{fileInfo.onnx.path}</code>
                </div>
                {fileInfo.onnx.size_display && (
                  <div>
                    <label className="form-label text-muted small">Size</label>
                    <p className="mb-0 fw-medium">{fileInfo.onnx.size_display}</p>
                  </div>
                )}
              </>
            ) : (
              <p className="text-muted mb-0">ONNX model not exported</p>
            )}
          </div>
        </div>
      </div>

      {/* PyTorch Model */}
      <div className="col-12">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <h6 className="mb-0">PyTorch Model</h6>
            {fileInfo.pytorch?.exists && (
              <span className="badge bg-success">Available</span>
            )}
            {!fileInfo.pytorch?.exists && (
              <span className="badge bg-danger">Not Saved</span>
            )}
          </div>
          <div className="card-body">
            {fileInfo.pytorch?.exists ? (
              <>
                <div className="mb-2">
                  <label className="form-label text-muted small">Path</label>
                  <code className="d-block small bg-light text-dark p-2 rounded">{fileInfo.pytorch.path}</code>
                </div>
                {fileInfo.pytorch.size_display && (
                  <div>
                    <label className="form-label text-muted small">Size</label>
                    <p className="mb-0 fw-medium">{fileInfo.pytorch.size_display}</p>
                  </div>
                )}
              </>
            ) : (
              <p className="text-muted mb-0">PyTorch model not saved</p>
            )}
          </div>
        </div>
      </div>

      {/* MLflow Integration */}
      {model.mlflow_run_id && (
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h6 className="mb-0">MLflow Integration</h6>
            </div>
            <div className="card-body">
              <div className="mb-2">
                <label className="form-label text-muted small">Run ID</label>
                <code className="d-block">{model.mlflow_run_id}</code>
              </div>
              {model.mlflow_experiment_id && (
                <div className="mb-2">
                  <label className="form-label text-muted small">Experiment ID</label>
                  <code className="d-block">{model.mlflow_experiment_id}</code>
                </div>
              )}
              <button className="btn btn-sm btn-outline-primary mt-2">
                <i className="ph ph-arrow-square-out me-1"></i>
                View in MLflow
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderDatasetSection = () => (
    <div className="card">
      <div className="card-header">
        <h6 className="mb-0">Dataset Information</h6>
      </div>
      <div className="card-body">
        {model.synthetic_dataset_id ? (
          <>
            <div className="mb-3">
              <label className="form-label text-muted small">Dataset ID</label>
              <code className="d-block">{model.synthetic_dataset_id}</code>
            </div>
            <button className="btn btn-sm btn-outline-primary">
              <i className="ph ph-database me-1"></i>
              View Dataset Details
            </button>
          </>
        ) : (
          <p className="text-muted mb-0">No dataset information available</p>
        )}
      </div>
    </div>
  );

  const renderActionsSection = () => (
    <div className="card">
      <div className="card-header">
        <h6 className="mb-0">Actions</h6>
      </div>
      <div className="card-body">
        <div className="d-grid gap-2">
          {/* Evolve Model - Continue training from this model's weights */}
          {model.pytorch_model_location && (
            <button
              onClick={() => setShowEvolveModal(true)}
              disabled={isLoading}
              className="btn btn-info d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-arrow-up-right"></i>
              Evolve Model (Train More Epochs)
            </button>
          )}
          
          {!model.is_active && (
            <button
              onClick={handleSetActive}
              disabled={isLoading}
              className="btn btn-success d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-check-circle"></i>
              Set as Active
            </button>
          )}
          {!model.is_production && (
            <button
              onClick={handleSetProduction}
              disabled={isLoading}
              className="btn btn-primary d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-rocket-launch"></i>
              Set as Production
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={isLoading}
            className="btn btn-outline-danger d-flex align-items-center justify-content-center gap-2"
          >
            <i className="ph ph-trash"></i>
            {isLoading ? 'Deleting...' : 'Delete Model'}
          </button>
        </div>
        
        {/* Info about model evolution */}
        {model.pytorch_model_location && (
          <div className="alert alert-info mt-3 mb-0 small">
            <i className="ph ph-info me-2"></i>
            <strong>Model Evolution:</strong> Create a new model by training additional epochs
            starting from this model's learned weights. The new model will have an incremented version number.
          </div>
        )}
      </div>
    </div>
  );

  if (!isOpen) return null;

  return (
    <>
      {/* Evolve Training Modal */}
      <EvolveTrainingModal
        model={model}
        isOpen={showEvolveModal}
        onClose={() => setShowEvolveModal(false)}
        onSuccess={(jobId) => {
          console.log('Evolution training job created:', jobId);
          // Optionally switch to Jobs tab or show notification
        }}
      />

      <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 1050 }}>
        <div className="modal-dialog modal-xl modal-dialog-scrollable" style={{ zIndex: 1051 }}>
          <div className="modal-content">
          {/* Modal Header */}
          <div className="modal-header">
            <div>
              <h5 className="modal-title mb-1">
                <InlineEditText
                  value={model.model_name}
                  onSave={(newName) => updateModelName(model.id, newName)}
                  placeholder="Model Name"
                />
              </h5>
              <p className="text-muted small mb-0">Version {model.version} - {model.model_type || 'N/A'}</p>
            </div>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
              disabled={isLoading}
            ></button>
          </div>

          {/* Modal Body with Tabs */}
          <div className="modal-body">
            {/* Navigation Tabs */}
            <ul className="nav nav-tabs mb-4">
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'overview' ? 'active' : ''}`}
                  onClick={() => setActiveSection('overview')}
                >
                  <i className="ph ph-info me-1"></i>
                  Overview
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'hyperparameters' ? 'active' : ''}`}
                  onClick={() => setActiveSection('hyperparameters')}
                >
                  <i className="ph ph-sliders me-1"></i>
                  Hyperparameters
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'metrics' ? 'active' : ''}`}
                  onClick={() => setActiveSection('metrics')}
                >
                  <i className="ph ph-chart-line me-1"></i>
                  Metrics
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'files' ? 'active' : ''}`}
                  onClick={() => setActiveSection('files')}
                >
                  <i className="ph ph-file me-1"></i>
                  Files
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'dataset' ? 'active' : ''}`}
                  onClick={() => setActiveSection('dataset')}
                >
                  <i className="ph ph-database me-1"></i>
                  Dataset
                </button>
              </li>
              <li className="nav-item">
                <button
                  className={`nav-link ${activeSection === 'actions' ? 'active' : ''}`}
                  onClick={() => setActiveSection('actions')}
                >
                  <i className="ph ph-gear me-1"></i>
                  Actions
                </button>
              </li>
            </ul>

            {/* Tab Content */}
            <div className="tab-content">
              {activeSection === 'overview' && renderOverviewSection()}
              {activeSection === 'hyperparameters' && renderHyperparametersSection()}
              {activeSection === 'metrics' && renderMetricsSection()}
              {activeSection === 'files' && renderFilesSection()}
              {activeSection === 'dataset' && renderDatasetSection()}
              {activeSection === 'actions' && renderActionsSection()}
            </div>
          </div>

          {/* Modal Footer */}
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={isLoading}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
    </>
  );
};
