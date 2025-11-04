/**
 * CreateJobDialog Component
 * 
 * Modal dialog for creating a new training job
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useTrainingStore } from '../../../../store/trainingStore';
import type { CreateJobRequest, ModelArchitecture, SyntheticDataset } from '../../types';
import api from '../../../../lib/api';

interface CreateJobDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

// Memoized dataset checkbox component to prevent unnecessary re-renders
const DatasetCheckbox = React.memo<{
  dataset: SyntheticDataset;
  isChecked: boolean;
  onToggle: (id: string) => void;
  disabled: boolean;
  isCompatible: boolean;
}>(({ dataset, isChecked, onToggle, disabled, isCompatible }) => {
  const handleChange = useCallback(() => {
    onToggle(dataset.id);
  }, [dataset.id, onToggle]);

  const datasetTypeBadge = dataset.dataset_type === 'iq_raw' 
    ? <span className="badge bg-info text-white ms-1">IQ</span>
    : <span className="badge bg-secondary ms-1">Features</span>;

  return (
    <div className="form-check mb-2">
      <input
        className="form-check-input"
        type="checkbox"
        id={`dataset-${dataset.id}`}
        checked={isChecked}
        onChange={handleChange}
        disabled={disabled || !isCompatible}
      />
      <label 
        className={`form-check-label d-flex justify-content-between align-items-center w-100 ${!isCompatible ? 'text-muted' : ''}`}
        htmlFor={`dataset-${dataset.id}`}
      >
        <span>
          <strong>{dataset.name}</strong> {datasetTypeBadge}
          {dataset.description && (
            <small className="text-muted d-block">{dataset.description}</small>
          )}
          {!isCompatible && (
            <small className="text-warning d-block">
              <i className="ph ph-warning me-1"></i>
              Incompatible with selected architecture
            </small>
          )}
        </span>
        <span className="badge bg-secondary ms-2">
          {dataset.num_samples.toLocaleString()} samples
        </span>
      </label>
    </div>
  );
});

export const CreateJobDialog: React.FC<CreateJobDialogProps> = ({ isOpen, onClose }) => {
  // Use shallow selector to prevent unnecessary re-renders
  const createJob = useTrainingStore(state => state.createJob);
  const datasets = useTrainingStore(state => state.datasets);
  const fetchDatasets = useTrainingStore(state => state.fetchDatasets);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [architectures, setArchitectures] = useState<ModelArchitecture[]>([]);
  const [loadingArchitectures, setLoadingArchitectures] = useState(false);
  // Lazy initialization: only create div once
  const modalRootRef = useRef<HTMLDivElement | null>(null);
  if (!modalRootRef.current) {
    modalRootRef.current = document.createElement('div');
  }
  const isMountedRef = useRef(false);

  const [formData, setFormData] = useState<CreateJobRequest>({
    job_name: '',
    config: {
      dataset_ids: [],
      epochs: 50,
      batch_size: 32,
      learning_rate: 0.001,
      model_architecture: 'triangulation',
      validation_split: 0.2,
      early_stopping_patience: 5,
    },
  });

  // Mount and unmount the modal root element
  useEffect(() => {
    if (isOpen) {
      const modalRoot = modalRootRef.current;
      if (!modalRoot) return;

      // Only append if not already mounted
      if (!isMountedRef.current) {
        document.body.appendChild(modalRoot);
        isMountedRef.current = true;
      }
      
      document.body.style.overflow = 'hidden';

      return () => {
        document.body.style.overflow = '';
        // Clean up: remove the modal root from DOM
        if (modalRoot && modalRoot.parentNode === document.body) {
          document.body.removeChild(modalRoot);
          isMountedRef.current = false;
        }
      };
    }
  }, [isOpen]);

  // Create a Set for fast lookup of selected dataset IDs
  const selectedDatasetIds = useMemo(() => 
    new Set(formData.config.dataset_ids), 
    [formData.config.dataset_ids]
  );

  // Fetch architectures from API
  useEffect(() => {
    if (isOpen && architectures.length === 0) {
      setLoadingArchitectures(true);
      api.get('/v1/training/architectures')
        .then(response => {
          setArchitectures(response.data.architectures || []);
        })
        .catch(error => {
          console.error('Failed to fetch architectures:', error);
          setError('Failed to load model architectures');
        })
        .finally(() => {
          setLoadingArchitectures(false);
        });
    }
  }, [isOpen, architectures.length]);

  // Fetch datasets and reset error when dialog opens (only if not already loaded)
  useEffect(() => {
    if (isOpen) {
      setError(null);
      // Only fetch if we don't have datasets yet
      if (datasets.length === 0) {
        fetchDatasets(true); // Silent fetch to avoid loading indicator
      }
    }
  }, [isOpen]); // Removed fetchDatasets and datasets from deps to prevent loops

  // Get selected architecture metadata
  const selectedArchitecture = useMemo(() => {
    return architectures.find(arch => arch.name === formData.config.model_architecture);
  }, [architectures, formData.config.model_architecture]);

  // Filter datasets based on selected architecture compatibility
  const compatibleDatasets = useMemo(() => {
    if (!selectedArchitecture) return datasets;
    
    const archDataType = selectedArchitecture.data_type;
    
    // 'both' is compatible with everything
    if (archDataType === 'both') return datasets;
    
    // Filter by matching data_type
    return datasets.filter(dataset => {
      const datasetType = dataset.dataset_type || 'feature_based';
      return datasetType === archDataType;
    });
  }, [datasets, selectedArchitecture]);

  // Check if a dataset is compatible with selected architecture
  const isDatasetCompatible = useCallback((dataset: SyntheticDataset): boolean => {
    if (!selectedArchitecture) return true;
    
    const archDataType = selectedArchitecture.data_type;
    if (archDataType === 'both') return true;
    
    const datasetType = dataset.dataset_type || 'feature_based';
    return datasetType === archDataType;
  }, [selectedArchitecture]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const parsedValue = type === 'number' ? parseFloat(value) : value;
    
    setFormData(prev => {
      // Handle nested config fields
      if (name in prev.config) {
        return {
          ...prev,
          config: {
            ...prev.config,
            [name]: parsedValue,
          },
        };
      } else {
        return {
          ...prev,
          [name]: parsedValue,
        };
      }
    });
  }, []);

  const handleDatasetToggle = useCallback((datasetId: string) => {
    setFormData(prev => {
      const currentIds = prev.config.dataset_ids;
      const newIds = currentIds.includes(datasetId)
        ? currentIds.filter(id => id !== datasetId)
        : [...currentIds, datasetId];
      
      return {
        ...prev,
        config: {
          ...prev.config,
          dataset_ids: newIds,
        },
      };
    });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const jobId = await createJob(formData);
      console.log('Training job created:', jobId);
      
      // Reset form
      setFormData({
        job_name: '',
        config: {
          dataset_ids: [],
          epochs: 50,
          batch_size: 32,
          learning_rate: 0.001,
          model_architecture: 'ResNet-18',
          validation_split: 0.2,
          early_stopping_patience: 5,
        },
      });
      
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create training job');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return createPortal(
    <>
      {/* Modal Backdrop */}
      <div className="modal-backdrop fade show" onClick={onClose}></div>

      {/* Modal */}
      <div className="modal fade show" style={{ display: 'block' }} tabIndex={-1}>
        <div className="modal-dialog modal-dialog-centered modal-lg">
          <div className="modal-content">
            {/* Header */}
            <div className="modal-header">
              <h5 className="modal-title">
                <i className="ph ph-plus-circle me-2"></i>
                Create Training Job
              </h5>
              <button 
                type="button" 
                className="btn-close" 
                onClick={onClose}
                disabled={isSubmitting}
              ></button>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit}>
              <div className="modal-body">
                {error && (
                  <div className="alert alert-danger" role="alert">
                    <i className="ph ph-warning-circle me-2"></i>
                    {error}
                  </div>
                )}

                <div className="row g-3">
                  {/* Job Name */}
                  <div className="col-12">
                    <label htmlFor="job_name" className="form-label">
                      Job Name <span className="text-danger">*</span>
                    </label>
                    <input
                      type="text"
                      id="job_name"
                      name="job_name"
                      value={formData.job_name}
                      onChange={handleChange}
                      required
                      className="form-control"
                      placeholder="e.g., localization-model-v1"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Dataset Selection */}
                  <div className="col-12">
                    <label className="form-label">
                      Training Datasets <span className="text-danger">*</span>
                    </label>
                    <div className="card">
                      <div className="card-body" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                        {datasets.length === 0 ? (
                          <p className="text-muted mb-0">
                            <i className="ph ph-info me-2"></i>
                            No datasets available. Create synthetic datasets first.
                          </p>
                        ) : (
                          <div className="form-check-group">
                            {datasets.map(dataset => (
                              <DatasetCheckbox
                                key={dataset.id}
                                dataset={dataset}
                                isChecked={selectedDatasetIds.has(dataset.id)}
                                onToggle={handleDatasetToggle}
                                disabled={isSubmitting}
                                isCompatible={isDatasetCompatible(dataset)}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    {formData.config.dataset_ids.length > 0 && (
                      <small className="form-text text-success">
                        <i className="ph ph-check-circle me-1"></i>
                        {formData.config.dataset_ids.length} dataset(s) selected
                      </small>
                    )}
                  </div>

                  {/* Model Architecture */}
                  <div className="col-12">
                    <label htmlFor="model_architecture" className="form-label">
                      Model Architecture
                    </label>
                    {loadingArchitectures ? (
                      <div className="text-center py-2">
                        <span className="spinner-border spinner-border-sm me-2"></span>
                        Loading architectures...
                      </div>
                    ) : (
                      <>
                        <select
                          id="model_architecture"
                          name="model_architecture"
                          value={formData.config.model_architecture}
                          onChange={handleChange}
                          className="form-select"
                          disabled={isSubmitting || architectures.length === 0}
                        >
                          {architectures.map(arch => (
                            <option key={arch.name} value={arch.name}>
                              {arch.display_name}
                            </option>
                          ))}
                        </select>
                        {selectedArchitecture && (
                          <small className="form-text text-muted">
                            <i className="ph ph-info me-1"></i>
                            {selectedArchitecture.description}
                            {compatibleDatasets.length < datasets.length && (
                              <span className="text-warning d-block mt-1">
                                <i className="ph ph-warning me-1"></i>
                                {compatibleDatasets.length} of {datasets.length} datasets compatible with this architecture
                              </span>
                            )}
                          </small>
                        )}
                      </>
                    )}
                  </div>

                  {/* Epochs */}
                  <div className="col-md-6">
                    <label htmlFor="epochs" className="form-label">
                      Epochs <span className="text-danger">*</span>
                    </label>
                    <input
                      type="number"
                      id="epochs"
                      name="epochs"
                      value={formData.config.epochs}
                      onChange={handleChange}
                      required
                      min="1"
                      max="1000"
                      className="form-control"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Batch Size */}
                  <div className="col-md-6">
                    <label htmlFor="batch_size" className="form-label">
                      Batch Size <span className="text-danger">*</span>
                    </label>
                    <input
                      type="number"
                      id="batch_size"
                      name="batch_size"
                      value={formData.config.batch_size}
                      onChange={handleChange}
                      required
                      min="1"
                      max="512"
                      className="form-control"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Learning Rate */}
                  <div className="col-md-6">
                    <label htmlFor="learning_rate" className="form-label">
                      Learning Rate <span className="text-danger">*</span>
                    </label>
                    <input
                      type="number"
                      id="learning_rate"
                      name="learning_rate"
                      value={formData.config.learning_rate}
                      onChange={handleChange}
                      required
                      min="0.0001"
                      max="1"
                      step="0.0001"
                      className="form-control"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Validation Split */}
                  <div className="col-md-6">
                    <label htmlFor="validation_split" className="form-label">
                      Validation Split
                    </label>
                    <input
                      type="number"
                      id="validation_split"
                      name="validation_split"
                      value={formData.config.validation_split}
                      onChange={handleChange}
                      min="0"
                      max="1"
                      step="0.05"
                      className="form-control"
                      disabled={isSubmitting}
                    />
                    <small className="form-text text-muted">
                      Fraction of data for validation (0.0 - 1.0)
                    </small>
                  </div>

                  {/* Early Stopping Patience */}
                  <div className="col-12">
                    <label htmlFor="early_stopping_patience" className="form-label">
                      Early Stopping Patience
                    </label>
                    <input
                      type="number"
                      id="early_stopping_patience"
                      name="early_stopping_patience"
                      value={formData.config.early_stopping_patience}
                      onChange={handleChange}
                      min="0"
                      max="50"
                      className="form-control"
                      disabled={isSubmitting}
                    />
                    <small className="form-text text-muted">
                      Stop if no improvement after N epochs (0 = disabled)
                    </small>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="modal-footer">
                <button
                  type="button"
                  onClick={onClose}
                  className="btn btn-secondary"
                  disabled={isSubmitting}
                >
                  <i className="ph ph-x me-2"></i>
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting || formData.config.dataset_ids.length === 0}
                  className="btn btn-primary"
                >
                  {isSubmitting ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2"></span>
                      Creating...
                    </>
                  ) : (
                    <>
                      <i className="ph ph-plus me-2"></i>
                      Create Job
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>,
    modalRootRef.current
  );
};
