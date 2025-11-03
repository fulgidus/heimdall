/**
 * CreateJobDialog Component
 * 
 * Modal dialog for creating a new training job
 */

import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useTrainingStore } from '../../../../store/trainingStore';
import type { CreateJobRequest } from '../../types';

interface CreateJobDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const modelArchitectures = [
  'ResNet-18',
  'ResNet-34',
  'ResNet-50',
  'EfficientNet-B0',
  'MobileNetV2',
];

export const CreateJobDialog: React.FC<CreateJobDialogProps> = ({ isOpen, onClose }) => {
  const { createJob } = useTrainingStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const modalRootRef = useRef<HTMLDivElement>(document.createElement('div'));
  const isMountedRef = useRef(false);

  const [formData, setFormData] = useState<CreateJobRequest>({
    job_name: '',
    epochs: 50,
    batch_size: 32,
    learning_rate: 0.001,
    model_architecture: 'ResNet-18',
    validation_split: 0.2,
    early_stopping_patience: 5,
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

  // Reset error when dialog opens
  useEffect(() => {
    if (isOpen) {
      setError(null);
    }
  }, [isOpen]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value,
    }));
  };

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
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        model_architecture: 'ResNet-18',
        validation_split: 0.2,
        early_stopping_patience: 5,
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

                  {/* Model Architecture */}
                  <div className="col-12">
                    <label htmlFor="model_architecture" className="form-label">
                      Model Architecture
                    </label>
                    <select
                      id="model_architecture"
                      name="model_architecture"
                      value={formData.model_architecture}
                      onChange={handleChange}
                      className="form-select"
                      disabled={isSubmitting}
                    >
                      {modelArchitectures.map(arch => (
                        <option key={arch} value={arch}>{arch}</option>
                      ))}
                    </select>
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
                      value={formData.epochs}
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
                      value={formData.batch_size}
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
                      value={formData.learning_rate}
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
                      value={formData.validation_split}
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
                      value={formData.early_stopping_patience}
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
                  disabled={isSubmitting}
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
