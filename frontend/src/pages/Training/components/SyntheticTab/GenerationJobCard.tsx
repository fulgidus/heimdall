/**
 * GenerationJobCard Component
 * 
 * Displays a synthetic data generation job with status, progress, and controls
 */

import React, { useState } from 'react';
import type { SyntheticGenerationJob } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';

interface GenerationJobCardProps {
  job: SyntheticGenerationJob;
}

const statusBadges: Record<string, string> = {
  pending: 'bg-light-warning',
  queued: 'bg-light-info',
  running: 'bg-light-primary',
  paused: 'bg-light-secondary',
  completed: 'bg-light-success',
  failed: 'bg-light-danger',
  cancelled: 'bg-light-secondary',
};

export const GenerationJobCard: React.FC<GenerationJobCardProps> = ({ job }) => {
  const { cancelGenerationJob, deleteGenerationJob } = useTrainingStore();
  const [isLoading, setIsLoading] = useState(false);

  const handleCancel = async () => {
    setIsLoading(true);
    try {
      await cancelGenerationJob(job.id);
    } catch (error: any) {
      console.error('Failed to cancel generation job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      await deleteGenerationJob(job.id);
    } catch (error) {
      console.error('Failed to delete generation job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const progressPercent = job.progress_percent || 0;
  const currentSamples = job.current || 0;
  const totalSamples = job.total || job.config?.num_samples || 0;

  return (
    <div className="card">
      <div className="card-body">
        {/* Header */}
        <div className="d-flex justify-content-between align-items-start mb-3">
          <div className="flex-grow-1">
            <h6 className="card-title mb-1">{job.job_name || job.name || job.config?.name || 'Unnamed Job'}</h6>
            <p className="text-muted small mb-0">
              <code className="small">{job.id.slice(0, 8)}</code>
            </p>
          </div>
          <span className={`badge ${statusBadges[job.status] || statusBadges.pending}`}>
            {job.status.toUpperCase()}
          </span>
        </div>

        {/* Progress Bar */}
        {job.status === 'running' && (
          <div className="mb-3">
            <div className="d-flex justify-content-between small text-muted mb-1">
              <span>
                {currentSamples.toLocaleString()} / {totalSamples.toLocaleString()} samples
              </span>
              <span>{progressPercent.toFixed(1)}%</span>
            </div>
            <div className="progress" style={{ height: '8px' }}>
              <div 
                className="progress-bar"
                role="progressbar"
                style={{ width: `${progressPercent}%` }}
                aria-valuenow={progressPercent}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
          </div>
        )}

        {/* Config Details */}
        {job.config && (
          <div className="row g-2 mb-3 small">
            <div className="col-6">
              <span className="text-muted">Target Dataset:</span>
              <div className="fw-medium text-truncate" title={job.config.name}>
                {job.config.name}
              </div>
            </div>
            <div className="col-6">
              <span className="text-muted">Samples:</span>
              <div className="fw-medium">{totalSamples.toLocaleString()}</div>
            </div>
            <div className="col-6">
              <span className="text-muted">Frequency:</span>
              <div className="fw-medium">{job.config.frequency_mhz || 'N/A'} MHz</div>
            </div>
            <div className="col-6">
              <span className="text-muted">Min Receivers:</span>
              <div className="fw-medium">{job.config.min_receivers || 'N/A'}</div>
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="small text-muted mb-3">
          <div>
            <i className="ph ph-calendar-blank me-1"></i>
            Created: {new Date(job.created_at).toLocaleString()}
          </div>
          {job.started_at && (
            <div>
              <i className="ph ph-play me-1"></i>
              Started: {new Date(job.started_at).toLocaleString()}
            </div>
          )}
          {job.completed_at && (
            <div>
              <i className="ph ph-check me-1"></i>
              Completed: {new Date(job.completed_at).toLocaleString()}
            </div>
          )}
          {job.estimated_completion && job.status === 'running' && (
            <div>
              <i className="ph ph-clock me-1"></i>
              ETA: {new Date(job.estimated_completion).toLocaleString()}
            </div>
          )}
        </div>

        {/* Error Message */}
        {job.error_message && (
          <div className="alert alert-danger py-2 small mb-3">
            <i className="ph ph-warning-circle me-1"></i>
            <strong>Error:</strong> {job.error_message}
          </div>
        )}

        {/* Success Message for Completed */}
        {job.status === 'completed' && job.dataset_id && (
          <div className="alert alert-success py-2 small mb-3">
            <i className="ph ph-check-circle me-1"></i>
            <strong>Complete!</strong> Dataset updated successfully.
          </div>
        )}

        {/* Action Buttons */}
        <div className="d-flex gap-2">
          {/* Cancel button for pending, queued, or running jobs */}
          {(job.status === 'pending' || job.status === 'queued' || job.status === 'running') && (
            <button
              onClick={handleCancel}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger w-100"
            >
              <i className="ph ph-x me-1"></i>
              {isLoading ? 'Cancelling...' : 'Cancel'}
            </button>
          )}

          {/* Delete button for completed/failed/cancelled jobs */}
          {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
            <button
              onClick={handleDelete}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger w-100"
            >
              <i className="ph ph-trash me-1"></i>
              {isLoading ? 'Deleting...' : 'Delete Job'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
