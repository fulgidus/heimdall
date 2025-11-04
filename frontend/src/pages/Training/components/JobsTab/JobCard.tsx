/**
 * JobCard Component
 * 
 * Displays a training job with status, progress, and actions
 */

import React from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import type { TrainingJob } from '../../types';

interface JobCardProps {
  job: TrainingJob;
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

export const JobCard: React.FC<JobCardProps> = ({ job }) => {
  const { pauseJob, resumeJob, cancelJob, deleteJob } = useTrainingStore();
  const [isLoading, setIsLoading] = React.useState(false);

  // Connect to WebSocket for real-time updates if job is running
  useTrainingWebSocket(job.status === 'running' ? job.id : null);

  const handlePause = async () => {
    setIsLoading(true);
    try {
      await pauseJob(job.id);
    } catch (error) {
      console.error('Failed to pause job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResume = async () => {
    setIsLoading(true);
    try {
      await resumeJob(job.id);
    } catch (error) {
      console.error('Failed to resume job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = async () => {
    setIsLoading(true);
    try {
      await cancelJob(job.id);
    } catch (error: any) {
      console.error('Failed to cancel job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      await deleteJob(job.id);
    } catch (error) {
      console.error('Failed to delete job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const progressPercent = job.progress_percent || 0;
  const currentEpoch = job.current_epoch || 0;
  const totalEpochs = job.total_epochs || 0;

  return (
    <div className="card">
      <div className="card-body">
        {/* Header */}
        <div className="d-flex justify-content-between align-items-start mb-3">
          <div className="flex-grow-1">
            <h5 className="card-title mb-1">{job.job_name || job.name || 'Unnamed Job'}</h5>
            <p className="text-muted small mb-0">ID: {job.id.slice(0, 8)}</p>
          </div>
          <span className={`badge ${statusBadges[job.status] || statusBadges.pending}`}>
            {job.status.toUpperCase()}
          </span>
        </div>

        {/* Progress Bar */}
        {(job.status === 'running' || job.status === 'paused') && (
          <div className="mb-3">
            <div className="d-flex justify-content-between small text-muted mb-1">
              <span>Epoch {currentEpoch} / {totalEpochs}</span>
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
        <div className="row g-2 mb-3 small">
          <div className="col-6">
            <span className="text-muted">Architecture:</span>
            <div className="fw-medium">{job.config.model_architecture || 'ResNet-18'}</div>
          </div>
          <div className="col-6">
            <span className="text-muted">Batch Size:</span>
            <div className="fw-medium">{job.config.batch_size}</div>
          </div>
          <div className="col-6">
            <span className="text-muted">Learning Rate:</span>
            <div className="fw-medium">{job.config.learning_rate}</div>
          </div>
          <div className="col-6">
            <span className="text-muted">Epochs:</span>
            <div className="fw-medium">{totalEpochs}</div>
          </div>
        </div>

        {/* Timestamps */}
        <div className="small text-muted mb-3">
          <div><i className="ph ph-calendar-blank me-1"></i>Created: {new Date(job.created_at).toLocaleString()}</div>
          {job.started_at && (
            <div><i className="ph ph-play me-1"></i>Started: {new Date(job.started_at).toLocaleString()}</div>
          )}
          {job.completed_at && (
            <div><i className="ph ph-check me-1"></i>Completed: {new Date(job.completed_at).toLocaleString()}</div>
          )}
          {job.estimated_completion && job.status === 'running' && (
            <div><i className="ph ph-clock me-1"></i>ETA: {new Date(job.estimated_completion).toLocaleString()}</div>
          )}
        </div>

        {/* Error Message */}
        {job.error_message && (
          <div className="alert alert-danger py-2 small mb-3">
            <i className="ph ph-warning-circle me-1"></i>
            <strong>Error:</strong> {job.error_message}
          </div>
        )}

        {/* Action Buttons */}
        <div className="d-flex gap-2">
          {job.status === 'running' && (
            <>
              <button
                onClick={handlePause}
                disabled={isLoading}
                className="btn btn-sm btn-outline-warning flex-fill"
              >
                <i className="ph ph-pause me-1"></i>
                {isLoading ? 'Pausing...' : 'Pause'}
              </button>
              <button
                onClick={handleCancel}
                disabled={isLoading}
                className="btn btn-sm btn-outline-danger flex-fill"
              >
                <i className="ph ph-x me-1"></i>
                {isLoading ? 'Cancelling...' : 'Cancel'}
              </button>
            </>
          )}
          
          {job.status === 'paused' && (
            <button
              onClick={handleResume}
              disabled={isLoading}
              className="btn btn-sm btn-outline-primary w-100"
            >
              <i className="ph ph-play me-1"></i>
              {isLoading ? 'Resuming...' : 'Resume'}
            </button>
          )}

          {(job.status === 'pending' || job.status === 'queued') && (
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
