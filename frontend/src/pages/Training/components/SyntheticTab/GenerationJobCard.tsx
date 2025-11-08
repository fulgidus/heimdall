/**
 * GenerationJobCard Component
 * 
 * Displays a synthetic data generation job with status, progress, and controls
 */

import React, { useState, useEffect, useRef } from 'react';
import type { SyntheticGenerationJob } from '../../types';
import { useTrainingStore } from '../../../../store/trainingStore';

interface GenerationJobCardProps {
  job: SyntheticGenerationJob;
}

interface ProgressSnapshot {
  samples: number;
  timestamp: number;
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
  const [samplesPerSecond, setSamplesPerSecond] = useState<number | null>(null);
  const [timeToCompletion, setTimeToCompletion] = useState<number | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteDatasetToo, setDeleteDatasetToo] = useState(false);
  
  // Track progress history for rate calculation (keep last 5 snapshots)
  const progressHistory = useRef<ProgressSnapshot[]>([]);
  const lastUpdateTime = useRef<number>(Date.now());

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
      await deleteGenerationJob(job.id, deleteDatasetToo);
      setShowDeleteDialog(false);
    } catch (error) {
      console.error('Failed to delete generation job:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const openDeleteDialog = () => {
    setDeleteDatasetToo(false); // Reset checkbox
    setShowDeleteDialog(true);
  };

  const progressPercent = job.progress_percent || 0;
  const currentSamples = job.current || 0;
  const totalSamples = job.total || job.config?.num_samples || 0;

  // Calculate Samples/s and TTC when progress updates
  useEffect(() => {
    if (job.status !== 'running' || currentSamples === 0) {
      setSamplesPerSecond(null);
      setTimeToCompletion(null);
      progressHistory.current = [];
      return;
    }

    const now = Date.now();
    
    // Add current snapshot to history
    progressHistory.current.push({
      samples: currentSamples,
      timestamp: now,
    });

    // Keep only last 5 snapshots (for smoothing)
    if (progressHistory.current.length > 5) {
      progressHistory.current.shift();
    }

    // Calculate rate if we have at least 2 snapshots
    if (progressHistory.current.length >= 2) {
      const oldest = progressHistory.current[0];
      const newest = progressHistory.current[progressHistory.current.length - 1];
      
      const samplesDelta = newest.samples - oldest.samples;
      const timeDelta = (newest.timestamp - oldest.timestamp) / 1000; // Convert to seconds

      if (timeDelta > 0 && samplesDelta > 0) {
        const rate = samplesDelta / timeDelta;
        setSamplesPerSecond(rate);

        // Calculate TTC
        const remainingSamples = totalSamples - currentSamples;
        const secondsRemaining = remainingSamples / rate;
        setTimeToCompletion(secondsRemaining);
      }
    }

    lastUpdateTime.current = now;
  }, [currentSamples, totalSamples, job.status]);

  // Format TTC as human-readable string
  const formatTTC = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${minutes}m ${secs}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

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
            
            {/* Samples/s and TTC */}
            {samplesPerSecond !== null && timeToCompletion !== null && (
              <div className="d-flex justify-content-between small text-muted mt-2">
                <span>
                  <i className="ph ph-lightning me-1"></i>
                  <strong>{samplesPerSecond.toFixed(1)}</strong> samples/s
                </span>
                <span>
                  <i className="ph ph-clock me-1"></i>
                  TTC: <strong>{formatTTC(timeToCompletion)}</strong>
                </span>
              </div>
            )}

            {/* Rejection Statistics (if available) */}
            {job.total_attempted !== undefined && job.total_attempted > 0 && (
              <div className="mt-2 small">
                <div className="text-muted mb-1">
                  <i className="ph ph-funnel me-1"></i>
                  Validation: {currentSamples}/{job.total_attempted} accepted ({((currentSamples / job.total_attempted) * 100).toFixed(1)}% success)
                </div>
                {job.total_rejections !== undefined && job.total_rejections > 0 && (
                  <div className="d-flex gap-2 flex-wrap">
                    {job.rejected_min_receivers !== undefined && job.rejected_min_receivers > 0 && (
                      <span className="badge bg-light-warning">
                        <i className="ph ph-users me-1"></i>
                        {job.rejected_min_receivers} receivers
                      </span>
                    )}
                    {job.rejected_min_snr !== undefined && job.rejected_min_snr > 0 && (
                      <span className="badge bg-light-warning">
                        <i className="ph ph-broadcast me-1"></i>
                        {job.rejected_min_snr} SNR
                      </span>
                    )}
                    {job.rejected_gdop !== undefined && job.rejected_gdop > 0 && (
                      <span className="badge bg-light-warning">
                        <i className="ph ph-crosshair me-1"></i>
                        {job.rejected_gdop} GDOP
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}
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
              onClick={openDeleteDialog}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger w-100"
            >
              <i className="ph ph-trash me-1"></i>
              Delete Job
            </button>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteDialog && (
        <div 
          className="modal show d-block" 
          style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
          onClick={() => setShowDeleteDialog(false)}
        >
          <div className="modal-dialog modal-dialog-centered" onClick={(e) => e.stopPropagation()}>
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">
                  <i className="ph ph-warning-circle text-warning me-2"></i>
                  Delete Generation Job
                </h5>
                <button 
                  type="button" 
                  className="btn-close" 
                  onClick={() => setShowDeleteDialog(false)}
                  disabled={isLoading}
                ></button>
              </div>
              <div className="modal-body">
                <p className="mb-3">
                  Are you sure you want to delete job <code>{job.id.slice(0, 8)}</code>?
                </p>
                
                {job.dataset_id && (
                  <div className="alert alert-info py-2 small mb-3">
                    <i className="ph ph-info me-2"></i>
                    This job created a dataset. You can optionally delete it too.
                  </div>
                )}

                {/* Checkbox to delete dataset */}
                {job.dataset_id && (
                  <div className="form-check mb-3">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="deleteDatasetCheckbox"
                      checked={deleteDatasetToo}
                      onChange={(e) => setDeleteDatasetToo(e.target.checked)}
                      disabled={isLoading}
                    />
                    <label className="form-check-label" htmlFor="deleteDatasetCheckbox">
                      <strong className="text-danger">Also delete the dataset</strong>
                      <div className="small text-muted mt-1">
                        Warning: This will permanently delete all generated samples and associated data from MinIO.
                        This action cannot be undone!
                      </div>
                    </label>
                  </div>
                )}

                {!job.dataset_id && (
                  <div className="alert alert-secondary py-2 small">
                    <i className="ph ph-info me-2"></i>
                    This job did not create any dataset.
                  </div>
                )}
              </div>
              <div className="modal-footer">
                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  onClick={() => setShowDeleteDialog(false)}
                  disabled={isLoading}
                >
                  Cancel
                </button>
                <button 
                  type="button" 
                  className="btn btn-danger" 
                  onClick={handleDelete}
                  disabled={isLoading}
                >
                  <i className="ph ph-trash me-1"></i>
                  {isLoading ? 'Deleting...' : (deleteDatasetToo ? 'Delete Job & Dataset' : 'Delete Job Only')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
