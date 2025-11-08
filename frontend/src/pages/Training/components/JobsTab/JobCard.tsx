/**
 * JobCard Component
 * 
 * Displays a training job with status, progress, and actions
 */

import React from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
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

// Interface for sparkline data points
interface MetricPoint {
  epoch: number;
  loss: number;
  distance?: number;
}

export const JobCard: React.FC<JobCardProps> = ({ job }) => {
  const { pauseJob, resumeJob, cancelJob, deleteJob, metrics } = useTrainingStore();
  const [isLoading, setIsLoading] = React.useState(false);

  // Connect to WebSocket for real-time updates if job is running
  useTrainingWebSocket(job.status === 'running' ? job.id : null);
  
  // Get metrics history from store for sparklines
  const jobMetrics = metrics.get(job.id) || [];
  
  // Prepare sparkline data from metrics (last 100 points)
  const lossHistory = React.useMemo(() => {
    return jobMetrics
      .filter(m => m.val_loss !== undefined && !isNaN(m.val_loss) && m.val_loss < 999999)
      .map(m => ({ epoch: m.epoch, loss: m.val_loss }))
      .slice(-100);
  }, [jobMetrics]);
  
  const distanceHistory = React.useMemo(() => {
    return jobMetrics
      .filter(m => m.val_distance_p68_m !== undefined && !isNaN(m.val_distance_p68_m))
      .map(m => ({ epoch: m.epoch, distance: m.val_distance_p68_m! }))
      .slice(-100);
  }, [jobMetrics]);
  
  // Log when job updates
  React.useEffect(() => {
    console.log('[JobCard] Job updated:', {
      id: job.id.slice(0, 8),
      status: job.status,
      progress: job.progress_percent,
      epoch: job.current_epoch,
      total: job.total_epochs,
    });
  }, [job.id, job.status, job.progress_percent, job.current_epoch, job.total_epochs]);

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
    console.log('[JobCard] handleCancel called for job:', {
      id: job.id.slice(0, 8),
      status: job.status,
      timestamp: new Date().toISOString(),
    });
    setIsLoading(true);
    try {
      await cancelJob(job.id);
      console.log('[JobCard] cancelJob successful');
    } catch (error: any) {
      console.error('[JobCard] Failed to cancel job:', error);
      console.error('[JobCard] Error details:', {
        message: error?.message,
        response: error?.response?.data,
        status: error?.response?.status,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    console.log('[JobCard] handleDelete called for job:', {
      id: job.id.slice(0, 8),
      status: job.status,
      timestamp: new Date().toISOString(),
    });
    setIsLoading(true);
    try {
      await deleteJob(job.id);
      console.log('[JobCard] deleteJob successful');
    } catch (error: any) {
      console.error('[JobCard] Failed to delete job:', error);
      console.error('[JobCard] Error details:', {
        message: error?.message,
        response: error?.response?.data,
        status: error?.response?.status,
      });
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
            
            {/* Real-time Loss Display */}
            {(job.train_loss != null || job.val_loss != null) && (
              <div className="d-flex justify-content-between small mt-2">
                {job.train_loss != null && (
                  <span className={job.train_loss > 100 ? 'text-danger fw-medium' : 'text-muted'}>
                    <i className="ph ph-chart-line me-1"></i>
                    Train Loss: {job.train_loss.toFixed(4)}
                  </span>
                )}
                {job.val_loss != null && (
                  <span className={job.val_loss > 100 ? 'text-danger fw-medium' : 'text-muted'}>
                    <i className="ph ph-chart-line-up me-1"></i>
                    Val Loss: {job.val_loss.toFixed(4)}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Config Details */}
        {job.config && (
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
        )}
        
        {/* Sparkline Charts */}
        {job.status === 'running' && lossHistory.length > 2 && (
          <div className="mb-3">
            <div className="small text-muted mb-1">
              <i className="ph ph-chart-line me-1"></i>
              Validation Loss History (Log Scale)
            </div>
            <ResponsiveContainer width="100%" height={80}>
              <LineChart data={lossHistory} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <YAxis 
                  hide 
                  domain={['auto', 'auto']} 
                  scale="log"
                />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#0d6efd" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {job.status === 'running' && distanceHistory.length > 2 && (
          <div className="mb-3">
            <div className="small text-muted mb-1">
              <i className="ph ph-map-pin me-1"></i>
              Localization Error P68 (meters)
            </div>
            <ResponsiveContainer width="100%" height={80}>
              <LineChart data={distanceHistory} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <YAxis 
                  hide 
                  domain={[0, 'auto']}
                />
                <Line 
                  type="monotone" 
                  dataKey="distance" 
                  stroke="#198754" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

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
                data-action="pause"
              >
                <i className="ph ph-pause me-1"></i>
                {isLoading ? 'Pausing...' : 'Pause'}
              </button>
              <button
                onClick={(e) => {
                  console.log('[JobCard] Cancel button clicked!', {
                    jobId: job.id.slice(0, 8),
                    jobStatus: job.status,
                    buttonElement: e.currentTarget,
                    dataAction: e.currentTarget.getAttribute('data-action'),
                  });
                  handleCancel();
                }}
                disabled={isLoading}
                className="btn btn-sm btn-outline-danger flex-fill"
                data-action="cancel"
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
              data-action="resume"
            >
              <i className="ph ph-play me-1"></i>
              {isLoading ? 'Resuming...' : 'Resume'}
            </button>
          )}

          {/* Resume button for cancelled/failed jobs with checkpoints */}
          {(job.status === 'cancelled' || job.status === 'failed') && job.checkpoint_path && (
            <button
              onClick={handleResume}
              disabled={isLoading}
              className="btn btn-sm btn-outline-success flex-fill"
              data-action="resume"
              title="Resume training from last checkpoint"
            >
              <i className="ph ph-arrow-clockwise me-1"></i>
              {isLoading ? 'Resuming...' : 'Resume from Checkpoint'}
            </button>
          )}

          {(job.status === 'pending' || job.status === 'queued') && (
            <button
              onClick={(e) => {
                console.log('[JobCard] Cancel button clicked (pending/queued)!', {
                  jobId: job.id.slice(0, 8),
                  jobStatus: job.status,
                  buttonElement: e.currentTarget,
                  dataAction: e.currentTarget.getAttribute('data-action'),
                });
                handleCancel();
              }}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger w-100"
              data-action="cancel"
            >
              <i className="ph ph-x me-1"></i>
              {isLoading ? 'Cancelling...' : 'Cancel'}
            </button>
          )}

          {/* Delete button for completed jobs, or failed/cancelled without checkpoints */}
          {(job.status === 'completed' || 
            (job.status === 'failed' && !job.checkpoint_path) || 
            (job.status === 'cancelled' && !job.checkpoint_path)) && (
            <button
              onClick={(e) => {
                console.log('[JobCard] Delete button clicked!', {
                  jobId: job.id.slice(0, 8),
                  jobStatus: job.status,
                  buttonElement: e.currentTarget,
                  dataAction: e.currentTarget.getAttribute('data-action'),
                });
                handleDelete();
              }}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger w-100"
              data-action="delete"
            >
              <i className="ph ph-trash me-1"></i>
              {isLoading ? 'Deleting...' : 'Delete Job'}
            </button>
          )}

          {/* Show both Resume and Delete for failed/cancelled jobs with checkpoints */}
          {(job.status === 'failed' || job.status === 'cancelled') && job.checkpoint_path && (
            <button
              onClick={(e) => {
                console.log('[JobCard] Delete button clicked!', {
                  jobId: job.id.slice(0, 8),
                  jobStatus: job.status,
                  buttonElement: e.currentTarget,
                  dataAction: e.currentTarget.getAttribute('data-action'),
                });
                handleDelete();
              }}
              disabled={isLoading}
              className="btn btn-sm btn-outline-danger flex-fill"
              data-action="delete"
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
