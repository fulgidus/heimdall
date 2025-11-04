/**
 * JobsTab Component
 * 
 * Displays all training jobs with create, pause, resume, cancel actions
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { useWebSocket } from '../../../../contexts/WebSocketContext';
import { JobCard } from './JobCard';
import { CreateJobDialog } from './CreateJobDialog';

export const JobsTab: React.FC = () => {
  const { jobs, fetchJobs, handleJobUpdate, isLoading, error } = useTrainingStore();
  const { subscribe } = useWebSocket();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

  // Fetch jobs on mount and subscribe to real-time WebSocket updates
  useEffect(() => {
    fetchJobs();
    
    // Subscribe to real-time training job updates
    const unsubscribeProgress = subscribe('training:progress', (data: any) => {
      console.log('[JobsTab] Received training:progress event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, ...data });
      }
    });

    // Subscribe to batch-level progress updates (real-time, every ~1 second)
    const unsubscribeBatchProgress = subscribe('training:batch_progress', (data: any) => {
      console.log('[JobsTab] Received training:batch_progress event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, ...data });
      }
    });

    const unsubscribeStarted = subscribe('training:started', (data: any) => {
      console.log('[JobsTab] Received training:started event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, status: 'running', ...data });
      }
    });

    const unsubscribeCompleted = subscribe('training:completed', (data: any) => {
      console.log('[JobsTab] Received training:completed event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, status: 'completed', ...data });
      }
    });

    const unsubscribeFailed = subscribe('training:failed', (data: any) => {
      console.log('[JobsTab] Received training:failed event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, status: 'failed', ...data });
      }
    });

    const unsubscribePaused = subscribe('training:paused', (data: any) => {
      console.log('[JobsTab] Received training:paused event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, status: 'paused', ...data });
      }
    });

    const unsubscribeResumed = subscribe('training:resumed', (data: any) => {
      console.log('[JobsTab] Received training:resumed event:', data);
      if (data.job_id) {
        handleJobUpdate({ id: data.job_id, status: 'running', ...data });
      }
    });
    
    return () => {
      unsubscribeProgress();
      unsubscribeBatchProgress();
      unsubscribeStarted();
      unsubscribeCompleted();
      unsubscribeFailed();
      unsubscribePaused();
      unsubscribeResumed();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Zustand actions are stable, safe to omit from dependencies

  // Filter out synthetic generation jobs (they have their own tab)
  const trainingJobs = jobs.filter(job => {
    // @ts-ignore - job_type exists on backend response but not in TrainingJob type
    return !job.job_type || job.job_type === 'training';
  });

  return (
    <div>
      {/* Header */}
      <div className="d-flex justify-content-between align-items-start mb-4">
        <div>
          <h3 className="mb-1">Training Jobs</h3>
          <p className="text-muted mb-0">
            Manage and monitor training jobs
          </p>
        </div>
        <button
          onClick={() => setIsCreateDialogOpen(true)}
          className="btn btn-primary d-flex align-items-center gap-2"
        >
          <i className="ph ph-plus"></i>
          New Training Job
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="alert alert-danger d-flex align-items-center" role="alert">
          <i className="ph ph-warning-circle me-2"></i>
          <div>
            <strong>Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && trainingJobs.length === 0 && (
        <div className="text-center py-5">
          <div className="spinner-border text-primary mb-3" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="text-muted">Loading training jobs...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && trainingJobs.length === 0 && (
        <div className="card border-dashed">
          <div className="card-body text-center py-5">
            <i className="ph ph-clipboard-text display-4 text-muted mb-3"></i>
            <h5 className="card-title mb-2">No Training Jobs</h5>
            <p className="text-muted mb-3">Get started by creating your first training job</p>
            <button
              onClick={() => setIsCreateDialogOpen(true)}
              className="btn btn-primary"
            >
              <i className="ph ph-plus me-2"></i>
              Create Training Job
            </button>
          </div>
        </div>
      )}

      {/* Jobs Grid */}
      {trainingJobs.length > 0 && (
        <div className="row g-3">
          {trainingJobs.map(job => (
            <div key={job.id} className="col-12 col-md-6 col-lg-4">
              <JobCard job={job} />
            </div>
          ))}
        </div>
      )}

      {/* Create Job Dialog */}
      <CreateJobDialog
        isOpen={isCreateDialogOpen}
        onClose={() => setIsCreateDialogOpen(false)}
      />
    </div>
  );
};
