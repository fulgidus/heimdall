/**
 * JobsTab Component
 * 
 * Displays all training jobs with create, pause, resume, cancel actions
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { useWebSocket } from '../../../../contexts/WebSocketContext';
import { useAuth } from '../../../../hooks/useAuth';
import { JobCard } from './JobCard';
import { CreateJobDialog } from './CreateJobDialog';
import { listModelArchitectures, listSyntheticDatasets } from '../../../../services/api/training';

interface JobsTabProps {
  onJobCreated?: () => void;  // Optional callback when job is created
}

export const JobsTab: React.FC<JobsTabProps> = ({ onJobCreated }) => {
  const { jobs, fetchJobs, handleJobUpdate, isLoading, error, createJob, deleteAllJobs } = useTrainingStore();
  const { subscribe } = useWebSocket();
  const { isOperator } = useAuth(); // For permission checks
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isCreatingBulkJobs, setIsCreatingBulkJobs] = useState(false);
  const [isDeletingAllJobs, setIsDeletingAllJobs] = useState(false);

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
        // Map backend field names to frontend field names
        handleJobUpdate({ 
          id: data.job_id,
          current_epoch: data.epoch,           // Backend sends 'epoch', frontend expects 'current_epoch'
          total_epochs: data.total_epochs,
          progress_percent: data.progress_percent,
          status: 'running',                   // Ensure status stays 'running'
          // Include other fields that might be present
          ...data,
        });
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
    // Only show jobs explicitly marked as 'training'
    return job.job_type === 'training';
  });

  // Handler for creating training jobs for all architectures
  const handleTrainAllArchitectures = async () => {
    setIsCreatingBulkJobs(true);
    try {
      // Fetch all architectures
      console.log('[JobsTab] Fetching all model architectures...');
      const architecturesResponse = await listModelArchitectures();
      const architectures = architecturesResponse.architectures;
      console.log(`[JobsTab] Found ${architectures.length} architectures`);
      
      // Fetch all datasets
      console.log('[JobsTab] Fetching all synthetic datasets...');
      const datasetsResponse = await listSyntheticDatasets();
      const datasets = datasetsResponse.datasets;
      console.log(`[JobsTab] Found ${datasets.length} datasets`);
      
      if (datasets.length === 0) {
        alert('No synthetic datasets available. Please create a dataset first.');
        setIsCreatingBulkJobs(false);
        return;
      }
      
      const datasetIds = datasets.map(d => d.id);
      
      // Create a job for each architecture
      console.log(`[JobsTab] Creating ${architectures.length} training jobs...`);
      const jobPromises = architectures.map(async (arch) => {
        const jobConfig = {
          job_name: `Bulk ${arch.id} ${new Date().toISOString()}`,
          config: {
            dataset_ids: datasetIds, // Required array of dataset UUIDs
            model_architecture: arch.id,
            batch_size: 8, // Small default batch size
            learning_rate: 0.001,
            epochs: 500, // Note: 'epochs', not 'total_epochs'
            validation_split: 0.15,
            early_stop_patience: 50, // Very patient early stopping
          }
        };
        
        console.log(`[JobsTab] Creating job for ${arch.display_name} (${arch.id})`);
        return createJob(jobConfig);
      });
      
      await Promise.all(jobPromises);
      
      // Refresh jobs list
      await fetchJobs();
      
      console.log(`[JobsTab] Successfully created ${architectures.length} training jobs!`);
      alert(`Successfully created ${architectures.length} training jobs!\n\nEach job will train for 500 epochs using all available datasets with no early stopping.`);
    } catch (error) {
      console.error('[JobsTab] Error creating bulk jobs:', error);
      alert('Failed to create some training jobs. Check console for details.');
    } finally {
      setIsCreatingBulkJobs(false);
    }
  };

  // Handler for deleting all training jobs
  const handleDeleteAllJobs = async () => {
    const trainingJobs = jobs.filter(job => job.job_type === 'training');
    
    if (trainingJobs.length === 0) {
      alert('No training jobs to delete.');
      return;
    }
    
    const confirmed = window.confirm(
      `Are you sure you want to delete ALL ${trainingJobs.length} training jobs?\n\n` +
      `This will:\n` +
      `- Cancel all running/paused jobs\n` +
      `- Delete all job records\n` +
      `- This action cannot be undone!\n\n` +
      `Click OK to proceed or Cancel to abort.`
    );
    
    if (!confirmed) {
      return;
    }
    
    setIsDeletingAllJobs(true);
    try {
      console.log(`[JobsTab] Deleting all ${trainingJobs.length} training jobs...`);
      await deleteAllJobs();
      
      // Refresh jobs list
      await fetchJobs();
      
      console.log('[JobsTab] Successfully deleted all training jobs!');
      alert(`Successfully deleted all ${trainingJobs.length} training jobs!`);
    } catch (error) {
      console.error('[JobsTab] Error deleting all jobs:', error);
      alert('Failed to delete some training jobs. Check console for details.\n\nRemaining jobs have been removed from the UI.');
    } finally {
      setIsDeletingAllJobs(false);
    }
  };

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
        {isOperator && (
          <div className="d-flex gap-2">
            <button
              onClick={handleDeleteAllJobs}
              disabled={isDeletingAllJobs || isLoading || trainingJobs.length === 0}
              className="btn btn-danger d-flex align-items-center gap-2"
              title="Delete all training jobs (running jobs will be cancelled first)"
            >
              {isDeletingAllJobs ? (
                <>
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                  Deleting...
                </>
              ) : (
                <>
                  <i className="ph ph-trash"></i>
                  Delete All Jobs
                </>
              )}
            </button>
            <button
              onClick={handleTrainAllArchitectures}
              disabled={isCreatingBulkJobs || isLoading}
              className="btn btn-success d-flex align-items-center gap-2"
              title="Create training jobs for all architectures using all datasets (500 epochs, no early stopping)"
            >
              {isCreatingBulkJobs ? (
                <>
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                  Creating Jobs...
                </>
              ) : (
                <>
                  <i className="ph ph-lightning"></i>
                  Train All Architectures
                </>
              )}
            </button>
            <button
              onClick={() => setIsCreateDialogOpen(true)}
              className="btn btn-primary d-flex align-items-center gap-2"
            >
              <i className="ph ph-plus"></i>
              New Training Job
            </button>
          </div>
        )}
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
            <p className="text-muted mb-3">
              {isOperator 
                ? 'Get started by creating your first training job'
                : 'No training jobs have been created yet'}
            </p>
            {isOperator && (
              <button
                onClick={() => setIsCreateDialogOpen(true)}
                className="btn btn-primary"
              >
                <i className="ph ph-plus me-2"></i>
                Create Training Job
              </button>
            )}
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
        onJobCreated={onJobCreated}
      />
    </div>
  );
};
