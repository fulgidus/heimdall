/**
 * SyntheticTab Component
 * 
 * Tab for managing synthetic dataset generation with real-time WebSocket updates
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { useWebSocket } from '../../../../contexts/WebSocketContext';
import { DatasetCard } from './DatasetCard';
import { GenerateDataDialog } from './GenerateDataDialog';
import { GenerationJobCard } from './GenerationJobCard';

export const SyntheticTab: React.FC = () => {
  const { datasets, generationJobs, fetchDatasets, fetchGenerationJobs, handleGenerationJobUpdate, isLoading } = useTrainingStore();
  const { subscribe } = useWebSocket();
  const [isGenerateDialogOpen, setIsGenerateDialogOpen] = useState(false);

  useEffect(() => {
    // Initial fetch (with loading spinner)
    fetchDatasets();
    fetchGenerationJobs();
    
    // Subscribe to real-time updates via WebSocket
    const unsubscribeJob = subscribe('training_job_update', (data: any) => {
      console.log('[SyntheticTab] Received training_job_update:', data);
      
      // Event now includes complete job data from backend:
      // { job_id, status, action, current_progress, total_progress, progress_percent, 
      //   progress_message, job_name, job_type, dataset_id, created_at, result }
      
      if (data.job_id) {
        // Update job directly in store using complete data from event
        const jobUpdate: any = {
          id: data.job_id,
          name: data.job_name || 'Unnamed Job',
          status: data.status,
          job_type: data.job_type || 'synthetic_generation',
          current: data.current_progress,
          total: data.total_progress,
          progress_percent: data.progress_percent,
          progress_message: data.progress_message,
          dataset_id: data.dataset_id,
        };
        
        // For completed jobs, include result data
        if (data.action === 'completed' && data.result) {
          jobUpdate.result = data.result;
        }
        
        console.log('[SyntheticTab] Updating job in store with complete data:', jobUpdate);
        handleGenerationJobUpdate(jobUpdate);
        
        // For 'started' action, also refresh the full list after a short delay
        // to ensure any additional jobs are loaded (in case of multiple concurrent jobs)
        if (data.action === 'started') {
          setTimeout(() => fetchGenerationJobs(true), 1000);
        }
        
        // For 'completed' action, also refresh datasets to show the new dataset
        if (data.action === 'completed') {
          setTimeout(() => {
            fetchGenerationJobs(true);
            fetchDatasets(true);
          }, 500);
        }
      } else {
        console.warn('[SyntheticTab] Received training_job_update without job_id, ignoring');
      }
    });

    const unsubscribeDataset = subscribe('dataset_update', (data: any) => {
      console.log('[SyntheticTab] Received dataset update:', data);
      // Refresh datasets silently (without loading spinner)
      fetchDatasets(true);
    });

    // Subscribe to dataset generation progress events
    const unsubscribeProgress = subscribe('dataset:generation_progress', (data: any) => {
      console.log('[SyntheticTab] ===== DATASET GENERATION PROGRESS EVENT =====');
      console.log('[SyntheticTab] Event data:', JSON.stringify(data, null, 2));
      console.log('[SyntheticTab] Updating generation job directly from WebSocket data...');
      
      // Update the generation job directly from WebSocket data
      // Backend sends: { job_id, current, total, progress_percent, message }
      if (data.job_id) {
        handleGenerationJobUpdate({
          id: data.job_id,
          status: 'running', // Progress events mean job is running
          current: data.current,                    // Current samples generated
          total: data.total,                        // Total samples to generate
          progress_percent: data.progress_percent,  // Computed percentage
        });
        console.log('[SyntheticTab] Generation job updated in store:', {
          job_id: data.job_id,
          current: data.current,
          total: data.total,
          progress_percent: data.progress_percent,
        });
      } else {
        console.warn('[SyntheticTab] Missing job_id in progress event, falling back to API fetch');
        fetchGenerationJobs(true);
      }
    });
    
    return () => {
      unsubscribeJob();
      unsubscribeDataset();
      unsubscribeProgress();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Zustand actions are stable, safe to omit from dependencies

  return (
    <>
      {/* Header with Generate Button */}
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h4 className="mb-1">Synthetic Datasets</h4>
          <p className="text-muted mb-0">
            Generate synthetic RF localization data for training
          </p>
        </div>
        <button
          onClick={() => setIsGenerateDialogOpen(true)}
          className="btn btn-primary d-flex align-items-center gap-2"
        >
          <i className="ph ph-plus-circle"></i>
          Generate Dataset
        </button>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-5">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading datasets...</span>
          </div>
          <p className="text-muted mt-3">Loading datasets...</p>
        </div>
      )}

      {/* Active Generation Jobs */}
      {generationJobs.length > 0 && (
        <div className="mb-4">
          <h5 className="mb-3">
            <i className="ph ph-spinner-gap me-2"></i>
            Active Generation Jobs
            <span className="badge bg-primary ms-2">{generationJobs.length}</span>
          </h5>
          <div className="row g-3">
            {generationJobs.map((job) => (
              <div key={job.id} className="col-12 col-md-6 col-lg-4">
                <GenerationJobCard job={job} />
              </div>
            ))}
          </div>
          <hr className="my-4" />
        </div>
      )}

      {/* Empty State */}
      {!isLoading && datasets.length === 0 && generationJobs.length === 0 && (
        <div className="text-center py-5">
          <i className="ph ph-database" style={{ fontSize: '3rem', color: 'var(--bs-gray-400)' }}></i>
          <h5 className="mt-3 mb-2">No Datasets Yet</h5>
          <p className="text-muted mb-4">
            Generate your first synthetic dataset to start training models
          </p>
          <button
            onClick={() => setIsGenerateDialogOpen(true)}
            className="btn btn-primary d-flex align-items-center gap-2 mx-auto"
          >
            <i className="ph ph-plus-circle"></i>
            Generate Dataset
          </button>
        </div>
      )}

      {/* Dataset Grid */}
      {!isLoading && datasets.length > 0 && (
        <>
          <h5 className="mb-3">
            <i className="ph ph-database me-2"></i>
            Completed Datasets
            <span className="badge bg-success ms-2">{datasets.length}</span>
          </h5>
          <div className="row g-3">
            {datasets.map((dataset) => (
              <div key={dataset.id} className="col-12 col-md-6 col-lg-4">
                <DatasetCard dataset={dataset} />
              </div>
            ))}
          </div>
        </>
      )}

      {/* Generate Data Dialog */}
      <GenerateDataDialog
        isOpen={isGenerateDialogOpen}
        onClose={() => setIsGenerateDialogOpen(false)}
      />
    </>
  );
};
