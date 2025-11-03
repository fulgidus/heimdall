/**
 * SyntheticTab Component
 * 
 * Tab for managing synthetic dataset generation with real-time WebSocket updates
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { useWebSocket } from '../../../../contexts/WebSocketContext';
import { DatasetCard } from './DatasetCard';
import { GenerateDataDialog } from './GenerateDataDialog';
import { GenerationJobCard } from './GenerationJobCard';

export const SyntheticTab: React.FC = () => {
  const { datasets, generationJobs, fetchDatasets, fetchGenerationJobs, isLoading } = useTrainingStore();
  const { subscribe } = useWebSocket();
  const [isGenerateDialogOpen, setIsGenerateDialogOpen] = useState(false);

  useEffect(() => {
    // Initial fetch (with loading spinner)
    fetchDatasets();
    fetchGenerationJobs();
    
    // Subscribe to real-time updates via WebSocket
    const unsubscribeJob = subscribe('training_job_update', (data: any) => {
      console.log('[SyntheticTab] Received training job update:', data);
      // Refresh generation jobs silently (without loading spinner)
      fetchGenerationJobs(true);
    });

    const unsubscribeDataset = subscribe('dataset_update', (data: any) => {
      console.log('[SyntheticTab] Received dataset update:', data);
      // Refresh datasets silently (without loading spinner)
      fetchDatasets(true);
    });
    
    return () => {
      unsubscribeJob();
      unsubscribeDataset();
    };
  }, [fetchDatasets, fetchGenerationJobs, subscribe]);

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
