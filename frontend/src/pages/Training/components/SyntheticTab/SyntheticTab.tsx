/**
 * SyntheticTab Component
 * 
 * Tab for managing synthetic dataset generation
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { DatasetCard } from './DatasetCard';
import { GenerateDataDialog } from './GenerateDataDialog';

export const SyntheticTab: React.FC = () => {
  const { datasets, fetchDatasets, isLoading } = useTrainingStore();
  const [isGenerateDialogOpen, setIsGenerateDialogOpen] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

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

      {/* Empty State */}
      {!isLoading && datasets.length === 0 && (
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
        <div className="row g-3">
          {datasets.map((dataset) => (
            <div key={dataset.id} className="col-12 col-md-6 col-lg-4">
              <DatasetCard dataset={dataset} />
            </div>
          ))}
        </div>
      )}

      {/* Generate Data Dialog */}
      <GenerateDataDialog
        isOpen={isGenerateDialogOpen}
        onClose={() => setIsGenerateDialogOpen(false)}
      />
    </>
  );
};
