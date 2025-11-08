/**
 * ModelsTab Component
 * 
 * Displays grid of trained models with import functionality
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { ModelCard } from './ModelCard';
import { ImportDialog } from './ImportDialog';

export const ModelsTab: React.FC = () => {
  const { models, fetchModels, isLoading, error } = useTrainingStore();
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);

  useEffect(() => {
    fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Zustand actions are stable, safe to omit from dependencies

  return (
    <div>
      {/* Header with Import Button */}
      <div className="d-flex justify-content-between align-items-start mb-4">
        <div>
          <h2 className="h3 mb-2">Trained Models</h2>
          <p className="text-muted small">
            {models.length} {models.length === 1 ? 'model' : 'models'} available
          </p>
        </div>
        <button
          onClick={() => setIsImportDialogOpen(true)}
          className="btn btn-success d-flex align-items-center gap-2"
        >
          <i className="ph ph-upload"></i>
          Import Model
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="alert alert-danger d-flex align-items-start mb-4" role="alert">
          <i className="ph ph-warning-circle me-2 mt-1"></i>
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading && models.length === 0 && (
        <div className="text-center py-5">
          <div className="spinner-border text-primary mb-3" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="text-muted">Loading models...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && models.length === 0 && (
        <div className="card border-dashed text-center py-5">
          <div className="card-body">
            <i className="ph ph-cpu text-muted mb-3" style={{ fontSize: '4rem' }}></i>
            <h5 className="mb-2">No models yet</h5>
            <p className="text-muted small mb-4">
              Train your first model or import an existing one
            </p>
            <button
              onClick={() => setIsImportDialogOpen(true)}
              className="btn btn-primary d-inline-flex align-items-center gap-2"
            >
              <i className="ph ph-upload"></i>
              Import Model
            </button>
          </div>
        </div>
      )}

      {/* Models Grid */}
      {models.length > 0 && (
        <div className="row g-4">
          {models.map((model) => (
            <div key={model.id} className="col-12 col-md-6 col-lg-4">
              <ModelCard model={model} />
            </div>
          ))}
        </div>
      )}

      {/* Import Dialog */}
      <ImportDialog
        isOpen={isImportDialogOpen}
        onClose={() => setIsImportDialogOpen(false)}
      />
    </div>
  );
};
