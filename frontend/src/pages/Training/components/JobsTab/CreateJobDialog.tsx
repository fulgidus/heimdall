/**
 * CreateJobDialog Component
 * 
 * Wrapper that uses the new ModelSelectionModal for creating training jobs
 */

import React from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import ModelSelectionModal from '@/components/ModelSelectionModal';
import type { ModelArchitecture } from '@/services/api/training';

interface CreateJobDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onJobCreated?: () => void;  // Optional callback when job is successfully created
}

interface TrainingConfig {
  job_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  dataset_ids: string[];  // Array for multiple dataset selection
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
}

export const CreateJobDialog: React.FC<CreateJobDialogProps> = ({ isOpen, onClose, onJobCreated }) => {
  const createJob = useTrainingStore(state => state.createJob);

  const handleSelectModel = async (
    architecture: ModelArchitecture,
    trainingConfig: TrainingConfig
  ) => {
    try {
      // Generate job name if not provided
      const jobName = trainingConfig.job_name.trim() || 
        `${architecture.display_name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;

      // Build job request matching the backend CreateTrainingJobRequest schema
      const jobRequest = {
        job_name: jobName,
        model_architecture: architecture.id,  // Required top-level field
        batch_size: trainingConfig.batch_size,
        learning_rate: trainingConfig.learning_rate,
        total_epochs: trainingConfig.epochs,  // Backend expects 'total_epochs' not 'epochs'
        early_stopping_patience: trainingConfig.early_stopping_patience,
        train_split: trainingConfig.train_ratio,
        val_split: trainingConfig.val_ratio,
        dataset_id: trainingConfig.dataset_ids[0] || null,  // Backend expects single dataset_id, not array
        config: {
          dataset_ids: trainingConfig.dataset_ids,  // Keep full array in additional config
          epochs: trainingConfig.epochs,  // Required by CreateJobRequest type
          batch_size: trainingConfig.batch_size,  // Required by CreateJobRequest type
          learning_rate: trainingConfig.learning_rate,  // Required by CreateJobRequest type
          model_architecture: architecture.id,  // Optional but included for completeness
          validation_split: trainingConfig.val_ratio,  // Optional
          early_stop_patience: trainingConfig.early_stopping_patience,  // Optional
        }
      };

      const jobId = await createJob(jobRequest);
      console.log('Training job created with new architecture:', jobId, architecture.display_name);
    } catch (error) {
      console.error('Failed to create training job:', error);
    }
  };

  return (
    <ModelSelectionModal
      isOpen={isOpen}
      onClose={onClose}
      onSelectModel={handleSelectModel}
      onNavigateToConfig={onJobCreated}
    />
  );
};
