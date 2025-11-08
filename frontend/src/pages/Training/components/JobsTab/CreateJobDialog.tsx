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

      // Build job request matching the backend TrainingJobRequest schema
      const jobRequest = {
        job_name: jobName,
        config: {
          dataset_ids: trainingConfig.dataset_ids,  // Required array of dataset UUIDs
          model_architecture: architecture.id,
          batch_size: trainingConfig.batch_size,
          learning_rate: trainingConfig.learning_rate,
          epochs: trainingConfig.epochs,  // Note: 'epochs', not 'total_epochs'
          validation_split: trainingConfig.val_ratio,
          early_stop_patience: trainingConfig.early_stopping_patience,
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
