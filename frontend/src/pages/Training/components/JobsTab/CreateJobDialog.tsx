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
}

interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  dataset_ids: string[];  // Array for multiple dataset selection
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
}

export const CreateJobDialog: React.FC<CreateJobDialogProps> = ({ isOpen, onClose }) => {
  const createJob = useTrainingStore(state => state.createJob);

  const handleSelectModel = async (
    architecture: ModelArchitecture,
    trainingConfig: TrainingConfig
  ) => {
    try {
      // Build job request matching the backend CreateTrainingJobRequest schema
      const jobRequest = {
        job_name: `${architecture.display_name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
        model_architecture: architecture.id,
        total_epochs: trainingConfig.epochs,
        batch_size: trainingConfig.batch_size,
        learning_rate: trainingConfig.learning_rate,
        early_stopping_patience: trainingConfig.early_stopping_patience,
        train_split: trainingConfig.train_ratio,
        val_split: trainingConfig.val_ratio,
        use_gpu: true,  // Default to GPU
        num_workers: 4,  // Default
        optimizer: 'adam',  // Default
        scheduler: 'reduce_on_plateau',  // Default
        // Pass dataset_ids array in config (required by training task)
        config: {
          dataset_ids: trainingConfig.dataset_ids,
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
    />
  );
};
