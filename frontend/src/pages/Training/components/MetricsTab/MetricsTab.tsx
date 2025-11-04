/**
 * MetricsTab Component
 * 
 * Displays training metrics charts for a selected job
 */

import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '../../../../store/trainingStore';
import { LossChart } from './LossChart';
import { AccuracyChart } from './AccuracyChart';
import { LearningRateChart } from './LearningRateChart';
import { DistanceErrorChart } from './DistanceErrorChart';
import { DistancePercentilesChart } from './DistancePercentilesChart';
import { UncertaintyCalibrationChart } from './UncertaintyCalibrationChart';
import { GDOPHealthChart } from './GDOPHealthChart';
import { GradientHealthChart } from './GradientHealthChart';

export const MetricsTab: React.FC = () => {
  const { jobs, metrics, fetchMetrics, error } = useTrainingStore();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Filter out synthetic generation jobs (only show training jobs)
  const trainingJobs = jobs.filter(job => {
    // @ts-ignore - job_type exists on backend response but not in TrainingJob type
    return !job.job_type || job.job_type === 'training';
  });

  // Auto-select first running job or first job
  useEffect(() => {
    if (!selectedJobId && trainingJobs.length > 0) {
      const runningJob = trainingJobs.find(job => job.status === 'running');
      setSelectedJobId(runningJob?.id || trainingJobs[0].id);
    }
  }, [trainingJobs, selectedJobId]);

  // Fetch metrics when job is selected (WebSocket will handle real-time updates)
  useEffect(() => {
    if (selectedJobId) {
      fetchMetrics(selectedJobId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJobId]); // Only re-run when job selection changes, not on every jobs update

  const selectedMetrics = selectedJobId ? metrics.get(selectedJobId) || [] : [];
  const selectedJob = trainingJobs.find(j => j.id === selectedJobId);

  return (
    <div>
      {/* Header */}
      <div className="d-flex justify-content-between align-items-start mb-4">
        <div>
          <h2 className="h3 mb-2">Training Metrics</h2>
          <p className="text-muted small">
            View detailed training progress and performance metrics
          </p>
        </div>
      </div>

      {/* Job Selector */}
      {trainingJobs.length > 0 ? (
        <div className="card mb-4">
          <div className="card-body">
            <label htmlFor="job-select" className="form-label fw-medium">
              Select Training Job
            </label>
            <select
              id="job-select"
              value={selectedJobId || ''}
              onChange={(e) => setSelectedJobId(e.target.value)}
              className="form-select"
              style={{ maxWidth: '450px' }}
            >
              {trainingJobs.map(job => (
                <option key={job.id} value={job.id}>
                  {job.job_name || job.name || 'Unnamed Job'} ({job.status}) - Epoch {job.current_epoch || 0}/{job.total_epochs}
                </option>
              ))}
            </select>
            
            {selectedJob && (
              <div className="mt-3 d-flex flex-wrap gap-3 small text-muted">
                <div>
                  <span className="fw-medium">Status:</span>
                  <span className={`ms-2 badge ${
                    selectedJob.status === 'running' ? 'bg-light-primary text-primary' :
                    selectedJob.status === 'completed' ? 'bg-light-success text-success' :
                    selectedJob.status === 'failed' ? 'bg-light-danger text-danger' :
                    'bg-light text-dark'
                  }`}>
                    {selectedJob.status.toUpperCase()}
                  </span>
                </div>
                <div>
                  <span className="fw-medium">Epochs:</span>
                  <span className="ms-2">{selectedJob.current_epoch || 0} / {selectedJob.total_epochs}</span>
                </div>
                {selectedJob.progress_percent !== undefined && (
                  <div>
                    <span className="fw-medium">Progress:</span>
                    <span className="ms-2">{selectedJob.progress_percent.toFixed(1)}%</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="card border-dashed text-center py-5 mb-4">
          <div className="card-body">
            <i className="ph ph-chart-bar text-muted mb-3" style={{ fontSize: '4rem' }}></i>
            <h5 className="mb-2">No Training Jobs</h5>
            <p className="text-muted small mb-0">Create a training job to view metrics</p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="alert alert-danger d-flex align-items-start mb-4" role="alert">
          <i className="ph ph-warning-circle me-2 mt-1"></i>
          <div>
            <strong>Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Charts */}
      {selectedJobId && selectedMetrics.length === 0 && (
        <div className="card text-center py-5">
          <div className="card-body">
            <i className="ph ph-chart-bar text-muted mb-3" style={{ fontSize: '4rem' }}></i>
            <h5 className="mb-2">No Metrics Yet</h5>
            <p className="text-muted small mb-0">
              Metrics will appear once training starts
            </p>
          </div>
        </div>
      )}

      {selectedJobId && selectedMetrics.length > 0 && (
        <div className="row g-4">
          {/* Basic Metrics */}
          <div className="col-12">
            <h5 className="mb-3">Basic Metrics</h5>
          </div>
          <div className="col-12">
            <LossChart metrics={selectedMetrics} />
          </div>
          <div className="col-12">
            <AccuracyChart metrics={selectedMetrics} />
          </div>
          <div className="col-12">
            <LearningRateChart metrics={selectedMetrics} />
          </div>

          {/* Distance Errors */}
          <div className="col-12 mt-5">
            <h5 className="mb-3">Distance Errors</h5>
          </div>
          <div className="col-12">
            <DistanceErrorChart metrics={selectedMetrics} />
          </div>
          <div className="col-12">
            <DistancePercentilesChart metrics={selectedMetrics} />
          </div>

          {/* Model Quality */}
          <div className="col-12 mt-5">
            <h5 className="mb-3">Model Quality</h5>
          </div>
          <div className="col-12">
            <UncertaintyCalibrationChart metrics={selectedMetrics} />
          </div>
          <div className="col-12">
            <GDOPHealthChart metrics={selectedMetrics} />
          </div>

          {/* Training Health */}
          <div className="col-12 mt-5">
            <h5 className="mb-3">Training Health</h5>
          </div>
          <div className="col-12">
            <GradientHealthChart metrics={selectedMetrics} />
          </div>
        </div>
      )}
    </div>
  );
};
