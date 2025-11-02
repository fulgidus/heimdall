/**
 * Training Dashboard
 *
 * Main page for ML training pipeline:
 * - Synthetic data generation
 * - Training job management
 * - Model management
 *
 * Real-time updates via WebSocket (silent background refresh)
 */

import React, { useEffect, useState } from 'react';
import { useWebSocket } from '@/contexts/WebSocketContext';
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Alert,
  Badge,
  Table,
  Spinner,
  Modal,
  Form,
} from 'react-bootstrap';
import {
  PlayCircle,
  Database,
  Brain,
  PlusCircle,
  Trash,
  CheckCircle,
  XCircle,
  Clock,
  StopCircle,
  Eye,
  Copy,
  Pause,
  Play,
} from 'lucide-react';
import {
  listTrainingJobs,
  listSyntheticDatasets,
  listModels,
  generateSyntheticData,
  cancelTrainingJob,
  pauseTrainingJob,
  resumeTrainingJob,
  deleteTrainingJob,
  deleteSyntheticDataset,
  deleteModel,
  deployModel,
  getSyntheticDatasetSamples,
  type TrainingJob,
  type SyntheticDataset,
  type TrainedModel,
  type SyntheticSampleResponse,
} from '@/services/api/training';

/**
 * Helper function to detect if a job is synthetic data generation
 */
const isSyntheticDataJob = (job: TrainingJob): boolean => {
  return job.total_epochs === 0 || job.job_name.toLowerCase().includes('synthetic');
};

/**
 * Calculate ETA for a running job
 */
const calculateETA = (job: TrainingJob): string => {
  if (!job.started_at) return 'Calculating...';

  // For synthetic data jobs, use current/total samples
  if (isSyntheticDataJob(job)) {
    if (!job.current || !job.total || job.current === 0) return 'Calculating...';

    const elapsed = Date.now() - new Date(job.started_at).getTime();
    const rate = job.current / elapsed; // samples per ms
    const remaining = job.total - job.current;
    const etaMs = remaining / rate;

    const hours = Math.floor(etaMs / 3600000);
    const minutes = Math.floor((etaMs % 3600000) / 60000);
    const seconds = Math.floor((etaMs % 60000) / 1000);

    if (hours > 0) return `~${hours}h ${minutes}m`;
    if (minutes > 0) return `~${minutes}m ${seconds}s`;
    return `~${seconds}s`;
  }

  // For training jobs, use current_epoch/total_epochs
  if (!job.current_epoch || !job.total_epochs || job.current_epoch === 0) return 'Calculating...';

  const elapsed = Date.now() - new Date(job.started_at).getTime();
  const rate = job.current_epoch / elapsed; // epochs per ms
  const remaining = job.total_epochs - job.current_epoch;
  const etaMs = remaining / rate;

  const hours = Math.floor(etaMs / 3600000);
  const minutes = Math.floor((etaMs % 3600000) / 60000);
  const seconds = Math.floor((etaMs % 60000) / 1000);

  if (hours > 0) return `~${hours}h ${minutes}m`;
  if (minutes > 0) return `~${minutes}m ${seconds}s`;
  return `~${seconds}s`;
};

const TrainingDashboard: React.FC = () => {
  // WebSocket for real-time updates
  const { subscribe } = useWebSocket();

  // State
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [datasets, setDatasets] = useState<SyntheticDataset[]>([]);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDataModal, setShowDataModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<SyntheticDataset | null>(null);
  const [datasetSamples, setDatasetSamples] = useState<SyntheticSampleResponse[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  
  // Power unit toggle (Watt or dBm)
  const [powerUnit, setPowerUnit] = useState<'watt' | 'dbm'>('watt');
  const [powerValueWatt, setPowerValueWatt] = useState(2.0); // Default 2W = ~33dBm
  
  // Helper functions for power conversion
  const wattToDbm = (watt: number): number => {
    return 10 * Math.log10(watt * 1000);
  };
  
  const dbmToWatt = (dbm: number): number => {
    return Math.pow(10, dbm / 10) / 1000;
  };
  
  // Data generation form
  // NOTE: train_ratio/val_ratio/test_ratio removed - splits calculated at training time
  // Production Baseline defaults (balanced quality/performance with SRTM terrain)
  const [dataForm, setDataForm] = useState({
    name: '',
    num_samples: 10000,
    inside_ratio: 0.75,    // Production baseline
    // RF generation parameters (production baseline)
    frequency_mhz: 144.0,
    tx_power_dbm: wattToDbm(2.0), // 2W = ~33dBm
    min_snr_db: 3.0,       // Reasonable minimum SNR
    min_receivers: 3,      // Minimum triangulation
    max_gdop: 100.0,       // Acceptable geometry
    use_real_terrain: true, // Use SRTM terrain data (production baseline)
  });

  // Load data silently (no loading spinner after initial load)
  const loadData = async (silent = false) => {
    try {
      if (!silent) setInitialLoading(true);
      const [jobsResp, datasetsResp, modelsResp] = await Promise.all([
        listTrainingJobs(undefined, 10, 0),
        listSyntheticDatasets(10, 0),
        listModels(false, 10, 0),
      ]);

      setJobs(jobsResp.jobs);
      setDatasets(datasetsResp.datasets);
      setModels(modelsResp.models);
      setError(null);
    } catch (err) {
      console.error('Failed to load training data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      if (!silent) setInitialLoading(false);
    }
  };

  useEffect(() => {
    loadData(false);
    
    // Silent refresh every 10 seconds (backup for WebSocket)
    const interval = setInterval(() => loadData(true), 10000);
    
    // WebSocket real-time updates (silent and transparent)
    const unsubscribeTrainingJob = subscribe('training_job_update', (data: any) => {
      console.log('[TrainingDashboard] Received training job update:', data);
      // Refresh jobs silently without spinner
      loadData(true);
    });

    const unsubscribeDataset = subscribe('dataset_update', (data: any) => {
      console.log('[TrainingDashboard] Received dataset update:', data);
      // Refresh datasets silently
      loadData(true);
    });

    const unsubscribeModel = subscribe('model_update', (data: any) => {
      console.log('[TrainingDashboard] Received model update:', data);
      // Refresh models silently
      loadData(true);
    });

    return () => {
      clearInterval(interval);
      unsubscribeTrainingJob();
      unsubscribeDataset();
      unsubscribeModel();
    };
  }, [subscribe]);

  // Handle data generation
  const handleGenerateData = async () => {
    try {
      await generateSyntheticData(dataForm);
      setShowDataModal(false);
      // Reset form to production baseline defaults
      setPowerValueWatt(2.0);
      setPowerUnit('watt');
      setDataForm({
        name: '',
        num_samples: 10000,
        inside_ratio: 0.75,
        frequency_mhz: 144.0,
        tx_power_dbm: wattToDbm(2.0),
        min_snr_db: 3.0,
        min_receivers: 3,
        max_gdop: 100.0,
        use_real_terrain: true,
      });
      loadData(); // Refresh
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate data');
    }
  };

  // Handle cancel job
  const handleCancelJob = async (jobId: string) => {
    if (!confirm('Cancel this running job?')) return;

    try {
      await cancelTrainingJob(jobId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job');
    }
  };

  // Handle pause job
  const handlePauseJob = async (jobId: string) => {
    if (!confirm('Pause this training job? It will complete the current epoch before pausing.')) return;

    try {
      await pauseTrainingJob(jobId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pause job');
    }
  };

  // Handle resume job
  const handleResumeJob = async (jobId: string) => {
    if (!confirm('Resume this training job from where it was paused?')) return;

    try {
      await resumeTrainingJob(jobId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resume job');
    }
  };

  // Handle delete job
  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Delete this training job?')) return;

    try {
      await deleteTrainingJob(jobId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete job');
    }
  };

  // Handle delete dataset
  const handleDeleteDataset = async (datasetId: string) => {
    if (!confirm('Delete this dataset? This will remove all samples.')) return;
    
    try {
      await deleteSyntheticDataset(datasetId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
    }
  };

  // Handle delete model
  const handleDeleteModel = async (modelId: string) => {
    if (!confirm('Delete this model?')) return;
    
    try {
      await deleteModel(modelId);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  // Handle deploy model
  const handleDeployModel = async (modelId: string) => {
    try {
      await deployModel(modelId, false);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deploy model');
    }
  };

  // Handle view dataset details
  const handleViewDetails = async (dataset: SyntheticDataset) => {
    setSelectedDataset(dataset);
    setShowDetailsModal(true);
    setLoadingSamples(true);
    
    try {
      // Fetch real samples from backend
      const response = await getSyntheticDatasetSamples(dataset.id, 10, 0);
      setDatasetSamples(response.samples);
    } catch (err) {
      console.error('Failed to load samples:', err);
      setError(err instanceof Error ? err.message : 'Failed to load samples');
      setDatasetSamples([]);
    } finally {
      setLoadingSamples(false);
    }
  };

  // Handle clone job - opens modal with pre-filled values from selected job
  const handleCloneJob = (job: TrainingJob) => {
    const config = job.config;
    
    // Pre-fill the form with job config values
    const newDataForm = {
      name: `${config.name || job.job_name} (Copy)`,
      num_samples: config.num_samples || 10000,
      inside_ratio: config.inside_ratio || 0.7,
      frequency_mhz: config.frequency_mhz || 144.0,
      tx_power_dbm: config.tx_power_dbm || 33.0,
      min_snr_db: config.min_snr_db !== undefined ? config.min_snr_db : 0.0,
      min_receivers: config.min_receivers || 2,
      max_gdop: config.max_gdop || 500.0,
      use_real_terrain: config.use_real_terrain || false,
    };
    
    setDataForm(newDataForm);
    
    // Update power value and unit based on cloned job
    const powerInWatt = dbmToWatt(newDataForm.tx_power_dbm);
    setPowerValueWatt(powerInWatt);
    
    // Auto-select Watt if the value is a "nice" number in watts, otherwise use dBm
    if (powerInWatt >= 1 && powerInWatt <= 100 && Number.isInteger(powerInWatt)) {
      setPowerUnit('watt');
    } else {
      setPowerUnit('dbm');
    }
    
    // Open the modal
    setShowDataModal(true);
  };

  // Get status badge
  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { variant: string; icon: React.ReactNode }> = {
      pending: { variant: 'secondary', icon: <Clock size={14} /> },
      queued: { variant: 'info', icon: <Clock size={14} /> },
      running: { variant: 'primary', icon: <Spinner animation="border" size="sm" /> },
      paused: { variant: 'dark', icon: <Pause size={14} /> },
      completed: { variant: 'success', icon: <CheckCircle size={14} /> },
      failed: { variant: 'danger', icon: <XCircle size={14} /> },
      cancelled: { variant: 'warning', icon: <XCircle size={14} /> },
    };

    const { variant, icon } = statusMap[status] || statusMap.pending;

    return (
      <Badge bg={variant} className="d-flex align-items-center gap-1">
        {icon}
        {status}
      </Badge>
    );
  };

  if (initialLoading) {
    return (
      <Container className="mt-4">
        <div className="text-center">
          <Spinner animation="border" />
          <p className="mt-2">Loading training dashboard...</p>
        </div>
      </Container>
    );
  }

  return (
    <Container fluid className="mt-4">
      <h1 className="mb-4">
        <Brain className="me-2" />
        ML Training Pipeline
      </h1>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Synthetic Datasets Section */}
      <Card className="mb-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">
            <Database className="me-2" />
            Synthetic Datasets
          </h5>
          <Button 
            variant="primary" 
            size="sm"
            onClick={() => setShowDataModal(true)}
          >
            <PlusCircle size={16} className="me-1" />
            Generate New Dataset
          </Button>
        </Card.Header>
        <Card.Body>
          {datasets.length === 0 ? (
            <Alert variant="info">
              No datasets yet. Generate synthetic training data to get started.
            </Alert>
          ) : (
            <Table striped hover responsive>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Samples</th>
                  <th>Quality Metrics</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.id}>
                    <td><strong>{dataset.name}</strong></td>
                    <td>{dataset.num_samples.toLocaleString()}</td>
                    <td>
                      {dataset.quality_metrics ? (
                        <div style={{ fontSize: '0.875rem' }}>
                          <div><strong>SNR:</strong> {dataset.quality_metrics.avg_snr_db?.toFixed(1)}dB (min: {dataset.quality_metrics.min_snr_db?.toFixed(1)}, max: {dataset.quality_metrics.max_snr_db?.toFixed(1)})</div>
                          <div><strong>GDOP:</strong> {dataset.quality_metrics.avg_gdop?.toFixed(2)} (min: {dataset.quality_metrics.min_gdop?.toFixed(2)}, max: {dataset.quality_metrics.max_gdop?.toFixed(2)})</div>
                          <div><strong>Receivers:</strong> {dataset.quality_metrics.avg_receivers?.toFixed(1)} avg</div>
                          <div><strong>Distance:</strong> {dataset.quality_metrics.avg_distance_km?.toFixed(1)}km avg (max: {dataset.quality_metrics.max_distance_km?.toFixed(1)}km)</div>
                        </div>
                      ) : (
                        <span className="text-muted">-</span>
                      )}
                    </td>
                    <td>{new Date(dataset.created_at).toLocaleString()}</td>
                    <td>
                      <Button
                        variant="outline-primary"
                        size="sm"
                        className="me-2"
                        onClick={() => handleViewDetails(dataset)}
                      >
                        <Eye size={14} />
                      </Button>
                      <Button
                        variant="outline-danger"
                        size="sm"
                        onClick={() => handleDeleteDataset(dataset.id)}
                      >
                        <Trash size={14} />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card.Body>
      </Card>

      {/* Training Jobs Section */}
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <PlayCircle className="me-2" />
            Recent Training Jobs
          </h5>
        </Card.Header>
        <Card.Body>
          {jobs.length === 0 ? (
            <Alert variant="info">
              No training jobs yet.
            </Alert>
          ) : (
            <Table striped hover responsive>
              <thead>
                <tr>
                  <th>Job Name</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Loss</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td><strong>{job.job_name}</strong></td>
                    <td>{getStatusBadge(job.status)}</td>
                    <td>
                      {job.status === 'running' ? (
                        <div>
                          <div className="progress mb-1">
                            <div
                              className="progress-bar progress-bar-striped progress-bar-animated"
                              style={{ width: `${job.progress_percent}%` }}
                            />
                          </div>
                          <small className="d-block">
                            {isSyntheticDataJob(job) ? (
                              <>
                                {job.current || 0}/{job.total || 0} samples ({job.progress_percent.toFixed(1)}%)
                              </>
                            ) : (
                              <>
                                {job.current_epoch}/{job.total_epochs} epochs
                              </>
                            )}
                          </small>
                          <small className="text-muted d-block">ETA: {calculateETA(job)}</small>
                        </div>
                      ) : (
                        `${job.progress_percent.toFixed(0)}%`
                      )}
                    </td>
                    <td>
                      {job.train_loss && job.val_loss ? (
                        <small>
                          Train: {job.train_loss.toFixed(4)}<br />
                          Val: {job.val_loss.toFixed(4)}
                        </small>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>{new Date(job.created_at).toLocaleString()}</td>
                    <td>
                      <div className="d-flex gap-1">
                        {/* Pause button for running training jobs (not synthetic data) */}
                        {job.status === 'running' && !isSyntheticDataJob(job) && (
                          <Button
                            variant="outline-secondary"
                            size="sm"
                            onClick={() => handlePauseJob(job.id)}
                            title="Pause training"
                          >
                            <Pause size={14} />
                          </Button>
                        )}
                        {/* Resume button for paused jobs */}
                        {job.status === 'paused' && (
                          <Button
                            variant="outline-success"
                            size="sm"
                            onClick={() => handleResumeJob(job.id)}
                            title="Resume training"
                          >
                            <Play size={14} />
                          </Button>
                        )}
                        {/* Cancel button for running/queued/pending jobs */}
                        {(job.status === 'running' || job.status === 'queued' || job.status === 'pending') && (
                          <Button
                            variant="outline-warning"
                            size="sm"
                            onClick={() => handleCancelJob(job.id)}
                            title="Cancel job"
                          >
                            <StopCircle size={14} />
                          </Button>
                        )}
                        {/* Clone button for completed/cancelled/failed jobs */}
                        {(job.status === 'completed' || job.status === 'cancelled' || job.status === 'failed') && (
                          <Button
                            variant="outline-primary"
                            size="sm"
                            onClick={() => handleCloneJob(job)}
                            title="Clone job configuration"
                          >
                            <Copy size={14} />
                          </Button>
                        )}
                        {/* Delete button for non-running jobs */}
                        <Button
                          variant="outline-danger"
                          size="sm"
                          onClick={() => handleDeleteJob(job.id)}
                          disabled={job.status === 'running' || job.status === 'queued' || job.status === 'pending' || job.status === 'paused'}
                          title="Delete job"
                        >
                          <Trash size={14} />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card.Body>
      </Card>

      {/* Trained Models Section */}
      <Card>
        <Card.Header>
          <h5 className="mb-0">
            <Brain className="me-2" />
            Trained Models
          </h5>
        </Card.Header>
        <Card.Body>
          {models.length === 0 ? (
            <Alert variant="info">
              No trained models yet.
            </Alert>
          ) : (
            <Table striped hover responsive>
              <thead>
                <tr>
                  <th>Model Name</th>
                  <th>Version</th>
                  <th>Accuracy</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id}>
                    <td><strong>{model.model_name}</strong></td>
                    <td>v{model.version}</td>
                    <td>
                      {model.accuracy_meters ? (
                        `±${model.accuracy_meters.toFixed(0)}m`
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>
                      {model.is_active && (
                        <Badge bg="success" className="me-1">Active</Badge>
                      )}
                      {model.is_production && (
                        <Badge bg="primary">Production</Badge>
                      )}
                    </td>
                    <td>{new Date(model.created_at).toLocaleString()}</td>
                    <td>
                      {!model.is_active && (
                        <Button
                          variant="outline-success"
                          size="sm"
                          className="me-1"
                          onClick={() => handleDeployModel(model.id)}
                        >
                          Deploy
                        </Button>
                      )}
                      {!model.is_active && (
                        <Button
                          variant="outline-danger"
                          size="sm"
                          onClick={() => handleDeleteModel(model.id)}
                        >
                          <Trash size={14} />
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card.Body>
      </Card>

      {/* Dataset Details Modal */}
      <Modal
        show={showDetailsModal}
        onHide={() => setShowDetailsModal(false)}
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>
            <Database className="me-2" />
            Dataset Details: {selectedDataset?.name}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedDataset && (
            <>
              {/* Dataset Overview */}
              <Card className="mb-3">
                <Card.Header>
                  <h6 className="mb-0">Overview</h6>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={6}>
                      <p><strong>ID:</strong> <code>{selectedDataset.id}</code></p>
                      <p><strong>Name:</strong> {selectedDataset.name}</p>
                      <p><strong>Description:</strong> {selectedDataset.description || 'N/A'}</p>
                      <p><strong>Total Samples:</strong> {selectedDataset.num_samples.toLocaleString()}</p>
                    </Col>
                    <Col md={6}>
                      <p><strong>Storage Table:</strong> <code>{selectedDataset.storage_table}</code></p>
                      <p><strong>Created:</strong> {new Date(selectedDataset.created_at).toLocaleString()}</p>
                      <p><strong>Created by Job:</strong> {selectedDataset.created_by_job_id ? <code>{selectedDataset.created_by_job_id}</code> : 'N/A'}</p>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>

              {/* Generation Config */}
              <Card className="mb-3">
                <Card.Header>
                  <h6 className="mb-0">Generation Configuration</h6>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={6}>
                      <p><strong>Frequency:</strong> {selectedDataset.config.frequency_mhz || 'N/A'} MHz</p>
                      <p><strong>TX Power:</strong> {selectedDataset.config.tx_power_dbm || 'N/A'} dBm</p>
                      <p><strong>Inside Ratio:</strong> {selectedDataset.config.inside_ratio ? (selectedDataset.config.inside_ratio * 100).toFixed(0) + '%' : 'N/A'}</p>
                    </Col>
                    <Col md={6}>
                      <p><strong>Min SNR:</strong> {selectedDataset.config.min_snr_db ?? 'N/A'} dB</p>
                      <p><strong>Min Receivers:</strong> {selectedDataset.config.min_receivers || 'N/A'}</p>
                      <p><strong>Max GDOP:</strong> {selectedDataset.config.max_gdop || 'N/A'}</p>
                      <p><strong>Real Terrain:</strong> {selectedDataset.config.use_real_terrain ? 'Yes (SRTM)' : 'No'}</p>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>

              {/* Quality Metrics */}
              {selectedDataset.quality_metrics && (
                <Card className="mb-3">
                  <Card.Header>
                    <h6 className="mb-0">Quality Metrics</h6>
                  </Card.Header>
                  <Card.Body>
                    <Row>
                      <Col md={6}>
                        <h6 className="text-primary">Signal-to-Noise Ratio (SNR)</h6>
                        <p><strong>Average:</strong> {selectedDataset.quality_metrics.avg_snr_db?.toFixed(2)} dB</p>
                        <p><strong>Minimum:</strong> {selectedDataset.quality_metrics.min_snr_db?.toFixed(2)} dB</p>
                        <p><strong>Maximum:</strong> {selectedDataset.quality_metrics.max_snr_db?.toFixed(2)} dB</p>
                      </Col>
                      <Col md={6}>
                        <h6 className="text-primary">Geometric Dilution of Precision (GDOP)</h6>
                        <p><strong>Average:</strong> {selectedDataset.quality_metrics.avg_gdop?.toFixed(3)}</p>
                        <p><strong>Minimum:</strong> {selectedDataset.quality_metrics.min_gdop?.toFixed(3)}</p>
                        <p><strong>Maximum:</strong> {selectedDataset.quality_metrics.max_gdop?.toFixed(3)}</p>
                      </Col>
                    </Row>
                    <Row className="mt-3">
                      <Col md={6}>
                        <h6 className="text-primary">Receiver Coverage</h6>
                        <p><strong>Average Receivers:</strong> {selectedDataset.quality_metrics.avg_receivers?.toFixed(2)}</p>
                      </Col>
                      <Col md={6}>
                        <h6 className="text-primary">Distance Statistics</h6>
                        <p><strong>Average Distance:</strong> {selectedDataset.quality_metrics.avg_distance_km?.toFixed(2)} km</p>
                        <p><strong>Maximum Distance:</strong> {selectedDataset.quality_metrics.max_distance_km?.toFixed(2)} km</p>
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              )}

              {/* Sample Preview */}
              <Card>
                <Card.Header>
                  <h6 className="mb-0">Sample Preview (first 10 samples)</h6>
                </Card.Header>
                <Card.Body>
                  {loadingSamples ? (
                    <div className="text-center p-3">
                      <Spinner animation="border" size="sm" />
                      <p className="mt-2 mb-0">Loading samples...</p>
                    </div>
                  ) : datasetSamples.length > 0 ? (
                    <Table striped hover size="sm">
                      <thead>
                        <tr>
                          <th>Sample ID</th>
                          <th>Timestamp</th>
                          <th>TX Latitude</th>
                          <th>TX Longitude</th>
                          <th>Frequency</th>
                          <th>TX Power</th>
                          <th>Receivers</th>
                          <th>GDOP</th>
                          <th>Split</th>
                        </tr>
                      </thead>
                      <tbody>
                        {datasetSamples.map((sample) => (
                          <tr key={sample.id}>
                            <td>{sample.id}</td>
                            <td>{new Date(sample.timestamp).toLocaleString()}</td>
                            <td>{sample.tx_lat.toFixed(6)}°</td>
                            <td>{sample.tx_lon.toFixed(6)}°</td>
                            <td>{(sample.frequency_hz / 1e6).toFixed(2)} MHz</td>
                            <td>{sample.tx_power_dbm.toFixed(1)} dBm</td>
                            <td>{sample.num_receivers}</td>
                            <td>{sample.gdop.toFixed(2)}</td>
                            <td><Badge bg="secondary">{sample.split}</Badge></td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  ) : (
                    <Alert variant="warning">
                      No samples available for this dataset.
                    </Alert>
                  )}
                </Card.Body>
              </Card>
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowDetailsModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>

      {/* Data Generation Modal */}
      <Modal show={showDataModal} onHide={() => setShowDataModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Generate Synthetic Training Data</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Dataset Name</Form.Label>
              <Form.Control
                type="text"
                value={dataForm.name}
                onChange={(e) => setDataForm({ ...dataForm, name: e.target.value })}
                placeholder="e.g., italian_northwest_v1"
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Number of Samples</Form.Label>
              <Form.Control
                type="number"
                value={dataForm.num_samples}
                onChange={(e) => setDataForm({ ...dataForm, num_samples: parseInt(e.target.value) })}
                min={1000}
                max={100000}
              />
              <Form.Text>1,000 - 100,000 samples (recommended: 10,000)</Form.Text>
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Inside Network Ratio</Form.Label>
              <Form.Control
                type="number"
                value={dataForm.inside_ratio}
                onChange={(e) => setDataForm({ ...dataForm, inside_ratio: parseFloat(e.target.value) })}
                step={0.05}
                min={0}
                max={1}
              />
              <Form.Text>70% inside, 30% outside recommended</Form.Text>
            </Form.Group>

            <hr />
            <h6 className="mb-3">Signal Quality Constraints</h6>
            <Alert variant="info" className="mb-3">
              These parameters control which synthetic samples are accepted.
              Stricter values = higher quality but fewer samples.
              Relaxing these values will increase sample generation success rate.
            </Alert>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Frequency (MHz)</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.frequency_mhz}
                    onChange={(e) => setDataForm({ ...dataForm, frequency_mhz: parseFloat(e.target.value) })}
                    step={0.1}
                    min={50}
                    max={3000}
                  />
                  <Form.Text>VHF/UHF frequency (e.g., 144.0 for 2m band)</Form.Text>
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label className="d-flex align-items-center justify-content-between">
                    <span>TX Power</span>
                    <div className="btn-group btn-group-sm">
                      <button
                        type="button"
                        className={`btn ${powerUnit === 'watt' ? 'btn-primary' : 'btn-outline-secondary'}`}
                        onClick={() => setPowerUnit('watt')}
                        aria-pressed={powerUnit === 'watt'}
                      >
                        Watt
                      </button>
                      <button
                        type="button"
                        className={`btn ${powerUnit === 'dbm' ? 'btn-primary' : 'btn-outline-secondary'}`}
                        onClick={() => setPowerUnit('dbm')}
                        aria-pressed={powerUnit === 'dbm'}
                      >
                        dBm
                      </button>
                    </div>
                  </Form.Label>
                  {powerUnit === 'watt' ? (
                    <>
                      <Form.Control
                        type="number"
                        value={powerValueWatt}
                        onChange={(e) => {
                          const watt = parseFloat(e.target.value);
                          setPowerValueWatt(watt);
                          setDataForm({ ...dataForm, tx_power_dbm: wattToDbm(watt) });
                        }}
                        step={0.1}
                        min={0.001}
                        max={100}
                      />
                      <Form.Text>
                        Transmitter power in Watt (2W typical) = {dataForm.tx_power_dbm.toFixed(2)} dBm
                      </Form.Text>
                    </>
                  ) : (
                    <>
                      <Form.Control
                        type="number"
                        value={dataForm.tx_power_dbm}
                        onChange={(e) => {
                          const dbm = parseFloat(e.target.value);
                          setDataForm({ ...dataForm, tx_power_dbm: dbm });
                          setPowerValueWatt(dbmToWatt(dbm));
                        }}
                        step={1}
                        min={0}
                        max={60}
                      />
                      <Form.Text>
                        Transmitter power in dBm (30-40 dBm typical) = {powerValueWatt.toFixed(3)} W
                      </Form.Text>
                    </>
                  )}
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Min SNR (dB)</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.min_snr_db}
                    onChange={(e) => setDataForm({ ...dataForm, min_snr_db: parseFloat(e.target.value) })}
                    step={0.5}
                    min={-5}
                    max={20}
                  />
                  <Form.Text>Minimum signal-to-noise ratio (0 = more samples)</Form.Text>
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Min Receivers</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.min_receivers}
                    onChange={(e) => setDataForm({ ...dataForm, min_receivers: parseInt(e.target.value) })}
                    step={1}
                    min={2}
                    max={7}
                  />
                  <Form.Text>Minimum detecting stations (2 = more samples)</Form.Text>
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Max GDOP</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.max_gdop}
                    onChange={(e) => setDataForm({ ...dataForm, max_gdop: parseFloat(e.target.value) })}
                    step={5}
                    min={5}
                    max={200}
                  />
                  <Form.Text>Max geometric dilution (100 = balanced for ML)</Form.Text>
                </Form.Group>
              </Col>
            </Row>

            <hr />
            <h6 className="mb-3">Terrain Options</h6>
            <Form.Group className="mb-3">
              <Form.Check
                type="checkbox"
                label="Use Real Terrain Data (SRTM)"
                checked={dataForm.use_real_terrain}
                onChange={(e) => setDataForm({ ...dataForm, use_real_terrain: e.target.checked })}
              />
              <Form.Text className="text-muted">
                Enable realistic RF propagation with SRTM elevation data. Requires terrain tiles to be downloaded in Terrain Management.
              </Form.Text>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowDataModal(false)}>
            Cancel
          </Button>
          <Button 
            variant="primary" 
            onClick={handleGenerateData}
            disabled={!dataForm.name || dataForm.num_samples < 1000}
          >
            Generate Dataset
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default TrainingDashboard;
