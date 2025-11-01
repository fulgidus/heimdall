/**
 * Training Dashboard
 * 
 * Main page for ML training pipeline:
 * - Synthetic data generation
 * - Training job management  
 * - Model management
 */

import React, { useEffect, useState } from 'react';
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
} from 'lucide-react';
import {
  listTrainingJobs,
  listSyntheticDatasets,
  listModels,
  generateSyntheticData,
  deleteTrainingJob,
  deleteSyntheticDataset,
  deleteModel,
  deployModel,
  type TrainingJob,
  type SyntheticDataset,
  type TrainedModel,
} from '@/services/api/training';

const TrainingDashboard: React.FC = () => {
  // State
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [datasets, setDatasets] = useState<SyntheticDataset[]>([]);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDataModal, setShowDataModal] = useState(false);
  
  // Data generation form
  const [dataForm, setDataForm] = useState({
    name: '',
    num_samples: 10000,
    inside_ratio: 0.7,
    train_ratio: 0.7,
    val_ratio: 0.15,
    test_ratio: 0.15,
  });

  // Load data
  const loadData = async () => {
    try {
      setLoading(true);
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
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Refresh every 5 seconds
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle data generation
  const handleGenerateData = async () => {
    try {
      await generateSyntheticData(dataForm);
      setShowDataModal(false);
      setDataForm({
        name: '',
        num_samples: 10000,
        inside_ratio: 0.7,
        train_ratio: 0.7,
        val_ratio: 0.15,
        test_ratio: 0.15,
      });
      loadData(); // Refresh
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate data');
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

  // Get status badge
  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { variant: string; icon: React.ReactNode }> = {
      pending: { variant: 'secondary', icon: <Clock size={14} /> },
      queued: { variant: 'info', icon: <Clock size={14} /> },
      running: { variant: 'primary', icon: <Spinner animation="border" size="sm" /> },
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

  if (loading && jobs.length === 0) {
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
                  <th>Train/Val/Test</th>
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
                      {dataset.train_count}/{dataset.val_count}/{dataset.test_count}
                    </td>
                    <td>
                      {dataset.quality_metrics ? (
                        <small>
                          SNR: {dataset.quality_metrics.avg_snr_db?.toFixed(1)}dB,
                          GDOP: {dataset.quality_metrics.avg_gdop?.toFixed(2)}
                        </small>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>{new Date(dataset.created_at).toLocaleString()}</td>
                    <td>
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
                          <div className="progress">
                            <div
                              className="progress-bar progress-bar-striped progress-bar-animated"
                              style={{ width: `${job.progress_percent}%` }}
                            />
                          </div>
                          <small>{job.current_epoch}/{job.total_epochs} epochs</small>
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
                      <Button
                        variant="outline-danger"
                        size="sm"
                        onClick={() => handleDeleteJob(job.id)}
                        disabled={job.status === 'running'}
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

            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Training Split</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.train_ratio}
                    onChange={(e) => setDataForm({ ...dataForm, train_ratio: parseFloat(e.target.value) })}
                    step={0.05}
                    min={0}
                    max={1}
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Validation Split</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.val_ratio}
                    onChange={(e) => setDataForm({ ...dataForm, val_ratio: parseFloat(e.target.value) })}
                    step={0.05}
                    min={0}
                    max={1}
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Test Split</Form.Label>
                  <Form.Control
                    type="number"
                    value={dataForm.test_ratio}
                    onChange={(e) => setDataForm({ ...dataForm, test_ratio: parseFloat(e.target.value) })}
                    step={0.05}
                    min={0}
                    max={1}
                  />
                </Form.Group>
              </Col>
            </Row>

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
