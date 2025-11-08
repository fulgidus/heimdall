import React, { useEffect, useState } from 'react';
import classNames from 'classnames';
import Modal from './Modal';
import type { 
  ModelArchitecture, 
  ModelCategory,
  ComplexityLevel,
  DataType,
  SyntheticDataset
} from '@/services/api/training';
import { 
  listModelArchitectures,
  getRecommendedModelArchitecture,
  listSyntheticDatasets
} from '@/services/api/training';

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectModel: (architecture: ModelArchitecture, trainingConfig: TrainingConfig) => void;
  onNavigateToConfig?: () => void;  // Callback when navigating to config tab (to switch to jobs tab)
}

interface TrainingConfig {
  job_name: string;  // Training job name
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  dataset_ids: string[];  // Changed to array for multiple selection
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
}

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  job_name: '',  // Will be generated if empty
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.001,
  early_stopping_patience: 10,
  dataset_ids: [],  // Changed to empty array
  train_ratio: 0.7,
  val_ratio: 0.15,
  test_ratio: 0.15,
};

// Category emoji mapping
const CATEGORY_EMOJIS: Record<ModelCategory, string> = {
  spectrogram: '🖼️',
  iq_raw_cnn: '📡',
  transformer: '🔬',
  temporal: '🌊',
  hybrid: '🧠',
  features: '🧮',
};

// Speed emoji mapping
const SPEED_EMOJIS = {
  very_fast: '🚀',
  fast: '🏃',
  moderate: '🚶',
  slow: '🐢',
};

// Complexity emoji mapping
const COMPLEXITY_EMOJIS = {
  low: '📦',
  medium: '📚',
  high: '🏪',
  very_high: '🏢',
};

// Quality emoji mapping
const QUALITY_EMOJIS = {
  excellent: '💎',
  good: '💍',
  fair: '⚪',
  experimental: '⚫',
};

const ModelArchitectureCard: React.FC<{
  architecture: ModelArchitecture;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ architecture, isSelected, onSelect }) => {
  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <span key={i} className={i < rating ? 'text-yellow-400' : 'text-gray-600'}>
        ★
      </span>
    ));
  };

  const renderBadges = () => {
    const badges = [];
    if (architecture.badges.recommended) badges.push({ text: 'RECOMMENDED', color: 'bg-neon-blue', textColor: 'text-white' });
    if (architecture.badges.maximum_accuracy) badges.push({ text: 'MAX ACCURACY', color: 'bg-sea-green', textColor: 'text-white' });
    if (architecture.badges.fastest) badges.push({ text: 'FASTEST', color: 'bg-light-green', textColor: 'text-oxford-blue' });
    if (architecture.badges.best_ratio) badges.push({ text: 'BEST RATIO', color: 'bg-amber-500', textColor: 'text-oxford-blue' });
    if (architecture.badges.experimental) badges.push({ text: 'EXPERIMENTAL', color: 'bg-red-500', textColor: 'text-white' });
    if (architecture.badges.production_ready) badges.push({ text: 'PRODUCTION', color: 'bg-green-600', textColor: 'text-white' });
    if (architecture.badges.memory_efficient) badges.push({ text: 'MEM EFFICIENT', color: 'bg-purple-500', textColor: 'text-white' });
    if (architecture.badges.gpu_optimized) badges.push({ text: 'GPU OPTIMIZED', color: 'bg-orange-500', textColor: 'text-oxford-blue' });

    return badges.map((badge, idx) => (
      <span
        key={idx}
        className={classNames(
          'text-xs px-2 py-1 rounded font-semibold',
          badge.color,
          badge.textColor
        )}
      >
        {badge.text}
      </span>
    ));
  };

  return (
    <div
      onClick={onSelect}
      className={classNames(
        'card p-3 cursor-pointer',
        isSelected
          ? 'border-primary bg-primary bg-opacity-10 shadow'
          : 'border-secondary'
      )}
      style={{ cursor: 'pointer' }}
    >
      {/* Header */}
      <div className="d-flex justify-content-between align-items-start mb-3">
        <div className="d-flex align-items-center gap-2">
          <span style={{ fontSize: '2rem' }}>{architecture.emoji}</span>
          <div>
            <h5 className="mb-1">{architecture.display_name}</h5>
            <small className="text-muted">{architecture.class_name}</small>
          </div>
        </div>
        {isSelected && <span className="text-primary" style={{ fontSize: '1.5rem' }}>✓</span>}
      </div>

      {/* Badges */}
      {renderBadges().length > 0 && (
        <div className="d-flex flex-wrap gap-1 mb-3">{renderBadges()}</div>
      )}

      {/* Description */}
      <p className="small text-muted mb-3">{architecture.metadata.description}</p>

      {/* Metadata indicators */}
      <div className="d-flex gap-3 mb-3 small">
        <div className="d-flex align-items-center gap-1">
          <span>{CATEGORY_EMOJIS[architecture.category]}</span>
          <span className="text-muted text-capitalize">{architecture.category.replace('_', ' ')}</span>
        </div>
        <div className="d-flex align-items-center gap-1">
          <span>{SPEED_EMOJIS[architecture.speed_rating]}</span>
          <span className="text-muted">{architecture.speed_rating.replace('_', ' ')}</span>
        </div>
        <div className="d-flex align-items-center gap-1">
          <span>{COMPLEXITY_EMOJIS[architecture.complexity]}</span>
          <span className="text-muted">{architecture.complexity}</span>
        </div>
      </div>

      {/* Star Ratings */}
      <div className="mb-3">
        <div className="d-flex justify-content-between align-items-center small mb-1">
          <span className="text-muted">Accuracy</span>
          <span>{renderStars(architecture.star_ratings.accuracy)}</span>
        </div>
        <div className="d-flex justify-content-between align-items-center small mb-1">
          <span className="text-muted">Speed</span>
          <span>{renderStars(architecture.star_ratings.speed)}</span>
        </div>
        <div className="d-flex justify-content-between align-items-center small mb-1">
          <span className="text-muted">Efficiency</span>
          <span>{renderStars(architecture.star_ratings.efficiency)}</span>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="row g-2 small bg-dark bg-opacity-25 p-2 rounded">
        <div className="col-6">
          <span className="text-muted">Accuracy: </span>
          <span className="fw-bold">
            ±{architecture.performance.accuracy_meters_mean.toFixed(0)}m
          </span>
        </div>
        <div className="col-6">
          <span className="text-muted">Inference: </span>
          <span className="fw-bold">
            {architecture.performance.inference_ms_mean.toFixed(0)}ms
          </span>
        </div>
        <div className="col-6">
          <span className="text-muted">Params: </span>
          <span className="fw-bold">
            {architecture.parameters_millions.toFixed(1)}M
          </span>
        </div>
        <div className="col-6">
          <span className="text-muted">Size: </span>
          <span className="fw-bold">
            {architecture.model_size_mb.toFixed(0)}MB
          </span>
        </div>
      </div>
    </div>
  );
};

const ModelSelectionModal: React.FC<ModelSelectionModalProps> = ({ isOpen, onClose, onSelectModel, onNavigateToConfig }) => {
  const [activeTab, setActiveTab] = useState<'select' | 'config'>('select');
  const [architectures, setArchitectures] = useState<ModelArchitecture[]>([]);
  const [filteredArchitectures, setFilteredArchitectures] = useState<ModelArchitecture[]>([]);
  const [selectedArchitecture, setSelectedArchitecture] = useState<ModelArchitecture | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Datasets
  const [datasets, setDatasets] = useState<SyntheticDataset[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);

  // Filters
  const [categoryFilter, setCategoryFilter] = useState<ModelCategory | 'all'>('all');
  const [complexityFilter, setComplexityFilter] = useState<ComplexityLevel | 'all'>('all');

  useEffect(() => {
    if (isOpen) {
      loadArchitectures();
      loadDatasets();
    }
  }, [isOpen]);

  useEffect(() => {
    applyFilters();
  }, [architectures, categoryFilter, complexityFilter]);

  const loadArchitectures = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listModelArchitectures();
      const archs = response.architectures || [];
      setArchitectures(archs);
      setFilteredArchitectures(archs);
    } catch (err) {
      setError('Failed to load model architectures');
      console.error(err);
      // Ensure we have empty arrays even on error
      setArchitectures([]);
      setFilteredArchitectures([]);
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    setLoadingDatasets(true);
    try {
      const response = await listSyntheticDatasets(100, 0); // Get up to 100 datasets
      setDatasets(response.datasets || []);
    } catch (err) {
      console.error('Failed to load datasets:', err);
      setDatasets([]);
    } finally {
      setLoadingDatasets(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...architectures];

    if (categoryFilter !== 'all') {
      filtered = filtered.filter(arch => arch.category === categoryFilter);
    }

    if (complexityFilter !== 'all') {
      filtered = filtered.filter(arch => arch.complexity === complexityFilter);
    }

    setFilteredArchitectures(filtered);
  };

  const handleSelectArchitecture = (architecture: ModelArchitecture) => {
    setSelectedArchitecture(architecture);
  };

  const handleNextToConfig = () => {
    if (selectedArchitecture) {
      setActiveTab('config');
      // Callback to switch to jobs tab on the main page
      if (onNavigateToConfig) {
        onNavigateToConfig();
      }
    }
  };

  const handleBackToSelection = () => {
    setActiveTab('select');
  };

  const handleStartTraining = () => {
    if (selectedArchitecture) {
      onSelectModel(selectedArchitecture, trainingConfig);
      onClose();
    }
  };

  // Validation: Check if selected datasets have samples > 0
  const selectedDatasetsWithSamples = trainingConfig.dataset_ids
    .map(id => datasets.find(d => d.id === id))
    .filter(d => d && d.num_samples > 0);

  const hasValidDatasets = selectedDatasetsWithSamples.length > 0;
  const ratiosSumToOne = (trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio) === 1.0;
  const canStartTraining = hasValidDatasets && ratiosSumToOne;

  const modelSelectionTab = (
    <div>
      {/* Filters */}
      <div className="mb-4 row g-3">
        <div className="col-md-6">
          <label className="form-label">Category</label>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value as ModelCategory | 'all')}
            className="form-select"
          >
            <option value="all">All Categories</option>
            <option value="spectrogram">🖼️ Spectrogram</option>
            <option value="iq_raw_cnn">📡 IQ Raw CNN</option>
            <option value="transformer">🔬 Transformer</option>
            <option value="temporal">🌊 Temporal</option>
            <option value="hybrid">🧠 Hybrid</option>
            <option value="features">🧮 Features</option>
          </select>
        </div>
        <div className="col-md-6">
          <label className="form-label">Complexity</label>
          <select
            value={complexityFilter}
            onChange={(e) => setComplexityFilter(e.target.value as ComplexityLevel | 'all')}
            className="form-select"
          >
            <option value="all">All Levels</option>
            <option value="low">📦 Low</option>
            <option value="medium">📚 Medium</option>
            <option value="high">🏪 High</option>
            <option value="very_high">🏢 Very High</option>
          </select>
        </div>
      </div>

      {/* Architecture Cards Grid */}
      {loading ? (
        <div className="text-center py-5 text-muted">Loading architectures...</div>
      ) : error ? (
        <div className="text-center py-5 text-danger">{error}</div>
      ) : !filteredArchitectures || filteredArchitectures.length === 0 ? (
        <div className="text-center py-5 text-muted">No architectures found</div>
      ) : (
        <div className="row g-3" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
          {filteredArchitectures.map(arch => (
            <div key={arch.id} className="col-md-6 col-lg-4">
              <ModelArchitectureCard
                architecture={arch}
                isSelected={selectedArchitecture?.id === arch.id}
                onSelect={() => handleSelectArchitecture(arch)}
              />
            </div>
          ))}
        </div>
      )}

      {/* Selection Info */}
      {selectedArchitecture && (
        <div className="mt-4 p-3 bg-primary bg-opacity-10 border border-primary rounded">
          <h6 className="mb-2">Selected: {selectedArchitecture.display_name}</h6>
          <div className="small text-muted">
            <p className="mb-1"><strong>Recommended for:</strong> {selectedArchitecture.metadata.recommended_for.join(', ')}</p>
            <p className="mb-1"><strong>Training time:</strong> {selectedArchitecture.metadata.training_time_estimate}</p>
            <p className="mb-0"><strong>Dataset size:</strong> {selectedArchitecture.metadata.dataset_size_recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );

  // Filter datasets based on selected architecture's data_type
  const filteredDatasets = selectedArchitecture
    ? datasets.filter(dataset => {
        // Match dataset type with model data_type
        // TODO: Add proper field in dataset schema for data_type matching
        // For now, show all datasets
        return true;
      })
    : [];

  const trainingConfigTab = (
    <div>
      {/* Job Name */}
      <div className="mb-3">
        <label className="form-label">
          Job Name
          <span className="text-danger ms-1">*</span>
        </label>
        <input
          type="text"
          value={trainingConfig.job_name}
          onChange={(e) => setTrainingConfig({ ...trainingConfig, job_name: e.target.value })}
          placeholder={selectedArchitecture ? `${selectedArchitecture.display_name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}` : 'my-training-job'}
          className="form-control"
          required
        />
        <small className="text-muted">
          A unique name to identify this training job. Leave empty to auto-generate.
        </small>
      </div>

      {/* Dataset Selection */}
      <div className="mb-3">
        <label className="form-label">
          Select Datasets
          {selectedArchitecture && (
            <span className="text-muted ms-2">(Compatible with {selectedArchitecture.data_type})</span>
          )}
        </label>
        {loadingDatasets ? (
          <div className="text-center py-3 text-muted">
            <div className="spinner-border spinner-border-sm me-2" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            Loading datasets...
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="alert alert-warning mb-0">
            <i className="bi bi-exclamation-triangle me-2"></i>
            No compatible datasets found. You can proceed with the default dataset.
          </div>
        ) : (
          <div className="border rounded p-3" style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {filteredDatasets.map(dataset => (
              <div key={dataset.id} className="form-check mb-2">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id={`dataset-${dataset.id}`}
                  checked={trainingConfig.dataset_ids.includes(dataset.id)}
                  onChange={(e) => {
                    const newIds = e.target.checked
                      ? [...trainingConfig.dataset_ids, dataset.id]
                      : trainingConfig.dataset_ids.filter(id => id !== dataset.id);
                    setTrainingConfig({ ...trainingConfig, dataset_ids: newIds });
                  }}
                />
                <label className="form-check-label" htmlFor={`dataset-${dataset.id}`} style={{ cursor: 'pointer' }}>
                  <strong>{dataset.name}</strong>
                  <span className="text-muted ms-2">({dataset.num_samples.toLocaleString()} samples)</span>
                  {dataset.description && (
                    <div className="small text-muted mt-1">{dataset.description}</div>
                  )}
                </label>
              </div>
            ))}
          </div>
        )}
        {trainingConfig.dataset_ids.length > 0 && (
          <small className="d-block mt-2">
            {selectedDatasetsWithSamples.length > 0 ? (
              <span className="text-success">
                <i className="bi bi-check-circle-fill me-1"></i>
                {selectedDatasetsWithSamples.length} dataset{selectedDatasetsWithSamples.length !== 1 ? 's' : ''} selected with valid samples
              </span>
            ) : (
              <span className="text-danger">
                <i className="bi bi-exclamation-triangle-fill me-1"></i>
                Selected dataset{trainingConfig.dataset_ids.length !== 1 ? 's have' : ' has'} 0 samples. Training cannot proceed.
              </span>
            )}
          </small>
        )}
      </div>

      {/* Training Parameters */}
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <label className="form-label">Epochs</label>
          <input
            type="number"
            value={trainingConfig.epochs}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
            min={1}
            max={500}
            className="form-control"
          />
        </div>
        <div className="col-md-6">
          <label className="form-label">Batch Size</label>
          <input
            type="number"
            value={trainingConfig.batch_size}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
            min={1}
            max={256}
            className="form-control"
          />
        </div>
        <div className="col-md-6">
          <label className="form-label">Learning Rate</label>
          <input
            type="number"
            value={trainingConfig.learning_rate}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
            step={0.0001}
            min={0.00001}
            max={0.1}
            className="form-control"
          />
        </div>
        <div className="col-md-6">
          <label className="form-label">
            Early Stop Patience
            <small className="text-muted ms-2">(0 = disabled)</small>
          </label>
          <input
            type="number"
            value={trainingConfig.early_stopping_patience}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, early_stopping_patience: parseInt(e.target.value) })}
            min={0}
            max={50}
            className="form-control"
          />
        </div>
      </div>

      {/* Data Split Ratios */}
      <div className="mb-3">
        <label className="form-label">Data Split Ratios</label>
        <div className="row g-3">
          <div className="col-md-4">
            <label className="form-label small">Train</label>
            <input
              type="number"
              value={trainingConfig.train_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, train_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.1}
              max={0.9}
              className="form-control"
            />
          </div>
          <div className="col-md-4">
            <label className="form-label small">Validation</label>
            <input
              type="number"
              value={trainingConfig.val_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, val_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.05}
              max={0.5}
              className="form-control"
            />
          </div>
          <div className="col-md-4">
            <label className="form-label small">Test</label>
            <input
              type="number"
              value={trainingConfig.test_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, test_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.05}
              max={0.5}
              className="form-control"
            />
          </div>
        </div>
        <small className="text-muted d-block mt-1">
          Total: {(trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio).toFixed(2)}
          {(trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio) !== 1.0 && (
            <span className="text-danger ms-2">⚠️ Must sum to 1.0</span>
          )}
        </small>
      </div>

      {/* Selected Model Summary */}
      {selectedArchitecture && (
        <div className="p-3 bg-primary bg-opacity-10 border border-primary rounded">
          <h6 className="mb-2">Training: {selectedArchitecture.display_name}</h6>
          <div className="small text-muted">
            <p className="mb-1">Expected training time: {selectedArchitecture.metadata.training_time_estimate}</p>
            <p className="mb-0">Model parameters: {selectedArchitecture.parameters_millions.toFixed(1)}M</p>
          </div>
        </div>
      )}
    </div>
  );

  const footer = (
    <>
      {activeTab === 'select' ? (
        <>
          <button
            onClick={onClose}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleNextToConfig}
            disabled={!selectedArchitecture}
            className="btn btn-primary"
          >
            Next: Configure Training →
          </button>
        </>
      ) : (
        <>
          <button
            onClick={handleBackToSelection}
            className="btn btn-secondary"
          >
            ← Back to Selection
          </button>
          <button
            onClick={handleStartTraining}
            disabled={!canStartTraining}
            className="btn btn-success"
            title={!hasValidDatasets ? 'Please select at least one dataset with samples > 0' : (!ratiosSumToOne ? 'Data split ratios must sum to 1.0' : '')}
          >
            Start Training 🚀
          </button>
        </>
      )}
    </>
  );

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create Training Job" size="xl" footer={footer}>
      {/* Bootstrap Tab Navigation */}
      <ul className="nav nav-tabs mb-4">
        <li className="nav-item">
          <button
            className={`nav-link ${activeTab === 'select' ? 'active' : ''}`}
            onClick={() => setActiveTab('select')}
          >
            🎯 Select Model
          </button>
        </li>
        <li className="nav-item">
          <button
            className={`nav-link ${activeTab === 'config' ? 'active' : ''}`}
            onClick={() => {
              // Prevent navigating to config tab if no model is selected
              if (!selectedArchitecture) {
                return;
              }
              setActiveTab('config');
            }}
            disabled={!selectedArchitecture}
          >
            ⚙️ Training Config
          </button>
        </li>
      </ul>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'select' && modelSelectionTab}
        {activeTab === 'config' && trainingConfigTab}
      </div>
    </Modal>
  );
};

export default ModelSelectionModal;
