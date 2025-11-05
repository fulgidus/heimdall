import React, { useEffect, useState } from 'react';
import classNames from 'classnames';
import Modal from './Modal';
import Tabs from './Tabs';
import { 
  ModelArchitecture, 
  ModelCategory,
  ComplexityLevel,
  listModelArchitectures,
  getRecommendedModelArchitecture 
} from '@/services/api/training';

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectModel: (architecture: ModelArchitecture, trainingConfig: TrainingConfig) => void;
}

interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  dataset_id?: string;
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
}

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.001,
  early_stopping_patience: 10,
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
    if (architecture.badges.recommended) badges.push({ text: 'RECOMMENDED', color: 'bg-neon-blue' });
    if (architecture.badges.maximum_accuracy) badges.push({ text: 'MAX ACCURACY', color: 'bg-sea-green' });
    if (architecture.badges.fastest) badges.push({ text: 'FASTEST', color: 'bg-light-green' });
    if (architecture.badges.best_ratio) badges.push({ text: 'BEST RATIO', color: 'bg-amber-500' });
    if (architecture.badges.experimental) badges.push({ text: 'EXPERIMENTAL', color: 'bg-red-500' });
    if (architecture.badges.production_ready) badges.push({ text: 'PRODUCTION', color: 'bg-green-600' });
    if (architecture.badges.memory_efficient) badges.push({ text: 'MEM EFFICIENT', color: 'bg-purple-500' });
    if (architecture.badges.gpu_optimized) badges.push({ text: 'GPU OPTIMIZED', color: 'bg-orange-500' });

    return badges.map((badge, idx) => (
      <span
        key={idx}
        className={classNames(
          'text-xs px-2 py-1 rounded font-semibold text-white',
          badge.color
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
        'border rounded-lg p-4 cursor-pointer transition-all duration-200 hover:shadow-lg',
        isSelected
          ? 'border-neon-blue bg-neon-blue bg-opacity-10 shadow-md'
          : 'border-neon-blue border-opacity-20 bg-oxford-blue hover:border-neon-blue hover:border-opacity-50'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-3xl">{architecture.emoji}</span>
          <div>
            <h3 className="text-lg font-bold text-white">{architecture.display_name}</h3>
            <p className="text-xs text-french-gray">{architecture.class_name}</p>
          </div>
        </div>
        {isSelected && <span className="text-neon-blue text-2xl">✓</span>}
      </div>

      {/* Badges */}
      {renderBadges().length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">{renderBadges()}</div>
      )}

      {/* Description */}
      <p className="text-sm text-french-gray mb-3">{architecture.metadata.description}</p>

      {/* Metadata indicators */}
      <div className="flex gap-4 mb-3 text-sm">
        <div className="flex items-center gap-1">
          <span>{CATEGORY_EMOJIS[architecture.category]}</span>
          <span className="text-french-gray capitalize">{architecture.category.replace('_', ' ')}</span>
        </div>
        <div className="flex items-center gap-1">
          <span>{SPEED_EMOJIS[architecture.speed_rating]}</span>
          <span className="text-french-gray">{architecture.speed_rating.replace('_', ' ')}</span>
        </div>
        <div className="flex items-center gap-1">
          <span>{COMPLEXITY_EMOJIS[architecture.complexity]}</span>
          <span className="text-french-gray">{architecture.complexity}</span>
        </div>
      </div>

      {/* Star Ratings */}
      <div className="space-y-1 mb-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-french-gray">Accuracy</span>
          <span>{renderStars(architecture.star_ratings.accuracy)}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-french-gray">Speed</span>
          <span>{renderStars(architecture.star_ratings.speed)}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-french-gray">Efficiency</span>
          <span>{renderStars(architecture.star_ratings.efficiency)}</span>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 gap-2 text-xs bg-black bg-opacity-30 p-2 rounded">
        <div>
          <span className="text-french-gray">Accuracy: </span>
          <span className="text-white font-semibold">
            ±{architecture.performance.accuracy_meters_mean.toFixed(0)}m
          </span>
        </div>
        <div>
          <span className="text-french-gray">Inference: </span>
          <span className="text-white font-semibold">
            {architecture.performance.inference_ms_mean.toFixed(0)}ms
          </span>
        </div>
        <div>
          <span className="text-french-gray">Params: </span>
          <span className="text-white font-semibold">
            {architecture.parameters_millions.toFixed(1)}M
          </span>
        </div>
        <div>
          <span className="text-french-gray">Size: </span>
          <span className="text-white font-semibold">
            {architecture.model_size_mb.toFixed(0)}MB
          </span>
        </div>
      </div>
    </div>
  );
};

const ModelSelectionModal: React.FC<ModelSelectionModalProps> = ({ isOpen, onClose, onSelectModel }) => {
  const [activeTab, setActiveTab] = useState<'select' | 'config'>('select');
  const [architectures, setArchitectures] = useState<ModelArchitecture[]>([]);
  const [filteredArchitectures, setFilteredArchitectures] = useState<ModelArchitecture[]>([]);
  const [selectedArchitecture, setSelectedArchitecture] = useState<ModelArchitecture | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [categoryFilter, setCategoryFilter] = useState<ModelCategory | 'all'>('all');
  const [complexityFilter, setComplexityFilter] = useState<ComplexityLevel | 'all'>('all');

  useEffect(() => {
    if (isOpen) {
      loadArchitectures();
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
      setArchitectures(response.architectures);
      setFilteredArchitectures(response.architectures);
    } catch (err) {
      setError('Failed to load model architectures');
      console.error(err);
    } finally {
      setLoading(false);
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

  const modelSelectionTab = (
    <div>
      {/* Filters */}
      <div className="mb-4 flex gap-4">
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Category</label>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value as ModelCategory | 'all')}
            className="bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2 text-sm"
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
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Complexity</label>
          <select
            value={complexityFilter}
            onChange={(e) => setComplexityFilter(e.target.value as ComplexityLevel | 'all')}
            className="bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2 text-sm"
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
        <div className="text-center py-8 text-french-gray">Loading architectures...</div>
      ) : error ? (
        <div className="text-center py-8 text-red-500">{error}</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
          {filteredArchitectures.map(arch => (
            <ModelArchitectureCard
              key={arch.id}
              architecture={arch}
              isSelected={selectedArchitecture?.id === arch.id}
              onSelect={() => handleSelectArchitecture(arch)}
            />
          ))}
        </div>
      )}

      {/* Selection Info */}
      {selectedArchitecture && (
        <div className="mt-4 p-4 bg-neon-blue bg-opacity-10 border border-neon-blue rounded">
          <h4 className="text-white font-bold mb-2">Selected: {selectedArchitecture.display_name}</h4>
          <div className="text-sm text-french-gray space-y-1">
            <p><strong>Recommended for:</strong> {selectedArchitecture.metadata.recommended_for.join(', ')}</p>
            <p><strong>Training time:</strong> {selectedArchitecture.metadata.training_time_estimate}</p>
            <p><strong>Dataset size:</strong> {selectedArchitecture.metadata.dataset_size_recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );

  const trainingConfigTab = (
    <div className="space-y-4">
      {/* Dataset Selection */}
      <div>
        <label className="block text-sm font-medium text-french-gray mb-1">
          Dataset ID (optional)
        </label>
        <input
          type="text"
          value={trainingConfig.dataset_id || ''}
          onChange={(e) => setTrainingConfig({ ...trainingConfig, dataset_id: e.target.value })}
          placeholder="Leave empty for default dataset"
          className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
        />
      </div>

      {/* Training Parameters */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Epochs</label>
          <input
            type="number"
            value={trainingConfig.epochs}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
            min={1}
            max={500}
            className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Batch Size</label>
          <input
            type="number"
            value={trainingConfig.batch_size}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
            min={1}
            max={256}
            className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Learning Rate</label>
          <input
            type="number"
            value={trainingConfig.learning_rate}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
            step={0.0001}
            min={0.00001}
            max={0.1}
            className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-french-gray mb-1">Early Stop Patience</label>
          <input
            type="number"
            value={trainingConfig.early_stopping_patience}
            onChange={(e) => setTrainingConfig({ ...trainingConfig, early_stopping_patience: parseInt(e.target.value) })}
            min={1}
            max={50}
            className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
          />
        </div>
      </div>

      {/* Data Split Ratios */}
      <div>
        <label className="block text-sm font-medium text-french-gray mb-2">Data Split Ratios</label>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-french-gray mb-1">Train</label>
            <input
              type="number"
              value={trainingConfig.train_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, train_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.1}
              max={0.9}
              className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-xs text-french-gray mb-1">Validation</label>
            <input
              type="number"
              value={trainingConfig.val_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, val_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.05}
              max={0.5}
              className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-xs text-french-gray mb-1">Test</label>
            <input
              type="number"
              value={trainingConfig.test_ratio}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, test_ratio: parseFloat(e.target.value) })}
              step={0.05}
              min={0.05}
              max={0.5}
              className="w-full bg-oxford-blue border border-neon-blue border-opacity-20 text-white rounded px-3 py-2"
            />
          </div>
        </div>
        <p className="text-xs text-french-gray mt-1">
          Total: {(trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio).toFixed(2)}
          {(trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio) !== 1.0 && (
            <span className="text-red-500 ml-2">⚠️ Must sum to 1.0</span>
          )}
        </p>
      </div>

      {/* Selected Model Summary */}
      {selectedArchitecture && (
        <div className="mt-4 p-4 bg-neon-blue bg-opacity-10 border border-neon-blue rounded">
          <h4 className="text-white font-bold mb-2">Training: {selectedArchitecture.display_name}</h4>
          <div className="text-sm text-french-gray">
            <p>Expected training time: {selectedArchitecture.metadata.training_time_estimate}</p>
            <p>Model parameters: {selectedArchitecture.parameters_millions.toFixed(1)}M</p>
          </div>
        </div>
      )}
    </div>
  );

  const tabs = [
    {
      id: 'select',
      label: 'Select Model',
      icon: '🎯',
      content: modelSelectionTab,
    },
    {
      id: 'config',
      label: 'Training Config',
      icon: '⚙️',
      content: trainingConfigTab,
    },
  ];

  const footer = (
    <>
      {activeTab === 'select' ? (
        <>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-transparent border border-neon-blue border-opacity-30 text-white rounded hover:bg-neon-blue hover:bg-opacity-10 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleNextToConfig}
            disabled={!selectedArchitecture}
            className={classNames(
              'px-4 py-2 rounded transition-colors',
              selectedArchitecture
                ? 'bg-neon-blue text-white hover:bg-opacity-80'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            )}
          >
            Next: Configure Training →
          </button>
        </>
      ) : (
        <>
          <button
            onClick={handleBackToSelection}
            className="px-4 py-2 bg-transparent border border-neon-blue border-opacity-30 text-white rounded hover:bg-neon-blue hover:bg-opacity-10 transition-colors"
          >
            ← Back to Selection
          </button>
          <button
            onClick={handleStartTraining}
            disabled={(trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio) !== 1.0}
            className={classNames(
              'px-4 py-2 rounded transition-colors',
              (trainingConfig.train_ratio + trainingConfig.val_ratio + trainingConfig.test_ratio) === 1.0
                ? 'bg-sea-green text-white hover:bg-opacity-80'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            )}
          >
            Start Training 🚀
          </button>
        </>
      )}
    </>
  );

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create Training Job" size="xl" footer={footer}>
      <Tabs
        tabs={tabs}
        defaultTabId="select"
        onChange={(tabId) => {
          // Prevent navigating to config tab if no model is selected
          if (tabId === 'config' && !selectedArchitecture) {
            return;
          }
          setActiveTab(tabId as 'select' | 'config');
        }}
      />
    </Modal>
  );
};

export default ModelSelectionModal;
