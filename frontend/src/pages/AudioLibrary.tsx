import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  listAudioSamples,
  getAudioLibraryStats,
  uploadAudioSample,
  toggleAudioSampleEnabled,
  deleteAudioSample,
  downloadAudioSample,
  getCategoryWeights,
  updateCategoryWeights,
  formatFileSize,
  formatDuration,
  downloadFile,
  getProcessingStatusBadge,
  type AudioSample,
  type AudioLibraryStats,
  type AudioCategory,
  type CategoryWeights,
} from '../services/api/audioLibrary';

const CATEGORIES: AudioCategory[] = ['voice', 'music', 'documentary', 'conference', 'custom'];

const AudioLibrary: React.FC = () => {
  // State
  const [samples, setSamples] = useState<AudioSample[]>([]);
  const [stats, setStats] = useState<AudioLibraryStats | null>(null);
  const [weights, setWeights] = useState<CategoryWeights>({
    voice: 0.4,
    music: 0.3,
    documentary: 0.2,
    conference: 0.1,
    custom: 0.0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadCategory, setUploadCategory] = useState<AudioCategory>('voice');
  const [uploadDescription, setUploadDescription] = useState('');
  const [uploadTags, setUploadTags] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter state
  const [filterCategory, setFilterCategory] = useState<AudioCategory | ''>('');
  const [filterEnabled, setFilterEnabled] = useState<'all' | 'enabled' | 'disabled'>('all');

  // Polling state for monitoring preprocessing
  const [pollingIds, setPollingIds] = useState<Set<string>>(new Set());
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load data
  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params: any = {};
      if (filterCategory) params.category = filterCategory;
      if (filterEnabled !== 'all') params.enabled = filterEnabled === 'enabled';

      const [samplesData, statsData, weightsData] = await Promise.all([
        listAudioSamples(params),
        getAudioLibraryStats(),
        getCategoryWeights(),
      ]);

      setSamples(samplesData);
      setStats(statsData);
      setWeights(weightsData);

      // Auto-start polling for any files in PENDING or PROCESSING status
      const processingIds = samplesData
        .filter(s => s.processing_status === 'PENDING' || s.processing_status === 'PROCESSING')
        .map(s => s.id);
      
      if (processingIds.length > 0) {
        setPollingIds(new Set(processingIds));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load audio library');
      console.error('Failed to load audio library:', err);
    } finally {
      setIsLoading(false);
    }
  }, [filterCategory, filterEnabled]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Poll for preprocessing status updates
  useEffect(() => {
    if (pollingIds.size === 0) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    // Poll every 2 seconds
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const samplesData = await listAudioSamples({});
        setSamples(samplesData);

        // Check if any polled items are now ready or failed
        const updatedPollingIds = new Set(pollingIds);
        let hasStatusChange = false;

        samplesData.forEach(sample => {
          if (pollingIds.has(sample.id)) {
            if (sample.processing_status === 'READY') {
              updatedPollingIds.delete(sample.id);
              hasStatusChange = true;
              setSuccessMessage(`${sample.filename} is ready for training!`);
              setTimeout(() => setSuccessMessage(null), 3000);
            } else if (sample.processing_status === 'FAILED') {
              updatedPollingIds.delete(sample.id);
              hasStatusChange = true;
              setError(`${sample.filename} preprocessing failed`);
            }
          }
        });

        if (hasStatusChange) {
          setPollingIds(updatedPollingIds);
          // Reload stats when status changes
          const statsData = await getAudioLibraryStats();
          setStats(statsData);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [pollingIds]);

  // File selection handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('audio/')) {
        setError('Please select an audio file');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (!file.type.startsWith('audio/')) {
        setError('Please drop an audio file');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  // Upload handler
  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const tags = uploadTags
        .split(',')
        .map(tag => tag.trim())
        .filter(tag => tag.length > 0);

      const uploadedSample = await uploadAudioSample({
        file: selectedFile,
        category: uploadCategory,
        description: uploadDescription || undefined,
        tags: tags.length > 0 ? tags : undefined,
      });

      setSuccessMessage(`Successfully uploaded ${selectedFile.name}. Preprocessing chunks...`);
      
      // Reset form
      setSelectedFile(null);
      setUploadDescription('');
      setUploadTags('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      // Reload data
      await loadData();

      // Start polling if status is PENDING or PROCESSING
      if (uploadedSample.processing_status === 'PENDING' || uploadedSample.processing_status === 'PROCESSING') {
        setPollingIds(prev => new Set(prev).add(uploadedSample.id));
      }

      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  // Enable/disable handler
  const handleToggleEnabled = async (id: string, currentEnabled: boolean) => {
    try {
      await toggleAudioSampleEnabled(id, !currentEnabled);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update sample');
      console.error('Toggle enabled failed:', err);
    }
  };

  // Delete handler
  const handleDelete = async (id: string, filename: string) => {
    if (!window.confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    try {
      await deleteAudioSample(id);
      setSuccessMessage(`Deleted ${filename}`);
      await loadData();
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete sample');
      console.error('Delete failed:', err);
    }
  };

  // Download handler
  const handleDownload = async (id: string, filename: string) => {
    try {
      const blob = await downloadAudioSample(id);
      downloadFile(blob, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download file');
      console.error('Download failed:', err);
    }
  };

  // Category weights handlers
  const handleWeightChange = (category: AudioCategory, value: number) => {
    setWeights(prev => ({
      ...prev,
      [category]: value,
    }));
  };

  const handleSaveWeights = async () => {
    try {
      const normalized = await updateCategoryWeights(weights);
      setWeights(normalized);
      setSuccessMessage('Category weights updated successfully');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update weights');
      console.error('Update weights failed:', err);
    }
  };

  const handleResetWeights = () => {
    setWeights({
      voice: 0.4,
      music: 0.3,
      documentary: 0.2,
      conference: 0.1,
      custom: 0.0,
    });
  };

  // Calculate normalized percentages for display
  const getNormalizedPercentages = () => {
    const total = weights.voice + weights.music + weights.documentary + weights.conference + weights.custom;
    if (total === 0) {
      return {
        voice: 20,
        music: 20,
        documentary: 20,
        conference: 20,
        custom: 20,
      };
    }
    return {
      voice: (weights.voice / total) * 100,
      music: (weights.music / total) * 100,
      documentary: (weights.documentary / total) * 100,
      conference: (weights.conference / total) * 100,
      custom: (weights.custom / total) * 100,
    };
  };

  return (
    <>
      {/* Breadcrumb */}
      <div className="page-header">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ul className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/dashboard">Home</a>
                </li>
                <li className="breadcrumb-item">
                  <a href="#">Training</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Audio Library
                </li>
              </ul>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h2 className="mb-0">Audio Library</h2>
                <p className="text-muted mt-2">
                  Manage audio samples for ML training dataset generation
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        {/* Statistics Cards */}
        {stats && (
          <>
            <div className="col-md-3 col-sm-6">
              <div className="card">
                <div className="card-body">
                  <div className="d-flex align-items-center">
                    <div className="flex-shrink-0">
                      <div className="avtar avtar-s bg-light-primary">
                        <i className="ph ph-music-notes-simple text-primary"></i>
                      </div>
                    </div>
                    <div className="flex-grow-1 ms-3">
                      <h6 className="mb-0">Total Files</h6>
                      <p className="text-muted mb-0">{stats.total_files}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-3 col-sm-6">
              <div className="card">
                <div className="card-body">
                  <div className="d-flex align-items-center">
                    <div className="flex-shrink-0">
                      <div className="avtar avtar-s bg-light-success">
                        <i className="ph ph-check-circle text-success"></i>
                      </div>
                    </div>
                    <div className="flex-grow-1 ms-3">
                      <h6 className="mb-0">Ready</h6>
                      <p className="text-muted mb-0">
                        {samples.filter(s => s.processing_status === 'READY').length} / {stats.total_files}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-3 col-sm-6">
              <div className="card">
                <div className="card-body">
                  <div className="d-flex align-items-center">
                    <div className="flex-shrink-0">
                      <div className="avtar avtar-s bg-light-info">
                        <i className="ph ph-clock text-info"></i>
                      </div>
                    </div>
                    <div className="flex-grow-1 ms-3">
                      <h6 className="mb-0">Total Duration</h6>
                      <p className="text-muted mb-0">{formatDuration(stats.total_duration_seconds)}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-3 col-sm-6">
              <div className="card">
                <div className="card-body">
                  <div className="d-flex align-items-center">
                    <div className="flex-shrink-0">
                      <div className="avtar avtar-s bg-light-warning">
                        <i className="ph ph-hard-drives text-warning"></i>
                      </div>
                    </div>
                    <div className="flex-grow-1 ms-3">
                      <h6 className="mb-0">Total Size</h6>
                      <p className="text-muted mb-0">{formatFileSize(stats.total_size_bytes)}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Processing Status Alert */}
        {pollingIds.size > 0 && (
          <div className="col-12">
            <div className="alert alert-info alert-dismissible fade show" role="alert">
              <div className="d-flex align-items-center justify-content-between mb-2">
                <div>
                  <i className="ph ph-hourglass me-2"></i>
                  <strong>Processing {pollingIds.size} file{pollingIds.size > 1 ? 's' : ''}...</strong>
                  {' '}Audio chunks are being generated. This usually takes a few seconds.
                </div>
                <button
                  type="button"
                  className="btn-close"
                  onClick={() => setPollingIds(new Set())}
                  aria-label="Close"
                ></button>
              </div>
              <div className="progress" style={{ height: '6px' }}>
                <div 
                  className="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                  role="progressbar" 
                  style={{ width: '100%' }}
                  aria-label="Processing files"
                />
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Upload Audio Sample</h5>
            </div>
            <div className="card-body">
              {/* Error Message */}
              {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                  <button
                    type="button"
                    className="btn-close"
                    onClick={() => setError(null)}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              {/* Success Message */}
              {successMessage && (
                <div className="alert alert-success alert-dismissible fade show" role="alert">
                  <i className="ph ph-check-circle me-2"></i>
                  {successMessage}
                  <button
                    type="button"
                    className="btn-close"
                    onClick={() => setSuccessMessage(null)}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              <div className="row">
                {/* Drag and Drop Area */}
                <div className="col-md-12 mb-3">
                  <div
                    className="border border-2 border-dashed rounded p-4 text-center"
                    style={{ minHeight: '150px', cursor: 'pointer' }}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/*"
                      onChange={handleFileSelect}
                      style={{ display: 'none' }}
                    />
                    <i className="ph ph-upload-simple" style={{ fontSize: '3rem' }}></i>
                    <p className="mt-2 mb-0">
                      {selectedFile ? (
                        <>
                          <strong>{selectedFile.name}</strong>
                          <br />
                          <small className="text-muted">{formatFileSize(selectedFile.size)}</small>
                        </>
                      ) : (
                        <>
                          Drag and drop an audio file here, or click to select
                          <br />
                          <small className="text-muted">Supported formats: WAV, MP3, FLAC, OGG</small>
                        </>
                      )}
                    </p>
                  </div>
                </div>

                {/* Upload Form */}
                {selectedFile && (
                  <>
                    <div className="col-md-6 mb-3">
                      <label className="form-label">Category *</label>
                      <select
                        className="form-select"
                        value={uploadCategory}
                        onChange={e => setUploadCategory(e.target.value as AudioCategory)}
                      >
                        {CATEGORIES.map(cat => (
                          <option key={cat} value={cat}>
                            {cat.charAt(0).toUpperCase() + cat.slice(1)}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label className="form-label">Tags (comma-separated)</label>
                      <input
                        type="text"
                        className="form-control"
                        placeholder="italian, male, 40s"
                        value={uploadTags}
                        onChange={e => setUploadTags(e.target.value)}
                      />
                    </div>

                    <div className="col-md-12 mb-3">
                      <label className="form-label">Description</label>
                      <textarea
                        className="form-control"
                        rows={2}
                        placeholder="Optional description of the audio sample..."
                        value={uploadDescription}
                        onChange={e => setUploadDescription(e.target.value)}
                      />
                    </div>

                    <div className="col-md-12">
                      {/* Upload Progress Bar */}
                      {isUploading && (
                        <div className="mb-3">
                          <div className="d-flex align-items-center justify-content-between mb-2">
                            <small className="text-muted">
                              <i className="ph ph-upload-simple me-1"></i>
                              Uploading and processing audio file...
                            </small>
                            <small className="text-muted">
                              <i className="ph ph-spinner me-1"></i>
                              This may take a few seconds
                            </small>
                          </div>
                          <div className="progress" style={{ height: '8px' }}>
                            <div 
                              className="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                              role="progressbar" 
                              style={{ width: '100%' }}
                              aria-label="Uploading audio file"
                            />
                          </div>
                        </div>
                      )}
                      
                      <button
                        className="btn btn-primary me-2"
                        onClick={handleUpload}
                        disabled={isUploading}
                      >
                        {isUploading ? (
                          <>
                            <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                            Uploading...
                          </>
                        ) : (
                          <>
                            <i className="ph ph-upload me-2"></i>
                            Upload
                          </>
                        )}
                      </button>
                      <button
                        className="btn btn-outline-secondary"
                        onClick={() => {
                          setSelectedFile(null);
                          if (fileInputRef.current) {
                            fileInputRef.current.value = '';
                          }
                        }}
                        disabled={isUploading}
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Category Weights */}
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Category Weights</h5>
              <p className="text-muted mb-0 mt-1">
                Control sampling probability for each category during training
              </p>
            </div>
            <div className="card-body">
              <div className="row">
                {CATEGORIES.map(category => {
                  const percentages = getNormalizedPercentages();
                  const percentage = percentages[category];
                  return (
                    <div key={category} className="col-md-12 mb-3">
                      <div className="d-flex align-items-center justify-content-between mb-2">
                        <label className="form-label mb-0">
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </label>
                        <div className="d-flex align-items-center gap-3">
                          <span className="badge bg-light-primary">
                            Raw: {weights[category].toFixed(2)}
                          </span>
                          <span className="badge bg-light-success">
                            {percentage.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <input
                        type="range"
                        className="form-range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={weights[category]}
                        onChange={(e) => handleWeightChange(category, parseFloat(e.target.value))}
                      />
                    </div>
                  );
                })}
                <div className="col-md-12">
                  <div className="alert alert-info mb-3">
                    <i className="ph ph-info me-2"></i>
                    <strong>How it works:</strong> Set raw weights (0.0-1.0) for each category. 
                    The system automatically normalizes them into percentages that sum to 100%.
                    Higher weights mean more samples from that category will be used in training.
                  </div>
                  <button
                    className="btn btn-primary me-2"
                    onClick={handleSaveWeights}
                  >
                    <i className="ph ph-floppy-disk me-2"></i>
                    Save Weights
                  </button>
                  <button
                    className="btn btn-outline-secondary"
                    onClick={handleResetWeights}
                  >
                    <i className="ph ph-arrow-counter-clockwise me-2"></i>
                    Reset to Defaults
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* File List */}
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <div className="d-flex align-items-center justify-content-between">
                <h5 className="mb-0">Audio Samples</h5>
                <button
                  className="btn btn-sm btn-outline-primary"
                  onClick={loadData}
                  disabled={isLoading}
                >
                  <i className="ph ph-arrows-clockwise me-1"></i>
                  Refresh
                </button>
              </div>
            </div>
            <div className="card-body">
              {/* Filters */}
              <div className="row mb-3">
                <div className="col-md-4">
                  <label className="form-label">Filter by Category</label>
                  <select
                    className="form-select form-select-sm"
                    value={filterCategory}
                    onChange={e => setFilterCategory(e.target.value as AudioCategory | '')}
                  >
                    <option value="">All Categories</option>
                    {CATEGORIES.map(cat => (
                      <option key={cat} value={cat}>
                        {cat.charAt(0).toUpperCase() + cat.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="col-md-4">
                  <label className="form-label">Filter by Status</label>
                  <select
                    className="form-select form-select-sm"
                    value={filterEnabled}
                    onChange={e => setFilterEnabled(e.target.value as 'all' | 'enabled' | 'disabled')}
                  >
                    <option value="all">All</option>
                    <option value="enabled">Enabled Only</option>
                    <option value="disabled">Disabled Only</option>
                  </select>
                </div>
              </div>

              {/* Loading State */}
              {isLoading && (
                <div className="text-center py-5">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              )}

              {/* Empty State */}
              {!isLoading && samples.length === 0 && (
                <div className="text-center py-5">
                  <i className="ph ph-file-audio" style={{ fontSize: '4rem', opacity: 0.3 }}></i>
                  <p className="text-muted mt-3">No audio samples found</p>
                  <p className="text-muted">Upload your first audio file to get started</p>
                </div>
              )}

              {/* Table */}
              {!isLoading && samples.length > 0 && (
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th>Filename</th>
                        <th>Category</th>
                        <th>Duration</th>
                        <th>Size</th>
                        <th>Tags</th>
                        <th>Processing</th>
                        <th>Enabled</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samples.map(sample => (
                        <tr key={sample.id}>
                          <td>
                            <div className="d-flex align-items-center">
                              <i className="ph ph-file-audio me-2"></i>
                              <div>
                                <div>{sample.filename}</div>
                                {sample.description && (
                                  <small className="text-muted">{sample.description}</small>
                                )}
                              </div>
                            </div>
                          </td>
                          <td>
                            <span className="badge bg-light-primary">
                              {sample.category.charAt(0).toUpperCase() + sample.category.slice(1)}
                            </span>
                          </td>
                          <td>{formatDuration(sample.duration_seconds)}</td>
                          <td>{formatFileSize(sample.file_size_bytes)}</td>
                          <td>
                            {sample.tags.length > 0 ? (
                              <div className="d-flex flex-wrap gap-1">
                                {sample.tags.slice(0, 3).map(tag => (
                                  <span key={tag} className="badge bg-light-secondary">
                                    {tag}
                                  </span>
                                ))}
                                {sample.tags.length > 3 && (
                                  <span className="badge bg-light-secondary">
                                    +{sample.tags.length - 3}
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="text-muted">—</span>
                            )}
                          </td>
                          <td>
                            {(() => {
                              const statusBadge = getProcessingStatusBadge(sample.processing_status);
                              const isProcessing = sample.processing_status === 'PENDING' || sample.processing_status === 'PROCESSING';
                              
                              // Calculate progress percentage for PROCESSING state
                              // Expected chunks = floor(duration_seconds / 0.2) since each chunk is 200ms (0.2 seconds)
                              const expectedChunks = Math.floor(sample.duration_seconds / 0.2);
                              const currentChunks = sample.total_chunks || 0;
                              const progressPercent = expectedChunks > 0 
                                ? Math.min(100, Math.round((currentChunks / expectedChunks) * 100))
                                : 0;
                              
                              return (
                                <div style={{ minWidth: '180px' }}>
                                  <div className="d-flex align-items-center gap-2 mb-1">
                                    <span className={`badge ${statusBadge.colorClass}`}>
                                      <i className={`ph ${statusBadge.icon} me-1`}></i>
                                      {statusBadge.text}
                                    </span>
                                    {sample.processing_status === 'READY' && sample.total_chunks !== null && (
                                      <small className="text-muted">
                                        {sample.total_chunks} chunks
                                      </small>
                                    )}
                                    {/* Show progress info for PROCESSING state */}
                                    {sample.processing_status === 'PROCESSING' && currentChunks > 0 && (
                                      <small className="text-muted">
                                        {currentChunks}/{expectedChunks} ({progressPercent}%)
                                      </small>
                                    )}
                                  </div>
                                  
                                  {/* Progress bar for PENDING/PROCESSING states */}
                                  {isProcessing && (
                                    <div className="progress" style={{ height: '4px' }}>
                                      <div 
                                        className="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                                        role="progressbar" 
                                        style={{ width: sample.processing_status === 'PROCESSING' && progressPercent > 0 ? `${progressPercent}%` : '100%' }}
                                        aria-label={sample.processing_status === 'PROCESSING' ? `Processing: ${progressPercent}%` : "Processing audio file"}
                                      />
                                    </div>
                                  )}
                                  
                                  {/* Success progress bar for READY state */}
                                  {sample.processing_status === 'READY' && (
                                    <div className="progress" style={{ height: '4px' }}>
                                      <div 
                                        className="progress-bar bg-success" 
                                        role="progressbar" 
                                        style={{ width: '100%' }}
                                        aria-label="Processing complete"
                                      />
                                    </div>
                                  )}
                                  
                                  {/* Error progress bar for FAILED state */}
                                  {sample.processing_status === 'FAILED' && (
                                    <div className="progress" style={{ height: '4px' }}>
                                      <div 
                                        className="progress-bar bg-danger" 
                                        role="progressbar" 
                                        style={{ width: '100%' }}
                                        aria-label="Processing failed"
                                      />
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </td>
                          <td>
                            <div className="form-check form-switch">
                              <input
                                className="form-check-input"
                                type="checkbox"
                                checked={sample.enabled}
                                onChange={() => handleToggleEnabled(sample.id, sample.enabled)}
                                disabled={sample.processing_status !== 'READY'}
                              />
                            </div>
                          </td>
                          <td>
                            <div className="btn-group btn-group-sm">
                              <button
                                className="btn btn-outline-secondary"
                                onClick={() => handleDownload(sample.id, sample.filename)}
                                title="Download"
                              >
                                <i className="ph ph-download-simple"></i>
                              </button>
                              <button
                                className="btn btn-outline-danger"
                                onClick={() => handleDelete(sample.id, sample.filename)}
                                title="Delete"
                              >
                                <i className="ph ph-trash"></i>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AudioLibrary;
