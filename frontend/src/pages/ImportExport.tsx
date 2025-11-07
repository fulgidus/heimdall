/**
 * Import/Export Page
 *
 * Allows users to export and import Heimdall system data
 * to/from .heimdall JSON files for backup, migration, and sharing.
 */

import { useState, useEffect } from 'react';
import { Download, Upload, Info, AlertCircle, CheckCircle } from 'lucide-react';
import {
  getExportMetadata,
  exportData,
  importData,
  downloadHeimdallFile,
  loadHeimdallFile,
  type MetadataResponse,
  type ExportRequest,
  type ImportRequest,
  type HeimdallFile,
  type ImportResponse,
  type SampleSetExportConfig,
} from '@/services/api/import-export';
import { useAuthStore } from '@/store/authStore';
import SampleSetRangeSelector from '@/components/SampleSetRangeSelector';

export default function ImportExport() {
  // Get authenticated user info
  const { user } = useAuthStore();

  // State for metadata
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [loadingMetadata, setLoadingMetadata] = useState(true);

  // State for export
  const [exportLoading, setExportLoading] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);

  // State for export form - username and name auto-populated from auth
  const [exportForm, setExportForm] = useState({
    username: user?.email || '',
    name: user?.name || '',
    description: '',
    include_settings: true,
    include_sources: true,
    include_websdrs: true,
    include_sessions: true,
  });

  // State for sample set ranges (dataset_id -> {enabled, offset, limit})
  const [sampleSetRanges, setSampleSetRanges] = useState<
    Map<string, { enabled: boolean; offset: number; limit: number | null }>
  >(new Map());
  
  // State for selected models and audio library
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedAudioLibrary, setSelectedAudioLibrary] = useState<Set<string>>(new Set());

  // State for import
  const [importLoading, setImportLoading] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importSuccess, setImportSuccess] = useState<ImportResponse | null>(null);
  const [importFile, setImportFile] = useState<HeimdallFile | null>(null);

  // State for import form
  const [importForm, setImportForm] = useState({
    import_settings: true,
    import_sources: true,
    import_websdrs: true,
    import_sessions: true,
    import_sample_sets: true,
    import_models: true,
    import_audio_library: true,
    overwrite_existing: false,
  });

  // Load metadata on mount
  useEffect(() => {
    loadMetadata();
  }, []);

  // Auto-select all sample sets (with full range), models, and audio library when metadata loads
  useEffect(() => {
    if (metadata) {
      // Initialize sample set ranges - all enabled with full range by default
      if (metadata.sample_sets && metadata.sample_sets.length > 0) {
        const newRanges = new Map<string, { enabled: boolean; offset: number; limit: number | null }>();
        metadata.sample_sets.forEach(s => {
          newRanges.set(s.id, { enabled: true, offset: 0, limit: null });
        });
        setSampleSetRanges(newRanges);
      }
      // Select all models that have ONNX by default
      if (metadata.models && metadata.models.length > 0) {
        setSelectedModels(new Set(metadata.models.filter(m => m.has_onnx).map(m => m.id)));
      }
      // Select all audio library items by default
      if (metadata.audio_library && metadata.audio_library.length > 0) {
        setSelectedAudioLibrary(new Set(metadata.audio_library.map(a => a.id)));
      }
    }
  }, [metadata]);

  // Auto-populate username and name from authenticated user
  useEffect(() => {
    if (user) {
      setExportForm(prev => ({
        ...prev,
        username: user.email || '',
        name: user.name || '',
      }));
    }
  }, [user]);

  const loadMetadata = async () => {
    try {
      setLoadingMetadata(true);
      const data = await getExportMetadata();
      setMetadata(data);
    } catch (error) {
      console.error('Error loading metadata:', error);
    } finally {
      setLoadingMetadata(false);
    }
  };

  const handleExport = async () => {
    if (!user) {
      setExportError('You must be logged in to export data');
      return;
    }

    try {
      setExportLoading(true);
      setExportError(null);
      setExportSuccess(false);

      // Build sample_set_configs from enabled ranges
      const sampleSetConfigs: SampleSetExportConfig[] = [];
      sampleSetRanges.forEach((range, datasetId) => {
        if (range.enabled) {
          sampleSetConfigs.push({
            dataset_id: datasetId,
            sample_offset: range.offset,
            sample_limit: range.limit,
          });
        }
      });

      const request: ExportRequest = {
        creator: {
          username: user.email,
          name: user.name || undefined,
        },
        description: exportForm.description || undefined,
        include_settings: exportForm.include_settings,
        include_sources: exportForm.include_sources,
        include_websdrs: exportForm.include_websdrs,
        include_sessions: exportForm.include_sessions,
        sample_set_configs: sampleSetConfigs.length > 0 ? sampleSetConfigs : null,
        model_ids: selectedModels.size > 0 ? Array.from(selectedModels) : null,
        audio_library_ids: selectedAudioLibrary.size > 0 ? Array.from(selectedAudioLibrary) : null,
      };

      const response = await exportData(request);

      // Download the file
      const timestamp = new Date().toISOString().split('T')[0];
      downloadHeimdallFile(response.file, `heimdall-export-${timestamp}.heimdall`);

      setExportSuccess(true);
      setTimeout(() => setExportSuccess(false), 3000);
    } catch (error) {
      console.error('Export error:', error);
      setExportError(error instanceof Error ? error.message : 'Export failed');
    } finally {
      setExportLoading(false);
    }
  };

  const handleLoadFile = async () => {
    try {
      const file = await loadHeimdallFile();
      setImportFile(file);
      setImportError(null);
      setImportSuccess(null); // Clear previous import success message
    } catch (error) {
      console.error('Error loading file:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load file';
      // Don't show error for cancellation
      if (!errorMessage.includes('cancelled')) {
        setImportError(errorMessage);
      }
    }
  };

  const handleImport = async () => {
    if (!importFile) {
      setImportError('Please load a .heimdall file first');
      return;
    }

    try {
      setImportLoading(true);
      setImportError(null);
      setImportSuccess(null);

      const request: ImportRequest = {
        heimdall_file: importFile,
        import_settings: importForm.import_settings,
        import_sources: importForm.import_sources,
        import_websdrs: importForm.import_websdrs,
        import_sessions: importForm.import_sessions,
        import_sample_sets: importForm.import_sample_sets,
        import_models: importForm.import_models,
        import_audio_library: importForm.import_audio_library,
        overwrite_existing: importForm.overwrite_existing,
      };

      const response = await importData(request);
      setImportSuccess(response);

      // Reload metadata to reflect changes
      await loadMetadata();
    } catch (error) {
      console.error('Import error:', error);
      setImportError(error instanceof Error ? error.message : 'Import failed');
    } finally {
      setImportLoading(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="container-fluid py-4">
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Import/Export Data</h5>
              <p className="text-muted mb-0">
                Save and restore your Heimdall SDR configuration and data
              </p>
            </div>
            <div className="card-body">
              {/* Metadata Overview */}
              <div className="alert alert-info d-flex align-items-start mb-4">
                <Info className="me-2 flex-shrink-0" size={20} />
                <div className="flex-grow-1">
                  <strong>Available Data</strong>
                  {loadingMetadata ? (
                    <p className="mb-0">Loading...</p>
                  ) : metadata ? (
                    <div className="mt-2">
                      <div className="row">
                        <div className="col-md-2">
                          <small className="d-block text-muted">Sources</small>
                          <strong>{metadata.sources_count}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">WebSDRs</small>
                          <strong>{metadata.websdrs_count}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">Sessions</small>
                          <strong>{metadata.sessions_count}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">Sample Sets</small>
                          <strong>{metadata.sample_sets.length}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">Models</small>
                          <strong>{metadata.models.length}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">Audio Files</small>
                          <strong>{metadata.audio_library.length}</strong>
                        </div>
                        <div className="col-md-2">
                          <small className="d-block text-muted">Est. Size</small>
                          <strong>
                            {formatBytes(
                              metadata.estimated_sizes.sources +
                                metadata.estimated_sizes.websdrs +
                                metadata.estimated_sizes.sessions +
                                metadata.estimated_sizes.settings +
                                metadata.estimated_sizes.sample_sets +
                                metadata.estimated_sizes.models +
                                metadata.estimated_sizes.audio_library
                            )}
                          </strong>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="mb-0">Failed to load metadata</p>
                  )}
                </div>
              </div>

              <div className="row">
                {/* Export Section */}
                <div className="col-md-6">
                  <h6 className="mb-3">
                    <Download className="me-2" size={18} />
                    Export Data
                  </h6>

                  {/* Creator Info - Auto-populated from authenticated user */}
                  <div className="mb-3">
                    <label className="form-label">Username (from your account)</label>
                    <input
                      type="text"
                      className="form-control"
                      value={exportForm.username}
                      readOnly
                      disabled
                      placeholder="Not logged in"
                    />
                    <small className="form-text text-muted">
                      Automatically set from your logged-in account
                    </small>
                  </div>

                  <div className="mb-3">
                    <label className="form-label">Full Name (from your account)</label>
                    <input
                      type="text"
                      className="form-control"
                      value={exportForm.name}
                      readOnly
                      disabled
                      placeholder="Not logged in"
                    />
                    <small className="form-text text-muted">
                      Automatically set from your logged-in account
                    </small>
                  </div>

                  <div className="mb-3">
                    <label className="form-label">Description</label>
                    <textarea
                      className="form-control"
                      rows={2}
                      value={exportForm.description}
                      onChange={e => setExportForm({ ...exportForm, description: e.target.value })}
                      placeholder="Describe this export..."
                    />
                  </div>

                  {/* Section Selection */}
                  <div className="mb-3">
                    <label className="form-label d-block">Include Sections</label>
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="export-settings"
                        checked={exportForm.include_settings}
                        onChange={e =>
                          setExportForm({ ...exportForm, include_settings: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="export-settings">
                        Settings
                      </label>
                    </div>
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="export-sources"
                        checked={exportForm.include_sources}
                        onChange={e =>
                          setExportForm({ ...exportForm, include_sources: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="export-sources">
                        Known Sources ({metadata?.sources_count || 0})
                      </label>
                    </div>
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="export-websdrs"
                        checked={exportForm.include_websdrs}
                        onChange={e =>
                          setExportForm({ ...exportForm, include_websdrs: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="export-websdrs">
                        WebSDRs ({metadata?.websdrs_count || 0})
                      </label>
                    </div>
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="export-sessions"
                        checked={exportForm.include_sessions}
                        onChange={e =>
                          setExportForm({ ...exportForm, include_sessions: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="export-sessions">
                        Recording Sessions ({metadata?.sessions_count || 0})
                      </label>
                    </div>

                    {/* Sample Sets Selection with Range Support */}
                    {metadata && metadata.sample_sets && metadata.sample_sets.length > 0 && (
                      <div className="mt-3">
                        <label className="form-label d-block">
                          <strong>Sample Sets ({metadata.sample_sets.length})</strong>
                        </label>
                        <div
                          style={{
                            maxHeight: '400px',
                            overflowY: 'auto',
                            border: '1px solid #dee2e6',
                            borderRadius: '4px',
                            padding: '8px',
                          }}
                        >
                          {metadata.sample_sets.map(sampleSet => {
                            const range = sampleSetRanges.get(sampleSet.id) || {
                              enabled: false,
                              offset: 0,
                              limit: null,
                            };
                            return (
                              <SampleSetRangeSelector
                                key={sampleSet.id}
                                sampleSet={sampleSet}
                                enabled={range.enabled}
                                range={{ offset: range.offset, limit: range.limit }}
                                onEnabledChange={enabled => {
                                  const newRanges = new Map(sampleSetRanges);
                                  newRanges.set(sampleSet.id, { ...range, enabled });
                                  setSampleSetRanges(newRanges);
                                }}
                                onRangeChange={(offset, limit) => {
                                  const newRanges = new Map(sampleSetRanges);
                                  newRanges.set(sampleSet.id, { enabled: range.enabled, offset, limit });
                                  setSampleSetRanges(newRanges);
                                }}
                              />
                            );
                          })}
                        </div>
                        <small className="form-text text-muted">
                          {Array.from(sampleSetRanges.values()).filter(r => r.enabled).length} of{' '}
                          {metadata.sample_sets.length} selected
                        </small>
                      </div>
                    )}

                    {/* Models Selection */}
                    {metadata && metadata.models && metadata.models.length > 0 && (
                      <div className="mt-3">
                        <label className="form-label d-block">
                          <strong>Models ({metadata.models.length})</strong>
                        </label>
                        <div
                          style={{
                            maxHeight: '200px',
                            overflowY: 'auto',
                            border: '1px solid #dee2e6',
                            borderRadius: '4px',
                            padding: '8px',
                          }}
                        >
                          {metadata.models.map(model => (
                            <div key={model.id} className="form-check">
                              <input
                                className="form-check-input"
                                type="checkbox"
                                id={`model-${model.id}`}
                                checked={selectedModels.has(model.id)}
                                onChange={e => {
                                  const newSet = new Set(selectedModels);
                                  if (e.target.checked) {
                                    newSet.add(model.id);
                                  } else {
                                    newSet.delete(model.id);
                                  }
                                  setSelectedModels(newSet);
                                }}
                                disabled={!model.has_onnx}
                              />
                              <label className="form-check-label" htmlFor={`model-${model.id}`}>
                                {model.model_name} v{model.version}
                                {!model.has_onnx && ' (No ONNX)'}
                              </label>
                            </div>
                          ))}
                        </div>
                        <small className="form-text text-muted">
                          {selectedModels.size} of {metadata.models.length} selected
                        </small>
                      </div>
                    )}

                    {/* Audio Library Selection */}
                    {metadata && metadata.audio_library && metadata.audio_library.length > 0 && (
                      <div className="mt-3">
                        <label className="form-label d-block">
                          <strong>Audio Library ({metadata.audio_library.length})</strong>
                        </label>
                        <div
                          style={{
                            maxHeight: '200px',
                            overflowY: 'auto',
                            border: '1px solid #dee2e6',
                            borderRadius: '4px',
                            padding: '8px',
                          }}
                        >
                          {metadata.audio_library.map(audio => (
                            <div key={audio.id} className="form-check">
                              <input
                                className="form-check-input"
                                type="checkbox"
                                id={`audio-${audio.id}`}
                                checked={selectedAudioLibrary.has(audio.id)}
                                onChange={e => {
                                  const newSet = new Set(selectedAudioLibrary);
                                  if (e.target.checked) {
                                    newSet.add(audio.id);
                                  } else {
                                    newSet.delete(audio.id);
                                  }
                                  setSelectedAudioLibrary(newSet);
                                }}
                              />
                              <label className="form-check-label" htmlFor={`audio-${audio.id}`}>
                                {audio.filename} ({audio.total_chunks} chunks,{' '}
                                {formatBytes(audio.file_size_bytes)})
                              </label>
                            </div>
                          ))}
                        </div>
                        <small className="form-text text-muted">
                          {selectedAudioLibrary.size} of {metadata.audio_library.length} selected
                        </small>
                      </div>
                    )}
                  </div>

                  {exportError && (
                    <div className="alert alert-danger d-flex align-items-center">
                      <AlertCircle className="me-2" size={18} />
                      {exportError}
                    </div>
                  )}

                  {exportSuccess && (
                    <div className="alert alert-success d-flex align-items-center">
                      <CheckCircle className="me-2" size={18} />
                      Export successful! File downloaded.
                    </div>
                  )}

                  <button
                    className="btn btn-primary w-100"
                    onClick={handleExport}
                    disabled={exportLoading || !user}
                  >
                    {exportLoading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" />
                        Exporting...
                      </>
                    ) : (
                      <>
                        <Download className="me-2" size={18} />
                        {!user ? 'Login Required to Export' : 'Export & Download'}
                      </>
                    )}
                  </button>
                </div>

                {/* Import Section */}
                <div className="col-md-6">
                  <h6 className="mb-3">
                    <Upload className="me-2" size={18} />
                    Import Data
                  </h6>

                  <div className="mb-3">
                    <button className="btn btn-outline-primary w-100" onClick={handleLoadFile}>
                      <Upload className="me-2" size={18} />
                      Load .heimdall File
                    </button>
                  </div>

                  {importFile && (
                    <>
                      <div className="alert alert-success mb-3">
                        <strong>File Loaded</strong>
                        <div className="mt-2">
                          <small className="d-block">Version: {importFile.metadata.version}</small>
                          <small className="d-block">
                            Created: {new Date(importFile.metadata.created_at).toLocaleString()}
                          </small>
                          <small className="d-block">
                            Creator: {importFile.metadata.creator.username}
                          </small>
                          {importFile.metadata.description && (
                            <small className="d-block mt-1">
                              {importFile.metadata.description}
                            </small>
                          )}
                        </div>
                      </div>

                      {/* Section Selection */}
                      <div className="mb-3">
                        <label className="form-label d-block">Import Sections</label>
                        {importFile.sections.settings && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-settings"
                              checked={importForm.import_settings}
                              onChange={e =>
                                setImportForm({ ...importForm, import_settings: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-settings">
                              Settings
                            </label>
                          </div>
                        )}
                        {importFile.sections.sources && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-sources"
                              checked={importForm.import_sources}
                              onChange={e =>
                                setImportForm({ ...importForm, import_sources: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-sources">
                              Sources ({importFile.sections.sources.length})
                            </label>
                          </div>
                        )}
                        {importFile.sections.websdrs && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-websdrs"
                              checked={importForm.import_websdrs}
                              onChange={e =>
                                setImportForm({ ...importForm, import_websdrs: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-websdrs">
                              WebSDRs ({importFile.sections.websdrs.length})
                            </label>
                          </div>
                        )}
                        {importFile.sections.sessions && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-sessions"
                              checked={importForm.import_sessions}
                              onChange={e =>
                                setImportForm({ ...importForm, import_sessions: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-sessions">
                              Sessions ({importFile.sections.sessions.length})
                            </label>
                          </div>
                        )}
                        {importFile.sections.sample_sets && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-sample-sets"
                              checked={importForm.import_sample_sets}
                              onChange={e =>
                                setImportForm({ ...importForm, import_sample_sets: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-sample-sets">
                              Sample Sets ({importFile.sections.sample_sets.length})
                            </label>
                          </div>
                        )}
                        {importFile.sections.models && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-models"
                              checked={importForm.import_models}
                              onChange={e =>
                                setImportForm({ ...importForm, import_models: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-models">
                              Models ({importFile.sections.models.length})
                            </label>
                          </div>
                        )}
                        {importFile.sections.audio_library && (
                          <div className="form-check">
                            <input
                              className="form-check-input"
                              type="checkbox"
                              id="import-audio-library"
                              checked={importForm.import_audio_library}
                              onChange={e =>
                                setImportForm({ ...importForm, import_audio_library: e.target.checked })
                              }
                            />
                            <label className="form-check-label" htmlFor="import-audio-library">
                              Audio Library ({importFile.sections.audio_library.length})
                            </label>
                          </div>
                        )}
                      </div>

                      <div className="mb-3">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            id="import-overwrite"
                            checked={importForm.overwrite_existing}
                            onChange={e =>
                              setImportForm({ ...importForm, overwrite_existing: e.target.checked })
                            }
                          />
                          <label className="form-check-label" htmlFor="import-overwrite">
                            Overwrite existing data
                          </label>
                          <small className="form-text text-muted d-block">
                            If unchecked, existing items will be skipped
                          </small>
                        </div>
                      </div>
                    </>
                  )}

                  {importError && (
                    <div className="alert alert-danger d-flex align-items-center">
                      <AlertCircle className="me-2" size={18} />
                      {importError}
                    </div>
                  )}

                  {importSuccess && (
                    <div className="alert alert-success">
                      <div className="d-flex align-items-center">
                        <CheckCircle className="me-2" size={18} />
                        <strong>{importSuccess.message}</strong>
                      </div>
                      <div className="mt-2">
                        <small className="d-block">
                          Sources: {importSuccess.imported_counts.sources || 0}
                        </small>
                        <small className="d-block">
                          WebSDRs: {importSuccess.imported_counts.websdrs || 0}
                        </small>
                        <small className="d-block">
                          Sessions: {importSuccess.imported_counts.sessions || 0}
                        </small>
                        <small className="d-block">
                          Sample Sets: {importSuccess.imported_counts.sample_sets || 0}
                        </small>
                        <small className="d-block">
                          Models: {importSuccess.imported_counts.models || 0}
                        </small>
                        <small className="d-block">
                          Audio Library: {importSuccess.imported_counts.audio_library || 0}
                        </small>
                      </div>
                      {importSuccess.errors.length > 0 && (
                        <div className="mt-2">
                          <small className="text-danger">Errors:</small>
                          {importSuccess.errors.map((error, idx) => (
                            <small key={idx} className="d-block text-danger">
                              {error}
                            </small>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <button
                    className="btn btn-success w-100"
                    onClick={handleImport}
                    disabled={!importFile || importLoading}
                  >
                    {importLoading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" />
                        Importing...
                      </>
                    ) : (
                      <>
                        <Upload className="me-2" size={18} />
                        Confirm Import
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
