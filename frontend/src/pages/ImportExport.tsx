import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  exportData,
  importData,
  getExportMetadata,
  parseHeimdallFile,
  serializeHeimdallFile,
  validateHeimdallFile,
  formatFileSize,
} from '../services/api/import-export';
import type {
  ExportRequest,
  ImportRequest,
  HeimdallFile,
  ExportMetadataResponse,
} from '../services/api/import-export';

const ImportExport: React.FC = () => {
  const [metadata, setMetadata] = useState<ExportMetadataResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [exportStatus, setExportStatus] = useState<string>('');
  const [importStatus, setImportStatus] = useState<string>('');
  const [importedFile, setImportedFile] = useState<HeimdallFile | null>(null);

  // Export options
  const [exportOptions, setExportOptions] = useState({
    include_settings: true,
    include_sources: true,
    include_websdrs: true,
    include_sessions: false,
    include_training_model: false,
    include_inference_model: false,
  });

  // Import options
  const [importOptions, setImportOptions] = useState({
    import_settings: true,
    import_sources: true,
    import_websdrs: true,
    import_sessions: false,
    import_training_model: false,
    import_inference_model: false,
    overwrite_existing: false,
  });

  // User info for export
  const [creatorInfo, setCreatorInfo] = useState({
    username: 'user',
    name: 'Heimdall User',
  });

  const [exportDescription, setExportDescription] = useState('');
  const isTauriApp = '__TAURI__' in window;

  useEffect(() => {
    loadMetadata();
    
    // Listen for file open event from Tauri (when app opened with .heimdall file)
    if (isTauriApp) {
      const setupFileOpenListener = async () => {
        try {
          const { listen } = await import('@tauri-apps/api/event');
          const unlisten = await listen<string>('open-heimdall-file', async (event) => {
            const filePath = event.payload;
            console.log('Received open-heimdall-file event:', filePath);
            setImportStatus(`Loading file: ${filePath}...`);
            
            try {
              const result = await invoke<{ success: boolean; content?: string; message: string }>(
                'load_heimdall_file_from_path',
                { path: filePath }
              );
              
              if (result.success && result.content) {
                const heimdallFile = parseHeimdallFile(result.content);
                if (validateHeimdallFile(heimdallFile)) {
                  setImportedFile(heimdallFile);
                  setImportStatus(`✅ File loaded from: ${filePath}\nReview and confirm import below.`);
                } else {
                  setImportStatus(`❌ Invalid .heimdall file format: ${filePath}`);
                }
              } else {
                setImportStatus(`❌ Failed to load file: ${result.message}`);
              }
            } catch (error) {
              console.error('Failed to load file from path:', error);
              setImportStatus(`❌ Failed to load file: ${error}`);
            }
          });
          
          return unlisten;
        } catch (error) {
          console.error('Failed to setup file open listener:', error);
        }
      };
      
      setupFileOpenListener();
    }
  }, [isTauriApp]);

  const loadMetadata = async () => {
    try {
      const data = await getExportMetadata();
      setMetadata(data);
    } catch (error) {
      console.error('Failed to load export metadata:', error);
    }
  };

  const handleExport = async () => {
    setLoading(true);
    setExportStatus('Preparing export...');

    try {
      const request: ExportRequest = {
        creator: creatorInfo,
        description: exportDescription || undefined,
        ...exportOptions,
      };

      const heimdallFile = await exportData(request);
      const content = serializeHeimdallFile(heimdallFile);

      if (isTauriApp) {
        // Use Tauri file dialog
        const result = await invoke<{ success: boolean; message: string; path?: string }>(
          'save_heimdall_file',
          {
            content,
            defaultFilename: `heimdall-export-${new Date().toISOString().split('T')[0]}.heimdall`,
          }
        );

        if (result.success) {
          setExportStatus(`✅ Export saved successfully to: ${result.path}`);
        } else {
          setExportStatus(`❌ ${result.message}`);
        }
      } else {
        // Web browser: trigger download
        const blob = new Blob([content], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `heimdall-export-${new Date().toISOString().split('T')[0]}.heimdall`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setExportStatus('✅ Export downloaded successfully');
      }

      await loadMetadata();
    } catch (error) {
      console.error('Export failed:', error);
      setExportStatus(`❌ Export failed: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleImportFromDialog = async () => {
    if (!isTauriApp) {
      setImportStatus('❌ File dialog is only available in desktop app');
      return;
    }

    setLoading(true);
    setImportStatus('Opening file dialog...');

    try {
      const result = await invoke<{ success: boolean; content?: string; message: string }>(
        'load_heimdall_file'
      );

      if (result.success && result.content) {
        const heimdallFile = parseHeimdallFile(result.content);
        if (validateHeimdallFile(heimdallFile)) {
          setImportedFile(heimdallFile);
          setImportStatus('✅ File loaded successfully. Review and confirm import below.');
        } else {
          setImportStatus('❌ Invalid .heimdall file format');
        }
      } else {
        setImportStatus(result.message);
      }
    } catch (error) {
      console.error('File load failed:', error);
      setImportStatus(`❌ Failed to load file: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleImportFromWeb = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setImportStatus('Reading file...');

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const heimdallFile = parseHeimdallFile(content);
        if (validateHeimdallFile(heimdallFile)) {
          setImportedFile(heimdallFile);
          setImportStatus('✅ File loaded successfully. Review and confirm import below.');
        } else {
          setImportStatus('❌ Invalid .heimdall file format');
        }
      } catch (error) {
        console.error('Failed to parse file:', error);
        setImportStatus(`❌ Failed to parse file: ${error}`);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const handleConfirmImport = async () => {
    if (!importedFile) return;

    setLoading(true);
    setImportStatus('Importing data...');

    try {
      const request: ImportRequest = {
        file_content: importedFile,
        ...importOptions,
      };

      const result = await importData(request);

      if (result.success) {
        let statusMsg = `✅ Import completed successfully!\n\n`;
        statusMsg += `Imported:\n`;
        Object.entries(result.imported_counts).forEach(([key, count]) => {
          statusMsg += `  - ${key}: ${count}\n`;
        });
        if (result.warnings.length > 0) {
          statusMsg += `\nWarnings:\n`;
          result.warnings.forEach((w) => (statusMsg += `  - ${w}\n`));
        }
        setImportStatus(statusMsg);
        setImportedFile(null);
        await loadMetadata();
      } else {
        let statusMsg = `❌ Import failed: ${result.message}\n\n`;
        if (result.errors.length > 0) {
          statusMsg += `Errors:\n`;
          result.errors.forEach((e) => (statusMsg += `  - ${e}\n`));
        }
        setImportStatus(statusMsg);
      }
    } catch (error) {
      console.error('Import failed:', error);
      setImportStatus(`❌ Import failed: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container-fluid py-4">
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">
                <i className="fas fa-exchange-alt me-2"></i>
                Import / Export
              </h3>
              <p className="text-muted mb-0">
                Save and restore your Heimdall configuration and data using .heimdall files
              </p>
            </div>
            <div className="card-body">
              {/* Metadata Overview */}
              {metadata && (
                <div className="alert alert-info mb-4">
                  <h5>Available Data</h5>
                  <div className="row">
                    <div className="col-md-3">
                      <strong>Sources:</strong> {metadata.available_sources_count}
                    </div>
                    <div className="col-md-3">
                      <strong>WebSDRs:</strong> {metadata.available_websdrs_count}
                    </div>
                    <div className="col-md-3">
                      <strong>Sessions:</strong> {metadata.available_sessions_count}
                    </div>
                    <div className="col-md-3">
                      <strong>Est. Size:</strong> {formatFileSize(metadata.estimated_size_bytes)}
                    </div>
                  </div>
                </div>
              )}

              {/* Export Section */}
              <div className="row mb-5">
                <div className="col-12">
                  <h4 className="mb-3">
                    <i className="fas fa-upload me-2"></i>
                    Export Data
                  </h4>

                  <div className="row mb-3">
                    <div className="col-md-6">
                      <label className="form-label">Username</label>
                      <input
                        type="text"
                        className="form-control"
                        value={creatorInfo.username}
                        onChange={(e) =>
                          setCreatorInfo({ ...creatorInfo, username: e.target.value })
                        }
                      />
                    </div>
                    <div className="col-md-6">
                      <label className="form-label">Full Name</label>
                      <input
                        type="text"
                        className="form-control"
                        value={creatorInfo.name}
                        onChange={(e) => setCreatorInfo({ ...creatorInfo, name: e.target.value })}
                      />
                    </div>
                  </div>

                  <div className="mb-3">
                    <label className="form-label">Description (optional)</label>
                    <textarea
                      className="form-control"
                      rows={2}
                      value={exportDescription}
                      onChange={(e) => setExportDescription(e.target.value)}
                      placeholder="Describe what this export contains..."
                    />
                  </div>

                  <div className="mb-3">
                    <label className="form-label d-block">Select sections to export:</label>
                    <div className="row">
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_settings}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_settings: e.target.checked,
                              })
                            }
                            id="export-settings"
                          />
                          <label className="form-check-label" htmlFor="export-settings">
                            User Settings
                          </label>
                        </div>
                      </div>
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_sources}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_sources: e.target.checked,
                              })
                            }
                            id="export-sources"
                          />
                          <label className="form-check-label" htmlFor="export-sources">
                            Known Sources ({metadata?.available_sources_count || 0})
                          </label>
                        </div>
                      </div>
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_websdrs}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_websdrs: e.target.checked,
                              })
                            }
                            id="export-websdrs"
                          />
                          <label className="form-check-label" htmlFor="export-websdrs">
                            WebSDR Stations ({metadata?.available_websdrs_count || 0})
                          </label>
                        </div>
                      </div>
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_sessions}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_sessions: e.target.checked,
                              })
                            }
                            id="export-sessions"
                          />
                          <label className="form-check-label" htmlFor="export-sessions">
                            Recording Sessions ({metadata?.available_sessions_count || 0})
                          </label>
                        </div>
                      </div>
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_training_model}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_training_model: e.target.checked,
                              })
                            }
                            id="export-training"
                            disabled={!metadata?.has_training_model}
                          />
                          <label className="form-check-label" htmlFor="export-training">
                            Training Model
                            {!metadata?.has_training_model && ' (not available)'}
                          </label>
                        </div>
                      </div>
                      <div className="col-md-4">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={exportOptions.include_inference_model}
                            onChange={(e) =>
                              setExportOptions({
                                ...exportOptions,
                                include_inference_model: e.target.checked,
                              })
                            }
                            id="export-inference"
                            disabled={!metadata?.has_inference_model}
                          />
                          <label className="form-check-label" htmlFor="export-inference">
                            Inference Model
                            {!metadata?.has_inference_model && ' (not available)'}
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>

                  <button
                    className="btn btn-primary"
                    onClick={handleExport}
                    disabled={loading}
                  >
                    <i className="fas fa-download me-2"></i>
                    Export Data
                  </button>

                  {exportStatus && (
                    <div className="alert alert-info mt-3" style={{ whiteSpace: 'pre-wrap' }}>
                      {exportStatus}
                    </div>
                  )}
                </div>
              </div>

              <hr className="my-4" />

              {/* Import Section */}
              <div className="row">
                <div className="col-12">
                  <h4 className="mb-3">
                    <i className="fas fa-download me-2"></i>
                    Import Data
                  </h4>

                  <div className="mb-3">
                    {isTauriApp ? (
                      <button
                        className="btn btn-secondary"
                        onClick={handleImportFromDialog}
                        disabled={loading}
                      >
                        <i className="fas fa-folder-open me-2"></i>
                        Open .heimdall File
                      </button>
                    ) : (
                      <div>
                        <label htmlFor="file-input" className="btn btn-secondary">
                          <i className="fas fa-folder-open me-2"></i>
                          Choose .heimdall File
                        </label>
                        <input
                          id="file-input"
                          type="file"
                          accept=".heimdall"
                          onChange={handleImportFromWeb}
                          style={{ display: 'none' }}
                        />
                      </div>
                    )}
                  </div>

                  {importedFile && (
                    <>
                      <div className="alert alert-success mb-3">
                        <h5>File Loaded</h5>
                        <p className="mb-2">
                          <strong>Version:</strong> {importedFile.metadata.version}
                          <br />
                          <strong>Created:</strong>{' '}
                          {new Date(importedFile.metadata.created_at).toLocaleString()}
                          <br />
                          <strong>Creator:</strong> {importedFile.metadata.creator.name} (@
                          {importedFile.metadata.creator.username})
                          <br />
                          {importedFile.metadata.description && (
                            <>
                              <strong>Description:</strong> {importedFile.metadata.description}
                            </>
                          )}
                        </p>
                        <div className="row">
                          <div className="col-md-3">
                            <strong>Sources:</strong> {importedFile.sections.sources?.length || 0}
                          </div>
                          <div className="col-md-3">
                            <strong>WebSDRs:</strong> {importedFile.sections.websdrs?.length || 0}
                          </div>
                          <div className="col-md-3">
                            <strong>Sessions:</strong> {importedFile.sections.sessions?.length || 0}
                          </div>
                          <div className="col-md-3">
                            <strong>Settings:</strong>{' '}
                            {importedFile.sections.settings ? 'Yes' : 'No'}
                          </div>
                        </div>
                      </div>

                      <div className="mb-3">
                        <label className="form-label d-block">Select sections to import:</label>
                        <div className="row">
                          {importedFile.sections.settings && (
                            <div className="col-md-4">
                              <div className="form-check">
                                <input
                                  className="form-check-input"
                                  type="checkbox"
                                  checked={importOptions.import_settings}
                                  onChange={(e) =>
                                    setImportOptions({
                                      ...importOptions,
                                      import_settings: e.target.checked,
                                    })
                                  }
                                  id="import-settings"
                                />
                                <label className="form-check-label" htmlFor="import-settings">
                                  User Settings
                                </label>
                              </div>
                            </div>
                          )}
                          {importedFile.sections.sources && (
                            <div className="col-md-4">
                              <div className="form-check">
                                <input
                                  className="form-check-input"
                                  type="checkbox"
                                  checked={importOptions.import_sources}
                                  onChange={(e) =>
                                    setImportOptions({
                                      ...importOptions,
                                      import_sources: e.target.checked,
                                    })
                                  }
                                  id="import-sources"
                                />
                                <label className="form-check-label" htmlFor="import-sources">
                                  Sources ({importedFile.sections.sources.length})
                                </label>
                              </div>
                            </div>
                          )}
                          {importedFile.sections.websdrs && (
                            <div className="col-md-4">
                              <div className="form-check">
                                <input
                                  className="form-check-input"
                                  type="checkbox"
                                  checked={importOptions.import_websdrs}
                                  onChange={(e) =>
                                    setImportOptions({
                                      ...importOptions,
                                      import_websdrs: e.target.checked,
                                    })
                                  }
                                  id="import-websdrs"
                                />
                                <label className="form-check-label" htmlFor="import-websdrs">
                                  WebSDRs ({importedFile.sections.websdrs.length})
                                </label>
                              </div>
                            </div>
                          )}
                          {importedFile.sections.sessions && (
                            <div className="col-md-4">
                              <div className="form-check">
                                <input
                                  className="form-check-input"
                                  type="checkbox"
                                  checked={importOptions.import_sessions}
                                  onChange={(e) =>
                                    setImportOptions({
                                      ...importOptions,
                                      import_sessions: e.target.checked,
                                    })
                                  }
                                  id="import-sessions"
                                />
                                <label className="form-check-label" htmlFor="import-sessions">
                                  Sessions ({importedFile.sections.sessions.length})
                                </label>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="mb-3">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={importOptions.overwrite_existing}
                            onChange={(e) =>
                              setImportOptions({
                                ...importOptions,
                                overwrite_existing: e.target.checked,
                              })
                            }
                            id="overwrite-existing"
                          />
                          <label className="form-check-label" htmlFor="overwrite-existing">
                            Overwrite existing items with same ID/name
                          </label>
                        </div>
                      </div>

                      <button
                        className="btn btn-success"
                        onClick={handleConfirmImport}
                        disabled={loading}
                      >
                        <i className="fas fa-check me-2"></i>
                        Confirm Import
                      </button>
                      <button
                        className="btn btn-secondary ms-2"
                        onClick={() => {
                          setImportedFile(null);
                          setImportStatus('');
                        }}
                        disabled={loading}
                      >
                        Cancel
                      </button>
                    </>
                  )}

                  {importStatus && (
                    <div className="alert alert-info mt-3" style={{ whiteSpace: 'pre-wrap' }}>
                      {importStatus}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImportExport;
