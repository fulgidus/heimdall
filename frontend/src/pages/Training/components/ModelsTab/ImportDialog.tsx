/**
 * ImportDialog Component
 * 
 * Modal for importing .heimdall model bundles
 */

import React, { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useTrainingStore } from '../../../../store/trainingStore';
import { usePortal } from '@/hooks/usePortal';

interface ImportDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ImportDialog: React.FC<ImportDialogProps> = ({ isOpen, onClose }) => {
  const { importModel, isLoading } = useTrainingStore();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(isOpen);

  const handleFileSelect = (file: File) => {
    setError(null);

    // Validate file extension
    if (!file.name.endsWith('.heimdall')) {
      setError('Invalid file type. Please select a .heimdall file');
      return;
    }

    // Validate file size (max 500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Maximum size is 500MB');
      return;
    }

    setSelectedFile(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleImport = async () => {
    if (!selectedFile) return;

    setError(null);
    try {
      await importModel(selectedFile);
      onClose();
      // Reset state
      setSelectedFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
    }
  };

  const handleClose = () => {
    if (isLoading) return;
    setSelectedFile(null);
    setError(null);
    onClose();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  if (!isOpen || !portalTarget) return null;

  return createPortal(
    <>
      {/* Modal Backdrop */}
      <div className="modal-backdrop fade show" onClick={handleClose}></div>

      {/* Modal */}
      <div className="modal fade show" style={{ display: 'block' }} tabIndex={-1}>
        <div className="modal-dialog modal-dialog-scrollable">
          <div className="modal-content">
            {/* Header */}
            <div className="modal-header">
              <h5 className="modal-title">
                <i className="ph ph-upload me-2"></i>
                Import Model
              </h5>
              <button 
                type="button" 
                className="btn-close" 
                onClick={handleClose}
                disabled={isLoading}
              ></button>
            </div>

            {/* Body */}
            <div className="modal-body">
              {/* Error Message */}
              {error && (
                <div className="alert alert-danger">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                </div>
              )}

              {/* File Drop Zone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  border border-dashed rounded p-4 text-center mb-3
                  ${isDragOver ? 'border-primary bg-light text-dark' : 'border-secondary'}
                  ${isLoading ? 'opacity-50' : 'cursor-pointer'}
                `}
                style={{ minHeight: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                {selectedFile ? (
                  <div className="text-center">
                    <div className="mb-3">
                      <i className="ph ph-check-circle text-success" style={{ fontSize: '3rem' }}></i>
                    </div>
                    <div>
                      <p className="mb-1 fw-medium">{selectedFile.name}</p>
                      <p className="text-muted small">{formatFileSize(selectedFile.size)}</p>
                    </div>
                    {!isLoading && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFile(null);
                          setError(null);
                        }}
                        className="btn btn-link btn-sm mt-2"
                      >
                        Choose different file
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="mb-3">
                      <i className="ph ph-cloud-arrow-up text-muted" style={{ fontSize: '3rem' }}></i>
                    </div>
                    <div>
                      <p className="mb-1 fw-medium">
                        Drop .heimdall file here or click to browse
                      </p>
                      <p className="text-muted small">Maximum file size: 500MB</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Hidden File Input */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".heimdall"
                onChange={handleFileInputChange}
                className="d-none"
                disabled={isLoading}
              />

              {/* Import Progress */}
              {isLoading && (
                <div className="mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <span className="small">Importing model...</span>
                    <span className="text-muted small">Please wait</span>
                  </div>
                  <div className="progress" style={{ height: '8px' }}>
                    <div 
                      className="progress-bar progress-bar-striped progress-bar-animated" 
                      style={{ width: '100%' }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Info Box */}
              <div className="alert alert-info mb-0">
                <i className="ph ph-info me-2"></i>
                <strong>Import will include:</strong>
                <ul className="mb-0 mt-2 ps-3">
                  <li>ONNX model weights</li>
                  <li>Training configuration (if included)</li>
                  <li>Metrics history (if included)</li>
                  <li>Normalization parameters (if included)</li>
                </ul>
              </div>
            </div>

            {/* Footer */}
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={handleClose}
                disabled={isLoading}
              >
                <i className="ph ph-x me-2"></i>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleImport}
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2"></span>
                    Importing...
                  </>
                ) : (
                  <>
                    <i className="ph ph-upload me-2"></i>
                    Import
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>,
    portalTarget
  );
};
