/**
 * Delete Confirmation Modal Component
 * 
 * Bootstrap 5 modal for confirming WebSDR deletion
 */

import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { WebSDRConfig } from '@/services/api/types';

interface DeleteConfirmModalProps {
    show: boolean;
    onHide: () => void;
    onConfirm: (hardDelete: boolean) => Promise<void>;
    websdr: WebSDRConfig | null;
}

const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({ show, onHide, onConfirm, websdr }) => {
    const [hardDelete, setHardDelete] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [modalRoot] = useState(() => document.createElement('div'));

    // Mount and unmount the modal root element
    useEffect(() => {
        if (show) {
            document.body.appendChild(modalRoot);
            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';

            return () => {
                // Restore body scroll
                document.body.style.overflow = '';
                // Clean up: remove the modal root from DOM
                if (modalRoot.parentNode) {
                    modalRoot.parentNode.removeChild(modalRoot);
                }
            };
        }
    }, [show, modalRoot]);

    const handleConfirm = async () => {
        setIsDeleting(true);
        try {
            await onConfirm(hardDelete);
            onHide();
        } catch (error) {
            console.error('Delete error:', error);
        } finally {
            setIsDeleting(false);
        }
    };

    if (!show || !websdr) return null;

    return createPortal(
        <>
            {/* Modal Backdrop */}
            <div className="modal-backdrop fade show" onClick={onHide}></div>

            {/* Modal */}
            <div className="modal fade show" style={{ display: 'block' }} tabIndex={-1}>
                <div className="modal-dialog modal-dialog-centered">
                    <div className="modal-content">
                        <div className="modal-header bg-danger text-white">
                            <h5 className="modal-title">
                                <i className="ph ph-warning me-2"></i>
                                Confirm Deletion
                            </h5>
                            <button type="button" className="btn-close btn-close-white" onClick={onHide}></button>
                        </div>

                        <div className="modal-body">
                            <div className="alert alert-warning">
                                <i className="ph ph-warning-circle me-2"></i>
                                <strong>Warning:</strong> This action cannot be undone!
                            </div>

                            <p className="mb-3">
                                Are you sure you want to delete the WebSDR station:
                            </p>

                            <div className="card bg-light mb-3">
                                <div className="card-body">
                                    <h6 className="mb-1">
                                        <i className="ph ph-radio-button me-2"></i>
                                        {websdr.name}
                                    </h6>
                                    <small className="text-muted">
                                        <i className="ph ph-map-pin me-1"></i>
                                        {websdr.location_description || 'No location'}
                                    </small>
                                </div>
                            </div>

                            <div className="form-check form-switch mb-3">
                                <input
                                    className="form-check-input"
                                    type="checkbox"
                                    id="hardDelete"
                                    checked={hardDelete}
                                    onChange={(e) => setHardDelete(e.target.checked)}
                                    disabled={isDeleting}
                                />
                                <label className="form-check-label" htmlFor="hardDelete">
                                    <strong>Permanently delete</strong> (hard delete)
                                </label>
                            </div>

                            <div className="alert alert-info mb-0">
                                <i className="ph ph-info me-2"></i>
                                {hardDelete ? (
                                    <span>
                                        The station will be <strong>permanently removed</strong> from the database.
                                        This cannot be undone!
                                    </span>
                                ) : (
                                    <span>
                                        The station will be <strong>deactivated</strong> (soft delete).
                                        You can reactivate it later if needed.
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="modal-footer">
                            <button
                                type="button"
                                className="btn btn-outline-secondary"
                                onClick={onHide}
                                disabled={isDeleting}
                            >
                                <i className="ph ph-x me-2"></i>Cancel
                            </button>
                            <button
                                type="button"
                                className={`btn ${hardDelete ? 'btn-danger' : 'btn-warning'}`}
                                onClick={handleConfirm}
                                disabled={isDeleting}
                            >
                                {isDeleting ? (
                                    <>
                                        <span className="spinner-border spinner-border-sm me-2"></span>
                                        Deleting...
                                    </>
                                ) : (
                                    <>
                                        <i className="ph ph-trash me-2"></i>
                                        {hardDelete ? 'Permanently Delete' : 'Deactivate'}
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </>,
        modalRoot
    );
};

export default DeleteConfirmModal;
