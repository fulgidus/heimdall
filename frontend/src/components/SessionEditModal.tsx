import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import type { RecordingSessionWithDetails } from '@/services/api/session';
import { usePortal } from '@/hooks/usePortal';

interface SessionEditModalProps {
  session: RecordingSessionWithDetails;
  onSave: (
    sessionId: string,
    updates: {
      session_name?: string;
      notes?: string;
      approval_status?: 'pending' | 'approved' | 'rejected';
    }
  ) => Promise<void>;
  onClose: () => void;
}

const SessionEditModal: React.FC<SessionEditModalProps> = ({ session, onSave, onClose }) => {
  const [sessionName, setSessionName] = useState(session.session_name);
  const [notes, setNotes] = useState(session.notes || '');
  const [approvalStatus, setApprovalStatus] = useState(session.approval_status || 'pending');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(true);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSaving(true);

    try {
      const updates: {
        session_name?: string;
        notes?: string;
        approval_status?: 'pending' | 'approved' | 'rejected';
      } = {};

      if (sessionName !== session.session_name) {
        updates.session_name = sessionName;
      }
      if (notes !== (session.notes || '')) {
        updates.notes = notes;
      }
      if (approvalStatus !== session.approval_status) {
        updates.approval_status = approvalStatus as 'pending' | 'approved' | 'rejected';
      }

      if (Object.keys(updates).length === 0) {
        onClose();
        return;
      }

      await onSave(session.id, updates);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update session');
    } finally {
      setIsSaving(false);
    }
  };

  if (!portalTarget) return null;

  return createPortal(
    <>
      {/* Modal Backdrop */}
      <div className="modal-backdrop fade show" onClick={onClose} style={{ zIndex: 1040 }}></div>

      {/* Modal */}
      <div
        className="modal fade show"
        style={{ display: 'block', zIndex: 1050 }}
        tabIndex={-1}
        role="dialog"
      >
        <div className="modal-dialog modal-dialog-centered">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Edit Session</h5>
              <button
                type="button"
                className="btn-close"
                onClick={onClose}
                disabled={isSaving}
              ></button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="modal-body">
                {error && (
                  <div className="alert alert-danger" role="alert">
                    {error}
                  </div>
                )}

                {/* Session Name */}
                <div className="mb-3">
                  <label htmlFor="sessionName" className="form-label">
                    Session Name <span className="text-danger">*</span>
                  </label>
                  <input
                    type="text"
                    className="form-control"
                    id="sessionName"
                    value={sessionName}
                    onChange={e => setSessionName(e.target.value)}
                    required
                    maxLength={255}
                    disabled={isSaving}
                  />
                </div>

                {/* Notes */}
                <div className="mb-3">
                  <label htmlFor="notes" className="form-label">
                    Notes
                  </label>
                  <textarea
                    className="form-control"
                    id="notes"
                    rows={4}
                    value={notes}
                    onChange={e => setNotes(e.target.value)}
                    disabled={isSaving}
                    placeholder="Optional notes about this session..."
                  />
                </div>

                {/* Approval Status */}
                <div className="mb-3">
                  <label htmlFor="approvalStatus" className="form-label">
                    Approval Status
                  </label>
                  <select
                    className="form-select"
                    id="approvalStatus"
                    value={approvalStatus}
                    onChange={e =>
                      setApprovalStatus(e.target.value as 'pending' | 'approved' | 'rejected')
                    }
                    disabled={isSaving}
                  >
                    <option value="pending">Pending</option>
                    <option value="approved">Approved</option>
                    <option value="rejected">Rejected</option>
                  </select>
                </div>

                {/* Session Info (Read-only) */}
                <div className="border-top pt-3 mt-3">
                  <h6 className="text-muted mb-2">Session Information</h6>
                  <div className="row g-2">
                    <div className="col-6">
                      <small className="text-muted d-block">Source</small>
                      <span>{session.source_name}</span>
                    </div>
                    <div className="col-6">
                      <small className="text-muted d-block">Frequency</small>
                      <span>
                        {session.source_frequency
                          ? (session.source_frequency / 1e6).toFixed(3) + ' MHz'
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="col-6">
                      <small className="text-muted d-block">Status</small>
                      <span
                        className={`badge ${
                          session.status === 'completed'
                            ? 'bg-light-success'
                            : session.status === 'in_progress'
                              ? 'bg-light-primary'
                              : session.status === 'failed'
                                ? 'bg-light-danger'
                                : 'bg-light-warning'
                        }`}
                      >
                        {session.status}
                      </span>
                    </div>
                    <div className="col-6">
                      <small className="text-muted d-block">Measurements</small>
                      <span>{session.measurements_count || 0}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={onClose}
                  disabled={isSaving}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isSaving || !sessionName.trim()}
                >
                  {isSaving ? (
                    <>
                      <span
                        className="spinner-border spinner-border-sm me-2"
                        role="status"
                        aria-hidden="true"
                      ></span>
                      Saving...
                    </>
                  ) : (
                    'Save Changes'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>,
    portalTarget
  );
};

export default SessionEditModal;
