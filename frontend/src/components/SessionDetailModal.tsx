/**
 * Session Detail Modal Component
 * Displays full session metadata, spectrograms, and approval controls
 */

import React, { useEffect, useState } from 'react';
import {
  X,
  CheckCircle,
  XCircle,
  Clock,
  MapPin,
  Radio,
  Calendar,
  AlertTriangle,
} from 'lucide-react';
import { Button } from './ui/button';
import { SpectrogramViewer } from './SpectrogramViewer';
import { useSessions } from '@/hooks/useSessions';
import type { RecordingSessionWithDetails } from '@/services/api/session';

interface SessionDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string | null;
}

// Mock WebSDR data - in Phase 8, this will come from the session metadata
const MOCK_WEBSDRS = [
  { id: 1, name: 'Torino Nord', snr: 25.3 },
  { id: 2, name: 'Genova Est', snr: 18.7 },
  { id: 3, name: 'Alessandria', snr: 22.1 },
  { id: 4, name: 'Cuneo', snr: 15.4 },
  { id: 5, name: 'Asti', snr: 19.8 },
  { id: 6, name: 'Savona', snr: 12.3 },
  { id: 7, name: 'Imperia', snr: 16.5 },
];

export const SessionDetailModal: React.FC<SessionDetailModalProps> = ({
  isOpen,
  onClose,
  sessionId,
}) => {
  const { currentSession, fetchSession, approveSession, rejectSession, isLoading } = useSessions();
  const [showRejectDialog, setShowRejectDialog] = useState(false);
  const [rejectComment, setRejectComment] = useState('');
  const [actionInProgress, setActionInProgress] = useState(false);

  useEffect(() => {
    if (isOpen && sessionId) {
      fetchSession(sessionId);
    }
  }, [isOpen, sessionId, fetchSession]);

  if (!isOpen) return null;

  const handleApprove = async () => {
    if (!sessionId) return;

    setActionInProgress(true);
    try {
      await approveSession(sessionId);
      onClose();
    } catch (error) {
      console.error('Failed to approve session:', error);
      alert('Failed to approve session. Please try again.');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleReject = async () => {
    if (!sessionId) return;

    setActionInProgress(true);
    try {
      // Note: Backend doesn't support comments yet, but we keep the UI ready
      await rejectSession(sessionId);
      setShowRejectDialog(false);
      setRejectComment('');
      onClose();
    } catch (error) {
      console.error('Failed to reject session:', error);
      alert('Failed to reject session. Please try again.');
    } finally {
      setActionInProgress(false);
    }
  };

  const session = currentSession as RecordingSessionWithDetails | null;

  const getApprovalBadge = (status?: string) => {
    switch (status) {
      case 'approved':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-green-900/20 text-green-400">
            <CheckCircle className="w-3 h-3" />
            APPROVED
          </span>
        );
      case 'rejected':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-red-900/20 text-red-400">
            <XCircle className="w-3 h-3" />
            REJECTED
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-yellow-900/20 text-yellow-400">
            <Clock className="w-3 h-3" />
            PENDING APPROVAL
          </span>
        );
    }
  };

  const formatDate = (dateString?: string | null) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const formatDuration = (seconds?: number | null) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
        <div
          className="bg-slate-900 rounded-lg shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden border border-slate-700"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-slate-700 flex items-center justify-between bg-slate-800">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-white">
                {session?.session_name || 'Loading...'}
              </h2>
              {session && getApprovalBadge(session.approval_status)}
            </div>
            <button
              onClick={onClose}
              className="p-1 hover:bg-slate-700 rounded transition-colors text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-4 overflow-y-auto max-h-[calc(90vh-140px)]">
            {isLoading && !session ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
                  <p className="text-slate-400">Loading session details...</p>
                </div>
              </div>
            ) : session ? (
              <div className="space-y-6">
                {/* Session Metadata */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Left Column */}
                  <div className="space-y-3">
                    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                      <h3 className="text-sm font-semibold text-slate-400 mb-3">
                        Session Information
                      </h3>

                      <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-400">Created:</span>
                          <span className="text-white">{formatDate(session.created_at)}</span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-400">Duration:</span>
                          <span className="text-white">
                            {formatDuration(session.duration_seconds)}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Radio className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-400">Frequency:</span>
                          <span className="text-white">
                            {session.source_frequency
                              ? `${(session.source_frequency / 1e6).toFixed(3)} MHz`
                              : 'N/A'}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <MapPin className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-400">Measurements:</span>
                          <span className="text-white">{session.measurements_count || 0}</span>
                        </div>
                      </div>
                    </div>

                    {/* Source Information */}
                    {session.source_name && (
                      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                        <h3 className="text-sm font-semibold text-slate-400 mb-3">Known Source</h3>

                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="text-slate-400">Name:</span>
                            <span className="text-white ml-2">{session.source_name}</span>
                          </div>

                          {session.source_latitude && session.source_longitude && (
                            <div>
                              <span className="text-slate-400">Location:</span>
                              <span className="text-white ml-2">
                                {session.source_latitude.toFixed(6)},{' '}
                                {session.source_longitude.toFixed(6)}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Right Column */}
                  <div className="space-y-3">
                    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                      <h3 className="text-sm font-semibold text-slate-400 mb-3">Status</h3>

                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="text-slate-400">Processing:</span>
                          <span className="text-white ml-2 capitalize">{session.status}</span>
                        </div>

                        <div>
                          <span className="text-slate-400">Approval:</span>
                          <span className="text-white ml-2 capitalize">
                            {session.approval_status}
                          </span>
                        </div>

                        {session.celery_task_id && (
                          <div>
                            <span className="text-slate-400">Task ID:</span>
                            <span className="text-white ml-2 font-mono text-xs">
                              {session.celery_task_id}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Notes */}
                    {session.notes && (
                      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                        <h3 className="text-sm font-semibold text-slate-400 mb-2">Notes</h3>
                        <p className="text-sm text-white">{session.notes}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Spectrograms Grid */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">
                    WebSDR Spectrograms ({MOCK_WEBSDRS.length} Receivers)
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {MOCK_WEBSDRS.map(websdr => (
                      <SpectrogramViewer
                        key={websdr.id}
                        sessionId={session.id}
                        websdrId={websdr.id}
                        websdrName={websdr.name}
                        snr={websdr.snr}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Session not found</p>
              </div>
            )}
          </div>

          {/* Footer - Approval Actions */}
          {session && session.approval_status === 'pending' && (
            <div className="px-6 py-4 border-t border-slate-700 bg-slate-800 flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => setShowRejectDialog(true)}
                disabled={actionInProgress}
                className="border-red-700 text-red-400 hover:bg-red-900/20"
              >
                <XCircle className="w-4 h-4 mr-2" />
                Reject
              </Button>

              <Button
                onClick={handleApprove}
                disabled={actionInProgress}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                {actionInProgress ? 'Approving...' : 'Approve for Training'}
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Reject Confirmation Dialog */}
      {showRejectDialog && (
        <div className="fixed inset-0 flex items-center justify-center z-[60] p-4">
          <div
            className="fixed inset-0 bg-black bg-opacity-70"
            onClick={() => setShowRejectDialog(false)}
          />

          <div className="bg-slate-800 rounded-lg shadow-2xl w-full max-w-md border border-slate-700 z-10">
            <div className="px-6 py-4 border-b border-slate-700">
              <h3 className="text-lg font-bold text-white">Reject Session</h3>
            </div>

            <div className="px-6 py-4">
              <p className="text-slate-300 mb-4">
                Are you sure you want to reject this session? It will not be used for training.
              </p>

              <label className="block text-sm text-slate-400 mb-2">Reason (optional):</label>
              <textarea
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white placeholder-slate-500 focus:outline-none focus:border-purple-500"
                rows={3}
                placeholder="Enter reason for rejection..."
                value={rejectComment}
                onChange={e => setRejectComment(e.target.value)}
              />
            </div>

            <div className="px-6 py-4 border-t border-slate-700 flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setShowRejectDialog(false);
                  setRejectComment('');
                }}
                disabled={actionInProgress}
                className="border-slate-700"
              >
                Cancel
              </Button>

              <Button
                onClick={handleReject}
                disabled={actionInProgress}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                {actionInProgress ? 'Rejecting...' : 'Confirm Rejection'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SessionDetailModal;
