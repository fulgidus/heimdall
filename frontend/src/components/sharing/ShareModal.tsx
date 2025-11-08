/**
 * ShareModal Component
 * 
 * Modal for managing resource sharing with users.
 * Orchestrates UserSearch and ShareList components.
 * 
 * Features:
 * - Search and add new users
 * - View existing shares
 * - Edit permissions (read/edit)
 * - Remove shares
 * - Real-time updates
 * 
 * Generic component - can be used for constellations, sources, models, etc.
 */

import React, { useState, useEffect } from 'react';
import Modal from '../Modal';
import { UserSearch } from './UserSearch';
import { ShareList } from './ShareList';
import type { UserProfile } from '../../services/api/users';

interface ShareModalProps {
  isOpen: boolean;
  onClose: () => void;
  resourceType: 'constellation' | 'source' | 'model';
  resourceId: string;
  resourceName: string;
  ownerId: string;
  currentUserId: string;
  shares: Array<{
    id: string;
    user_id: string;
    permission: 'read' | 'edit';
    shared_by: string;
    shared_at: string;
  }>;
  onAddShare: (userId: string, permission: 'read' | 'edit') => Promise<void>;
  onUpdatePermission: (shareId: string, permission: 'read' | 'edit') => Promise<void>;
  onRemoveShare: (shareId: string) => Promise<void>;
  onRefresh?: () => Promise<void>;
}

export const ShareModal: React.FC<ShareModalProps> = ({
  isOpen,
  onClose,
  resourceType,
  resourceId,
  resourceName,
  ownerId,
  currentUserId,
  shares,
  onAddShare,
  onUpdatePermission,
  onRemoveShare,
  onRefresh,
}) => {
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);
  const [selectedPermission, setSelectedPermission] = useState<'read' | 'edit'>('read');
  const [isAdding, setIsAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setSelectedUser(null);
      setSelectedPermission('read');
      setIsAdding(false);
      setError(null);
      setSuccessMessage(null);
    }
  }, [isOpen]);

  const handleSelectUser = (user: UserProfile) => {
    setSelectedUser(user);
    setError(null);
  };

  const handleAddShare = async () => {
    if (!selectedUser) return;

    try {
      setIsAdding(true);
      setError(null);
      setSuccessMessage(null);

      await onAddShare(selectedUser.user_id, selectedPermission);

      setSuccessMessage(
        `Successfully shared with ${selectedUser.email || selectedUser.username || 'user'}`
      );
      setSelectedUser(null);
      setSelectedPermission('read');

      // Refresh shares list
      if (onRefresh) {
        await onRefresh();
      }

      // Auto-hide success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      console.error('Failed to add share:', err);
      setError('Failed to share. Please try again.');
    } finally {
      setIsAdding(false);
    }
  };

  const handleUpdatePermission = async (shareId: string, permission: 'read' | 'edit') => {
    try {
      setError(null);
      setSuccessMessage(null);
      await onUpdatePermission(shareId, permission);
      setSuccessMessage('Permission updated successfully');

      // Refresh shares list
      if (onRefresh) {
        await onRefresh();
      }

      // Auto-hide success message
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      console.error('Failed to update permission:', err);
      setError('Failed to update permission. Please try again.');
    }
  };

  const handleRemoveShare = async (shareId: string) => {
    try {
      setError(null);
      setSuccessMessage(null);
      await onRemoveShare(shareId);
      setSuccessMessage('Access removed successfully');

      // Refresh shares list
      if (onRefresh) {
        await onRefresh();
      }

      // Auto-hide success message
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      console.error('Failed to remove share:', err);
      setError('Failed to remove access. Please try again.');
    }
  };

  const getResourceTypeLabel = (): string => {
    switch (resourceType) {
      case 'constellation':
        return 'Constellation';
      case 'source':
        return 'Source';
      case 'model':
        return 'Model';
      default:
        return 'Resource';
    }
  };

  const getExcludedUserIds = (): string[] => {
    // Exclude users who already have access
    return shares.map(share => share.user_id);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Share ${getResourceTypeLabel()}: ${resourceName}`}
      size="lg"
    >
      <div className="modal-body">
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

        {/* Add User Section */}
        <div className="card mb-4">
          <div className="card-header">
            <h5 className="mb-0">
              <i className="ph ph-user-plus me-2"></i>
              Add User
            </h5>
          </div>
          <div className="card-body">
            {/* User Search */}
            <div className="mb-3">
              <label className="form-label">Search for user</label>
              <UserSearch
                onSelectUser={handleSelectUser}
                excludeUserIds={getExcludedUserIds()}
                placeholder="Search by email or username..."
              />
            </div>

            {/* Selected User & Permission */}
            {selectedUser && (
              <div className="card bg-light">
                <div className="card-body">
                  <div className="row align-items-center">
                    <div className="col-md-6">
                      <label className="form-label mb-2">Selected User</label>
                      <div className="d-flex align-items-center">
                        <div className="avtar avtar-s bg-primary me-2">
                          <i className="ph ph-user"></i>
                        </div>
                        <div>
                          <div className="fw-medium">{selectedUser.email || selectedUser.username}</div>
                          {selectedUser.organization && (
                            <small className="text-muted">{selectedUser.organization}</small>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="col-md-4">
                      <label className="form-label">Permission Level</label>
                      <select
                        className="form-select"
                        value={selectedPermission}
                        onChange={(e) => setSelectedPermission(e.target.value as 'read' | 'edit')}
                        disabled={isAdding}
                      >
                        <option value="read">Read Only</option>
                        <option value="edit">Can Edit</option>
                      </select>
                    </div>
                    <div className="col-md-2 text-end">
                      <label className="form-label d-block">&nbsp;</label>
                      <button
                        className="btn btn-primary w-100"
                        onClick={handleAddShare}
                        disabled={isAdding}
                      >
                        {isAdding ? (
                          <>
                            <span className="spinner-border spinner-border-sm me-2"></span>
                            Adding...
                          </>
                        ) : (
                          <>
                            <i className="ph ph-plus me-2"></i>
                            Add
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Existing Shares Section */}
        <div className="card">
          <div className="card-header">
            <div className="d-flex justify-content-between align-items-center">
              <h5 className="mb-0">
                <i className="ph ph-users me-2"></i>
                Shared With ({shares.length})
              </h5>
              {onRefresh && (
                <button
                  className="btn btn-sm btn-outline-secondary"
                  onClick={onRefresh}
                  disabled={isAdding}
                >
                  <i className="ph ph-arrows-clockwise"></i>
                </button>
              )}
            </div>
          </div>
          <div className="card-body p-0">
            <ShareList
              shares={shares}
              ownerId={ownerId}
              currentUserId={currentUserId}
              onUpdatePermission={handleUpdatePermission}
              onRemoveShare={handleRemoveShare}
              isLoading={isAdding}
            />
          </div>
        </div>

        {/* Help Text */}
        <div className="alert alert-info mt-3 mb-0" role="alert">
          <i className="ph ph-info me-2"></i>
          <strong>Permission Levels:</strong>
          <ul className="mb-0 mt-2">
            <li><strong>Read Only:</strong> Can view the {resourceType} but cannot make changes</li>
            <li><strong>Can Edit:</strong> Can view and modify the {resourceType}</li>
          </ul>
        </div>
      </div>

      <div className="modal-footer">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onClose}
          disabled={isAdding}
        >
          Close
        </button>
      </div>
    </Modal>
  );
};

export default ShareModal;
