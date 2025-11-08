/**
 * ShareList Component
 * 
 * Display and manage existing shares for a resource.
 * Features:
 * - List all users with access
 * - Display permission level (read/edit)
 * - Edit permission level (dropdown)
 * - Remove share (with confirmation)
 * - Show who shared and when
 * - Owner cannot be removed
 */

import React, { useState } from 'react';
import type { ConstellationShare } from '../../services/api/constellations';
import type { UserProfile } from '../../services/api/users';

interface ShareListProps {
  shares: ConstellationShare[];
  ownerId: string;
  currentUserId: string;
  onUpdatePermission: (shareId: string, permission: 'read' | 'edit') => Promise<void>;
  onRemoveShare: (shareId: string) => Promise<void>;
  isLoading?: boolean;
}

export const ShareList: React.FC<ShareListProps> = ({
  shares,
  ownerId,
  currentUserId,
  onUpdatePermission,
  onRemoveShare,
  isLoading = false,
}) => {
  const [editingShareId, setEditingShareId] = useState<string | null>(null);
  const [removingShareId, setRemovingShareId] = useState<string | null>(null);

  const handlePermissionChange = async (shareId: string, newPermission: 'read' | 'edit') => {
    try {
      setEditingShareId(shareId);
      await onUpdatePermission(shareId, newPermission);
    } finally {
      setEditingShareId(null);
    }
  };

  const handleRemoveShare = async (shareId: string) => {
    try {
      setRemovingShareId(shareId);
      await onRemoveShare(shareId);
    } finally {
      setRemovingShareId(null);
    }
  };

  const getPermissionBadgeClass = (permission: string) => {
    switch (permission) {
      case 'edit':
        return 'bg-success';
      case 'read':
        return 'bg-info';
      default:
        return 'bg-secondary';
    }
  };

  const getPermissionIcon = (permission: string) => {
    switch (permission) {
      case 'edit':
        return 'ph-pencil';
      case 'read':
        return 'ph-eye';
      default:
        return 'ph-question';
    }
  };

  const formatDate = (dateString: string): string => {
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

      if (diffDays === 0) return 'Today';
      if (diffDays === 1) return 'Yesterday';
      if (diffDays < 7) return `${diffDays} days ago`;
      if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
      if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
      return date.toLocaleDateString();
    } catch {
      return dateString;
    }
  };

  const canEditShare = (share: ConstellationShare): boolean => {
    // Only owner can edit shares
    return currentUserId === ownerId;
  };

  if (shares.length === 0) {
    return (
      <div className="text-center py-4">
        <i className="ph ph-share-network d-block mb-2 text-muted" style={{ fontSize: '3rem' }}></i>
        <p className="text-muted mb-0">Not shared with anyone yet</p>
        <small className="text-muted">Use the search above to share with users</small>
      </div>
    );
  }

  return (
    <div className="list-group">
      {shares.map(share => {
        const isEditing = editingShareId === share.id;
        const isRemoving = removingShareId === share.id;
        const isOwner = share.user_id === ownerId;
        const canEdit = canEditShare(share);

        return (
          <div
            key={share.id}
            className={`list-group-item ${isLoading ? 'opacity-50' : ''}`}
          >
            <div className="d-flex align-items-center">
              {/* User Avatar */}
              <div className="flex-shrink-0 me-3">
                <div className={`avtar avtar-s ${isOwner ? 'bg-primary' : 'bg-light-secondary'}`}>
                  <i className={`ph ${isOwner ? 'ph-crown' : 'ph-user'}`}></i>
                </div>
              </div>

              {/* User Info */}
              <div className="flex-grow-1">
                <div className="d-flex align-items-center mb-1">
                  <h6 className="mb-0 me-2">{share.user_id}</h6>
                  {isOwner && (
                    <span className="badge bg-primary me-2">Owner</span>
                  )}
                  <span className={`badge ${getPermissionBadgeClass(share.permission)}`}>
                    <i className={`ph ${getPermissionIcon(share.permission)} me-1`}></i>
                    {share.permission}
                  </span>
                </div>
                <p className="text-muted mb-0 small">
                  {isOwner ? (
                    'Created this constellation'
                  ) : (
                    <>
                      Shared by {share.shared_by} • {formatDate(share.shared_at)}
                    </>
                  )}
                </p>
              </div>

              {/* Actions */}
              {!isOwner && canEdit && (
                <div className="flex-shrink-0 ms-3">
                  <div className="btn-group">
                    {/* Permission Dropdown */}
                    <div className="dropdown">
                      <button
                        className="btn btn-sm btn-outline-secondary dropdown-toggle"
                        type="button"
                        data-bs-toggle="dropdown"
                        aria-expanded="false"
                        disabled={isEditing || isRemoving || isLoading}
                      >
                        {isEditing ? (
                          <span className="spinner-border spinner-border-sm me-1"></span>
                        ) : (
                          <i className="ph ph-gear me-1"></i>
                        )}
                        Change
                      </button>
                      <ul className="dropdown-menu">
                        <li>
                          <button
                            className="dropdown-item"
                            onClick={() => handlePermissionChange(share.id, 'read')}
                            disabled={share.permission === 'read'}
                          >
                            <i className="ph ph-eye me-2"></i>
                            Read Only
                          </button>
                        </li>
                        <li>
                          <button
                            className="dropdown-item"
                            onClick={() => handlePermissionChange(share.id, 'edit')}
                            disabled={share.permission === 'edit'}
                          >
                            <i className="ph ph-pencil me-2"></i>
                            Can Edit
                          </button>
                        </li>
                      </ul>
                    </div>

                    {/* Remove Button */}
                    <button
                      className="btn btn-sm btn-outline-danger"
                      onClick={() => handleRemoveShare(share.id)}
                      disabled={isEditing || isRemoving || isLoading}
                      title="Remove access"
                    >
                      {isRemoving ? (
                        <span className="spinner-border spinner-border-sm"></span>
                      ) : (
                        <i className="ph ph-trash"></i>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Non-owner, can't edit (view-only) */}
              {!isOwner && !canEdit && (
                <div className="flex-shrink-0 ms-3">
                  <span className="text-muted small">
                    <i className="ph ph-lock-simple"></i>
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ShareList;
