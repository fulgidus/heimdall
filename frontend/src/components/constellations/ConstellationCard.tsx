/**
 * ConstellationCard Component
 * 
 * Display card for constellation information with permission-based action buttons.
 * Shows constellation name, description, member count, ownership status, and permission level.
 */

import React from 'react';
import type { Constellation } from '../../services/api/constellations';
import { useAuth } from '../../hooks/useAuth';
import { usePermissions } from '../../hooks/usePermissions';

interface ConstellationCardProps {
  constellation: Constellation;
  onEdit?: (constellation: Constellation) => void;
  onDelete?: (constellation: Constellation) => void;
  onShare?: (constellation: Constellation) => void;
  onViewDetails?: (constellation: Constellation) => void;
}

export const ConstellationCard: React.FC<ConstellationCardProps> = ({
  constellation,
  onEdit,
  onDelete,
  onShare,
  onViewDetails,
}) => {
  const { user } = useAuth();
  const {
    canEditConstellation,
    canDeleteConstellation,
    canShareConstellation,
  } = usePermissions();

  // Determine ownership and permission level
  const isOwner = user?.id === constellation.owner_id;
  const userShare = constellation.shares?.find(s => s.user_id === user?.id);
  const permissionLevel = isOwner ? 'owner' : userShare?.permission || 'none';

  // Permission checks
  const canEdit = canEditConstellation(constellation);
  const canDelete = canDeleteConstellation(constellation);
  const canShare = canShareConstellation(constellation);

  // Badge color mapping
  const getBadgeColor = (permission: string) => {
    switch (permission) {
      case 'owner':
        return 'bg-primary';
      case 'edit':
        return 'bg-success';
      case 'read':
        return 'bg-info';
      default:
        return 'bg-secondary';
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="card h-100">
      <div className="card-body">
        {/* Header with badges */}
        <div className="d-flex justify-content-between align-items-start mb-3">
          <div className="flex-grow-1">
            <h5 className="mb-1 text-truncate" title={constellation.name}>
              {constellation.name}
            </h5>
            {constellation.description && (
              <p className="text-muted small mb-0 text-truncate" title={constellation.description}>
                {constellation.description}
              </p>
            )}
          </div>
          <div className="d-flex flex-column gap-1 ms-2">
            {/* Permission badge */}
            <span className={`badge ${getBadgeColor(permissionLevel)} text-uppercase`}>
              {permissionLevel}
            </span>
            {/* Member count badge */}
            <span className="badge bg-light text-dark">
              <i className="ph ph-broadcast me-1"></i>
              {constellation.member_count} WebSDR{constellation.member_count !== 1 ? 's' : ''}
            </span>
          </div>
        </div>

        {/* Metadata */}
        <div className="mb-3">
          <div className="d-flex justify-content-between mb-2">
            <span className="text-muted small">Created:</span>
            <span className="fw-medium small">{formatDate(constellation.created_at)}</span>
          </div>
          <div className="d-flex justify-content-between mb-2">
            <span className="text-muted small">Last Updated:</span>
            <span className="fw-medium small">{formatDate(constellation.updated_at)}</span>
          </div>
          <div className="d-flex justify-content-between">
            <span className="text-muted small">ID:</span>
            <code className="small">{constellation.id.slice(0, 8)}</code>
          </div>
        </div>

        {/* Share information (for shared constellations) */}
        {!isOwner && userShare && (
          <div className="mb-3 p-2 bg-light border rounded">
            <p className="small text-muted mb-1">
              <i className="ph ph-share-network me-1"></i>
              Shared by: <strong>{userShare.shared_by}</strong>
            </p>
            <p className="small text-muted mb-0">
              Shared on: {formatDate(userShare.shared_at)}
            </p>
          </div>
        )}

        {/* Additional shares count (for owners) */}
        {isOwner && constellation.shares && constellation.shares.length > 0 && (
          <div className="mb-3 p-2 bg-light border rounded">
            <p className="small text-muted mb-0">
              <i className="ph ph-users me-1"></i>
              Shared with <strong>{constellation.shares.length}</strong> user{constellation.shares.length !== 1 ? 's' : ''}
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="d-grid gap-2">
          {/* View Details (always available if callback provided) */}
          {onViewDetails && (
            <button
              onClick={() => onViewDetails(constellation)}
              className="btn btn-outline-primary btn-sm d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-info"></i>
              View Details
            </button>
          )}

          {/* Edit (requires edit permission) */}
          {canEdit && onEdit && (
            <button
              onClick={() => onEdit(constellation)}
              className="btn btn-outline-secondary btn-sm d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-pencil-simple"></i>
              Edit
            </button>
          )}

          {/* Share (owner only) */}
          {canShare && onShare && (
            <button
              onClick={() => onShare(constellation)}
              className="btn btn-outline-info btn-sm d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-share-network"></i>
              Manage Sharing
            </button>
          )}

          {/* Delete (owner only) */}
          {canDelete && onDelete && (
            <button
              onClick={() => onDelete(constellation)}
              className="btn btn-outline-danger btn-sm d-flex align-items-center justify-content-center gap-2"
            >
              <i className="ph ph-trash"></i>
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConstellationCard;
