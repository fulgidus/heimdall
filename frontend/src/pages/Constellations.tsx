/**
 * Constellations Page
 * 
 * Manage constellation groups (collections of WebSDR receivers).
 * Provides CRUD operations with permission-based access control.
 */

import React, { useState, useEffect } from 'react';
import { ConstellationCard } from '../components/constellations/ConstellationCard';
import { ConstellationForm } from '../components/constellations/ConstellationForm';
import Modal from '../components/Modal';
import { ShareModal } from '../components/sharing';
import { useAuth } from '../hooks/useAuth';
import { usePermissions } from '../hooks/usePermissions';
import {
  getConstellations,
  createConstellation,
  updateConstellation,
  deleteConstellation,
  getConstellationShares,
  createConstellationShare,
  updateConstellationShare,
  deleteConstellationShare,
  type Constellation,
  type ConstellationShare,
  type CreateConstellationRequest,
  type UpdateConstellationRequest,
} from '../services/api/constellations';

type FilterType = 'all' | 'owned' | 'shared';

const Constellations: React.FC = () => {
  const { user } = useAuth();
  const { canCreateConstellation } = usePermissions();

  // State management
  const [constellations, setConstellations] = useState<Constellation[]>([]);
  const [filteredConstellations, setFilteredConstellations] = useState<Constellation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterType>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Modal state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [selectedConstellation, setSelectedConstellation] = useState<Constellation | null>(null);
  const [constellationShares, setConstellationShares] = useState<ConstellationShare[]>([]);
  const [isLoadingShares, setIsLoadingShares] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Load constellations
  const loadConstellations = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getConstellations();
      setConstellations(data);
    } catch (err) {
      console.error('Failed to load constellations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load constellations');
    } finally {
      setIsLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    loadConstellations();
  }, []);

  // Apply filters and search
  useEffect(() => {
    let filtered = [...constellations];

    // Apply ownership filter
    if (filter === 'owned') {
      filtered = filtered.filter(c => c.owner_id === user?.id);
    } else if (filter === 'shared') {
      filtered = filtered.filter(c => c.owner_id !== user?.id);
    }

    // Apply search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        c =>
          c.name.toLowerCase().includes(query) ||
          c.description?.toLowerCase().includes(query) ||
          c.id.toLowerCase().includes(query)
      );
    }

    setFilteredConstellations(filtered);
  }, [constellations, filter, searchQuery, user]);

  // Handle create constellation
  const handleCreate = async (data: CreateConstellationRequest & { websdr_ids?: string[] }) => {
    setIsSubmitting(true);
    try {
      const newConstellation = await createConstellation({
        name: data.name,
        description: data.description,
      });

      // TODO: Add WebSDR members if specified
      // For now, we'll need to call addConstellationMember for each websdr_id
      // This will be handled in the ShareModal/details page later

      await loadConstellations();
      setIsCreateModalOpen(false);
    } catch (err) {
      console.error('Failed to create constellation:', err);
      throw err; // Let form handle the error
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle edit constellation
  const handleEdit = async (data: UpdateConstellationRequest) => {
    if (!selectedConstellation) return;

    setIsSubmitting(true);
    try {
      await updateConstellation(selectedConstellation.id, data);
      await loadConstellations();
      setIsEditModalOpen(false);
      setSelectedConstellation(null);
    } catch (err) {
      console.error('Failed to update constellation:', err);
      throw err;
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle delete constellation
  const handleDelete = async () => {
    if (!selectedConstellation) return;

    setIsSubmitting(true);
    try {
      await deleteConstellation(selectedConstellation.id);
      await loadConstellations();
      setIsDeleteModalOpen(false);
      setSelectedConstellation(null);
    } catch (err) {
      console.error('Failed to delete constellation:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete constellation');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Open modals
  const openEditModal = (constellation: Constellation) => {
    setSelectedConstellation(constellation);
    setIsEditModalOpen(true);
  };

  const openDeleteModal = (constellation: Constellation) => {
    setSelectedConstellation(constellation);
    setIsDeleteModalOpen(true);
  };

  const openShareModal = async (constellation: Constellation) => {
    setSelectedConstellation(constellation);
    setIsLoadingShares(true);
    setIsShareModalOpen(true);
    
    try {
      const shares = await getConstellationShares(constellation.id);
      setConstellationShares(shares);
    } catch (err) {
      console.error('Failed to load constellation shares:', err);
      setError(err instanceof Error ? err.message : 'Failed to load shares');
      setConstellationShares([]);
    } finally {
      setIsLoadingShares(false);
    }
  };

  const handleAddShare = async (userId: string, permission: 'read' | 'edit') => {
    if (!selectedConstellation) return;

    try {
      const newShare = await createConstellationShare(selectedConstellation.id, {
        user_id: userId,
        permission,
      });
      setConstellationShares(prev => [...prev, newShare]);
    } catch (err) {
      console.error('Failed to create share:', err);
      throw err; // Let ShareModal handle the error
    }
  };

  const handleUpdatePermission = async (shareId: string, permission: 'read' | 'edit') => {
    if (!selectedConstellation) return;

    // Find the share to get the user_id
    const share = constellationShares.find(s => s.id === shareId);
    if (!share) {
      throw new Error('Share not found');
    }

    try {
      const updatedShare = await updateConstellationShare(
        selectedConstellation.id,
        share.user_id,
        { permission }
      );
      setConstellationShares(prev =>
        prev.map(s => (s.id === shareId ? updatedShare : s))
      );
    } catch (err) {
      console.error('Failed to update share permission:', err);
      throw err; // Let ShareModal handle the error
    }
  };

  const handleRemoveShare = async (shareId: string) => {
    if (!selectedConstellation) return;

    // Find the share to get the user_id
    const share = constellationShares.find(s => s.id === shareId);
    if (!share) {
      throw new Error('Share not found');
    }

    try {
      await deleteConstellationShare(selectedConstellation.id, share.user_id);
      setConstellationShares(prev => prev.filter(s => s.id !== shareId));
    } catch (err) {
      console.error('Failed to remove share:', err);
      throw err; // Let ShareModal handle the error
    }
  };

  const handleCloseShareModal = () => {
    setIsShareModalOpen(false);
    setSelectedConstellation(null);
    setConstellationShares([]);
    // Optionally refresh constellation list to update share counts
    loadConstellations();
  };

  const openDetailsModal = (constellation: Constellation) => {
    // TODO: Implement details view/modal
    console.log('Details view not yet implemented for:', constellation.id);
    alert('Details view coming soon!');
  };

  // Calculate counts for filter badges
  const ownedCount = constellations.filter(c => c.owner_id === user?.id).length;
  const sharedCount = constellations.filter(c => c.owner_id !== user?.id).length;

  return (
    <>
      {/* Breadcrumb */}
      <nav className="page-header" aria-label="Breadcrumb">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ol className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/">Home</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Constellations
                </li>
              </ol>
            </div>
            <div className="col-md-12">
              <div className="page-header-title d-flex align-items-center justify-content-between">
                <h1 className="mb-0">
                  <i className="ph ph-circles-three me-2"></i>
                  Constellations
                </h1>
                {canCreateConstellation() && (
                  <button
                    className="btn btn-primary"
                    onClick={() => setIsCreateModalOpen(true)}
                  >
                    <i className="ph ph-plus me-2"></i>
                    Create Constellation
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              {/* Filters and Search */}
              <div className="row mb-4">
                <div className="col-md-6">
                  {/* Filter Tabs */}
                  <ul className="nav nav-pills" role="tablist">
                    <li className="nav-item" role="presentation">
                      <button
                        className={`nav-link ${filter === 'all' ? 'active' : ''}`}
                        onClick={() => setFilter('all')}
                        type="button"
                      >
                        All <span className="badge bg-light text-dark ms-1">{constellations.length}</span>
                      </button>
                    </li>
                    <li className="nav-item" role="presentation">
                      <button
                        className={`nav-link ${filter === 'owned' ? 'active' : ''}`}
                        onClick={() => setFilter('owned')}
                        type="button"
                      >
                        My Constellations <span className="badge bg-light text-dark ms-1">{ownedCount}</span>
                      </button>
                    </li>
                    <li className="nav-item" role="presentation">
                      <button
                        className={`nav-link ${filter === 'shared' ? 'active' : ''}`}
                        onClick={() => setFilter('shared')}
                        type="button"
                      >
                        Shared With Me <span className="badge bg-light text-dark ms-1">{sharedCount}</span>
                      </button>
                    </li>
                  </ul>
                </div>
                <div className="col-md-6">
                  {/* Search */}
                  <div className="input-group">
                    <span className="input-group-text">
                      <i className="ph ph-magnifying-glass"></i>
                    </span>
                    <input
                      type="text"
                      className="form-control"
                      placeholder="Search constellations..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                    />
                    {searchQuery && (
                      <button
                        className="btn btn-outline-secondary"
                        type="button"
                        onClick={() => setSearchQuery('')}
                      >
                        <i className="ph ph-x"></i>
                      </button>
                    )}
                  </div>
                </div>
              </div>

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

              {/* Loading State */}
              {isLoading && (
                <div className="text-center py-5">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <p className="text-muted mt-3">Loading constellations...</p>
                </div>
              )}

              {/* Empty State */}
              {!isLoading && filteredConstellations.length === 0 && (
                <div className="text-center py-5">
                  <i className="ph ph-circles-three text-muted" style={{ fontSize: '4rem' }}></i>
                  <h4 className="mt-3 text-muted">
                    {searchQuery
                      ? 'No constellations match your search'
                      : filter === 'owned'
                        ? 'You haven\'t created any constellations yet'
                        : filter === 'shared'
                          ? 'No constellations have been shared with you'
                          : 'No constellations available'}
                  </h4>
                  {canCreateConstellation() && filter !== 'shared' && !searchQuery && (
                    <button
                      className="btn btn-primary mt-3"
                      onClick={() => setIsCreateModalOpen(true)}
                    >
                      <i className="ph ph-plus me-2"></i>
                      Create Your First Constellation
                    </button>
                  )}
                </div>
              )}

              {/* Constellation Grid */}
              {!isLoading && filteredConstellations.length > 0 && (
                <div className="row g-3">
                  {filteredConstellations.map(constellation => (
                    <div key={constellation.id} className="col-12 col-md-6 col-lg-4">
                      <ConstellationCard
                        constellation={constellation}
                        onEdit={openEditModal}
                        onDelete={openDeleteModal}
                        onShare={openShareModal}
                        onViewDetails={openDetailsModal}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Create Modal */}
      <Modal
        isOpen={isCreateModalOpen}
        onClose={() => !isSubmitting && setIsCreateModalOpen(false)}
        title="Create Constellation"
        size="lg"
      >
        <ConstellationForm
          mode="create"
          onSubmit={handleCreate}
          onCancel={() => setIsCreateModalOpen(false)}
          isSubmitting={isSubmitting}
        />
      </Modal>

      {/* Edit Modal */}
      <Modal
        isOpen={isEditModalOpen}
        onClose={() => !isSubmitting && setIsEditModalOpen(false)}
        title="Edit Constellation"
        size="md"
      >
        {selectedConstellation && (
          <ConstellationForm
            mode="edit"
            initialData={selectedConstellation}
            onSubmit={handleEdit}
            onCancel={() => setIsEditModalOpen(false)}
            isSubmitting={isSubmitting}
          />
        )}
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => !isSubmitting && setIsDeleteModalOpen(false)}
        title="Delete Constellation"
        size="md"
      >
        {selectedConstellation && (
          <div className="modal-body">
            <div className="alert alert-warning">
              <i className="ph ph-warning-circle me-2"></i>
              <strong>Warning:</strong> This action cannot be undone!
            </div>
            <p>
              Are you sure you want to delete the constellation{' '}
              <strong>"{selectedConstellation.name}"</strong>?
            </p>
            <p className="mb-0">
              All constellation members will be removed, but the WebSDR receivers themselves will not be affected.
            </p>
          </div>
        )}
        <div className="modal-footer">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={() => !isSubmitting && setIsDeleteModalOpen(false)}
            disabled={isSubmitting}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={handleDelete}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <span className="spinner-border spinner-border-sm me-2"></span>
                Deleting...
              </>
            ) : (
              <>
                <i className="ph ph-trash me-2"></i>
                Delete
              </>
            )}
          </button>
        </div>
      </Modal>

      {/* Share Modal */}
      {selectedConstellation && user && (
        <ShareModal
          isOpen={isShareModalOpen}
          onClose={handleCloseShareModal}
          resourceType="constellation"
          resourceId={selectedConstellation.id}
          resourceName={selectedConstellation.name}
          ownerId={selectedConstellation.owner_id}
          currentUserId={user.id}
          shares={constellationShares}
          onAddShare={handleAddShare}
          onUpdatePermission={handleUpdatePermission}
          onRemoveShare={handleRemoveShare}
          onRefresh={async () => {
            if (!selectedConstellation) return;
            setIsLoadingShares(true);
            try {
              const shares = await getConstellationShares(selectedConstellation.id);
              setConstellationShares(shares);
            } catch (err) {
              console.error('Failed to refresh shares:', err);
              setError(err instanceof Error ? err.message : 'Failed to refresh shares');
            } finally {
              setIsLoadingShares(false);
            }
          }}
        />
      )}
    </>
  );
};

export default Constellations;
