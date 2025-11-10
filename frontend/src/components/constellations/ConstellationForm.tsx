/**
 * ConstellationForm Component
 * 
 * Form for creating and editing constellations.
 * Allows user to set name, description, and select WebSDR members.
 */

import React, { useState, useEffect } from 'react';
import type { Constellation } from '../../services/api/constellations';
import { getWebSDRs } from '../../services/api/websdr';
import type { WebSDRConfig } from '../../services/api/schemas';

interface ConstellationFormProps {
  initialData?: Constellation;
  mode: 'create' | 'edit';
  onSubmit: (data: {
    name: string;
    description?: string;
    websdr_ids?: string[];
  }) => Promise<void>;
  onCancel: () => void;
  isSubmitting?: boolean;
}

export const ConstellationForm: React.FC<ConstellationFormProps> = ({
  initialData,
  mode,
  onSubmit,
  onCancel,
  isSubmitting = false,
}) => {
  // Form state
  const [name, setName] = useState(initialData?.name || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [selectedWebSDRs, setSelectedWebSDRs] = useState<string[]>(
    initialData?.members?.map(m => m.websdr_station_id) || []
  );

  // WebSDR list state
  const [websdrs, setWebsdrs] = useState<WebSDRConfig[]>([]);
  const [isLoadingWebSDRs, setIsLoadingWebSDRs] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Validation state
  const [errors, setErrors] = useState<{ name?: string; description?: string }>({});

  // Load WebSDRs on mount
  useEffect(() => {
    const loadWebSDRs = async () => {
      try {
        setIsLoadingWebSDRs(true);
        setLoadError(null);
        const data = await getWebSDRs();
        setWebsdrs(data);
      } catch (error) {
        console.error('Failed to load WebSDRs:', error);
        setLoadError(error instanceof Error ? error.message : 'Failed to load WebSDRs');
      } finally {
        setIsLoadingWebSDRs(false);
      }
    };

    loadWebSDRs();
  }, []);

  // Validate form
  const validate = (): boolean => {
    const newErrors: { name?: string; description?: string } = {};

    if (!name.trim()) {
      newErrors.name = 'Name is required';
    } else if (name.length > 255) {
      newErrors.name = 'Name must be 255 characters or less';
    }

    if (description && description.length > 1000) {
      newErrors.description = 'Description must be 1000 characters or less';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || undefined,
        websdr_ids: mode === 'create' ? selectedWebSDRs : undefined,
      });
    } catch (error) {
      console.error('Form submission error:', error);
    }
  };

  // Handle WebSDR selection toggle
  const toggleWebSDR = (websdrId: string) => {
    setSelectedWebSDRs(prev =>
      prev.includes(websdrId)
        ? prev.filter(id => id !== websdrId)
        : [...prev, websdrId]
    );
  };

  // Select all WebSDRs
  const selectAllWebSDRs = () => {
    setSelectedWebSDRs(websdrs.map(w => w.id));
  };

  // Deselect all WebSDRs
  const deselectAllWebSDRs = () => {
    setSelectedWebSDRs([]);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Name Field */}
      <div className="mb-3">
        <label htmlFor="constellation-name" className="form-label">
          Name <span className="text-danger">*</span>
        </label>
        <input
          type="text"
          className={`form-control ${errors.name ? 'is-invalid' : ''}`}
          id="constellation-name"
          value={name}
          onChange={e => {
            setName(e.target.value);
            setErrors(prev => ({ ...prev, name: undefined }));
          }}
          placeholder="Enter constellation name"
          maxLength={255}
          disabled={isSubmitting}
          required
        />
        {errors.name && <div className="invalid-feedback">{errors.name}</div>}
        <div className="form-text">
          {name.length}/255 characters
        </div>
      </div>

      {/* Description Field */}
      <div className="mb-3">
        <label htmlFor="constellation-description" className="form-label">
          Description
        </label>
        <textarea
          className={`form-control ${errors.description ? 'is-invalid' : ''}`}
          id="constellation-description"
          rows={3}
          value={description}
          onChange={e => {
            setDescription(e.target.value);
            setErrors(prev => ({ ...prev, description: undefined }));
          }}
          placeholder="Optional description of this constellation"
          maxLength={1000}
          disabled={isSubmitting}
        />
        {errors.description && <div className="invalid-feedback">{errors.description}</div>}
        <div className="form-text">
          {description.length}/1000 characters
        </div>
      </div>

      {/* WebSDR Member Selection (create mode only) */}
      {mode === 'create' && (
        <div className="mb-3">
          <label className="form-label">
            WebSDR Members
          </label>
          
          {isLoadingWebSDRs && (
            <div className="text-center py-3">
              <div className="spinner-border spinner-border-sm text-primary" role="status">
                <span className="visually-hidden">Loading WebSDRs...</span>
              </div>
              <p className="text-muted small mt-2">Loading WebSDRs...</p>
            </div>
          )}

          {loadError && (
            <div className="alert alert-warning" role="alert">
              <i className="ph ph-warning-circle me-2"></i>
              {loadError}
            </div>
          )}

          {!isLoadingWebSDRs && !loadError && websdrs.length === 0 && (
            <div className="alert alert-info" role="alert">
              <i className="ph ph-info me-2"></i>
              No WebSDRs available. Please configure WebSDRs first.
            </div>
          )}

          {!isLoadingWebSDRs && !loadError && websdrs.length > 0 && (
            <>
              <div className="d-flex gap-2 mb-2">
                <button
                  type="button"
                  className="btn btn-sm btn-outline-primary"
                  onClick={selectAllWebSDRs}
                  disabled={isSubmitting}
                >
                  Select All
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary"
                  onClick={deselectAllWebSDRs}
                  disabled={isSubmitting}
                >
                  Deselect All
                </button>
                <div className="ms-auto text-muted small align-self-center">
                  {selectedWebSDRs.length} of {websdrs.length} selected
                </div>
              </div>

              <div className="border rounded p-2" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {websdrs.map(websdr => (
                  <div key={websdr.id} className="form-check mb-2">
                    <input
                      type="checkbox"
                      className="form-check-input"
                      id={`websdr-${websdr.id}`}
                      checked={selectedWebSDRs.includes(websdr.id)}
                      onChange={() => toggleWebSDR(websdr.id)}
                      disabled={isSubmitting}
                    />
                    <label className="form-check-label d-flex justify-content-between align-items-center" htmlFor={`websdr-${websdr.id}`}>
                      <span>
                        <strong>{websdr.name}</strong>
                        {websdr.location_description && (
                          <span className="text-muted ms-2">
                            ({websdr.location_description})
                          </span>
                        )}
                      </span>
                      <span className="badge bg-light text-dark ms-2">
                        {websdr.latitude.toFixed(4)}, {websdr.longitude.toFixed(4)}
                      </span>
                    </label>
                  </div>
                ))}
              </div>

              <div className="form-text">
                Select WebSDR receivers to include in this constellation
              </div>
            </>
          )}
        </div>
      )}

      {/* Edit mode: Display current members */}
      {mode === 'edit' && initialData?.members && initialData.members.length > 0 && (
        <div className="mb-3">
          <label className="form-label">Current Members</label>
          <div className="alert alert-info">
            <i className="ph ph-info me-2"></i>
            This constellation has <strong>{initialData.member_count}</strong> WebSDR member(s).
            To add or remove members, use the constellation details page.
          </div>
        </div>
      )}

      {/* Form Actions */}
      <div className="d-flex justify-content-end gap-2">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={isSubmitting}
        >
          Cancel
        </button>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={isSubmitting || isLoadingWebSDRs}
        >
          {isSubmitting ? (
            <>
              <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              {mode === 'create' ? 'Creating...' : 'Saving...'}
            </>
          ) : (
            <>
              <i className={`ph ${mode === 'create' ? 'ph-plus' : 'ph-check'} me-2`}></i>
              {mode === 'create' ? 'Create Constellation' : 'Save Changes'}
            </>
          )}
        </button>
      </div>
    </form>
  );
};

export default ConstellationForm;
