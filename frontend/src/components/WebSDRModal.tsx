/**
 * WebSDR Create/Edit Modal Component
 *
 * Bootstrap 5 modal for creating and editing WebSDR stations
 */

import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { WebSDRConfig } from '@/services/api/types';

interface WebSDRModalProps {
  show: boolean;
  onHide: () => void;
  onSave: (data: Partial<WebSDRConfig>) => Promise<void>;
  websdr?: WebSDRConfig | null;
  mode: 'create' | 'edit';
}

interface FormData {
  name: string;
  url: string;
  latitude: string;
  longitude: string;
  location_description: string;
  country: string;
  admin_email: string;
  altitude_asl: string;
  frequency_min_hz: string;
  frequency_max_hz: string;
  timeout_seconds: string;
  retry_count: string;
  is_active: boolean;
}

interface FormErrors {
  [key: string]: string;
}

const WebSDRModal: React.FC<WebSDRModalProps> = ({ show, onHide, onSave, websdr, mode }) => {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    url: '',
    latitude: '',
    longitude: '',
    location_description: '',
    country: '',
    admin_email: '',
    altitude_asl: '',
    frequency_min_hz: '',
    frequency_max_hz: '',
    timeout_seconds: '30',
    retry_count: '3',
    is_active: true,
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isSaving, setIsSaving] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
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

  // Populate form when editing
  useEffect(() => {
    if (mode === 'edit' && websdr) {
      setFormData({
        name: websdr.name || '',
        url: websdr.url || '',
        latitude: websdr.latitude?.toString() || '',
        longitude: websdr.longitude?.toString() || '',
        location_description: websdr.location_description || '',
        country: websdr.country || '',
        admin_email: websdr.admin_email || '',
        altitude_asl: websdr.altitude_asl?.toString() || '',
        frequency_min_hz: websdr.frequency_min_hz?.toString() || '',
        frequency_max_hz: websdr.frequency_max_hz?.toString() || '',
        timeout_seconds: websdr.timeout_seconds?.toString() || '30',
        retry_count: websdr.retry_count?.toString() || '3',
        is_active: websdr.is_active ?? true,
      });
    } else if (mode === 'create') {
      // Reset form for create mode
      setFormData({
        name: '',
        url: '',
        latitude: '',
        longitude: '',
        location_description: '',
        country: '',
        admin_email: '',
        altitude_asl: '',
        frequency_min_hz: '',
        frequency_max_hz: '',
        timeout_seconds: '30',
        retry_count: '3',
        is_active: true,
      });
    }
    setErrors({});
    setFetchError(null);
  }, [mode, websdr, show]);

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // Required fields
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }

    if (!formData.url.trim()) {
      newErrors.url = 'URL is required';
    } else if (!/^https?:\/\/.+/.test(formData.url)) {
      newErrors.url = 'URL must start with http:// or https://';
    }

    // Latitude validation
    const lat = parseFloat(formData.latitude);
    if (!formData.latitude.trim()) {
      newErrors.latitude = 'Latitude is required';
    } else if (isNaN(lat) || lat < -90 || lat > 90) {
      newErrors.latitude = 'Latitude must be between -90 and 90';
    }

    // Longitude validation
    const lon = parseFloat(formData.longitude);
    if (!formData.longitude.trim()) {
      newErrors.longitude = 'Longitude is required';
    } else if (isNaN(lon) || lon < -180 || lon > 180) {
      newErrors.longitude = 'Longitude must be between -180 and 180';
    }

    // Optional email validation
    if (formData.admin_email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.admin_email)) {
      newErrors.admin_email = 'Invalid email format';
    }

    // Timeout validation
    const timeout = parseInt(formData.timeout_seconds);
    if (isNaN(timeout) || timeout < 1 || timeout > 300) {
      newErrors.timeout_seconds = 'Timeout must be between 1 and 300 seconds';
    }

    // Retry count validation
    const retry = parseInt(formData.retry_count);
    if (isNaN(retry) || retry < 0 || retry > 10) {
      newErrors.retry_count = 'Retry count must be between 0 and 10';
    }

    // Altitude validation
    if (formData.altitude_asl) {
      const altitude = parseInt(formData.altitude_asl);
      if (isNaN(altitude)) {
        newErrors.altitude_asl = 'Altitude must be a number';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    const checked = (e.target as HTMLInputElement).checked;

    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));

    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  const handleFetchFromUrl = async () => {
    if (!formData.url.trim()) {
      setFetchError('Please enter a URL first');
      return;
    }

    if (!/^https?:\/\/.+/.test(formData.url)) {
      setFetchError('URL must start with http:// or https://');
      return;
    }

    setIsFetching(true);
    setFetchError(null);

    try {
      const { fetchWebSDRInfo } = await import('@/services/api/websdr');
      const info = await fetchWebSDRInfo(formData.url);

      if (!info.success) {
        setFetchError(info.error_message || 'Failed to fetch WebSDR information');
        return;
      }

      // Populate form with fetched data
      setFormData(prev => ({
        ...prev,
        name: info.receiver_name || prev.name,
        location_description: info.location || prev.location_description,
        latitude: info.latitude?.toString() || prev.latitude,
        longitude: info.longitude?.toString() || prev.longitude,
        altitude_asl: info.altitude_asl?.toString() || prev.altitude_asl,
        admin_email: info.admin_email || prev.admin_email,
        frequency_min_hz: info.frequency_min_hz?.toString() || prev.frequency_min_hz,
        frequency_max_hz: info.frequency_max_hz?.toString() || prev.frequency_max_hz,
      }));

      // Show success feedback
      console.log('✅ Fetched WebSDR info:', info);
    } catch (error) {
      console.error('Fetch error:', error);
      setFetchError(error instanceof Error ? error.message : 'Failed to fetch WebSDR information');
    } finally {
      setIsFetching(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSaving(true);

    try {
      const payload: any = {
        name: formData.name.trim(),
        url: formData.url.trim(),
        latitude: parseFloat(formData.latitude),
        longitude: parseFloat(formData.longitude),
        timeout_seconds: parseInt(formData.timeout_seconds),
        retry_count: parseInt(formData.retry_count),
        is_active: formData.is_active,
      };

      // Optional fields
      if (formData.location_description.trim()) {
        payload.location_description = formData.location_description.trim();
      }
      if (formData.country.trim()) {
        payload.country = formData.country.trim();
      }
      if (formData.admin_email.trim()) {
        payload.admin_email = formData.admin_email.trim();
      }
      if (formData.altitude_asl) {
        payload.altitude_asl = parseInt(formData.altitude_asl);
      }
      if (formData.frequency_min_hz) {
        payload.frequency_min_hz = parseInt(formData.frequency_min_hz);
      }
      if (formData.frequency_max_hz) {
        payload.frequency_max_hz = parseInt(formData.frequency_max_hz);
      }

      await onSave(payload);
      onHide();
    } catch (error) {
      console.error('Save error:', error);
      setErrors({ submit: error instanceof Error ? error.message : 'Failed to save WebSDR' });
    } finally {
      setIsSaving(false);
    }
  };

  if (!show) return null;

  return createPortal(
    <>
      {/* Modal Backdrop */}
      <div className="modal-backdrop fade show" onClick={onHide}></div>

      {/* Modal */}
      <div className="modal fade show" style={{ display: 'block' }} tabIndex={-1}>
        <div className="modal-dialog modal-lg modal-dialog-scrollable">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">
                <i
                  className={`ph ${mode === 'create' ? 'ph-plus-circle' : 'ph-pencil-simple'} me-2`}
                ></i>
                {mode === 'create' ? 'Add New WebSDR Station' : 'Edit WebSDR Station'}
              </h5>
              <button type="button" className="btn-close" onClick={onHide}></button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="modal-body">
                {errors.submit && (
                  <div className="alert alert-danger">
                    <i className="ph ph-warning-circle me-2"></i>
                    {errors.submit}
                  </div>
                )}

                {/* Basic Information */}
                <div className="mb-4">
                  <h6 className="mb-3">
                    <i className="ph ph-info me-2"></i>Basic Information
                  </h6>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="name" className="form-label">
                        Station Name <span className="text-danger">*</span>
                      </label>
                      <input
                        type="text"
                        className={`form-control ${errors.name ? 'is-invalid' : ''}`}
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        placeholder="e.g., IW2MXM Milano"
                        required
                      />
                      {errors.name && <div className="invalid-feedback">{errors.name}</div>}
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="url" className="form-label">
                        WebSDR URL <span className="text-danger">*</span>
                      </label>
                      <div className="input-group">
                        <input
                          type="url"
                          className={`form-control ${errors.url ? 'is-invalid' : ''}`}
                          id="url"
                          name="url"
                          value={formData.url}
                          onChange={handleChange}
                          placeholder="http://websdr.example.com:8901"
                          required
                        />
                        <button
                          type="button"
                          className="btn btn-outline-primary"
                          onClick={handleFetchFromUrl}
                          disabled={isFetching || !formData.url.trim()}
                          title="Fetch station info from URL"
                        >
                          {isFetching ? (
                            <span className="spinner-border spinner-border-sm"></span>
                          ) : (
                            <i className="ph ph-download-simple"></i>
                          )}
                        </button>
                      </div>
                      {errors.url && <div className="invalid-feedback d-block">{errors.url}</div>}
                      {fetchError && (
                        <div className="text-danger small mt-1">
                          <i className="ph ph-warning-circle me-1"></i>
                          {fetchError}
                        </div>
                      )}
                      {!fetchError && isFetching && (
                        <small className="text-muted d-block mt-1">
                          Fetching WebSDR information...
                        </small>
                      )}
                    </div>
                  </div>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="location_description" className="form-label">
                        Location Description
                      </label>
                      <input
                        type="text"
                        className="form-control"
                        id="location_description"
                        name="location_description"
                        value={formData.location_description}
                        onChange={handleChange}
                        placeholder="e.g., Milano, Lombardia"
                      />
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="country" className="form-label">
                        Country
                      </label>
                      <input
                        type="text"
                        className="form-control"
                        id="country"
                        name="country"
                        value={formData.country}
                        onChange={handleChange}
                        placeholder="e.g., IT, Italy"
                      />
                    </div>
                  </div>

                  <div className="mb-3">
                    <label htmlFor="admin_email" className="form-label">
                      Administrator Email
                    </label>
                    <input
                      type="email"
                      className={`form-control ${errors.admin_email ? 'is-invalid' : ''}`}
                      id="admin_email"
                      name="admin_email"
                      value={formData.admin_email}
                      onChange={handleChange}
                      placeholder="admin@example.com"
                    />
                    {errors.admin_email && (
                      <div className="invalid-feedback">{errors.admin_email}</div>
                    )}
                  </div>
                </div>

                {/* Geographic Coordinates */}
                <div className="mb-4">
                  <h6 className="mb-3">
                    <i className="ph ph-map-pin me-2"></i>Geographic Coordinates
                  </h6>

                  <div className="row">
                    <div className="col-md-4 mb-3">
                      <label htmlFor="latitude" className="form-label">
                        Latitude <span className="text-danger">*</span>
                      </label>
                      <input
                        type="number"
                        step="0.000001"
                        className={`form-control ${errors.latitude ? 'is-invalid' : ''}`}
                        id="latitude"
                        name="latitude"
                        value={formData.latitude}
                        onChange={handleChange}
                        placeholder="45.464200"
                        required
                      />
                      {errors.latitude && <div className="invalid-feedback">{errors.latitude}</div>}
                      <small className="text-muted">Range: -90 to 90</small>
                    </div>

                    <div className="col-md-4 mb-3">
                      <label htmlFor="longitude" className="form-label">
                        Longitude <span className="text-danger">*</span>
                      </label>
                      <input
                        type="number"
                        step="0.000001"
                        className={`form-control ${errors.longitude ? 'is-invalid' : ''}`}
                        id="longitude"
                        name="longitude"
                        value={formData.longitude}
                        onChange={handleChange}
                        placeholder="9.188000"
                        required
                      />
                      {errors.longitude && (
                        <div className="invalid-feedback">{errors.longitude}</div>
                      )}
                      <small className="text-muted">Range: -180 to 180</small>
                    </div>

                    <div className="col-md-4 mb-3">
                      <label htmlFor="altitude_asl" className="form-label">
                        Altitude ASL (m)
                      </label>
                      <input
                        type="number"
                        className={`form-control ${errors.altitude_asl ? 'is-invalid' : ''}`}
                        id="altitude_asl"
                        name="altitude_asl"
                        value={formData.altitude_asl}
                        onChange={handleChange}
                        placeholder="120"
                      />
                      {errors.altitude_asl && (
                        <div className="invalid-feedback">{errors.altitude_asl}</div>
                      )}
                      <small className="text-muted">Above sea level</small>
                    </div>
                  </div>
                </div>

                {/* Frequency Range */}
                <div className="mb-4">
                  <h6 className="mb-3">
                    <i className="ph ph-wave-sine me-2"></i>Frequency Range
                  </h6>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="frequency_min_hz" className="form-label">
                        Min Frequency (Hz)
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="frequency_min_hz"
                        name="frequency_min_hz"
                        value={formData.frequency_min_hz}
                        onChange={handleChange}
                        placeholder="144000000"
                        readOnly
                      />
                      <small className="text-muted">Auto-updated from status.json</small>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="frequency_max_hz" className="form-label">
                        Max Frequency (Hz)
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        id="frequency_max_hz"
                        name="frequency_max_hz"
                        value={formData.frequency_max_hz}
                        onChange={handleChange}
                        placeholder="146000000"
                        readOnly
                      />
                      <small className="text-muted">Auto-updated from status.json</small>
                    </div>
                  </div>

                  {formData.frequency_min_hz && formData.frequency_max_hz && (
                    <div className="alert alert-info mb-0">
                      <i className="ph ph-info me-2"></i>
                      <strong>Frequency Range:</strong>{' '}
                      {(parseInt(formData.frequency_min_hz) / 1e6).toFixed(3)} MHz -{' '}
                      {(parseInt(formData.frequency_max_hz) / 1e6).toFixed(3)} MHz
                    </div>
                  )}
                </div>

                {/* Connection Settings */}
                <div className="mb-4">
                  <h6 className="mb-3">
                    <i className="ph ph-gear me-2"></i>Connection Settings
                  </h6>

                  <div className="row">
                    <div className="col-md-6 mb-3">
                      <label htmlFor="timeout_seconds" className="form-label">
                        Timeout (seconds)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="300"
                        className={`form-control ${errors.timeout_seconds ? 'is-invalid' : ''}`}
                        id="timeout_seconds"
                        name="timeout_seconds"
                        value={formData.timeout_seconds}
                        onChange={handleChange}
                      />
                      {errors.timeout_seconds && (
                        <div className="invalid-feedback">{errors.timeout_seconds}</div>
                      )}
                      <small className="text-muted">Range: 1-300 seconds</small>
                    </div>

                    <div className="col-md-6 mb-3">
                      <label htmlFor="retry_count" className="form-label">
                        Retry Count
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="10"
                        className={`form-control ${errors.retry_count ? 'is-invalid' : ''}`}
                        id="retry_count"
                        name="retry_count"
                        value={formData.retry_count}
                        onChange={handleChange}
                      />
                      {errors.retry_count && (
                        <div className="invalid-feedback">{errors.retry_count}</div>
                      )}
                      <small className="text-muted">Range: 0-10 attempts</small>
                    </div>
                  </div>
                </div>

                {/* Status */}
                <div className="mb-3">
                  <div className="form-check form-switch">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="is_active"
                      name="is_active"
                      checked={formData.is_active}
                      onChange={handleChange}
                    />
                    <label className="form-check-label" htmlFor="is_active">
                      <i className="ph ph-check-circle me-2"></i>
                      Station is Active
                    </label>
                  </div>
                  <small className="text-muted">
                    Inactive stations will not be used for acquisitions
                  </small>
                </div>
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-outline-secondary"
                  onClick={onHide}
                  disabled={isSaving}
                >
                  <i className="ph ph-x me-2"></i>Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={isSaving}>
                  {isSaving ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2"></span>
                      Saving...
                    </>
                  ) : (
                    <>
                      <i className="ph ph-check me-2"></i>
                      {mode === 'create' ? 'Create Station' : 'Save Changes'}
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>,
    modalRoot
  );
};

export default WebSDRModal;
