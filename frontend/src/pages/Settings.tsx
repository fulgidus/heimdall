import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../store/settingsStore';
import type { UserSettingsUpdate } from '../services/api/settings';

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'general' | 'api' | 'notifications' | 'advanced'>(
    'general'
  );
  
  const { settings, isLoading, error, isSaving, fetchSettings, updateSettings, resetSettings, clearError } = useSettingsStore();
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  
  // Local state for form fields
  const [formData, setFormData] = useState<UserSettingsUpdate>({});

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // Update form data when settings are loaded
  useEffect(() => {
    if (settings) {
      setFormData({
        theme: settings.theme,
        language: settings.language,
        timezone: settings.timezone,
        auto_refresh: settings.auto_refresh,
        refresh_interval: settings.refresh_interval,
        api_timeout: settings.api_timeout,
        retry_attempts: settings.retry_attempts,
        enable_caching: settings.enable_caching,
        email_notifications: settings.email_notifications,
        system_alerts: settings.system_alerts,
        performance_warnings: settings.performance_warnings,
        webhook_url: settings.webhook_url,
        debug_mode: settings.debug_mode,
        log_level: settings.log_level,
        max_concurrent_requests: settings.max_concurrent_requests,
      });
    }
  }, [settings]);

  const handleSave = async () => {
    try {
      clearError();
      setSuccessMessage(null);
      await updateSettings(formData);
      setSuccessMessage('Settings saved successfully!');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (_err) {
      // Error is already set in the store
    }
  };

  const handleReset = async () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      try {
        clearError();
        setSuccessMessage(null);
        await resetSettings();
        setSuccessMessage('Settings reset to defaults!');
        setTimeout(() => setSuccessMessage(null), 3000);
      } catch (_err) {
        // Error is already set in the store
      }
    }
  };

  if (isLoading && !settings) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '400px' }}>
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Breadcrumb */}
      <div className="page-header">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ul className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/dashboard">Home</a>
                </li>
                <li className="breadcrumb-item">
                  <a href="#">Settings</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Application Settings
                </li>
              </ul>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h2 className="mb-0">Settings</h2>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        {/* Settings Navigation */}
        <div className="col-lg-3">
          <div className="card">
            <div className="card-body">
              <div className="list-group list-group-flush">
                <a
                  href="#!"
                  className={`list-group-item list-group-item-action ${activeTab === 'general' ? 'active' : ''}`}
                  onClick={e => {
                    e.preventDefault();
                    setActiveTab('general');
                  }}
                >
                  <i className="ph ph-gear me-2"></i>
                  General
                </a>
                <a
                  href="#!"
                  className={`list-group-item list-group-item-action ${activeTab === 'api' ? 'active' : ''}`}
                  onClick={e => {
                    e.preventDefault();
                    setActiveTab('api');
                  }}
                >
                  <i className="ph ph-plugs-connected me-2"></i>
                  API Configuration
                </a>
                <a
                  href="#!"
                  className={`list-group-item list-group-item-action ${activeTab === 'notifications' ? 'active' : ''}`}
                  onClick={e => {
                    e.preventDefault();
                    setActiveTab('notifications');
                  }}
                >
                  <i className="ph ph-bell me-2"></i>
                  Notifications
                </a>
                <a
                  href="#!"
                  className={`list-group-item list-group-item-action ${activeTab === 'advanced' ? 'active' : ''}`}
                  onClick={e => {
                    e.preventDefault();
                    setActiveTab('advanced');
                  }}
                >
                  <i className="ph ph-code me-2"></i>
                  Advanced
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Settings Content */}
        <div className="col-lg-9">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">
                {activeTab === 'general' && 'General Settings'}
                {activeTab === 'api' && 'API Configuration'}
                {activeTab === 'notifications' && 'Notification Settings'}
                {activeTab === 'advanced' && 'Advanced Settings'}
              </h5>
            </div>
            <div className="card-body">
              {/* Error Message */}
              {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                  <i className="ph ph-warning-circle me-2"></i>
                  {error}
                  <button
                    type="button"
                    className="btn-close"
                    onClick={() => clearError()}
                    aria-label="Close"
                  ></button>
                </div>
              )}

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

              {/* General Settings */}
              {activeTab === 'general' && (
                <div className="row g-3">
                  <div className="col-md-6">
                    <label className="form-label">Theme</label>
                    <select
                      className="form-select"
                      value={formData.theme || 'dark'}
                      onChange={e => setFormData({ ...formData, theme: e.target.value as any })}
                    >
                      <option value="dark">Dark</option>
                      <option value="light">Light</option>
                      <option value="auto">Auto</option>
                    </select>
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Language</label>
                    <select
                      className="form-select"
                      value={formData.language || 'en'}
                      onChange={e => setFormData({ ...formData, language: e.target.value as any })}
                    >
                      <option value="en">English</option>
                      <option value="it">Italiano</option>
                    </select>
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Timezone</label>
                    <select
                      className="form-select"
                      value={formData.timezone || 'UTC'}
                      onChange={e => setFormData({ ...formData, timezone: e.target.value })}
                    >
                      <option value="UTC">UTC</option>
                      <option value="Europe/Rome">Europe/Rome</option>
                      <option value="America/New_York">America/New_York</option>
                    </select>
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Auto-refresh Interval (seconds)</label>
                    <input
                      type="number"
                      className="form-control"
                      value={formData.refresh_interval || 30}
                      onChange={e =>
                        setFormData({ ...formData, refresh_interval: parseInt(e.target.value) })
                      }
                      min="10"
                      max="300"
                    />
                  </div>
                  <div className="col-12">
                    <div className="form-check form-switch">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="autoRefresh"
                        checked={formData.auto_refresh ?? true}
                        onChange={e => setFormData({ ...formData, auto_refresh: e.target.checked })}
                      />
                      <label className="form-check-label" htmlFor="autoRefresh">
                        Enable auto-refresh for dashboard
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* API Configuration */}
              {activeTab === 'api' && (
                <div className="row g-3">
                  <div className="col-md-6">
                    <label className="form-label">API Timeout (ms)</label>
                    <input
                      type="number"
                      className="form-control"
                      value={formData.api_timeout || 30000}
                      onChange={e =>
                        setFormData({ ...formData, api_timeout: parseInt(e.target.value) })
                      }
                      min="5000"
                      max="60000"
                    />
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Retry Attempts</label>
                    <input
                      type="number"
                      className="form-control"
                      value={formData.retry_attempts || 3}
                      onChange={e =>
                        setFormData({ ...formData, retry_attempts: parseInt(e.target.value) })
                      }
                      min="0"
                      max="5"
                    />
                  </div>
                  <div className="col-12">
                    <div className="form-check form-switch">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="enableCaching"
                        checked={formData.enable_caching ?? true}
                        onChange={e =>
                          setFormData({ ...formData, enable_caching: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="enableCaching">
                        Enable API response caching
                      </label>
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="alert alert-info">
                      <i className="ph ph-info me-2"></i>
                      API Base URL:{' '}
                      <code>{import.meta.env.VITE_API_URL || 'http://localhost:8000'}</code>
                    </div>
                  </div>
                </div>
              )}

              {/* Notifications */}
              {activeTab === 'notifications' && (
                <div className="row g-3">
                  <div className="col-12">
                    <div className="form-check form-switch mb-3">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="emailNotifications"
                        checked={formData.email_notifications ?? true}
                        onChange={e =>
                          setFormData({ ...formData, email_notifications: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="emailNotifications">
                        Email Notifications
                      </label>
                    </div>
                    <div className="form-check form-switch mb-3">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="systemAlerts"
                        checked={formData.system_alerts ?? true}
                        onChange={e => setFormData({ ...formData, system_alerts: e.target.checked })}
                      />
                      <label className="form-check-label" htmlFor="systemAlerts">
                        System Alerts
                      </label>
                    </div>
                    <div className="form-check form-switch mb-3">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="performanceWarnings"
                        checked={formData.performance_warnings ?? true}
                        onChange={e =>
                          setFormData({ ...formData, performance_warnings: e.target.checked })
                        }
                      />
                      <label className="form-check-label" htmlFor="performanceWarnings">
                        Performance Warnings
                      </label>
                    </div>
                  </div>
                  <div className="col-12">
                    <label className="form-label">Webhook URL (Optional)</label>
                    <input
                      type="url"
                      className="form-control"
                      placeholder="https://your-webhook-url.com"
                      value={formData.webhook_url || ''}
                      onChange={e => setFormData({ ...formData, webhook_url: e.target.value || null })}
                    />
                    <small className="text-muted">Receive notifications via webhook</small>
                  </div>
                </div>
              )}

              {/* Advanced Settings */}
              {activeTab === 'advanced' && (
                <div className="row g-3">
                  <div className="col-md-6">
                    <label className="form-label">Log Level</label>
                    <select
                      className="form-select"
                      value={formData.log_level || 'info'}
                      onChange={e => setFormData({ ...formData, log_level: e.target.value as any })}
                    >
                      <option value="error">Error</option>
                      <option value="warn">Warn</option>
                      <option value="info">Info</option>
                      <option value="debug">Debug</option>
                    </select>
                  </div>
                  <div className="col-md-6">
                    <label className="form-label">Max Concurrent Requests</label>
                    <input
                      type="number"
                      className="form-control"
                      value={formData.max_concurrent_requests || 5}
                      onChange={e =>
                        setFormData({
                          ...formData,
                          max_concurrent_requests: parseInt(e.target.value),
                        })
                      }
                      min="1"
                      max="10"
                    />
                  </div>
                  <div className="col-12">
                    <div className="form-check form-switch">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="debugMode"
                        checked={formData.debug_mode ?? false}
                        onChange={e => setFormData({ ...formData, debug_mode: e.target.checked })}
                      />
                      <label className="form-check-label" htmlFor="debugMode">
                        Enable Debug Mode
                      </label>
                    </div>
                    <small className="text-muted">Show detailed error messages and API logs</small>
                  </div>
                  <div className="col-12">
                    <div className="alert alert-warning">
                      <i className="ph ph-warning me-2"></i>
                      <strong>Warning:</strong> Advanced settings can affect system performance.
                      Change only if you know what you're doing.
                    </div>
                  </div>
                </div>
              )}

              {/* Save Button */}
              <div className="row mt-4">
                <div className="col-12">
                  <button className="btn btn-primary" onClick={handleSave} disabled={isSaving}>
                    {isSaving ? (
                      <>
                        <span
                          className="spinner-border spinner-border-sm me-2"
                          role="status"
                        ></span>
                        Saving...
                      </>
                    ) : (
                      <>
                        <i className="ph ph-check me-2"></i>
                        Save Settings
                      </>
                    )}
                  </button>
                  <button 
                    className="btn btn-outline-secondary ms-2"
                    onClick={handleReset}
                    disabled={isSaving}
                  >
                    <i className="ph ph-arrow-counter-clockwise me-2"></i>
                    Reset to Defaults
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Settings;
