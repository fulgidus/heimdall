import React, { useState } from 'react';
import { useAuthStore } from '../store';

const Settings: React.FC = () => {
    const { user } = useAuthStore();
    const [activeTab, setActiveTab] = useState<'general' | 'api' | 'notifications' | 'advanced'>('general');
    const [isSaving, setIsSaving] = useState(false);

    // Settings state (TODO: Connect to backend settings API)
    const [settings, setSettings] = useState({
        // General
        theme: 'dark',
        language: 'en',
        timezone: 'UTC',
        autoRefresh: true,
        refreshInterval: 30,

        // API
        apiTimeout: 30000,
        retryAttempts: 3,
        enableCaching: true,

        // Notifications
        emailNotifications: true,
        systemAlerts: true,
        performanceWarnings: true,
        webhookUrl: '',

        // Advanced
        debugMode: false,
        logLevel: 'info',
        maxConcurrentRequests: 5,
    });

    const handleSave = async () => {
        setIsSaving(true);
        // TODO: Save to backend API
        await new Promise(resolve => setTimeout(resolve, 1000));
        setIsSaving(false);
    };

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item"><a href="#">Settings</a></li>
                                <li className="breadcrumb-item" aria-current="page">Application Settings</li>
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
                                    onClick={(e) => {
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
                                    onClick={(e) => {
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
                                    onClick={(e) => {
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
                                    onClick={(e) => {
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
                            {/* General Settings */}
                            {activeTab === 'general' && (
                                <div className="row g-3">
                                    <div className="col-md-6">
                                        <label className="form-label">Theme</label>
                                        <select
                                            className="form-select"
                                            value={settings.theme}
                                            onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
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
                                            value={settings.language}
                                            onChange={(e) => setSettings({ ...settings, language: e.target.value })}
                                        >
                                            <option value="en">English</option>
                                            <option value="it">Italiano</option>
                                        </select>
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Timezone</label>
                                        <select
                                            className="form-select"
                                            value={settings.timezone}
                                            onChange={(e) => setSettings({ ...settings, timezone: e.target.value })}
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
                                            value={settings.refreshInterval}
                                            onChange={(e) => setSettings({ ...settings, refreshInterval: parseInt(e.target.value) })}
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
                                                checked={settings.autoRefresh}
                                                onChange={(e) => setSettings({ ...settings, autoRefresh: e.target.checked })}
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
                                            value={settings.apiTimeout}
                                            onChange={(e) => setSettings({ ...settings, apiTimeout: parseInt(e.target.value) })}
                                            min="5000"
                                            max="60000"
                                        />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Retry Attempts</label>
                                        <input
                                            type="number"
                                            className="form-control"
                                            value={settings.retryAttempts}
                                            onChange={(e) => setSettings({ ...settings, retryAttempts: parseInt(e.target.value) })}
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
                                                checked={settings.enableCaching}
                                                onChange={(e) => setSettings({ ...settings, enableCaching: e.target.checked })}
                                            />
                                            <label className="form-check-label" htmlFor="enableCaching">
                                                Enable API response caching
                                            </label>
                                        </div>
                                    </div>
                                    <div className="col-12">
                                        <div className="alert alert-info">
                                            <i className="ph ph-info me-2"></i>
                                            API Base URL: <code>{import.meta.env.VITE_API_URL || 'http://localhost:8000'}</code>
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
                                                checked={settings.emailNotifications}
                                                onChange={(e) => setSettings({ ...settings, emailNotifications: e.target.checked })}
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
                                                checked={settings.systemAlerts}
                                                onChange={(e) => setSettings({ ...settings, systemAlerts: e.target.checked })}
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
                                                checked={settings.performanceWarnings}
                                                onChange={(e) => setSettings({ ...settings, performanceWarnings: e.target.checked })}
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
                                            value={settings.webhookUrl}
                                            onChange={(e) => setSettings({ ...settings, webhookUrl: e.target.value })}
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
                                            value={settings.logLevel}
                                            onChange={(e) => setSettings({ ...settings, logLevel: e.target.value })}
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
                                            value={settings.maxConcurrentRequests}
                                            onChange={(e) => setSettings({ ...settings, maxConcurrentRequests: parseInt(e.target.value) })}
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
                                                checked={settings.debugMode}
                                                onChange={(e) => setSettings({ ...settings, debugMode: e.target.checked })}
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
                                            <strong>Warning:</strong> Advanced settings can affect system performance. Change only if you know what you're doing.
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Save Button */}
                            <div className="row mt-4">
                                <div className="col-12">
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleSave}
                                        disabled={isSaving}
                                    >
                                        {isSaving ? (
                                            <>
                                                <span className="spinner-border spinner-border-sm me-2" role="status"></span>
                                                Saving...
                                            </>
                                        ) : (
                                            <>
                                                <i className="ph ph-check me-2"></i>
                                                Save Settings
                                            </>
                                        )}
                                    </button>
                                    <button className="btn btn-outline-secondary ms-2">
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
