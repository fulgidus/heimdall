import React, { useEffect, useState } from 'react';
import { useWebSDRStore } from '../store/websdrStore';
import WebSDRModal from '../components/WebSDRModal';
import DeleteConfirmModal from '../components/DeleteConfirmModal';
import type { WebSDRConfig } from '@/services/api/types';
import { createWebSocketManager } from '@/lib/websocket';
import type { WebSocketManager } from '@/lib/websocket';

const WebSDRManagement: React.FC = () => {
    const {
        websdrs,
        healthStatus,
        isLoading,
        error,
        fetchWebSDRs,
        checkHealth,
        refreshAll,
        // lastHealthCheck,  // Commented - not used after removing stats cards
        createWebSDR,
        updateWebSDR,
        deleteWebSDR,
    } = useWebSDRStore();

    const [isRefreshing, setIsRefreshing] = useState(false);
    const [selectedWebSDR, setSelectedWebSDR] = useState<string | null>(null);
    const [wsManager, setWsManager] = useState<WebSocketManager | null>(null);

    // Modal states
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showEditModal, setShowEditModal] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [editingWebSDR, setEditingWebSDR] = useState<WebSDRConfig | null>(null);
    const [deletingWebSDR, setDeletingWebSDR] = useState<WebSDRConfig | null>(null);

    useEffect(() => {
        // Initial data load
        const loadData = async () => {
            await fetchWebSDRs();
            await checkHealth();
        };
        loadData();

        // Setup WebSocket for real-time updates
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const hostname = window.location.hostname;
        const port = window.location.port || (protocol === 'wss' ? '443' : '80');
        const wsUrl = import.meta.env.VITE_SOCKET_URL || `${protocol}://${hostname}:${port}/ws`;
        
        const manager = createWebSocketManager(wsUrl);
        
        // Subscribe to WebSDR health updates
        manager.subscribe('websdrs_update', (data) => {
            console.log('[WebSDRManagement] Received real-time WebSDR health update:', data);
            // Update health status in store directly from WebSocket
            useWebSDRStore.setState((state) => ({
                healthStatus: data,
            }));
        });

        // Connect WebSocket
        manager.connect().catch((error) => {
            console.error('[WebSDRManagement] WebSocket connection failed:', error);
        });
        
        setWsManager(manager);

        // Fallback: Auto-refresh health every 30 seconds if WebSocket fails
        const interval = setInterval(() => {
            if (!manager.isConnected()) {
                console.log('[WebSDRManagement] WebSocket disconnected, using fallback polling');
                checkHealth();
            }
        }, 30000);

        return () => {
            clearInterval(interval);
            if (manager) {
                manager.disconnect();
            }
        };
    }, [fetchWebSDRs, checkHealth]);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await refreshAll();
        setIsRefreshing(false);
    };

    // Modal handlers
    const handleCreateClick = () => {
        setShowCreateModal(true);
    };

    const handleEditClick = (websdr: WebSDRConfig) => {
        setEditingWebSDR(websdr);
        setShowEditModal(true);
    };

    const handleDeleteClick = (websdr: WebSDRConfig) => {
        setDeletingWebSDR(websdr);
        setShowDeleteModal(true);
    };

    const handleCreateSave = async (data: Partial<WebSDRConfig>) => {
        await createWebSDR(data as Omit<WebSDRConfig, 'id'>);
        setShowCreateModal(false);
        await refreshAll();
    };

    const handleEditSave = async (data: Partial<WebSDRConfig>) => {
        if (editingWebSDR) {
            await updateWebSDR(editingWebSDR.id, data);
            setShowEditModal(false);
            setEditingWebSDR(null);
            await refreshAll();
        }
    };

    const handleDeleteConfirm = async (hardDelete: boolean) => {
        if (deletingWebSDR) {
            await deleteWebSDR(deletingWebSDR.id, hardDelete);
            setShowDeleteModal(false);
            setDeletingWebSDR(null);
            await refreshAll();
        }
    };

    // Calculate statistics - Commented because stats cards were removed
    // const onlineCount = Object.values(healthStatus).filter(h => h.status === 'online').length;
    // const totalCount = websdrs.length;
    // const activeCount = websdrs.filter(w => w.is_active).length;
    // const avgResponseTime = Object.values(healthStatus).reduce((sum, h) => sum + (h.response_time_ms || 0), 0) / (Object.keys(healthStatus).length || 1);

    return (
        <>
            {/* Breadcrumb */}
            <div className="page-header">
                <div className="page-block">
                    <div className="row align-items-center">
                        <div className="col-md-12">
                            <ul className="breadcrumb">
                                <li className="breadcrumb-item"><a href="/dashboard">Home</a></li>
                                <li className="breadcrumb-item"><a href="#">RF Operations</a></li>
                                <li className="breadcrumb-item" aria-current="page">WebSDR Management</li>
                            </ul>
                        </div>
                        <div className="col-md-12">
                            <div className="page-header-title">
                                <h2 className="mb-0">WebSDR Management</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error!</strong> {error}
                    <button type="button" className="btn-close" data-bs-dismiss="alert"></button>
                </div>
            )}

            {/* WebSDR List */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header d-flex align-items-center justify-content-between">
                            <h5 className="mb-0">Configured WebSDR Receivers</h5>
                            <div className="btn-group">
                                <button
                                    className="btn btn-sm btn-primary me-2"
                                    onClick={handleCreateClick}
                                >
                                    <i className="ph ph-plus"></i> Add New WebSDR
                                </button>
                                <button className="btn btn-sm btn-outline-primary" onClick={handleRefresh} disabled={isRefreshing}>
                                    <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
                                    Refresh
                                </button>
                            </div>
                        </div>
                        <div className="card-body">
                            {isLoading ? (
                                <div className="text-center py-5">
                                    <div className="spinner-border text-primary" role="status">
                                        <span className="visually-hidden">Loading...</span>
                                    </div>
                                    <p className="text-muted mt-2">Loading WebSDR receivers...</p>
                                </div>
                            ) : websdrs.length > 0 ? (
                                <div className="table-responsive">
                                    <table className="table table-hover mb-0">
                                        <thead>
                                            <tr>
                                                <th>Status</th>
                                                <th>Name</th>
                                                <th>Location</th>
                                                <th>Coordinates</th>
                                                <th>URL</th>
                                                <th>Response Time</th>
                                                <th>Active</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {websdrs.map((websdr) => {
                                                const health = healthStatus[websdr.id];
                                                const isOnline = health?.status === 'online';
                                                const responseTime = health?.response_time_ms || 0;

                                                return (
                                                    <tr key={websdr.id}>
                                                        <td>
                                                            <span className={`badge ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                <i className={`ph ${isOnline ? 'ph-check-circle' : 'ph-x-circle'}`}></i>
                                                                {isOnline ? ' Online' : ' Offline'}
                                                            </span>
                                                        </td>
                                                        <td>
                                                            <div className="d-flex align-items-center">
                                                                <div className={`avtar avtar-xs ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                    <i className="ph ph-radio-button"></i>
                                                                </div>
                                                                <span className="ms-2">{websdr.name}</span>
                                                            </div>
                                                        </td>
                                                        <td>{websdr.location_description || '-'}</td>
                                                        <td className="f-12">
                                                            {websdr.latitude.toFixed(4)}, {websdr.longitude.toFixed(4)}
                                                        </td>
                                                        <td>
                                                            <a
                                                                href={websdr.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="link-primary"
                                                            >
                                                                <i className="ph ph-arrow-square-out"></i>
                                                            </a>
                                                        </td>
                                                        <td>
                                                            {responseTime > 0 ? (
                                                                <span className={`badge ${responseTime < 200 ? 'bg-light-success' : responseTime < 500 ? 'bg-light-warning' : 'bg-light-danger'}`}>
                                                                    {responseTime}ms
                                                                </span>
                                                            ) : (
                                                                <span className="text-muted">-</span>
                                                            )}
                                                        </td>
                                                        <td>
                                                            <span className={`badge ${websdr.is_active ? 'bg-light-primary' : 'bg-light-secondary'}`}>
                                                                {websdr.is_active ? 'Yes' : 'No'}
                                                            </span>
                                                        </td>
                                                        <td>
                                                            <div className="btn-group">
                                                                <button
                                                                    className="btn btn-sm btn-link-secondary"
                                                                    onClick={() => setSelectedWebSDR(websdr.id)}
                                                                    title="View Details"
                                                                >
                                                                    <i className="ph ph-eye"></i>
                                                                </button>
                                                                <button
                                                                    className="btn btn-sm btn-link-primary"
                                                                    onClick={() => handleEditClick(websdr)}
                                                                    title="Edit"
                                                                >
                                                                    <i className="ph ph-pencil-simple"></i>
                                                                </button>
                                                                <button
                                                                    className="btn btn-sm btn-link-danger"
                                                                    onClick={() => handleDeleteClick(websdr)}
                                                                    title="Delete"
                                                                >
                                                                    <i className="ph ph-trash"></i>
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <div className="text-center py-5">
                                    <i className="ph ph-warning-circle f-40 text-warning mb-3"></i>
                                    <p className="text-muted mb-0">No WebSDR receivers configured</p>
                                    <button className="btn btn-primary mt-3" onClick={handleCreateClick}>
                                        <i className="ph ph-plus-circle"></i>
                                        Add WebSDR Receiver
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Selected WebSDR Details */}
            {selectedWebSDR && websdrs.find(w => w.id === selectedWebSDR) && (
                <div className="row">
                    <div className="col-12">
                        <div className="card">
                            <div className="card-header d-flex align-items-center justify-content-between">
                                <h5 className="mb-0">WebSDR Details</h5>
                                <button
                                    className="btn btn-sm btn-link-secondary"
                                    onClick={() => setSelectedWebSDR(null)}
                                >
                                    <i className="ph ph-x"></i>
                                </button>
                            </div>
                            <div className="card-body">
                                {(() => {
                                    const websdr = websdrs.find(w => w.id === selectedWebSDR);
                                    const health = healthStatus[selectedWebSDR];
                                    if (!websdr) return null;

                                    return (
                                        <div className="row">
                                            <div className="col-md-6">
                                                <h6>General Information</h6>
                                                <table className="table table-sm">
                                                    <tbody>
                                                        <tr>
                                                            <td className="text-muted">ID</td>
                                                            <td>{websdr.id}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Name</td>
                                                            <td>{websdr.name}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Location</td>
                                                            <td>{websdr.location_description || '-'}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Latitude</td>
                                                            <td>{websdr.latitude.toFixed(6)}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Longitude</td>
                                                            <td>{websdr.longitude.toFixed(6)}</td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">URL</td>
                                                            <td>
                                                                <a href={websdr.url} target="_blank" rel="noopener noreferrer">
                                                                    {websdr.url}
                                                                </a>
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-muted">Active</td>
                                                            <td>
                                                                <span className={`badge ${websdr.is_active ? 'bg-light-success' : 'bg-light-secondary'}`}>
                                                                    {websdr.is_active ? 'Yes' : 'No'}
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                            <div className="col-md-6">
                                                <h6>Health Status</h6>
                                                {health ? (
                                                    <table className="table table-sm">
                                                        <tbody>
                                                            <tr>
                                                                <td className="text-muted">Status</td>
                                                                <td>
                                                                    <span className={`badge ${health.status === 'online' ? 'bg-light-success' : 'bg-light-danger'}`}>
                                                                        {health.status}
                                                                    </span>
                                                                </td>
                                                            </tr>
                                                            <tr>
                                                                <td className="text-muted">Response Time</td>
                                                                <td>{health.response_time_ms || 0}ms</td>
                                                            </tr>
                                                            <tr>
                                                                <td className="text-muted">Last Check</td>
                                                                <td>{health.last_check || 'Never'}</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                ) : (
                                                    <p className="text-muted">No health data available</p>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })()}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Modals */}
            <WebSDRModal
                show={showCreateModal}
                onHide={() => setShowCreateModal(false)}
                onSave={handleCreateSave}
                mode="create"
            />

            <WebSDRModal
                show={showEditModal}
                onHide={() => {
                    setShowEditModal(false);
                    setEditingWebSDR(null);
                }}
                onSave={handleEditSave}
                websdr={editingWebSDR}
                mode="edit"
            />

            <DeleteConfirmModal
                show={showDeleteModal}
                onHide={() => {
                    setShowDeleteModal(false);
                    setDeletingWebSDR(null);
                }}
                onConfirm={handleDeleteConfirm}
                websdr={deletingWebSDR}
            />
        </>
    );
};

export default WebSDRManagement;
