/**
 * Modal Stress Test Component
 * 
 * This component demonstrates rapid modal opening/closing cycles
 * to verify that the Portal-based modal implementation prevents
 * the "Node.removeChild: The node to be removed is not a child of this node" bug.
 */

import React, { useState } from 'react';
import SessionEditModal from './SessionEditModal';
import type { RecordingSessionWithDetails } from '@/services/api/session';

const ModalStressTest: React.FC = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [openCount, setOpenCount] = useState(0);
    const [errorCount, setErrorCount] = useState(0);

    const mockSession: RecordingSessionWithDetails = {
        id: '123',
        known_source_id: '456',
        session_name: 'Test Session',
        notes: 'Testing rapid modal operations',
        approval_status: 'pending',
        status: 'completed',
        source_name: 'Test Source',
        source_frequency: 145000000,
        measurements_count: 10,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        session_start: '2024-01-01T00:00:00Z',
        duration_seconds: 60,
    };

    const handleOpenModal = () => {
        setIsModalOpen(true);
        setOpenCount(prev => prev + 1);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
    };

    const handleSave = async () => {
        // Mock save operation
        return Promise.resolve();
    };

    const runStressTest = async () => {
        console.log('Starting modal stress test...');
        setErrorCount(0);
        setOpenCount(0);

        // Capture console errors
        const originalError = console.error;
        let errors = 0;
        console.error = (...args: any[]) => {
            if (args[0]?.includes?.('removeChild') || args[0]?.includes?.('not a child')) {
                errors++;
                setErrorCount(prev => prev + 1);
            }
            originalError(...args);
        };

        // Rapid open/close cycles
        for (let i = 0; i < 50; i++) {
            setIsModalOpen(true);
            setOpenCount(prev => prev + 1);
            await new Promise(resolve => setTimeout(resolve, 10));
            setIsModalOpen(false);
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        // Restore console.error
        console.error = originalError;

        console.log(`Stress test complete: ${errors} errors detected`);
    };

    return (
        <div className="container mt-5">
            <div className="card">
                <div className="card-header">
                    <h5 className="mb-0">Modal Stress Test</h5>
                </div>
                <div className="card-body">
                    <p className="text-muted">
                        This test rapidly opens and closes modals to verify the Portal-based
                        implementation prevents DOM manipulation errors.
                    </p>

                    <div className="row g-3 mb-4">
                        <div className="col-md-4">
                            <div className="card bg-light">
                                <div className="card-body text-center">
                                    <h6 className="text-muted mb-1">Opens</h6>
                                    <h3 className="mb-0">{openCount}</h3>
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4">
                            <div className="card bg-light">
                                <div className="card-body text-center">
                                    <h6 className="text-muted mb-1">Status</h6>
                                    <h3 className="mb-0">
                                        {isModalOpen ? (
                                            <span className="badge bg-success">Open</span>
                                        ) : (
                                            <span className="badge bg-secondary">Closed</span>
                                        )}
                                    </h3>
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4">
                            <div className="card bg-light">
                                <div className="card-body text-center">
                                    <h6 className="text-muted mb-1">Errors</h6>
                                    <h3 className={`mb-0 ${errorCount > 0 ? 'text-danger' : 'text-success'}`}>
                                        {errorCount}
                                    </h3>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="d-flex gap-2">
                        <button
                            className="btn btn-primary"
                            onClick={handleOpenModal}
                        >
                            Open Modal
                        </button>
                        <button
                            className="btn btn-warning"
                            onClick={runStressTest}
                        >
                            Run Stress Test (50 cycles)
                        </button>
                    </div>

                    <div className="alert alert-info mt-3 mb-0">
                        <i className="ph ph-info me-2"></i>
                        <strong>How to use:</strong>
                        <ul className="mb-0 mt-2">
                            <li>Click "Open Modal" to manually test opening/closing</li>
                            <li>Click "Run Stress Test" to automatically open/close 50 times</li>
                            <li>Check console for any DOM manipulation errors</li>
                            <li>Error count should remain at 0 with Portal implementation</li>
                        </ul>
                    </div>
                </div>
            </div>

            {isModalOpen && (
                <SessionEditModal
                    session={mockSession}
                    onSave={handleSave}
                    onClose={handleCloseModal}
                />
            )}
        </div>
    );
};

export default ModalStressTest;
