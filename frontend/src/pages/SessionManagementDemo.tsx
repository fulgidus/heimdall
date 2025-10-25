/**
 * Session Management Demo Page
 * Demonstrates the new session management UI components
 */

import React from 'react';
import { SessionListEnhanced } from '@/components';

const SessionManagementDemo: React.FC = () => {
    return (
        <div className="min-h-screen bg-slate-950 p-8">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">
                        Session Management Interface
                    </h1>
                    <p className="text-slate-400">
                        View, filter, and approve recording sessions for ML training pipeline
                    </p>
                </div>

                {/* Session List Component */}
                <SessionListEnhanced autoRefresh={false} />
            </div>
        </div>
    );
};

export default SessionManagementDemo;
