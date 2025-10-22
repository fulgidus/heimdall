'use client';

import React, { useEffect, useState } from 'react';
import {
    Clock,
    CheckCircle,
    AlertCircle,
    RefreshCw,
    Download,
    Eye,
    Trash2,
    Zap,
} from 'lucide-react';
import { useSessionStore, RecordingSession } from '../store/sessionStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface SessionsListProps {
    onSessionSelect?: (sessionId: number) => void;
    autoRefresh?: boolean;
}

export const SessionsList: React.FC<SessionsListProps> = ({
    onSessionSelect,
    autoRefresh = true,
}) => {
    const { sessions, isLoading, error, fetchSessions } = useSessionStore();
    const [autoRefreshInterval, setAutoRefreshInterval] = useState<NodeJS.Timeout | null>(null);

    // Initial load
    useEffect(() => {
        fetchSessions();
    }, [fetchSessions]);

    // Auto-refresh
    useEffect(() => {
        if (autoRefresh) {
            const interval = setInterval(() => {
                fetchSessions();
            }, 5000); // Refresh every 5 seconds
            setAutoRefreshInterval(interval);
            return () => clearInterval(interval);
        }
    }, [autoRefresh, fetchSessions]);

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'pending':
                return 'text-yellow-400 bg-yellow-900/20';
            case 'processing':
                return 'text-blue-400 bg-blue-900/20';
            case 'completed':
                return 'text-green-400 bg-green-900/20';
            case 'failed':
                return 'text-red-400 bg-red-900/20';
            default:
                return 'text-slate-400 bg-slate-900/20';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'pending':
                return <Clock className="w-4 h-4" />;
            case 'processing':
                return <RefreshCw className="w-4 h-4 animate-spin" />;
            case 'completed':
                return <CheckCircle className="w-4 h-4" />;
            case 'failed':
                return <AlertCircle className="w-4 h-4" />;
            default:
                return <Zap className="w-4 h-4" />;
        }
    };

    const formatDate = (dateString: string | null) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleString();
    };

    const formatDuration = (seconds: number) => {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    };

    return (
        <Card className="bg-slate-900 border-slate-800 w-full">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="text-white">Recording Sessions Queue</CardTitle>
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={() => fetchSessions()}
                        className="border-slate-700 text-slate-300 hover:bg-slate-800"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Refresh
                    </Button>
                </div>
            </CardHeader>
            <CardContent>
                {isLoading && sessions.length === 0 ? (
                    <div className="flex items-center justify-center py-12">
                        <RefreshCw className="w-8 h-8 animate-spin text-purple-500 mr-3" />
                        <p className="text-slate-400">Loading sessions...</p>
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="text-center py-12">
                        <Zap className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                        <p className="text-slate-400">No recording sessions yet</p>
                        <p className="text-slate-500 text-sm">Create one using the form above</p>
                    </div>
                ) : (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        {sessions.map((session) => (
                            <div
                                key={session.id}
                                className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition"
                            >
                                <div className="flex items-start justify-between gap-4">
                                    {/* Left side - Info */}
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-2">
                                            <h3 className="text-white font-semibold truncate">{session.session_name}</h3>
                                            <span
                                                className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold ${getStatusColor(
                                                    session.status
                                                )}`}
                                            >
                                                {getStatusIcon(session.status)}
                                                {session.status.toUpperCase()}
                                            </span>
                                        </div>

                                        <div className="grid grid-cols-2 gap-2 text-xs text-slate-400 mb-2">
                                            <div>
                                                <span className="text-slate-500">Frequency:</span> {session.frequency_mhz.toFixed(3)} MHz
                                            </div>
                                            <div>
                                                <span className="text-slate-500">Duration:</span> {formatDuration(session.duration_seconds)}
                                            </div>
                                            <div>
                                                <span className="text-slate-500">Created:</span> {formatDate(session.created_at)}
                                            </div>
                                            <div>
                                                <span className="text-slate-500">Receivers:</span> {session.websdrs_enabled}
                                            </div>
                                        </div>

                                        {/* Error message if failed */}
                                        {session.status === 'failed' && session.error_message && (
                                            <div className="text-xs text-red-400 bg-red-900/20 px-2 py-1 rounded mt-2">
                                                <strong>Error:</strong> {session.error_message}
                                            </div>
                                        )}
                                    </div>

                                    {/* Right side - Actions */}
                                    <div className="flex gap-2 flex-shrink-0">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => onSessionSelect?.(session.id)}
                                            className="border-slate-700 text-slate-300 hover:bg-slate-700"
                                            title="View details"
                                        >
                                            <Eye className="w-4 h-4" />
                                        </Button>

                                        {session.status === 'completed' && session.minio_path && (
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                className="border-slate-700 text-green-400 hover:bg-slate-700"
                                                title="Download results"
                                            >
                                                <Download className="w-4 h-4" />
                                            </Button>
                                        )}

                                        {session.status === 'pending' && (
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                className="border-slate-700 text-red-400 hover:bg-slate-700"
                                                title="Cancel session"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default SessionsList;
