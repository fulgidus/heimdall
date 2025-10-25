/**
 * Enhanced Session List with Approval Management
 * Displays sessions with approval status badges, filters, and pagination
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
    Clock,
    CheckCircle,
    AlertCircle,
    RefreshCw,
    Eye,
    Zap,
    XCircle,
    Filter,
    ChevronLeft,
    ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useSessions } from '@/hooks/useSessions';
import { SessionDetailModal } from './SessionDetailModal';

interface SessionListEnhancedProps {
    autoRefresh?: boolean;
}

export const SessionListEnhanced: React.FC<SessionListEnhancedProps> = ({
    autoRefresh = true,
}) => {
    const {
        sessions,
        isLoading,
        statusFilter,
        approvalFilter,
        currentPage,
        totalPages,
        totalSessions,
        fetchSessions,
        setStatusFilter,
        setApprovalFilter,
        nextPage,
        previousPage,
    } = useSessions();

    const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
    const [showDetailModal, setShowDetailModal] = useState(false);

    // Initial load
    useEffect(() => {
        fetchSessions();
    }, [fetchSessions]);

    // Auto-refresh
    useEffect(() => {
        if (autoRefresh) {
            const interval = setInterval(() => {
                fetchSessions();
            }, 10000); // Refresh every 10 seconds
            return () => clearInterval(interval);
        }
    }, [autoRefresh, fetchSessions]);

    const handleSessionClick = (sessionId: number) => {
        setSelectedSessionId(sessionId);
        setShowDetailModal(true);
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'pending':
                return 'text-yellow-400 bg-yellow-900/20';
            case 'in_progress':
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
            case 'in_progress':
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

    const getApprovalBadge = (approvalStatus?: string) => {
        switch (approvalStatus) {
            case 'approved':
                return (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-green-900/20 text-green-400">
                        <CheckCircle className="w-3 h-3" />
                        APPROVED
                    </span>
                );
            case 'rejected':
                return (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-red-900/20 text-red-400">
                        <XCircle className="w-3 h-3" />
                        REJECTED
                    </span>
                );
            default:
                return (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold bg-yellow-900/20 text-yellow-400">
                        <Clock className="w-3 h-3" />
                        PENDING
                    </span>
                );
        }
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleString();
    };

    const formatDuration = (seconds?: number | null) => {
        if (!seconds) return 'N/A';
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    };

    return (
        <>
            <Card className="bg-slate-900 border-slate-800 w-full">
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <CardTitle className="text-white">Recording Sessions</CardTitle>
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

                    {/* Filters */}
                    <div className="flex flex-wrap gap-2 mt-4">
                        <div className="flex items-center gap-2">
                            <Filter className="w-4 h-4 text-slate-400" />
                            <span className="text-sm text-slate-400">Status:</span>
                            <Button
                                size="sm"
                                variant={statusFilter === null ? 'default' : 'outline'}
                                onClick={() => setStatusFilter(null)}
                                className="h-7 text-xs"
                            >
                                All
                            </Button>
                            <Button
                                size="sm"
                                variant={statusFilter === 'completed' ? 'default' : 'outline'}
                                onClick={() => setStatusFilter('completed')}
                                className="h-7 text-xs"
                            >
                                Completed
                            </Button>
                            <Button
                                size="sm"
                                variant={statusFilter === 'pending' ? 'default' : 'outline'}
                                onClick={() => setStatusFilter('pending')}
                                className="h-7 text-xs"
                            >
                                Pending
                            </Button>
                            <Button
                                size="sm"
                                variant={statusFilter === 'failed' ? 'default' : 'outline'}
                                onClick={() => setStatusFilter('failed')}
                                className="h-7 text-xs"
                            >
                                Failed
                            </Button>
                        </div>

                        <div className="flex items-center gap-2 ml-4">
                            <span className="text-sm text-slate-400">Approval:</span>
                            <Button
                                size="sm"
                                variant={approvalFilter === null ? 'default' : 'outline'}
                                onClick={() => setApprovalFilter(null)}
                                className="h-7 text-xs"
                            >
                                All
                            </Button>
                            <Button
                                size="sm"
                                variant={approvalFilter === 'pending' ? 'default' : 'outline'}
                                onClick={() => setApprovalFilter('pending')}
                                className="h-7 text-xs"
                            >
                                Pending
                            </Button>
                            <Button
                                size="sm"
                                variant={approvalFilter === 'approved' ? 'default' : 'outline'}
                                onClick={() => setApprovalFilter('approved')}
                                className="h-7 text-xs"
                            >
                                Approved
                            </Button>
                            <Button
                                size="sm"
                                variant={approvalFilter === 'rejected' ? 'default' : 'outline'}
                                onClick={() => setApprovalFilter('rejected')}
                                className="h-7 text-xs"
                            >
                                Rejected
                            </Button>
                        </div>
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
                            <p className="text-slate-400">No sessions found</p>
                            <p className="text-slate-500 text-sm">
                                {statusFilter || approvalFilter
                                    ? 'Try adjusting your filters'
                                    : 'Create one using the form above'}
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* Session List */}
                            <div className="space-y-3">
                                {sessions.map((session) => (
                                    <div
                                        key={session.id}
                                        className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition cursor-pointer"
                                        onClick={() => handleSessionClick(session.id)}
                                    >
                                        <div className="flex items-start justify-between gap-4">
                                            {/* Left side - Info */}
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2 mb-2 flex-wrap">
                                                    <h3 className="text-white font-semibold truncate">
                                                        {session.session_name}
                                                    </h3>
                                                    <span
                                                        className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold ${getStatusColor(
                                                            session.status
                                                        )}`}
                                                    >
                                                        {getStatusIcon(session.status)}
                                                        {session.status.toUpperCase()}
                                                    </span>
                                                    {getApprovalBadge(session.approval_status)}
                                                </div>

                                                <div className="grid grid-cols-2 gap-2 text-xs text-slate-400 mb-2">
                                                    <div>
                                                        <span className="text-slate-500">Source:</span>{' '}
                                                        {session.source_name || 'N/A'}
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500">Frequency:</span>{' '}
                                                        {session.source_frequency
                                                            ? `${(session.source_frequency / 1e6).toFixed(3)} MHz`
                                                            : 'N/A'}
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500">Duration:</span>{' '}
                                                        {formatDuration(session.duration_seconds)}
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500">Created:</span>{' '}
                                                        {formatDate(session.created_at)}
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500">Measurements:</span>{' '}
                                                        {session.measurements_count || 0}
                                                    </div>
                                                </div>

                                                {/* Notes preview */}
                                                {session.notes && (
                                                    <div className="text-xs text-slate-500 italic truncate">
                                                        {session.notes}
                                                    </div>
                                                )}
                                            </div>

                                            {/* Right side - Actions */}
                                            <div className="flex gap-2 shrink-0">
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleSessionClick(session.id);
                                                    }}
                                                    className="border-slate-700 text-slate-300 hover:bg-slate-700"
                                                    title="View details"
                                                >
                                                    <Eye className="w-4 h-4" />
                                                </Button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Pagination */}
                            {totalPages > 1 && (
                                <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-700">
                                    <div className="text-sm text-slate-400">
                                        Page {currentPage} of {totalPages} ({totalSessions} total)
                                    </div>
                                    <div className="flex gap-2">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={previousPage}
                                            disabled={currentPage === 1}
                                            className="border-slate-700"
                                        >
                                            <ChevronLeft className="w-4 h-4" />
                                            Previous
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={nextPage}
                                            disabled={currentPage === totalPages}
                                            className="border-slate-700"
                                        >
                                            Next
                                            <ChevronRight className="w-4 h-4" />
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </CardContent>
            </Card>

            {/* Session Detail Modal */}
            <SessionDetailModal
                isOpen={showDetailModal}
                onClose={() => {
                    setShowDetailModal(false);
                    setSelectedSessionId(null);
                }}
                sessionId={selectedSessionId}
            />
        </>
    );
};

export default SessionListEnhanced;
