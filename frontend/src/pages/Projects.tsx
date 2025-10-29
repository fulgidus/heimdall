'use client';

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LogOut,
    Home,
    MapPin,
    Radio,
    BarChart3,
    Zap,
    Radar,
    Menu,
    X,
    Plus,
    Square,
    Trash2,
    Clock,
    Loader,
    AlertCircle,
} from 'lucide-react';
import { useAuthStore } from '../store';
import { useSessionStore } from '../store/sessionStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

export const Projects: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const { sessions, isLoading, error, fetchSessions, createSession, deleteSession } = useSessionStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [newSessionName, setNewSessionName] = useState('');
    const [newSessionFrequency, setNewSessionFrequency] = useState('145.5');
    const [newSessionDuration, setNewSessionDuration] = useState('10');
    const [showNewSessionForm, setShowNewSessionForm] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [deletingSessionId, setDeletingSessionId] = useState<number | null>(null);

    // Load sessions on mount and set up polling
    useEffect(() => {
        fetchSessions();
        const interval = setInterval(() => fetchSessions(), 5000);
        return () => clearInterval(interval);
    }, [fetchSessions]);

    const handleCreateSession = async () => {
        if (!newSessionName.trim()) {
            alert('Session name required');
            return;
        }
        setSubmitting(true);
        try {
            await createSession({
                known_source_id: '00000000-0000-0000-0000-000000000000', // Default unknown source
                session_name: newSessionName,
                frequency_hz: parseFloat(newSessionFrequency) * 1_000_000, // Convert MHz to Hz
                duration_seconds: parseInt(newSessionDuration),
            });
            setNewSessionName('');
            setNewSessionFrequency('145.5');
            setNewSessionDuration('10');
            setShowNewSessionForm(false);
            await fetchSessions();
        } catch (err) {
            alert('Failed to create session: ' + (err instanceof Error ? err.message : 'Unknown error'));
        } finally {
            setSubmitting(false);
        }
    };

    const handleDeleteSession = async (sessionId: number, sessionName: string) => {
        const confirm = window.confirm(
            `Are you sure you want to delete "${sessionName}"?\n\nThis action cannot be undone.`
        );

        if (!confirm) return;

        setDeletingSessionId(sessionId);
        try {
            await deleteSession(sessionId);
            await fetchSessions();
        } catch (err) {
            alert('Failed to delete session: ' + (err instanceof Error ? err.message : 'Unknown error'));
        } finally {
            setDeletingSessionId(null);
        }
    };

    const getStatusColor = (status: 'pending' | 'in_progress' | 'processing' | 'completed' | 'failed') => {
        switch (status) {
            case 'processing':
            case 'pending':
                return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'completed':
                return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
            case 'failed':
                return 'bg-red-500/20 text-red-400 border-red-500/50';
            default:
                return 'bg-slate-700/20 text-slate-400 border-slate-700/50';
        }
    };

    const getStatusLabel = (status: 'pending' | 'in_progress' | 'processing' | 'completed' | 'failed') => {
        switch (status) {
            case 'processing':
            case 'in_progress':
                return 'Processing';
            case 'pending':
                return 'Pending';
            case 'completed':
                return 'Completed';
            case 'failed':
                return 'Failed';
            default:
                return status;
        }
    };

    const menuItems = [
        { icon: Home, label: 'Dashboard', path: '/dashboard', active: false },
        { icon: MapPin, label: 'Localization', path: '/localization', active: false },
        { icon: Radio, label: 'Recording Sessions', path: '/projects', active: true },
        { icon: BarChart3, label: 'Analytics', path: '/analytics', active: false },
        { icon: Zap, label: 'Settings', path: '/settings', active: false },
    ];

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const handleNavigation = (path: string) => {
        navigate(path);
        setSidebarOpen(false);
    };

    // Organize sessions by status
    const activeSessions = sessions.filter((s) => s.status === 'processing' || s.status === 'pending');
    const completedSessions = sessions.filter((s) => s.status === 'completed');
    const failedSessions = sessions.filter((s) => s.status === 'failed');

    const formatFrequency = (freq: number) => `${freq.toFixed(3)} MHz`;
    const formatDuration = (seconds: number) => `${seconds} sec`;
    const formatDateTime = (dateStr: string | null | undefined) => {
        if (!dateStr) return 'Not started';
        const date = new Date(dateStr);
        return date.toLocaleString('it-IT');
    };

    return (
        <div className="flex h-screen w-screen bg-slate-950">
            {/* Sidebar */}
            <aside
                className={`${sidebarOpen ? 'w-64' : 'w-0'} 
                bg-linear-to-b from-slate-900 to-slate-950 border-r border-slate-800 
                transition-all duration-300 overflow-hidden flex flex-col`}
            >
                {/* Logo Section */}
                <div className="p-6 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <Radar className="w-8 h-8 text-purple-500" />
                        <h1 className="text-xl font-bold text-white">Heimdall</h1>
                    </div>
                </div>

                {/* Menu Items */}
                <nav className="flex-1 px-4 py-6 flex flex-col gap-2 overflow-y-auto">
                    {menuItems.map((item, idx) => {
                        const Icon = item.icon;
                        return (
                            <button
                                key={idx}
                                onClick={() => handleNavigation(item.path)}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${item.active
                                    ? 'bg-purple-600/20 text-purple-400 border-l-2 border-purple-500'
                                    : 'text-slate-300 hover:bg-slate-800/50'
                                    }`}
                            >
                                <Icon className="w-5 h-5" />
                                <span className="font-medium">{item.label}</span>
                            </button>
                        );
                    })}
                </nav>

                {/* User Section */}
                <div className="p-4 border-t border-slate-800">
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button
                                variant="ghost"
                                className="w-full justify-start text-slate-300 hover:bg-slate-800"
                            >
                                <div className="w-8 h-8 rounded-full bg-linear-to-br from-purple-500 to-pink-500 flex items-center justify-center text-sm font-bold">
                                    AD
                                </div>
                                <span className="ml-2 text-sm">admin</span>
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-48">
                            <DropdownMenuItem onClick={() => handleNavigation('/profile')}>
                                Profile
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleNavigation('/settings')}>
                                Settings
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={handleLogout} className="text-red-400">
                                <LogOut className="w-4 h-4 mr-2" />
                                Logout
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-auto flex flex-col">
                {/* Header */}
                <header className="bg-slate-900 border-b border-slate-800 p-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => setSidebarOpen(!sidebarOpen)}
                                className="text-slate-400 hover:text-white"
                            >
                                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                            </Button>
                            <h1 className="text-3xl font-bold text-white">Recording Sessions</h1>
                        </div>
                        <Button
                            onClick={() => setShowNewSessionForm(!showNewSessionForm)}
                            className="bg-purple-600 hover:bg-purple-700 text-white"
                        >
                            <Plus className="w-4 h-4 mr-2" />
                            New Session
                        </Button>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6">
                        {/* Error Display */}
                        {error && (
                            <Card className="bg-red-500/10 border-red-500/50">
                                <CardContent className="p-4 flex items-center gap-3 text-red-400">
                                    <AlertCircle className="w-5 h-5 shrink-0" />
                                    <div>
                                        <p className="font-semibold">Error Loading Sessions</p>
                                        <p className="text-sm">{error}</p>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* New Session Form */}
                        {showNewSessionForm && (
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-6">
                                    <h3 className="text-white font-bold text-lg mb-4">Create New Session</h3>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-slate-300 text-sm font-medium mb-2">
                                                Session Name
                                            </label>
                                            <input
                                                type="text"
                                                value={newSessionName}
                                                onChange={(e) => setNewSessionName(e.target.value)}
                                                placeholder="e.g., Session Alpha - 2m Band"
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white placeholder-slate-500"
                                            />
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="block text-slate-300 text-sm font-medium mb-2">
                                                    Frequency (MHz)
                                                </label>
                                                <input
                                                    type="number"
                                                    step="0.001"
                                                    value={newSessionFrequency}
                                                    onChange={(e) => setNewSessionFrequency(e.target.value)}
                                                    placeholder="145.500"
                                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white placeholder-slate-500"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-slate-300 text-sm font-medium mb-2">
                                                    Duration (seconds)
                                                </label>
                                                <input
                                                    type="number"
                                                    value={newSessionDuration}
                                                    onChange={(e) => setNewSessionDuration(e.target.value)}
                                                    placeholder="10"
                                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white placeholder-slate-500"
                                                />
                                            </div>
                                        </div>
                                        <div className="flex gap-3 justify-end">
                                            <Button
                                                variant="outline"
                                                onClick={() => setShowNewSessionForm(false)}
                                                className="border-slate-700"
                                            >
                                                Cancel
                                            </Button>
                                            <Button
                                                onClick={handleCreateSession}
                                                disabled={submitting}
                                                className="bg-purple-600 hover:bg-purple-700 text-white"
                                            >
                                                {submitting ? (
                                                    <>
                                                        <Loader className="w-4 h-4 mr-2 animate-spin" />
                                                        Creating...
                                                    </>
                                                ) : (
                                                    'Create Session'
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Loading State */}
                        {isLoading && sessions.length === 0 && (
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-12 text-center">
                                    <Loader className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
                                    <p className="text-slate-300">Loading sessions...</p>
                                </CardContent>
                            </Card>
                        )}

                        {/* Active Sessions */}
                        <div className="mb-6">
                            <h2 className="text-xl font-bold text-white mb-4">
                                Active Sessions ({activeSessions.length})
                            </h2>
                            {activeSessions.length > 0 ? (
                                <div className="space-y-3">
                                    {activeSessions.map((session) => (
                                        <Card
                                            key={session.id}
                                            className="bg-slate-900 border-slate-800"
                                        >
                                            <CardContent className="p-6">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-3">
                                                            {session.status === 'processing' && (
                                                                <Loader className="w-5 h-5 text-green-400 animate-spin shrink-0" />
                                                            )}
                                                            <div>
                                                                <h3 className="text-white font-bold text-lg">
                                                                    {session.session_name}
                                                                </h3>
                                                                <p className="text-slate-400 text-sm">
                                                                    {formatFrequency(session.frequency_mhz)} • {session.websdrs_enabled} receivers
                                                                </p>
                                                                <div className="flex items-center gap-4 mt-3">
                                                                    <div className="flex items-center gap-2">
                                                                        <Clock className="w-4 h-4 text-cyan-500" />
                                                                        <span className="text-slate-300 text-sm">
                                                                            Started: {formatDateTime(session.started_at)}
                                                                        </span>
                                                                    </div>
                                                                    <span
                                                                        className={`px-3 py-1 rounded text-sm font-bold border ${getStatusColor(
                                                                            session.status
                                                                        )}`}
                                                                    >
                                                                        {getStatusLabel(session.status)}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="flex gap-2">
                                                        {session.status === 'pending' && (
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                className="border-red-500/50 text-red-400 hover:text-red-300"
                                                                onClick={() => handleDeleteSession(session.id, session.session_name)}
                                                                disabled={deletingSessionId === session.id}
                                                            >
                                                                {deletingSessionId === session.id ? (
                                                                    <Loader className="w-4 h-4 animate-spin" />
                                                                ) : (
                                                                    <Trash2 className="w-4 h-4" />
                                                                )}
                                                            </Button>
                                                        )}
                                                        {session.status === 'processing' && (
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                className="border-slate-700"
                                                                disabled
                                                            >
                                                                <Square className="w-4 h-4" />
                                                            </Button>
                                                        )}
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            ) : (
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6 text-center">
                                        <p className="text-slate-400">
                                            No active sessions. Create a new session to begin recording.
                                        </p>
                                    </CardContent>
                                </Card>
                            )}
                        </div>

                        {/* Completed Sessions */}
                        {completedSessions.length > 0 && (
                            <div className="mb-6">
                                <h2 className="text-xl font-bold text-white mb-4">
                                    Completed Sessions ({completedSessions.length})
                                </h2>
                                <div className="space-y-3">
                                    {completedSessions.map((session) => (
                                        <Card
                                            key={session.id}
                                            className="bg-slate-900 border-slate-800 hover:border-slate-700 cursor-pointer transition-colors"
                                        >
                                            <CardContent className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex-1">
                                                        <h3 className="text-white font-semibold">
                                                            {session.session_name}
                                                        </h3>
                                                        <div className="flex items-center gap-4 mt-2 text-sm text-slate-400">
                                                            <span>{formatFrequency(session.frequency_mhz)}</span>
                                                            <span>{session.websdrs_enabled} receivers</span>
                                                            <span>{formatDuration(session.duration_seconds)}</span>
                                                            <span>
                                                                Completed: {formatDateTime(session.completed_at)}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div className="flex gap-2">
                                                        <Button
                                                            size="sm"
                                                            variant="outline"
                                                            className="border-slate-700 text-red-400 hover:text-red-300"
                                                            onClick={() => handleDeleteSession(session.id, session.session_name)}
                                                            disabled={deletingSessionId === session.id}
                                                        >
                                                            {deletingSessionId === session.id ? (
                                                                <Loader className="w-4 h-4 animate-spin" />
                                                            ) : (
                                                                <Trash2 className="w-4 h-4" />
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Failed Sessions */}
                        {failedSessions.length > 0 && (
                            <div className="mb-6">
                                <h2 className="text-xl font-bold text-white mb-4">
                                    Failed Sessions ({failedSessions.length})
                                </h2>
                                <div className="space-y-3">
                                    {failedSessions.map((session) => (
                                        <Card
                                            key={session.id}
                                            className="bg-red-500/10 border-red-500/50"
                                        >
                                            <CardContent className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex-1">
                                                        <h3 className="text-white font-semibold">
                                                            {session.session_name}
                                                        </h3>
                                                        <p className="text-red-400 text-sm mt-1">
                                                            Error: {session.error_message || 'Unknown error'}
                                                        </p>
                                                        <div className="flex items-center gap-4 mt-2 text-sm text-slate-400">
                                                            <span>{formatFrequency(session.frequency_mhz)}</span>
                                                            <span>{formatDateTime(session.completed_at)}</span>
                                                        </div>
                                                    </div>
                                                    <div className="flex gap-2">
                                                        <Button
                                                            size="sm"
                                                            variant="outline"
                                                            className="border-red-500/50 text-red-400 hover:text-red-300"
                                                            onClick={() => handleDeleteSession(session.id, session.session_name)}
                                                            disabled={deletingSessionId === session.id}
                                                        >
                                                            {deletingSessionId === session.id ? (
                                                                <Loader className="w-4 h-4 animate-spin" />
                                                            ) : (
                                                                <Trash2 className="w-4 h-4" />
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Empty State */}
                        {!isLoading &&
                            sessions.length === 0 &&
                            !error && (
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-12 text-center">
                                        <Radio className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                                        <p className="text-slate-400 mb-4">
                                            No recording sessions yet
                                        </p>
                                        <Button
                                            onClick={() => setShowNewSessionForm(true)}
                                            className="bg-purple-600 hover:bg-purple-700 text-white"
                                        >
                                            <Plus className="w-4 h-4 mr-2" />
                                            Create First Session
                                        </Button>
                                    </CardContent>
                                </Card>
                            )}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Projects;
