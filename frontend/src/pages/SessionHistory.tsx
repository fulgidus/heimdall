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
    Check,
    Trash2,
    X as XIcon,
    RefreshCw,
    AlertCircle,
} from 'lucide-react';
import { useAuthStore, useSessionStore } from '../store';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Alert, AlertDescription } from '@/components/ui/alert';

export const SessionHistory: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const {
        sessions,
        analytics,
        isLoading,
        error,
        fetchSessions,
        fetchAnalytics,
        approveSession,
        rejectSession,
        deleteSession,
        setStatusFilter,
        totalSessions,
    } = useSessionStore();
    
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [filterStatus, setFilterStatus] = useState('all');
    const [searchTerm, setSearchTerm] = useState('');
    const [isRefreshing, setIsRefreshing] = useState(false);

    // Load sessions and analytics on mount
    useEffect(() => {
        fetchSessions();
        fetchAnalytics();
    }, [fetchSessions, fetchAnalytics]);

    // Handle refresh
    const handleRefresh = async () => {
        setIsRefreshing(true);
        await Promise.all([fetchSessions(), fetchAnalytics()]);
        setIsRefreshing(false);
    };

    // Handle status filter change
    const handleStatusFilterChange = (status: string) => {
        setFilterStatus(status);
        if (status === 'all') {
            setStatusFilter(null);
        } else {
            setStatusFilter(status);
        }
    };

    // Handle approval
    const handleApprove = async (id: string) => {
        try {
            await approveSession(id);
        } catch (error) {
            console.error('Failed to approve session:', error);
        }
    };

    // Handle rejection
    const handleReject = async (id: string) => {
        try {
            await rejectSession(id);
        } catch (error) {
            console.error('Failed to reject session:', error);
        }
    };

    // Handle delete
    const handleDelete = async (id: string) => {
        if (confirm('Are you sure you want to delete this session?')) {
            try {
                await deleteSession(id);
            } catch (error) {
                console.error('Failed to delete session:', error);
            }
        }
    };

    // Filter sessions by search term
    const filteredSessions = sessions.filter((session) =>
        session.session_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (session.source_frequency / 1000000).toFixed(3).includes(searchTerm)
    );

    const menuItems = [
        { icon: Home, label: 'Dashboard', path: '/dashboard', active: false },
        { icon: MapPin, label: 'Localization', path: '/localization', active: false },
        { icon: Radio, label: 'Recording Sessions', path: '/history', active: true },
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

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'completed':
                return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'pending':
            case 'in_progress':
                return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
            case 'failed':
                return 'bg-red-500/20 text-red-400 border-red-500/50';
            default:
                return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
        }
    };

    const getStatusLabel = (status: string) => {
        switch (status) {
            case 'completed':
                return '✓ Completed';
            case 'pending':
                return '⏳ Pending';
            case 'in_progress':
                return '▶ In Progress';
            case 'failed':
                return '✗ Failed';
            default:
                return status;
        }
    };

    const formatFrequency = (freqHz: number) => {
        return (freqHz / 1000000).toFixed(3) + ' MHz';
    };

    const formatDuration = (seconds?: number) => {
        if (!seconds) return 'N/A';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    };

    return (
        <div className="flex h-screen w-screen bg-slate-950">
            {/* Sidebar */}
            <aside
                className={`${sidebarOpen ? 'w-64' : 'w-0'} 
                bg-linear-to-b from-slate-900 to-slate-950 border-r border-slate-800 
                transition-all duration-300 overflow-hidden flex flex-col`}
            >
                <div className="p-6 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <Radar className="w-8 h-8 text-purple-500" />
                        <h1 className="text-xl font-bold text-white">Heimdall</h1>
                    </div>
                </div>

                <nav className="flex-1 px-4 py-6 flex flex-col gap-2 overflow-y-auto">
                    {menuItems.map((item, idx) => (
                        <button
                            key={idx}
                            onClick={() => handleNavigation(item.path)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                                item.active
                                    ? 'bg-purple-600/20 text-purple-400 border-l-2 border-purple-500'
                                    : 'text-slate-300 hover:bg-slate-800/50'
                            }`}
                        >
                            <item.icon className="w-5 h-5" />
                            <span className="font-medium">{item.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="p-4 border-t border-slate-800">
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="w-full justify-start text-slate-300">
                                <div className="w-8 h-8 rounded-full bg-linear-to-br from-purple-500 to-pink-500 flex items-center justify-center text-sm font-bold">
                                    AD
                                </div>
                                <span className="ml-2 text-sm">admin</span>
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
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
                            <h1 className="text-3xl font-bold text-white">Session History</h1>
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleRefresh}
                            disabled={isRefreshing}
                            className="border-slate-700 text-slate-300"
                        >
                            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                    </div>
                </header>

                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6 max-w-7xl mx-auto">
                        {/* Error Alert */}
                        {error && (
                            <Alert className="bg-red-900/20 border-red-800">
                                <AlertCircle className="h-4 w-4 text-red-500" />
                                <AlertDescription className="text-red-300">{error}</AlertDescription>
                            </Alert>
                        )}

                        {/* Analytics Cards */}
                        {analytics && (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6">
                                        <p className="text-slate-400 text-sm">Total Sessions</p>
                                        <p className="text-3xl font-bold text-white mt-2">
                                            {analytics.total_sessions}
                                        </p>
                                    </CardContent>
                                </Card>
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6">
                                        <p className="text-slate-400 text-sm">Completed</p>
                                        <p className="text-3xl font-bold text-green-400 mt-2">
                                            {analytics.completed_sessions}
                                        </p>
                                    </CardContent>
                                </Card>
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6">
                                        <p className="text-slate-400 text-sm">Success Rate</p>
                                        <p className="text-3xl font-bold text-cyan-400 mt-2">
                                            {analytics.success_rate.toFixed(1)}%
                                        </p>
                                    </CardContent>
                                </Card>
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6">
                                        <p className="text-slate-400 text-sm">Measurements</p>
                                        <p className="text-3xl font-bold text-purple-400 mt-2">
                                            {analytics.total_measurements}
                                        </p>
                                    </CardContent>
                                </Card>
                            </div>
                        )}

                        {/* Filters */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardContent className="p-4 flex gap-4 items-end">
                                <div className="flex-1">
                                    <label className="text-sm text-slate-400 mb-2 block">Search</label>
                                    <Input
                                        placeholder="Search by name or frequency..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="bg-slate-800 border-slate-700"
                                    />
                                </div>
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button variant="outline" className="border-slate-700">
                                            <span className="mr-2">Filter: {filterStatus}</span>
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent>
                                        <DropdownMenuItem onClick={() => handleStatusFilterChange('all')}>
                                            All
                                        </DropdownMenuItem>
                                        <DropdownMenuItem onClick={() => handleStatusFilterChange('completed')}>
                                            Completed
                                        </DropdownMenuItem>
                                        <DropdownMenuItem onClick={() => handleStatusFilterChange('pending')}>
                                            Pending
                                        </DropdownMenuItem>
                                        <DropdownMenuItem onClick={() => handleStatusFilterChange('failed')}>
                                            Failed
                                        </DropdownMenuItem>
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </CardContent>
                        </Card>

                        {/* Sessions Table */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardHeader>
                                <CardTitle className="text-white">Recording Sessions</CardTitle>
                            </CardHeader>
                            <CardContent>
                                {isLoading && filteredSessions.length === 0 ? (
                                    <div className="text-center py-12">
                                        <RefreshCw className="w-12 h-12 mx-auto mb-4 text-purple-500 animate-spin" />
                                        <p className="text-slate-400">Loading sessions...</p>
                                    </div>
                                ) : filteredSessions.length === 0 ? (
                                    <div className="text-center py-12">
                                        <p className="text-slate-400">No sessions found</p>
                                    </div>
                                ) : (
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead className="border-b border-slate-700">
                                                <tr>
                                                    <th className="text-left py-3 px-4 text-slate-400">Session</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Frequency</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Date</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Duration</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Status</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Approval</th>
                                                    <th className="text-left py-3 px-4 text-slate-400">Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-800">
                                                {filteredSessions.map((session) => (
                                                    <tr key={session.id} className="hover:bg-slate-800/50">
                                                        <td className="py-3 px-4">
                                                            <div>
                                                                <p className="text-white font-medium">
                                                                    {session.session_name}
                                                                </p>
                                                                <p className="text-slate-500 text-xs">
                                                                    {session.source_name}
                                                                </p>
                                                            </div>
                                                        </td>
                                                        <td className="py-3 px-4 text-cyan-400">
                                                            {formatFrequency(session.source_frequency)}
                                                        </td>
                                                        <td className="py-3 px-4 text-slate-300">
                                                            {new Date(session.session_start).toLocaleString()}
                                                        </td>
                                                        <td className="py-3 px-4 text-slate-300">
                                                            {formatDuration(session.duration_seconds)}
                                                        </td>
                                                        <td className="py-3 px-4">
                                                            <span
                                                                className={`px-2 py-1 rounded text-xs font-semibold border ${getStatusBadge(
                                                                    session.status
                                                                )}`}
                                                            >
                                                                {getStatusLabel(session.status)}
                                                            </span>
                                                        </td>
                                                        <td className="py-3 px-4">
                                                            <span
                                                                className={`px-2 py-1 rounded text-xs font-semibold border ${getStatusBadge(
                                                                    session.approval_status
                                                                )}`}
                                                            >
                                                                {session.approval_status}
                                                            </span>
                                                        </td>
                                                        <td className="py-3 px-4">
                                                            <div className="flex gap-2">
                                                                <Button
                                                                    size="sm"
                                                                    variant="ghost"
                                                                    className="text-green-400 hover:text-green-300"
                                                                    onClick={() => handleApprove(session.id)}
                                                                    disabled={session.approval_status === 'approved'}
                                                                >
                                                                    <Check className="w-4 h-4" />
                                                                </Button>
                                                                <Button
                                                                    size="sm"
                                                                    variant="ghost"
                                                                    className="text-red-400 hover:text-red-300"
                                                                    onClick={() => handleReject(session.id)}
                                                                    disabled={session.approval_status === 'rejected'}
                                                                >
                                                                    <XIcon className="w-4 h-4" />
                                                                </Button>
                                                                <Button
                                                                    size="sm"
                                                                    variant="ghost"
                                                                    className="text-slate-400 hover:text-slate-300"
                                                                    onClick={() => handleDelete(session.id)}
                                                                >
                                                                    <Trash2 className="w-4 h-4" />
                                                                </Button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Pagination Info */}
                        {totalSessions > 0 && (
                            <div className="text-center text-slate-400 text-sm">
                                Showing {filteredSessions.length} of {totalSessions} sessions
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default SessionHistory;
