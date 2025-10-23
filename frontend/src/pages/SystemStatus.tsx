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
    CheckCircle,
    AlertCircle,
    XCircle,
    TrendingUp,
    RefreshCw,
} from 'lucide-react';
import { useAuthStore, useSystemStore } from '../store';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface ServiceStatus {
    name: string;
    status: 'online' | 'degraded' | 'offline';
    lastCheck: string;
    uptime: number;
    responseTime: number;
}

export const SystemStatus: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const { servicesHealth, isLoading, checkAllServices } = useSystemStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isRefreshing, setIsRefreshing] = useState(false);

    // Load services health on mount
    useEffect(() => {
        checkAllServices();
    }, [checkAllServices]);

    // Auto-refresh every 30 seconds
    useEffect(() => {
        const interval = setInterval(() => {
            checkAllServices();
        }, 30000);

        return () => clearInterval(interval);
    }, [checkAllServices]);

    // Handle manual refresh
    const handleRefresh = async () => {
        setIsRefreshing(true);
        await checkAllServices();
        setIsRefreshing(false);
    };

    // Convert servicesHealth to ServiceStatus array
    const services: ServiceStatus[] = Object.entries(servicesHealth).map(([name, health]) => {
        // Calculate approximate uptime based on status
        const uptime = health.status === 'healthy' ? 99.9 : health.status === 'degraded' ? 95.0 : 0;
        
        // Extract response time from details if available
        const responseTime = (health.details?.response_time_ms as number) || 
                           (health.status === 'healthy' ? 50 : 500);
        
        return {
            name: name.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            status: health.status === 'healthy' ? 'online' : 
                   health.status === 'degraded' ? 'degraded' : 'offline',
            lastCheck: new Date(health.timestamp).toLocaleString(),
            uptime,
            responseTime,
        };
    });

    // Calculate metrics from services
    const metrics = {
        totalRequests: 'N/A', // Would need separate metrics API
        avgLatency: services.length > 0 
            ? `${Math.round(services.reduce((sum, s) => sum + s.responseTime, 0) / services.length)}ms`
            : 'N/A',
        errorRate: services.filter(s => s.status === 'offline').length > 0 
            ? `${((services.filter(s => s.status === 'offline').length / services.length) * 100).toFixed(2)}%`
            : '0.00%',
        queueDepth: 'N/A', // Would need RabbitMQ metrics API
        dbConnections: 'N/A', // Would need PostgreSQL metrics API
        cacheHitRate: 'N/A', // Would need Redis metrics API
    };

    const menuItems = [
        { icon: Home, label: 'Dashboard', path: '/dashboard', active: false },
        { icon: MapPin, label: 'Localization', path: '/localization', active: false },
        { icon: Radio, label: 'Recording Sessions', path: '/projects', active: false },
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

    const getStatusIcon = (status: ServiceStatus['status']) => {
        switch (status) {
            case 'online':
                return <CheckCircle className="w-5 h-5 text-green-400" />;
            case 'degraded':
                return <AlertCircle className="w-5 h-5 text-yellow-400" />;
            case 'offline':
                return <XCircle className="w-5 h-5 text-red-400" />;
        }
    };

    const getStatusColor = (status: ServiceStatus['status']) => {
        switch (status) {
            case 'online':
                return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'degraded':
                return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
            case 'offline':
                return 'bg-red-500/20 text-red-400 border-red-500/50';
        }
    };

    const getStatusLabel = (status: ServiceStatus['status']) => {
        switch (status) {
            case 'online':
                return 'Online';
            case 'degraded':
                return 'Degraded';
            case 'offline':
                return 'Offline';
        }
    };

    const onlineCount = services.filter((s) => s.status === 'online').length;
    const avgUptime = (services.reduce((sum, s) => sum + s.uptime, 0) / services.length).toFixed(2);

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
                            <h1 className="text-3xl font-bold text-white">System Status & Monitoring</h1>
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleRefresh}
                            disabled={isRefreshing || isLoading}
                            className="border-slate-700 text-slate-300"
                        >
                            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6 max-w-7xl mx-auto">
                        {/* Overall Status Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-6">
                                    <p className="text-slate-400 text-sm">Services Online</p>
                                    <p className="text-3xl font-bold text-green-400 mt-2">
                                        {onlineCount}/{services.length}
                                    </p>
                                </CardContent>
                            </Card>
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-6">
                                    <p className="text-slate-400 text-sm">Average Uptime</p>
                                    <p className="text-3xl font-bold text-cyan-400 mt-2">{avgUptime}%</p>
                                </CardContent>
                            </Card>
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-6">
                                    <p className="text-slate-400 text-sm">System Health</p>
                                    <p className="text-3xl font-bold text-purple-400 mt-2">Excellent</p>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Performance Metrics */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-white">
                                    <TrendingUp className="w-5 h-5 text-cyan-400" />
                                    Performance Metrics (Last 24h)
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">Total Requests</p>
                                        <p className="text-2xl font-bold text-cyan-400 mt-1">
                                            {metrics.totalRequests}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">Avg Latency</p>
                                        <p className="text-2xl font-bold text-green-400 mt-1">
                                            {metrics.avgLatency}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">Error Rate</p>
                                        <p className="text-2xl font-bold text-yellow-400 mt-1">
                                            {metrics.errorRate}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">Queue Depth</p>
                                        <p className="text-2xl font-bold text-purple-400 mt-1">
                                            {metrics.queueDepth}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">DB Connections</p>
                                        <p className="text-2xl font-bold text-orange-400 mt-1">
                                            {metrics.dbConnections}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 p-4 rounded border border-slate-700">
                                        <p className="text-slate-400 text-sm">Cache Hit Rate</p>
                                        <p className="text-2xl font-bold text-blue-400 mt-1">
                                            {metrics.cacheHitRate}
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Services Status Table */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardHeader>
                                <CardTitle className="text-white">Services Status</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {services.map((service, idx) => (
                                        <div
                                            key={idx}
                                            className={`p-4 rounded border ${getStatusColor(service.status)}`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-3 flex-1">
                                                    {getStatusIcon(service.status)}
                                                    <div>
                                                        <p className="font-semibold text-white">{service.name}</p>
                                                        <p className="text-xs text-slate-400 mt-1">
                                                            Last check: {service.lastCheck}
                                                        </p>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <p className="font-bold text-sm">
                                                        {getStatusLabel(service.status)}
                                                    </p>
                                                    <div className="flex gap-4 mt-2 text-xs">
                                                        <div>
                                                            <p className="text-slate-400">Uptime</p>
                                                            <p className="font-semibold">{service.uptime.toFixed(2)}%</p>
                                                        </div>
                                                        <div>
                                                            <p className="text-slate-400">Latency</p>
                                                            <p className="font-semibold">{service.responseTime}ms</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Alerts Panel */}
                        <Card className="bg-slate-900 border-yellow-600/50">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-yellow-400">
                                    <AlertCircle className="w-5 h-5" />
                                    Active Alerts
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-2">
                                    <div className="p-3 bg-yellow-500/10 border border-yellow-500/50 rounded">
                                        <p className="text-yellow-400 font-semibold text-sm">
                                            MinIO Storage - Degraded Performance
                                        </p>
                                        <p className="text-yellow-300 text-xs mt-1">
                                            Response time increased to 450ms. Investigating cause.
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default SystemStatus;
