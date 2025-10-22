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
    Edit2,
    Wifi,
    Zap as Activity,
    RefreshCw,
    AlertCircle,
} from 'lucide-react';
import { useAuthStore } from '../store';
import { useWebSDRStore } from '../store/websdrStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface DisplayWebSDR {
    id: number;
    name: string;
    url: string;
    location: string;
    latitude: number;
    longitude: number;
    status: 'online' | 'offline' | 'unknown';
    lastContact: string;
    uptime: number;
    avgSnr: number;
    enabled: boolean;
}

export const WebSDRManagement: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const {
        websdrs,
        healthStatus,
        isLoading,
        error,
        fetchWebSDRs,
        checkHealth,
        refreshAll,
    } = useWebSDRStore();

    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [editingId, setEditingId] = useState<number | null>(null);
    const [isRefreshing, setIsRefreshing] = useState(false);

    // Load WebSDRs and health status on mount
    useEffect(() => {
        const loadData = async () => {
            await fetchWebSDRs();
            await checkHealth();
        };
        loadData();
    }, [fetchWebSDRs, checkHealth]);

    // Periodically refresh health status (every 30 seconds)
    useEffect(() => {
        const interval = setInterval(() => {
            checkHealth();
        }, 30000); // 30 seconds

        return () => clearInterval(interval);
    }, [checkHealth]);

    // Handle manual refresh
    const handleRefresh = async () => {
        setIsRefreshing(true);
        try {
            await refreshAll();
        } finally {
            setIsRefreshing(false);
        }
    };

    // Convert backend data to display format
    const webSdrs: DisplayWebSDR[] = websdrs.map((ws) => {
        const health = healthStatus[ws.id];
        return {
            id: ws.id,
            name: ws.name,
            url: ws.url,
            location: ws.location_name,
            latitude: ws.latitude,
            longitude: ws.longitude,
            status: health?.status || 'unknown',
            lastContact: health?.last_check || 'Never',
            uptime: 0, // TODO: Calculate from historical data
            avgSnr: 0, // TODO: Calculate from measurements
            enabled: ws.is_active,
        };
    });


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

    const onlineCount = webSdrs.filter((w) => w.status === 'online').length;
    const avgUptime = webSdrs.length > 0
        ? (webSdrs.reduce((sum, w) => sum + (w.uptime || 0), 0) / webSdrs.length).toFixed(1)
        : '0.0';

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
                            <h1 className="text-3xl font-bold text-white">WebSDR Network Management</h1>
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleRefresh}
                            disabled={isRefreshing || isLoading}
                            className="border-slate-700 text-slate-300 hover:bg-slate-800"
                        >
                            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6 max-w-7xl mx-auto">
                        {/* Error Alert */}
                        {error && (
                            <Alert className="bg-red-900/20 border-red-800">
                                <AlertCircle className="h-4 w-4 text-red-500" />
                                <AlertDescription className="text-red-300">
                                    {error}
                                </AlertDescription>
                            </Alert>
                        )}

                        {/* Loading State */}
                        {isLoading && webSdrs.length === 0 && (
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="p-12 text-center">
                                    <RefreshCw className="w-12 h-12 mx-auto mb-4 text-purple-500 animate-spin" />
                                    <p className="text-slate-400 text-lg">Loading WebSDR configuration...</p>
                                </CardContent>
                            </Card>
                        )}

                        {/* Status Summary */}
                        {!isLoading || webSdrs.length > 0 ? (
                            <>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <Card className="bg-slate-900 border-slate-800">
                                        <CardContent className="p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-slate-400 text-sm">Online Receivers</p>
                                                    <p className="text-3xl font-bold text-green-400 mt-2">
                                                        {onlineCount}/{webSdrs.length}
                                                    </p>
                                                </div>
                                                <Wifi className="w-12 h-12 text-green-500 opacity-20" />
                                            </div>
                                        </CardContent>
                                    </Card>
                                    <Card className="bg-slate-900 border-slate-800">
                                        <CardContent className="p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-slate-400 text-sm">Average Uptime</p>
                                                    <p className="text-3xl font-bold text-cyan-400 mt-2">{avgUptime}%</p>
                                                </div>
                                                <Activity className="w-12 h-12 text-cyan-500 opacity-20" />
                                            </div>
                                        </CardContent>
                                    </Card>
                                    <Card className="bg-slate-900 border-slate-800">
                                        <CardContent className="p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-slate-400 text-sm">Network Status</p>
                                                    <p className="text-3xl font-bold text-purple-400 mt-2">Healthy</p>
                                                </div>
                                                <Radio className="w-12 h-12 text-purple-500 opacity-20" />
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>

                                {/* WebSDR Table */}
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardHeader>
                                        <CardTitle className="text-white">Receiver Configuration</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-sm">
                                                <thead className="border-b border-slate-700">
                                                    <tr>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Receiver Name
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Location
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            GPS Coordinates
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Status
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Uptime
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Avg SNR
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Last Contact
                                                        </th>
                                                        <th className="text-left py-3 px-4 text-slate-400 font-semibold">
                                                            Actions
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-slate-800">
                                                    {webSdrs.map((sdr) => (
                                                        <tr key={sdr.id} className="hover:bg-slate-800/50 transition">
                                                            <td className="py-3 px-4">
                                                                <div>
                                                                    <p className="text-white font-medium">{sdr.name}</p>
                                                                    <p className="text-slate-500 text-xs break-all">
                                                                        {sdr.url}
                                                                    </p>
                                                                </div>
                                                            </td>
                                                            <td className="py-3 px-4 text-slate-300">{sdr.location}</td>
                                                            <td className="py-3 px-4 text-slate-400 text-xs">
                                                                {sdr.latitude.toFixed(4)}, {sdr.longitude.toFixed(4)}
                                                            </td>
                                                            <td className="py-3 px-4">
                                                                {sdr.status === 'online' ? (
                                                                    <div className="flex items-center gap-2">
                                                                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                                                        <span className="text-green-400 font-semibold">
                                                                            Online
                                                                        </span>
                                                                    </div>
                                                                ) : sdr.status === 'offline' ? (
                                                                    <div className="flex items-center gap-2">
                                                                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                                                        <span className="text-red-400 font-semibold">
                                                                            Offline
                                                                        </span>
                                                                    </div>
                                                                ) : (
                                                                    <div className="flex items-center gap-2">
                                                                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                                                                        <span className="text-yellow-400 font-semibold">
                                                                            Unknown
                                                                        </span>
                                                                    </div>
                                                                )}
                                                            </td>
                                                            <td className="py-3 px-4">
                                                                <span
                                                                    className={`font-semibold ${sdr.uptime === 0
                                                                            ? 'text-slate-500'
                                                                            : sdr.uptime > 99
                                                                                ? 'text-green-400'
                                                                                : sdr.uptime > 95
                                                                                    ? 'text-yellow-400'
                                                                                    : 'text-red-400'
                                                                        }`}
                                                                >
                                                                    {sdr.uptime === 0 ? 'N/A' : `${sdr.uptime.toFixed(1)}%`}
                                                                </span>
                                                            </td>
                                                            <td className="py-3 px-4">
                                                                <span className={`font-semibold ${sdr.avgSnr === 0 ? 'text-slate-500' : 'text-cyan-400'}`}>
                                                                    {sdr.avgSnr === 0 ? 'N/A' : `${sdr.avgSnr.toFixed(1)} dB`}
                                                                </span>
                                                            </td>
                                                            <td className="py-3 px-4 text-slate-400 text-xs">
                                                                {sdr.lastContact}
                                                            </td>
                                                            <td className="py-3 px-4">
                                                                <div className="flex gap-2">
                                                                    <Button
                                                                        size="sm"
                                                                        variant="outline"
                                                                        className="border-slate-700"
                                                                        onClick={() =>
                                                                            setEditingId(
                                                                                editingId === sdr.id ? null : sdr.id
                                                                            )
                                                                        }
                                                                    >
                                                                        <Edit2 className="w-4 h-4" />
                                                                    </Button>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Test Panel */}
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardHeader>
                                        <CardTitle className="text-white">Network Diagnostics</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <Button
                                                className="bg-purple-600 hover:bg-purple-700 w-full"
                                                onClick={handleRefresh}
                                                disabled={isRefreshing}
                                            >
                                                Test All Connections
                                            </Button>
                                            <Button className="bg-purple-600 hover:bg-purple-700 w-full">
                                                Verify Frequencies
                                            </Button>
                                            <Button className="bg-purple-600 hover:bg-purple-700 w-full">
                                                Fetch IQ Test Data
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            </>
                        ) : null}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default WebSDRManagement;
