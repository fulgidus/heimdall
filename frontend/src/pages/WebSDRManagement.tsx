'use client';

import React, { useState } from 'react';
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
} from 'lucide-react';
import { useAuthStore } from '../store';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface WebSDR {
    id: string;
    name: string;
    url: string;
    location: string;
    latitude: number;
    longitude: number;
    status: 'online' | 'offline';
    lastContact: string;
    uptime: number;
    avgSnr: number;
    enabled: boolean;
}

export const WebSDRManagement: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [webSdrs] = useState<WebSDR[]>([
        {
            id: '1',
            name: 'Turin (Torino)',
            url: 'http://websdr.bzdmh.pl:8901/',
            location: 'Piemonte',
            latitude: 45.0703,
            longitude: 7.6869,
            status: 'online',
            lastContact: '2025-10-22 16:42:15',
            uptime: 99.8,
            avgSnr: 18.5,
            enabled: true,
        },
        {
            id: '2',
            name: 'Milan (Milano)',
            url: 'http://websdr-italy.mynetdomain.it:8902/',
            location: 'Lombardia',
            latitude: 45.4642,
            longitude: 9.1900,
            status: 'online',
            lastContact: '2025-10-22 16:41:58',
            uptime: 99.5,
            avgSnr: 16.2,
            enabled: true,
        },
        {
            id: '3',
            name: 'Genoa (Genova)',
            url: 'http://websdr-liguria.example.com:8903/',
            location: 'Liguria',
            latitude: 44.4056,
            longitude: 8.9463,
            status: 'online',
            lastContact: '2025-10-22 16:42:03',
            uptime: 98.9,
            avgSnr: 14.8,
            enabled: true,
        },
        {
            id: '4',
            name: 'Alessandria',
            url: 'http://websdr-alessandria.myhost.it:8904/',
            location: 'Piemonte',
            latitude: 44.9129,
            longitude: 8.6176,
            status: 'online',
            lastContact: '2025-10-22 16:42:11',
            uptime: 99.2,
            avgSnr: 17.1,
            enabled: true,
        },
        {
            id: '5',
            name: 'Piacenza',
            url: 'http://websdr-piacenza.local:8905/',
            location: 'Emilia-Romagna',
            latitude: 45.0549,
            longitude: 9.7088,
            status: 'offline',
            lastContact: '2025-10-22 14:12:30',
            uptime: 85.3,
            avgSnr: 0,
            enabled: false,
        },
        {
            id: '6',
            name: 'Savona',
            url: 'http://websdr-savona.radio.it:8906/',
            location: 'Liguria',
            latitude: 44.3105,
            longitude: 8.4817,
            status: 'online',
            lastContact: '2025-10-22 16:41:52',
            uptime: 99.1,
            avgSnr: 15.3,
            enabled: true,
        },
        {
            id: '7',
            name: 'La Spezia',
            url: 'http://websdr-laspezia.org:8907/',
            location: 'Liguria',
            latitude: 43.5436,
            longitude: 9.8263,
            status: 'online',
            lastContact: '2025-10-22 16:42:09',
            uptime: 97.8,
            avgSnr: 13.9,
            enabled: true,
        },
    ]);

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
    const avgUptime = (webSdrs.reduce((sum, w) => sum + w.uptime, 0) / webSdrs.length).toFixed(1);

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
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6 max-w-7xl mx-auto">
                        {/* Status Summary */}
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
                                                        ) : (
                                                            <div className="flex items-center gap-2">
                                                                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                                                <span className="text-red-400 font-semibold">
                                                                    Offline
                                                                </span>
                                                            </div>
                                                        )}
                                                    </td>
                                                    <td className="py-3 px-4">
                                                        <span
                                                            className={`font-semibold ${sdr.uptime > 99
                                                                    ? 'text-green-400'
                                                                    : sdr.uptime > 95
                                                                        ? 'text-yellow-400'
                                                                        : 'text-red-400'
                                                                }`}
                                                        >
                                                            {sdr.uptime.toFixed(1)}%
                                                        </span>
                                                    </td>
                                                    <td className="py-3 px-4">
                                                        <span className="text-cyan-400 font-semibold">
                                                            {sdr.avgSnr.toFixed(1)} dB
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
                                    <Button className="bg-purple-600 hover:bg-purple-700 w-full">
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
                    </div>
                </div>
            </main>
        </div>
    );
};

export default WebSDRManagement;
