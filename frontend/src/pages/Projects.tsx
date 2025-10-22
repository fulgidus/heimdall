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
    Plus,
    Play,
    Square,
    Trash2,
    Clock,
} from 'lucide-react';
import { useAuthStore } from '../store';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface RecordingSession {
    id: string;
    name: string;
    frequency: string;
    status: 'idle' | 'running' | 'completed';
    startTime: string;
    duration: string;
    receivers: number;
}

export const Projects: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [sessions] = useState<RecordingSession[]>([
        {
            id: '1',
            name: 'Session Alpha - 2m Band',
            frequency: '145.500 MHz',
            status: 'completed',
            startTime: '2025-10-22 14:30',
            duration: '15 min',
            receivers: 7,
        },
        {
            id: '2',
            name: 'Session Beta - 70cm Band',
            frequency: '432.500 MHz',
            status: 'running',
            startTime: '2025-10-22 15:45',
            duration: 'in progress',
            receivers: 7,
        },
        {
            id: '3',
            name: 'Session Gamma - 2m Band',
            frequency: '145.100 MHz',
            status: 'idle',
            startTime: '---',
            duration: '0 min',
            receivers: 0,
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

    const getStatusColor = (status: RecordingSession['status']) => {
        switch (status) {
            case 'running':
                return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'completed':
                return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
            default:
                return 'bg-slate-700/20 text-slate-400 border-slate-700/50';
        }
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
                        <Button className="bg-purple-600 hover:bg-purple-700 text-white">
                            <Plus className="w-4 h-4 mr-2" />
                            New Session
                        </Button>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6">
                        {/* Active Sessions */}
                        <div className="mb-6">
                            <h2 className="text-xl font-bold text-white mb-4">Active Session</h2>
                            {sessions
                                .filter((s) => s.status === 'running')
                                .map((session) => (
                                    <Card key={session.id} className="bg-slate-900 border-slate-800">
                                        <CardContent className="p-6">
                                            <div className="flex items-center justify-between">
                                                <div className="flex-1">
                                                    <h3 className="text-white font-bold text-lg">
                                                        {session.name}
                                                    </h3>
                                                    <p className="text-slate-400 text-sm">
                                                        {session.frequency} • {session.receivers} receivers
                                                    </p>
                                                    <div className="flex items-center gap-4 mt-3">
                                                        <div className="flex items-center gap-2">
                                                            <Clock className="w-4 h-4 text-cyan-500" />
                                                            <span className="text-slate-300">
                                                                Started: {session.startTime}
                                                            </span>
                                                        </div>
                                                        <span
                                                            className={`px-3 py-1 rounded text-sm font-bold border ${getStatusColor(
                                                                session.status
                                                            )}`}
                                                        >
                                                            {session.status.toUpperCase()}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="flex gap-2">
                                                    <Button
                                                        size="sm"
                                                        variant="outline"
                                                        className="border-slate-700"
                                                    >
                                                        <Square className="w-4 h-4" />
                                                    </Button>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            {sessions.filter((s) => s.status === 'running').length === 0 && (
                                <Card className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6 text-center">
                                        <p className="text-slate-400">No active sessions. Start a new session to begin recording.</p>
                                    </CardContent>
                                </Card>
                            )}
                        </div>

                        {/* Completed Sessions */}
                        <div className="mb-6">
                            <h2 className="text-xl font-bold text-white mb-4">Recent Sessions</h2>
                            <div className="space-y-3">
                                {sessions
                                    .filter((s) => s.status === 'completed')
                                    .map((session) => (
                                        <Card
                                            key={session.id}
                                            className="bg-slate-900 border-slate-800 hover:border-slate-700 cursor-pointer transition-colors"
                                        >
                                            <CardContent className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex-1">
                                                        <h3 className="text-white font-semibold">
                                                            {session.name}
                                                        </h3>
                                                        <div className="flex items-center gap-4 mt-2 text-sm text-slate-400">
                                                            <span>{session.frequency}</span>
                                                            <span>{session.receivers} receivers</span>
                                                            <span>{session.duration}</span>
                                                            <span>
                                                                Completed: {session.startTime}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div className="flex gap-2">
                                                        <Button
                                                            size="sm"
                                                            variant="outline"
                                                            className="border-slate-700"
                                                        >
                                                            <Play className="w-4 h-4" />
                                                        </Button>
                                                        <Button
                                                            size="sm"
                                                            variant="outline"
                                                            className="border-slate-700 text-red-400 hover:text-red-300"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </Button>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                            </div>
                        </div>

                        {/* Idle Sessions */}
                        <div>
                            <h2 className="text-xl font-bold text-white mb-4">Available Configurations</h2>
                            <div className="space-y-3">
                                {sessions
                                    .filter((s) => s.status === 'idle')
                                    .map((session) => (
                                        <Card
                                            key={session.id}
                                            className="bg-slate-900 border-slate-800 hover:border-slate-700 cursor-pointer transition-colors"
                                        >
                                            <CardContent className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div>
                                                        <h3 className="text-white font-semibold">
                                                            {session.name}
                                                        </h3>
                                                        <p className="text-slate-400 text-sm mt-1">
                                                            Frequency: {session.frequency}
                                                        </p>
                                                    </div>
                                                    <Button className="bg-purple-600 hover:bg-purple-700">
                                                        <Play className="w-4 h-4 mr-2" />
                                                        Start
                                                    </Button>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Projects;
