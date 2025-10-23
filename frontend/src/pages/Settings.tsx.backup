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
    Lock,
    User,
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

export const Settings: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const menuItems = [
        { icon: Home, label: 'Dashboard', path: '/dashboard', active: false },
        { icon: MapPin, label: 'Localization', path: '/localization', active: false },
        { icon: Radio, label: 'Recording Sessions', path: '/projects', active: false },
        { icon: BarChart3, label: 'Analytics', path: '/analytics', active: false },
        { icon: Zap, label: 'Settings', path: '/settings', active: true },
    ];

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const handleNavigation = (path: string) => {
        navigate(path);
        setSidebarOpen(false);
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
                            <h1 className="text-3xl font-bold text-white">System Settings</h1>
                        </div>
                        <div className="text-slate-400 text-sm">
                            {new Date().toLocaleString()}
                        </div>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-8 max-w-4xl">
                        {/* Profile Settings */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center gap-2">
                                    <User className="w-5 h-5 text-purple-500" />
                                    Profile Settings
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">
                                            First Name
                                        </label>
                                        <input
                                            type="text"
                                            defaultValue="John"
                                            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">
                                            Last Name
                                        </label>
                                        <input
                                            type="text"
                                            defaultValue="Doe"
                                            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Email Address
                                    </label>
                                    <input
                                        type="email"
                                        defaultValue="john@heimdall.com"
                                        className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                    />
                                </div>
                                <div className="flex gap-3 pt-4">
                                    <Button className="bg-purple-600 hover:bg-purple-700 text-white">
                                        Save Changes
                                    </Button>
                                    <Button variant="outline" className="border-slate-700 text-slate-300">
                                        Cancel
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Security */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center gap-2">
                                    <Lock className="w-5 h-5 text-cyan-500" />
                                    Security & Privacy
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Current Password
                                    </label>
                                    <input
                                        type="password"
                                        className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                    />
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">
                                            New Password
                                        </label>
                                        <input
                                            type="password"
                                            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">
                                            Confirm Password
                                        </label>
                                        <input
                                            type="password"
                                            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
                                        />
                                    </div>
                                </div>
                                <div className="flex gap-3 pt-4">
                                    <Button className="bg-purple-600 hover:bg-purple-700 text-white">
                                        Change Password
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

export default Settings;
