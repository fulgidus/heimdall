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
    Mail,
    Phone,
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

export const Profile: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);

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
                            <h1 className="text-3xl font-bold text-white">User Profile</h1>
                        </div>
                        <div className="text-slate-400 text-sm">
                            {new Date().toLocaleString()}
                        </div>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-8 max-w-4xl">
                        {/* User Info Card */}
                        <Card className="bg-slate-900 border-slate-800">
                            <CardContent className="pt-6">
                                <div className="flex flex-col sm:flex-row items-center gap-6">
                                    <div className="w-32 h-32 rounded-full bg-linear-to-br from-purple-500 to-pink-500 flex items-center justify-center text-4xl flex-shrink-0">
                                        👤
                                    </div>
                                    <div className="flex-1 text-center sm:text-left">
                                        <h1 className="text-3xl font-bold text-white mb-2">John Doe</h1>
                                        <p className="text-purple-400 font-semibold mb-3">Admin • Founder</p>
                                        <p className="text-slate-400">
                                            Passionate developer and RF enthusiast. Building Heimdall for the amateur radio community.
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Contact Information */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="pt-6 flex items-center gap-4">
                                    <Mail className="w-8 h-8 text-purple-500" />
                                    <div>
                                        <p className="text-sm text-slate-400">Email</p>
                                        <p className="text-white font-medium">john@heimdall.com</p>
                                    </div>
                                </CardContent>
                            </Card>
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="pt-6 flex items-center gap-4">
                                    <Phone className="w-8 h-8 text-cyan-500" />
                                    <div>
                                        <p className="text-sm text-slate-400">Phone</p>
                                        <p className="text-white font-medium">+1 (555) 123-4567</p>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Statistics */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="pt-6">
                                    <p className="text-slate-400 text-sm">Localizations Performed</p>
                                    <p className="text-3xl font-bold text-white mt-2">1,247</p>
                                </CardContent>
                            </Card>
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="pt-6">
                                    <p className="text-slate-400 text-sm">Average Accuracy</p>
                                    <p className="text-3xl font-bold text-white mt-2">±27.3m</p>
                                </CardContent>
                            </Card>
                            <Card className="bg-slate-900 border-slate-800">
                                <CardContent className="pt-6">
                                    <p className="text-slate-400 text-sm">Member Since</p>
                                    <p className="text-white font-medium mt-2">January 2024</p>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Actions */}
                        <div className="flex gap-3">
                            <Button
                                onClick={() => navigate('/settings')}
                                className="bg-purple-600 hover:bg-purple-700 text-white"
                            >
                                Edit Profile
                            </Button>
                            <Button variant="outline" className="border-slate-700 text-slate-300">
                                Download Data
                            </Button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Profile;
