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
    TrendingUp,
    Users,
    Activity,
    Zap as ZapIcon,
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

interface Metric {
    label: string;
    value: string;
    change: string;
    icon: React.ReactNode;
    color: string;
}

export const Analytics: React.FC = () => {
    const navigate = useNavigate();
    const { logout } = useAuthStore();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [timeRange] = useState('7d');

    const menuItems = [
        { icon: Home, label: 'Dashboard', path: '/dashboard', active: false },
        { icon: MapPin, label: 'Localization', path: '/localization', active: false },
        { icon: Radio, label: 'Recording Sessions', path: '/projects', active: false },
        { icon: BarChart3, label: 'Analytics', path: '/analytics', active: true },
        { icon: Zap, label: 'Settings', path: '/settings', active: false },
    ];

    const metrics: Metric[] = [
        {
            label: 'Total Localizations',
            value: '1,284',
            change: '+12.5% from last week',
            icon: <Activity className="w-5 h-5" />,
            color: 'text-cyan-400',
        },
        {
            label: 'Avg Accuracy',
            value: '±28.3m',
            change: '+2.1% improvement',
            icon: <TrendingUp className="w-5 h-5" />,
            color: 'text-green-400',
        },
        {
            label: 'Active Receivers',
            value: '7/7',
            change: '100% uptime',
            icon: <Radio className="w-5 h-5" />,
            color: 'text-purple-400',
        },
        {
            label: 'Avg Response Time',
            value: '234ms',
            change: '-18% faster',
            icon: <ZapIcon className="w-5 h-5" />,
            color: 'text-orange-400',
        },
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
                            <h1 className="text-3xl font-bold text-white">Analytics Dashboard</h1>
                        </div>
                        <div className="flex gap-2">
                            {['7d', '30d', '90d'].map((range) => (
                                <Button
                                    key={range}
                                    variant={timeRange === range ? 'default' : 'outline'}
                                    size="sm"
                                    className={
                                        timeRange === range
                                            ? 'bg-purple-600 hover:bg-purple-700'
                                            : 'border-slate-700'
                                    }
                                >
                                    {range === '7d' ? 'Last 7 Days' : range === '30d' ? 'Last 30 Days' : 'Last 90 Days'}
                                </Button>
                            ))}
                        </div>
                    </div>
                </header>

                {/* Content Area */}
                <div className="flex-1 overflow-auto p-6">
                    <div className="space-y-6 max-w-7xl mx-auto">
                        {/* Metrics Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {metrics.map((metric, idx) => (
                                <Card key={idx} className="bg-slate-900 border-slate-800">
                                    <CardContent className="p-6">
                                        <div className="flex items-start justify-between">
                                            <div>
                                                <p className="text-slate-400 text-sm font-medium">{metric.label}</p>
                                                <h3 className="text-3xl font-bold text-white mt-2">{metric.value}</h3>
                                                <p className="text-green-400 text-xs mt-2">{metric.change}</p>
                                            </div>
                                            <div className={`${metric.color}`}>{metric.icon}</div>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>

                        {/* Charts Section */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Localization Trend */}
                            <Card className="bg-slate-900 border-slate-800">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-white">
                                        <TrendingUp className="w-5 h-5 text-cyan-400" />
                                        Localization Trend
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="h-64 flex items-center justify-center text-slate-500 border border-slate-800 rounded-lg">
                                        📈 Chart visualization (7-day trend)
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Receiver Performance */}
                            <Card className="bg-slate-900 border-slate-800">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-white">
                                        <Activity className="w-5 h-5 text-green-400" />
                                        Receiver Performance
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {[
                                            { name: 'Turin', uptime: '99.8%', color: 'bg-green-500' },
                                            { name: 'Milan', uptime: '99.5%', color: 'bg-green-500' },
                                            { name: 'Genoa', uptime: '98.9%', color: 'bg-yellow-500' },
                                            { name: 'Alessandria', uptime: '99.2%', color: 'bg-green-500' },
                                        ].map((receiver, idx) => (
                                            <div key={idx} className="flex items-center justify-between">
                                                <span className="text-slate-300">{receiver.name}</span>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full ${receiver.color}`}
                                                            style={{
                                                                width: receiver.uptime,
                                                            }}
                                                        ></div>
                                                    </div>
                                                    <span className="text-slate-400 text-sm w-12">
                                                        {receiver.uptime}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Accuracy Distribution */}
                            <Card className="bg-slate-900 border-slate-800">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-white">
                                        <ZapIcon className="w-5 h-5 text-purple-400" />
                                        Accuracy Distribution
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {[
                                            { range: '±10m - ±20m', count: '234', percentage: 35 },
                                            { range: '±20m - ±30m', count: '456', percentage: 55 },
                                            { range: '±30m - ±50m', count: '156', percentage: 10 },
                                        ].map((dist, idx) => (
                                            <div key={idx}>
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-slate-300">{dist.range}</span>
                                                    <span className="text-cyan-400 font-bold">{dist.count}</span>
                                                </div>
                                                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-linear-to-r from-cyan-500 to-blue-500"
                                                        style={{ width: `${dist.percentage}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Response Times */}
                            <Card className="bg-slate-900 border-slate-800">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-white">
                                        <Users className="w-5 h-5 text-orange-400" />
                                        Response Time Stats
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        <div>
                                            <div className="flex justify-between mb-2">
                                                <span className="text-slate-400 text-sm">P50</span>
                                                <span className="text-white font-bold">185ms</span>
                                            </div>
                                            <div className="h-2 bg-slate-800 rounded-full">
                                                <div
                                                    className="h-full bg-green-500"
                                                    style={{ width: '30%' }}
                                                ></div>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between mb-2">
                                                <span className="text-slate-400 text-sm">P95</span>
                                                <span className="text-white font-bold">234ms</span>
                                            </div>
                                            <div className="h-2 bg-slate-800 rounded-full">
                                                <div
                                                    className="h-full bg-yellow-500"
                                                    style={{ width: '40%' }}
                                                ></div>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between mb-2">
                                                <span className="text-slate-400 text-sm">P99</span>
                                                <span className="text-white font-bold">342ms</span>
                                            </div>
                                            <div className="h-2 bg-slate-800 rounded-full">
                                                <div
                                                    className="h-full bg-red-500"
                                                    style={{ width: '60%' }}
                                                ></div>
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Analytics;
