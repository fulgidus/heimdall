import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Activity, 
    Radio, 
    Zap, 
    Target, 
    LogOut, 
    Menu, 
    X,
    Radar,
    TrendingUp,
    Clock,
    AlertCircle
} from 'lucide-react';
import { useAuthStore } from '../store';

interface StatCard {
    icon: React.ReactNode;
    label: string;
    value: string | number;
    unit?: string;
    status: 'active' | 'inactive' | 'warning';
}

interface ActivityLog {
    id: string;
    timestamp: string;
    action: string;
    source: string;
    status: 'success' | 'pending' | 'error';
}

const Dashboard: React.FC = () => {
    const navigate = useNavigate();
    const { user, logout } = useAuthStore();
    const [menuOpen, setMenuOpen] = useState(false);
    const [stats] = useState<StatCard[]>([
        {
            icon: <Radio className="w-8 h-8" />,
            label: 'Active WebSDR',
            value: 7,
            unit: '/ 7',
            status: 'active',
        },
        {
            icon: <Activity className="w-8 h-8" />,
            label: 'Signal Detection',
            value: 12,
            unit: 'events',
            status: 'active',
        },
        {
            icon: <Zap className="w-8 h-8" />,
            label: 'System Uptime',
            value: 99.8,
            unit: '%',
            status: 'active',
        },
        {
            icon: <Target className="w-8 h-8" />,
            label: 'Accuracy',
            value: 95.2,
            unit: '%',
            status: 'active',
        },
    ]);

    const [activities] = useState<ActivityLog[]>([
        {
            id: '1',
            timestamp: '2 min ago',
            action: 'RF signal detected',
            source: 'WebSDR Turin',
            status: 'success',
        },
        {
            id: '2',
            timestamp: '15 min ago',
            action: 'Localization computed',
            source: 'ML Pipeline',
            status: 'success',
        },
        {
            id: '3',
            timestamp: '1 hour ago',
            action: 'Model retraining completed',
            source: 'Training Service',
            status: 'success',
        },
        {
            id: '4',
            timestamp: '3 hours ago',
            action: 'Calibration sync',
            source: 'System Health',
            status: 'pending',
        },
    ]);

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active':
                return 'from-purple-500 to-indigo-600';
            case 'warning':
                return 'from-yellow-500 to-orange-600';
            default:
                return 'from-gray-500 to-gray-600';
        }
    };

    const getActivityStatusColor = (status: string) => {
        switch (status) {
            case 'success':
                return 'text-green-400';
            case 'pending':
                return 'text-yellow-400';
            case 'error':
                return 'text-red-400';
            default:
                return 'text-gray-400';
        }
    };

    return (
        <div className="min-h-screen bg-linear-to-br from-purple-600 via-purple-500 to-indigo-600">
            {/* Animated Background */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-10 w-72 h-72 bg-white rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
                <div className="absolute bottom-20 right-10 w-72 h-72 bg-indigo-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse" style={{ animationDelay: '2s' }}></div>
            </div>

            {/* Sidebar Navigation */}
            <div className={`fixed left-0 top-0 h-screen w-64 bg-linear-to-b from-purple-900 to-indigo-900 shadow-2xl transform transition-transform duration-300 z-40 ${menuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}>
                <div className="p-6 space-y-8">
                    {/* Logo */}
                    <div className="flex items-center gap-3">
                        <Radar className="w-8 h-8 text-indigo-300 animate-spin" style={{ animationDuration: '3s' }} />
                        <h1 className="text-2xl font-black text-white">Heimdall</h1>
                    </div>

                    {/* Navigation Menu */}
                    <nav className="space-y-4">
                        {[
                            { icon: Activity, label: 'Dashboard', active: true },
                            { icon: Radio, label: 'WebSDR Network', active: false },
                            { icon: TrendingUp, label: 'Localizations', active: false },
                            { icon: Clock, label: 'History', active: false },
                        ].map((item, idx) => (
                            <button
                                key={idx}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                                    item.active
                                        ? 'bg-linear-to-r from-purple-400 to-indigo-500 text-white shadow-lg'
                                        : 'text-purple-200 hover:bg-purple-800/50 hover:text-white'
                                }`}
                            >
                                <item.icon className="w-5 h-5" />
                                <span className="font-medium">{item.label}</span>
                            </button>
                        ))}
                    </nav>

                    {/* User Info */}
                    <div className="absolute bottom-6 left-6 right-6 pt-6 border-t border-purple-700">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 bg-linear-to-br from-purple-300 to-indigo-400 rounded-full flex items-center justify-center">
                                <span className="text-white font-bold">{user?.email?.[0].toUpperCase()}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-white font-semibold text-sm truncate">{user?.email}</p>
                                <p className="text-purple-200 text-xs">Administrator</p>
                            </div>
                        </div>
                        <button
                            onClick={handleLogout}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-500/20 text-red-300 rounded-xl hover:bg-red-500/30 transition-all border border-red-500/30"
                        >
                            <LogOut className="w-4 h-4" />
                            <span className="font-medium">Logout</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="lg:ml-64 relative z-10 min-h-screen p-4 lg:p-8">
                {/* Mobile Header */}
                <div className="lg:hidden flex items-center justify-between mb-8 backdrop-blur-sm bg-white/10 rounded-2xl p-4">
                    <h1 className="text-2xl font-black text-white">Heimdall</h1>
                    <button
                        onClick={() => setMenuOpen(!menuOpen)}
                        className="text-white hover:bg-white/20 p-2 rounded-lg transition-all"
                    >
                        {menuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>

                {/* Desktop Header */}
                <div className="hidden lg:flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-4xl font-black text-white mb-2">Welcome back! 🚀</h1>
                        <p className="text-purple-200">RF Source Localization System Status</p>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="flex items-center gap-2 px-6 py-3 bg-red-500/20 text-red-300 rounded-xl hover:bg-red-500/30 transition-all border border-red-500/30 font-semibold"
                    >
                        <LogOut className="w-5 h-5" />
                        Logout
                    </button>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-8">
                    {stats.map((stat, idx) => (
                        <div
                            key={idx}
                            className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all hover:shadow-2xl group cursor-pointer animate-fade-in"
                            style={{ animationDelay: `${idx * 0.1}s` }}
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className={`p-3 rounded-xl bg-linear-to-br ${getStatusColor(stat.status)} text-white shadow-lg group-hover:shadow-2xl transition-all`}>
                                    {stat.icon}
                                </div>
                                <div className={`w-2 h-2 rounded-full ${stat.status === 'active' ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></div>
                            </div>
                            <p className="text-purple-200 text-sm font-medium mb-2">{stat.label}</p>
                            <div className="flex items-baseline gap-2">
                                <p className="text-3xl font-black text-white">{stat.value}</p>
                                {stat.unit && <p className="text-purple-300 text-sm">{stat.unit}</p>}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Activity Feed - Spans 2 columns on desktop */}
                    <div className="lg:col-span-2 bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-6 pb-6 border-b border-white/20">
                            <Activity className="w-6 h-6 text-indigo-300" />
                            <h2 className="text-2xl font-bold text-white">Recent Activity</h2>
                        </div>

                        <div className="space-y-4">
                            {activities.map((activity) => (
                                <div
                                    key={activity.id}
                                    className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all"
                                >
                                    <div className={`mt-1 ${getActivityStatusColor(activity.status)}`}>
                                        {activity.status === 'success' && <Zap className="w-5 h-5" />}
                                        {activity.status === 'pending' && <Clock className="w-5 h-5" />}
                                        {activity.status === 'error' && <AlertCircle className="w-5 h-5" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-white font-semibold">{activity.action}</p>
                                        <p className="text-purple-300 text-sm mt-1">{activity.source}</p>
                                    </div>
                                    <p className="text-purple-400 text-sm font-medium whitespace-nowrap">{activity.timestamp}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="space-y-6">
                        {/* System Health */}
                        <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-green-400" />
                                System Health
                            </h3>
                            <div className="space-y-3">
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-purple-200 text-sm">CPU Usage</span>
                                        <span className="text-white font-semibold">45%</span>
                                    </div>
                                    <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                        <div className="h-full w-[45%] bg-linear-to-r from-purple-400 to-indigo-500 rounded-full"></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-purple-200 text-sm">Memory</span>
                                        <span className="text-white font-semibold">62%</span>
                                    </div>
                                    <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                        <div className="h-full w-[62%] bg-linear-to-r from-indigo-400 to-purple-500 rounded-full"></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-purple-200 text-sm">Disk</span>
                                        <span className="text-white font-semibold">38%</span>
                                    </div>
                                    <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                        <div className="h-full w-[38%] bg-linear-to-r from-purple-500 to-indigo-400 rounded-full"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Network Status */}
                        <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                <Radio className="w-5 h-5 text-cyan-400" />
                                Network Status
                            </h3>
                            <div className="space-y-3">
                                {['Turin', 'Milan', 'Genoa', 'Alessandria', 'Asti', 'Cuneo', 'Vercelli'].map((city, idx) => (
                                    <div key={idx} className="flex items-center justify-between">
                                        <span className="text-purple-200 text-sm">{city}</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                                            <span className="text-green-400 text-xs font-semibold">Online</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Custom Animations */}
            <style>{`
                @keyframes fade-in {
                    from {
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                .animate-fade-in {
                    animation: fade-in 0.6s ease-out forwards;
                    opacity: 0;
                }
            `}</style>
        </div>
    );
};

export default Dashboard;
