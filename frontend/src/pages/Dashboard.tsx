import React, { useEffect } from 'react';
import { Card, MainLayout } from '../components';
import { useDashboardStore } from '../store';
import { TrendingUp, Users, DollarSign, Zap } from 'lucide-react';

const StatCard: React.FC<{ icon: React.ReactNode; label: string; value: string | number; change: string }> = ({
    icon,
    label,
    value,
    change,
}) => (
    <Card variant="elevated" className="p-6">
        <div className="flex items-start justify-between">
            <div className="flex-1">
                <p className="text-sm text-french-gray mb-2">{label}</p>
                <p className="text-3xl font-bold text-light-green">{value}</p>
                <p className="text-xs text-sea-green mt-2">↑ {change} from last month</p>
            </div>
            <div className="text-neon-blue text-3xl opacity-50">{icon}</div>
        </div>
    </Card>
);

const Dashboard: React.FC = () => {
    const { metrics, fetchMetrics } = useDashboardStore();

    useEffect(() => {
        fetchMetrics();
    }, [fetchMetrics]);

    return (
        <MainLayout title="Dashboard">
            <div className="space-y-8">
                {/* Header */}
                <div>
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-right mb-2">
                        Welcome back! 👋
                    </h1>
                    <p className="text-french-gray">Here's what's happening with your business today.</p>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    <StatCard
                        icon={<Users size={32} />}
                        label="Active Users"
                        value={metrics.activeUsers}
                        change="12.5%"
                    />
                    <StatCard
                        icon={<DollarSign size={32} />}
                        label="Total Revenue"
                        value={`$${metrics.totalRevenue.toLocaleString()}`}
                        change="8.2%"
                    />
                    <StatCard
                        icon={<TrendingUp size={32} />}
                        label="Conversion Rate"
                        value={`${metrics.conversionRate}%`}
                        change="2.1%"
                    />
                    <StatCard
                        icon={<Zap size={32} />}
                        label="Performance"
                        value={`${metrics.performance}%`}
                        change="5.3%"
                    />
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Revenue Chart */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20">
                            <h3 className="text-lg font-semibold text-white">Revenue Trend</h3>
                        </div>
                        <div className="pt-6 h-64 flex items-center justify-center text-french-gray">
                            📊 Chart visualization goes here
                        </div>
                    </Card>

                    {/* Traffic Chart */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20">
                            <h3 className="text-lg font-semibold text-white">Traffic Sources</h3>
                        </div>
                        <div className="pt-6 h-64 flex items-center justify-center text-french-gray">
                            📈 Chart visualization goes here
                        </div>
                    </Card>
                </div>

                {/* Recent Activity */}
                <Card variant="elevated">
                    <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-4">
                        <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
                    </div>
                    <div className="space-y-3">
                        {[
                            { time: '2 hours ago', action: 'User registration peak', status: 'success' },
                            { time: '4 hours ago', action: 'System update completed', status: 'success' },
                            { time: '6 hours ago', action: 'API performance improved', status: 'success' },
                            { time: '1 day ago', action: 'Database backup completed', status: 'success' },
                        ].map((item, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-sea-green bg-opacity-10 border border-sea-green border-opacity-20">
                                <div>
                                    <p className="text-white font-medium">{item.action}</p>
                                    <p className="text-sm text-french-gray">{item.time}</p>
                                </div>
                                <span className="text-light-green text-sm font-semibold">✓ {item.status}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </MainLayout>
    );
};

export default Dashboard;
