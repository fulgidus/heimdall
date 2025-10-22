import React from 'react';
import { Card, MainLayout } from '../components';
import { BarChart3, TrendingUp, PieChart } from 'lucide-react';

const Analytics: React.FC = () => {
    return (
        <MainLayout title="Analytics">
            <div className="space-y-8">
                {/* Header */}
                <div>
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-right mb-2">
                        Analytics Dashboard 📊
                    </h1>
                    <p className="text-french-gray">Real-time insights into your platform's performance</p>
                </div>

                {/* Filters */}
                <Card variant="elevated" className="flex flex-wrap gap-3">
                    {['Last 7 days', 'Last 30 days', 'Last 3 months', 'Last 12 months'].map((filter) => (
                        <button
                            key={filter}
                            className="px-4 py-2 rounded-lg bg-neon-blue bg-opacity-10 text-neon-blue border border-neon-blue border-opacity-50 hover:bg-opacity-20 transition-all duration-200"
                        >
                            {filter}
                        </button>
                    ))}
                </Card>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* User Growth */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-6 flex items-center gap-3">
                            <TrendingUp className="text-sea-green" size={24} />
                            <h3 className="text-lg font-semibold text-white">User Growth</h3>
                        </div>
                        <div className="h-64 flex items-center justify-center text-french-gray">
                            📈 Chart visualization (Chart.js / Recharts)
                        </div>
                        <div className="mt-4 p-3 bg-sea-green bg-opacity-10 rounded text-sm text-light-green">
                            ↑ 23.5% increase from last month
                        </div>
                    </Card>

                    {/* Revenue Distribution */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-6 flex items-center gap-3">
                            <PieChart className="text-neon-blue" size={24} />
                            <h3 className="text-lg font-semibold text-white">Revenue Distribution</h3>
                        </div>
                        <div className="h-64 flex items-center justify-center text-french-gray">
                            📊 Pie chart visualization
                        </div>
                    </Card>

                    {/* Conversion Rate */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-6 flex items-center gap-3">
                            <BarChart3 className="text-light-green" size={24} />
                            <h3 className="text-lg font-semibold text-white">Conversion Rate by Channel</h3>
                        </div>
                        <div className="h-64 flex items-center justify-center text-french-gray">
                            📊 Bar chart visualization
                        </div>
                    </Card>

                    {/* Geographic Distribution */}
                    <Card variant="elevated">
                        <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-6">
                            <h3 className="text-lg font-semibold text-white">📍 Top Regions</h3>
                        </div>
                        <div className="space-y-3">
                            {[
                                { region: 'United States', users: '12,453', percentage: 45 },
                                { region: 'Europe', users: '8,234', percentage: 30 },
                                { region: 'Asia', users: '5,890', percentage: 21 },
                                { region: 'Other', users: '489', percentage: 4 },
                            ].map((item, idx) => (
                                <div key={idx} className="p-3 bg-sea-green bg-opacity-10 rounded">
                                    <div className="flex items-center justify-between mb-2">
                                        <p className="text-white font-medium">{item.region}</p>
                                        <p className="text-light-green font-semibold">{item.users}</p>
                                    </div>
                                    <div className="w-full bg-oxford-blue rounded-full h-2 overflow-hidden">
                                        <div
                                            className="bg-gradient-right h-full rounded-full"
                                            style={{ width: `${item.percentage}%` }}
                                        ></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>

                {/* Metrics Table */}
                <Card variant="elevated">
                    <div className="pb-4 border-b border-neon-blue border-opacity-20 mb-4">
                        <h3 className="text-lg font-semibold text-white">Performance Metrics</h3>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-neon-blue border-opacity-20">
                                    <th className="text-left py-3 px-4 text-french-gray font-semibold">Metric</th>
                                    <th className="text-left py-3 px-4 text-french-gray font-semibold">Current</th>
                                    <th className="text-left py-3 px-4 text-french-gray font-semibold">Previous</th>
                                    <th className="text-left py-3 px-4 text-french-gray font-semibold">Change</th>
                                </tr>
                            </thead>
                            <tbody>
                                {[
                                    { metric: 'Page Views', current: '245,123', previous: '198,456', change: '+23.5%' },
                                    { metric: 'Session Duration', current: '4m 32s', previous: '3m 48s', change: '+18.7%' },
                                    { metric: 'Bounce Rate', current: '32.4%', previous: '38.9%', change: '-16.8%' },
                                    { metric: 'CTR', current: '3.45%', previous: '2.89%', change: '+19.4%' },
                                ].map((row, idx) => (
                                    <tr key={idx} className="border-b border-neon-blue border-opacity-10 hover:bg-sea-green hover:bg-opacity-5 transition-colors">
                                        <td className="py-3 px-4 text-white">{row.metric}</td>
                                        <td className="py-3 px-4 text-light-green font-semibold">{row.current}</td>
                                        <td className="py-3 px-4 text-french-gray">{row.previous}</td>
                                        <td className="py-3 px-4 text-sea-green font-semibold">{row.change}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            </div>
        </MainLayout>
    );
};

export default Analytics;
