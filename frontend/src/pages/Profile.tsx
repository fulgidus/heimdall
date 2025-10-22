import React from 'react';
import { Card, MainLayout } from '../components';
import { Mail, MapPin, Phone, Globe } from 'lucide-react';

const Profile: React.FC = () => {
    return (
        <MainLayout title="Profile">
            <div className="space-y-8 max-w-4xl">
                {/* Header with Avatar */}
                <Card variant="elevated">
                    <div className="flex flex-col sm:flex-row items-center gap-6 mb-6">
                        <div className="w-32 h-32 rounded-full bg-gradient-right flex items-center justify-center text-5xl flex-shrink-0">
                            👤
                        </div>
                        <div className="flex-1 text-center sm:text-left">
                            <h1 className="text-3xl font-bold text-white mb-2">John Doe</h1>
                            <p className="text-neon-blue font-semibold mb-3">Admin • Founder</p>
                            <p className="text-french-gray max-w-md">
                                Passionate developer and product enthusiast. Building amazing products with cutting-edge technology.
                            </p>
                        </div>
                    </div>
                </Card>

                {/* Contact Information */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4">Contact Information</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <Card variant="elevated" className="flex items-center gap-4">
                            <Mail className="text-neon-blue" size={28} />
                            <div>
                                <p className="text-sm text-french-gray">Email</p>
                                <p className="text-white font-medium">john@heimdall.com</p>
                            </div>
                        </Card>
                        <Card variant="elevated" className="flex items-center gap-4">
                            <Phone className="text-sea-green" size={28} />
                            <div>
                                <p className="text-sm text-french-gray">Phone</p>
                                <p className="text-white font-medium">+1 (555) 123-4567</p>
                            </div>
                        </Card>
                        <Card variant="elevated" className="flex items-center gap-4">
                            <MapPin className="text-light-green" size={28} />
                            <div>
                                <p className="text-sm text-french-gray">Location</p>
                                <p className="text-white font-medium">San Francisco, CA</p>
                            </div>
                        </Card>
                        <Card variant="elevated" className="flex items-center gap-4">
                            <Globe className="text-neon-blue" size={28} />
                            <div>
                                <p className="text-sm text-french-gray">Website</p>
                                <p className="text-white font-medium">johndoe.dev</p>
                            </div>
                        </Card>
                    </div>
                </section>

                {/* Statistics */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4">Your Statistics</h2>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {[
                            { label: 'Projects', value: '12' },
                            { label: 'Teams', value: '3' },
                            { label: 'Contributions', value: '456' },
                            { label: 'Followers', value: '1.2K' },
                        ].map((stat, idx) => (
                            <Card key={idx} variant="elevated" className="text-center">
                                <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-right mb-1">
                                    {stat.value}
                                </p>
                                <p className="text-sm text-french-gray">{stat.label}</p>
                            </Card>
                        ))}
                    </div>
                </section>

                {/* Skills */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4">Skills & Expertise</h2>
                    <Card variant="elevated">
                        <div className="flex flex-wrap gap-3">
                            {[
                                'React',
                                'TypeScript',
                                'Node.js',
                                'Python',
                                'AWS',
                                'Docker',
                                'GraphQL',
                                'PostgreSQL',
                                'Redis',
                                'Machine Learning',
                            ].map((skill) => (
                                <span
                                    key={skill}
                                    className="px-4 py-2 rounded-full bg-neon-blue bg-opacity-20 text-light-green border border-neon-blue border-opacity-50"
                                >
                                    {skill}
                                </span>
                            ))}
                        </div>
                    </Card>
                </section>

                {/* Recent Activity */}
                <section>
                    <h2 className="text-2xl font-bold text-white mb-4">Recent Activity</h2>
                    <Card variant="elevated">
                        <div className="space-y-4">
                            {[
                                { date: 'Oct 22, 2025', activity: 'Updated project settings for Dashboard redesign' },
                                { date: 'Oct 21, 2025', activity: 'Created new team "Design System"' },
                                { date: 'Oct 20, 2025', activity: 'Completed Phase 7 - Frontend Development' },
                                { date: 'Oct 19, 2025', activity: 'Added 5 new team members' },
                            ].map((item, idx) => (
                                <div
                                    key={idx}
                                    className="p-4 rounded-lg bg-sea-green bg-opacity-10 border border-sea-green border-opacity-20"
                                >
                                    <div className="flex items-start justify-between">
                                        <p className="text-white">{item.activity}</p>
                                        <span className="text-xs text-french-gray whitespace-nowrap ml-2">{item.date}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </section>
            </div>
        </MainLayout>
    );
};

export default Profile;
