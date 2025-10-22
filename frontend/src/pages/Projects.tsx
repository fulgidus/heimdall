import React from 'react';
import { Card, MainLayout, Button } from '../components';
import { Plus, Edit2, Trash2, GitBranch, Calendar } from 'lucide-react';

const Projects: React.FC = () => {
    const projects = [
        {
            id: 1,
            name: 'Heimdall Core',
            description: 'Main platform infrastructure and analytics engine',
            status: 'active',
            progress: 87,
            team: 5,
            updated: '2 days ago',
        },
        {
            id: 2,
            name: 'Mobile App',
            description: 'React Native mobile application for iOS and Android',
            status: 'active',
            progress: 64,
            team: 3,
            updated: '5 days ago',
        },
        {
            id: 3,
            name: 'Dashboard UI Redesign',
            description: 'Modern dashboard with improved UX and accessibility',
            status: 'planning',
            progress: 32,
            team: 4,
            updated: '1 week ago',
        },
        {
            id: 4,
            name: 'API Documentation',
            description: 'Comprehensive API documentation and SDKs',
            status: 'completed',
            progress: 100,
            team: 2,
            updated: '2 weeks ago',
        },
    ];

    const statusColors: Record<string, string> = {
        active: 'text-light-green',
        planning: 'text-neon-blue',
        completed: 'text-sea-green',
    };

    const statusBgs: Record<string, string> = {
        active: 'bg-light-green',
        planning: 'bg-neon-blue',
        completed: 'bg-sea-green',
    };

    return (
        <MainLayout title="Projects">
            <div className="space-y-8">
                {/* Header */}
                <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-right mb-2">
                            Projects 📁
                        </h1>
                        <p className="text-french-gray">Manage and track all your projects</p>
                    </div>
                    <Button variant="accent" className="flex items-center gap-2">
                        <Plus size={20} />
                        New Project
                    </Button>
                </div>

                {/* Projects Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {projects.map((project) => (
                        <Card key={project.id} variant="elevated" className="hover:shadow-2xl transition-all duration-200 cursor-pointer">
                            <div className="mb-4 flex items-start justify-between">
                                <div className="flex-1">
                                    <h3 className="text-xl font-bold text-white mb-1">{project.name}</h3>
                                    <p className="text-sm text-french-gray">{project.description}</p>
                                </div>
                                <div className="flex gap-2">
                                    <button className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded transition-colors">
                                        <Edit2 size={18} className="text-neon-blue" />
                                    </button>
                                    <button className="p-2 hover:bg-red-500 hover:bg-opacity-20 rounded transition-colors">
                                        <Trash2 size={18} className="text-red-400" />
                                    </button>
                                </div>
                            </div>

                            {/* Progress Bar */}
                            <div className="mb-4">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-french-gray">Progress</span>
                                    <span className="text-sm font-semibold text-light-green">{project.progress}%</span>
                                </div>
                                <div className="w-full bg-oxford-blue rounded-full h-2 overflow-hidden border border-neon-blue border-opacity-20">
                                    <div
                                        className="bg-gradient-right h-full rounded-full transition-all duration-500"
                                        style={{ width: `${project.progress}%` }}
                                    ></div>
                                </div>
                            </div>

                            {/* Metadata */}
                            <div className="pt-4 border-t border-neon-blue border-opacity-20">
                                <div className="flex items-center justify-between flex-wrap gap-3 text-sm">
                                    <div className="flex items-center gap-2">
                                        <span className={`w-2 h-2 rounded-full ${statusBgs[project.status]}`}></span>
                                        <span className={`capitalize font-semibold ${statusColors[project.status]}`}>
                                            {project.status}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 text-french-gray">
                                        <GitBranch size={16} />
                                        <span>{project.team} members</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-french-gray">
                                        <Calendar size={16} />
                                        <span>{project.updated}</span>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                    <Card variant="elevated" className="text-center">
                        <p className="text-3xl font-bold text-light-green mb-1">4</p>
                        <p className="text-french-gray">Total Projects</p>
                    </Card>
                    <Card variant="elevated" className="text-center">
                        <p className="text-3xl font-bold text-neon-blue mb-1">2</p>
                        <p className="text-french-gray">Active Projects</p>
                    </Card>
                    <Card variant="elevated" className="text-center">
                        <p className="text-3xl font-bold text-sea-green mb-1">1</p>
                        <p className="text-french-gray">Completed</p>
                    </Card>
                </div>
            </div>
        </MainLayout>
    );
};

export default Projects;
