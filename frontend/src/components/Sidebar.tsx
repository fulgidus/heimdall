import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { X, LogOut } from 'lucide-react';
import Button from './Button';

interface SidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

const navigation = [
    { name: 'Dashboard', path: '/dashboard', icon: '📊' },
    { name: 'Projects', path: '/projects', icon: '📁' },
    { name: 'Analytics', path: '/analytics', icon: '📈' },
    { name: 'Settings', path: '/settings', icon: '⚙️' },
    { name: 'Profile', path: '/profile', icon: '👤' },
];

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
    const location = useLocation();

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-oxford-blue bg-opacity-50 lg:hidden z-30"
                    onClick={onClose}
                />
            )}

            {/* Sidebar */}
            <aside
                className={`fixed left-0 top-0 h-full w-64 bg-gradient-to-b from-oxford-blue to-sea-green border-r border-neon-blue border-opacity-20 transform transition-transform duration-300 ease-in-out z-40 lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'
                    }`}
            >
                {/* Logo */}
                <div className="p-6 border-b border-neon-blue border-opacity-20">
                    <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-right">
                        🚀 Heimdall
                    </h1>
                    <p className="text-sm text-french-gray mt-1">Analytics Platform</p>
                </div>

                {/* Navigation */}
                <nav className="p-4 flex-1">
                    {navigation.map((item) => {
                        const isActive = location.pathname === item.path;
                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all duration-200 ${isActive
                                        ? 'bg-neon-blue bg-opacity-20 text-light-green border border-light-green border-opacity-50'
                                        : 'text-french-gray hover:bg-sea-green hover:bg-opacity-10 hover:text-light-green'
                                    }`}
                            >
                                <span className="text-lg">{item.icon}</span>
                                <span className="font-medium">{item.name}</span>
                            </Link>
                        );
                    })}
                </nav>

                {/* Footer */}
                <div className="p-4 border-t border-neon-blue border-opacity-20">
                    <Button
                        variant="danger"
                        size="sm"
                        className="w-full flex items-center justify-center gap-2"
                    >
                        <LogOut size={16} />
                        Logout
                    </Button>
                </div>
            </aside>

            {/* Close Button for Mobile */}
            {isOpen && (
                <button
                    onClick={onClose}
                    className="fixed top-4 right-4 lg:hidden z-50 p-2 bg-neon-blue rounded-lg"
                >
                    <X size={24} />
                </button>
            )}
        </>
    );
};

export default Sidebar;
