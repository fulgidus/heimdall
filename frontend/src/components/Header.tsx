import React, { useState } from 'react';
import { Menu, Bell, User } from 'lucide-react';

interface HeaderProps {
    onMenuClick: () => void;
    title?: string;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick, title = 'Dashboard' }) => {
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    return (
        <header className="bg-oxford-blue border-b border-neon-blue border-opacity-20 sticky top-0 z-20 backdrop-blur-sm">
            <div className="h-16 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
                {/* Menu Button (Mobile) */}
                <button
                    onClick={onMenuClick}
                    className="lg:hidden p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                >
                    <Menu size={24} className="text-neon-blue" />
                </button>

                {/* Title */}
                <h2 className="text-xl font-bold text-french-gray flex-1 text-center lg:text-left lg:ml-0 ml-2">
                    {title}
                </h2>

                {/* Right Actions */}
                <div className="flex items-center gap-4">
                    {/* Notifications */}
                    <div className="relative">
                        <button
                            onClick={() => setShowNotifications(!showNotifications)}
                            className="relative p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                        >
                            <Bell size={20} className="text-neon-blue" />
                            <span className="absolute top-1 right-1 w-2 h-2 bg-light-green rounded-full"></span>
                        </button>
                        {showNotifications && (
                            <div className="absolute right-0 mt-2 w-64 bg-sea-green rounded-lg shadow-2xl p-4 text-sm text-white">
                                <div className="text-french-gray font-semibold mb-2">Notifications</div>
                                <div className="space-y-2">
                                    <p>✓ System update completed</p>
                                    <p>✓ New metrics available</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Profile */}
                    <div className="relative">
                        <button
                            onClick={() => setShowProfile(!showProfile)}
                            className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                        >
                            <User size={20} className="text-neon-blue" />
                        </button>
                        {showProfile && (
                            <div className="absolute right-0 mt-2 w-48 bg-sea-green rounded-lg shadow-2xl overflow-hidden">
                                <div className="p-4 border-b border-white border-opacity-10">
                                    <p className="font-semibold text-white">John Doe</p>
                                    <p className="text-sm text-french-gray">admin@heimdall.com</p>
                                </div>
                                <a href="#" className="block px-4 py-2 hover:bg-oxford-blue text-white">
                                    Profile
                                </a>
                                <a href="#" className="block px-4 py-2 hover:bg-oxford-blue text-white">
                                    Settings
                                </a>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
