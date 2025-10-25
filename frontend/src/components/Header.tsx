import React, { useState, useRef, useEffect } from 'react';
import { Menu, Bell, User } from 'lucide-react';

interface HeaderProps {
    onMenuClick: () => void;
    title?: string;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick, title = 'Dashboard' }) => {
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const notificationRef = useRef<HTMLDivElement>(null);
    const profileRef = useRef<HTMLDivElement>(null);

    // Close dropdowns when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
                setShowNotifications(false);
            }
            if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
                setShowProfile(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <header className="bg-oxford-blue border-b border-neon-blue border-opacity-20 sticky top-0 z-20 backdrop-blur-sm">
            <div className="h-16 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
                {/* Menu Button (Mobile) */}
                <button
                    onClick={onMenuClick}
                    className="lg:hidden p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                    aria-label="Toggle navigation menu"
                    aria-expanded="false"
                >
                    <Menu size={24} className="text-neon-blue" aria-hidden="true" />
                </button>

                {/* Title */}
                <h1 className="text-xl font-bold text-french-gray flex-1 text-center lg:text-left lg:ml-0 ml-2">
                    {title}
                </h1>

                {/* Right Actions */}
                <div className="flex items-center gap-4">
                    {/* Notifications */}
                    <div className="relative" ref={notificationRef}>
                        <button
                            onClick={() => setShowNotifications(!showNotifications)}
                            className="relative p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                            aria-label="Notifications"
                            aria-expanded={showNotifications}
                            aria-haspopup="true"
                        >
                            <Bell size={20} className="text-neon-blue" aria-hidden="true" />
                            <span 
                                className="absolute top-1 right-1 w-2 h-2 bg-light-green rounded-full"
                                aria-label="2 unread notifications"
                            ></span>
                        </button>
                        {showNotifications && (
                            <div 
                                className="absolute right-0 mt-2 w-64 bg-sea-green rounded-lg shadow-2xl p-4 text-sm text-white"
                                role="menu"
                                aria-label="Notifications menu"
                            >
                                <div className="text-french-gray font-semibold mb-2">Notifications</div>
                                <div className="space-y-2" role="list">
                                    <p role="listitem">✓ System update completed</p>
                                    <p role="listitem">✓ New metrics available</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Profile */}
                    <div className="relative" ref={profileRef}>
                        <button
                            onClick={() => setShowProfile(!showProfile)}
                            className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors"
                            aria-label="User profile menu"
                            aria-expanded={showProfile}
                            aria-haspopup="true"
                        >
                            <User size={20} className="text-neon-blue" aria-hidden="true" />
                        </button>
                        {showProfile && (
                            <div 
                                className="absolute right-0 mt-2 w-48 bg-sea-green rounded-lg shadow-2xl overflow-hidden"
                                role="menu"
                                aria-label="User menu"
                            >
                                <div className="p-4 border-b border-white border-opacity-10">
                                    <p className="font-semibold text-white">John Doe</p>
                                    <p className="text-sm text-french-gray">admin@heimdall.com</p>
                                </div>
                                <a 
                                    href="#" 
                                    className="block px-4 py-2 hover:bg-oxford-blue text-white"
                                    role="menuitem"
                                >
                                    Profile
                                </a>
                                <a 
                                    href="#" 
                                    className="block px-4 py-2 hover:bg-oxford-blue text-white"
                                    role="menuitem"
                                >
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
