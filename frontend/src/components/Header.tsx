import React, { useState, useRef, useEffect } from 'react';
import { Menu, Bell, User, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface HeaderProps {
  onMenuClick: () => void;
  title?: string;
  showBackButton?: boolean;
}

const Header: React.FC<HeaderProps> = ({
  onMenuClick,
  title = 'Dashboard',
  showBackButton = false,
}) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const navigate = useNavigate();
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
    <header className="bg-oxford-blue border-b border-neon-blue border-opacity-20 sticky top-0 z-20 backdrop-blur-sm sticky-top-mobile">
      <div className="h-16 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Left Side - Menu/Back Button */}
        <div className="flex items-center gap-2">
          {/* Menu Button (Mobile/Tablet) */}
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
            aria-label="Toggle navigation menu"
            aria-expanded="false"
          >
            <Menu size={24} className="text-neon-blue" aria-hidden="true" />
          </button>

          {/* Back Button (Mobile only, when applicable) */}
          {showBackButton && (
            <button
              onClick={() => navigate(-1)}
              className="md:hidden p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
              aria-label="Go back"
            >
              <ArrowLeft size={24} className="text-neon-blue" aria-hidden="true" />
            </button>
          )}
        </div>

        {/* Title */}
        <h1 className="text-lg sm:text-xl font-bold text-french-gray flex-1 text-center lg:text-left lg:ml-0 ml-2 mobile-text-base">
          {title}
        </h1>

        {/* Right Actions */}
        <div className="flex items-center gap-2 sm:gap-4">
          {/* Notifications */}
          <div className="relative" ref={notificationRef}>
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
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
                className="absolute right-0 mt-2 w-64 sm:w-80 bg-sea-green rounded-lg shadow-2xl p-4 text-sm text-white modal-fullscreen-mobile"
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
              className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
              aria-label="User profile menu"
              aria-expanded={showProfile}
              aria-haspopup="true"
            >
              <User size={20} className="text-neon-blue" aria-hidden="true" />
            </button>
            {showProfile && (
              <div
                className="absolute right-0 mt-2 w-48 sm:w-56 bg-sea-green rounded-lg shadow-2xl overflow-hidden"
                role="menu"
                aria-label="User menu"
              >
                <div className="p-4 border-b border-white border-opacity-10">
                  <p className="font-semibold text-white">John Doe</p>
                  <p className="text-sm text-french-gray">admin@heimdall.com</p>
                </div>
                <a
                  href="#"
                  className="block px-4 py-3 hover:bg-oxford-blue text-white touch-target"
                  role="menuitem"
                >
                  Profile
                </a>
                <a
                  href="#"
                  className="block px-4 py-3 hover:bg-oxford-blue text-white touch-target"
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
