import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { X } from 'lucide-react';

interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

const navigation = [
  { name: 'Dashboard', path: '/dashboard', icon: '📊' },

  { name: 'Localization', path: '/localization', icon: '🎯' },
  { name: 'Session History', path: '/sessions', icon: '📝' },
  { name: 'Analytics', path: '/analytics', icon: '📈' },
  { name: 'WebSDR Management', path: '/websdr-management', icon: '🛰️' },
  { name: 'System Status', path: '/system-status', icon: '⚙️' },
  { name: 'Settings', path: '/settings', icon: '🔧' },
  { name: 'Profile', path: '/profile', icon: '👤' },
];

const MobileMenu: React.FC<MobileMenuProps> = ({ isOpen, onClose }) => {
  const location = useLocation();

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-oxford-blue bg-opacity-70 z-40"
          onClick={onClose}
          style={{ backdropFilter: 'blur(4px)' }}
        />
      )}

      {/* Mobile Menu Drawer */}
      <div
        className={`fixed top-0 left-0 h-full w-80 max-w-[85vw] bg-gradient-to-b from-oxford-blue to-sea-green border-r border-neon-blue border-opacity-20 transform transition-transform duration-300 ease-in-out z-50 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        style={{ backgroundColor: 'var(--oxford-blue)' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neon-blue border-opacity-20">
          <div>
            <h1 className="text-xl font-bold text-neon-blue">🚀 Heimdall</h1>
            <p className="text-sm text-french-gray mt-1">RF Localization</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
            aria-label="Close menu"
          >
            <X size={24} className="text-neon-blue" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-4 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 160px)' }}>
          {navigation.map(item => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={onClose}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all duration-200 touch-target ${
                  isActive
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
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-neon-blue border-opacity-20">
          <button
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors touch-target"
            onClick={() => {
              // Handle logout
              console.log('Logout clicked');
            }}
          >
            <span>🚪</span>
            <span>Logout</span>
          </button>
        </div>
      </div>
    </>
  );
};

export default MobileMenu;
