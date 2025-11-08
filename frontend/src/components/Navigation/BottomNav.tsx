import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const BottomNav: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: 'ph-house' },
    { name: 'Sessions', path: '/sessions', icon: 'ph-record' },
    { name: 'Localization', path: '/localization', icon: 'ph-crosshair' },
    { name: 'More', path: '/settings', icon: 'ph-dots-three' },
  ];

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 bg-oxford-blue border-t border-neon-blue border-opacity-20 md:hidden z-30"
      style={{
        backgroundColor: 'var(--oxford-blue)',
        paddingBottom: 'env(safe-area-inset-bottom, 0px)', // iOS safe area
      }}
    >
      <div className="flex items-center justify-around h-16">
        {navItems.map(item => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex flex-col items-center justify-center flex-1 h-full touch-target ${
                isActive ? 'text-light-green' : 'text-french-gray'
              }`}
              style={{
                minWidth: '60px',
                transition: 'color 0.2s ease',
              }}
            >
              <i
                className={`ph ${item.icon} text-2xl mb-1`}
                style={{
                  color: isActive ? 'var(--light-green)' : 'var(--french-gray)',
                }}
              ></i>
              <span
                className="text-xs font-medium"
                style={{
                  color: isActive ? 'var(--light-green)' : 'var(--french-gray)',
                }}
              >
                {item.name}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
};

export default BottomNav;
