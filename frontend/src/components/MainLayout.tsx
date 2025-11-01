import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import MobileMenu from './Navigation/MobileMenu';
import BottomNav from './Navigation/BottomNav';

interface MainLayoutProps {
  children: React.ReactNode;
  title?: string;
  showBackButton?: boolean;
}

const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  title = 'Dashboard',
  showBackButton = false,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleMenuClick = () => {
    // On desktop, toggle sidebar; on mobile, toggle mobile menu
    if (window.innerWidth >= 1024) {
      setSidebarOpen(!sidebarOpen);
    } else {
      setMobileMenuOpen(!mobileMenuOpen);
    }
  };

  return (
    <div className="min-h-screen bg-oxford-blue">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      </div>

      {/* Mobile Menu */}
      <div className="lg:hidden">
        <MobileMenu isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />
      </div>

      <Header onMenuClick={handleMenuClick} title={title} showBackButton={showBackButton} />

      <main
        className="lg:ml-64 p-4 sm:p-6 lg:p-8 mobile-p-4"
        role="main"
        aria-label={title}
        style={{ paddingBottom: '5rem' }}
      >
        {children}
      </main>

      {/* Bottom Navigation for Mobile */}
      <BottomNav />
    </div>
  );
};

export default MainLayout;
