import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

interface MainLayoutProps {
    children: React.ReactNode;
    title?: string;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, title = 'Dashboard' }) => {
    const [sidebarOpen, setSidebarOpen] = useState(false);

    return (
        <div className="min-h-screen bg-oxford-blue">
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
            <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} title={title} />

            <main className="lg:ml-64 p-4 sm:p-6 lg:p-8">
                {children}
            </main>
        </div>
    );
};

export default MainLayout;
