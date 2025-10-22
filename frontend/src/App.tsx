import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Dashboard, Analytics, Settings, Projects, Profile, Login } from './pages';
import { useAuthStore } from './store';
import { SidebarProvider } from '@/components/ui/sidebar';
import './index.css';

interface ProtectedRouteProps {
    children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
    const { isAuthenticated } = useAuthStore();

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return <>{children}</>;
};

function App() {
    return (
        <Router>
            <Routes>
                {/* Public Routes */}
                <Route path="/login" element={<Login />} />

                {/* Protected Routes */}
                <Route
                    path="/dashboard"
                    element={
                        <SidebarProvider>
                            <ProtectedRoute>
                                <Dashboard />
                            </ProtectedRoute>
                        </SidebarProvider>
                    }
                />
                <Route
                    path="/analytics"
                    element={
                        <SidebarProvider>
                            <ProtectedRoute>
                                <Analytics />
                            </ProtectedRoute>
                        </SidebarProvider>
                    }
                />
                <Route
                    path="/projects"
                    element={
                        <SidebarProvider>
                            <ProtectedRoute>
                                <Projects />
                            </ProtectedRoute>
                        </SidebarProvider>
                    }
                />
                <Route
                    path="/settings"
                    element={
                        <SidebarProvider>
                            <ProtectedRoute>
                                <Settings />
                            </ProtectedRoute>
                        </SidebarProvider>
                    }
                />
                <Route
                    path="/profile"
                    element={
                        <SidebarProvider>
                            <ProtectedRoute>
                                <Profile />
                            </ProtectedRoute>
                        </SidebarProvider>
                    }
                />

                {/* Redirect */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
        </Router>
    );
}

export default App;
