import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { useAuthStore } from './store';
import DattaLayout from './components/layout/DattaLayout';
import { ErrorBoundary } from './components/ErrorBoundary';

// Lazy load all pages for code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Settings = lazy(() => import('./pages/Settings'));
const Projects = lazy(() => import('./pages/Projects'));
const Profile = lazy(() => import('./pages/Profile'));
const Localization = lazy(() => import('./pages/Localization'));
const Login = lazy(() => import('./pages/Login'));
const RecordingSession = lazy(() => import('./pages/RecordingSession'));
const SessionHistory = lazy(() => import('./pages/SessionHistory'));
const WebSDRManagement = lazy(() => import('./pages/WebSDRManagement'));
const SystemStatus = lazy(() => import('./pages/SystemStatus'));
const DataIngestion = lazy(() => import('./pages/DataIngestion'));
const SourcesManagement = lazy(() => import('./pages/SourcesManagement'));

// Loading fallback component
const LoadingFallback = () => (
    <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-gray-100"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
    </div>
);

interface ProtectedRouteProps {
    children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
    const { isAuthenticated } = useAuthStore();
    
    // Allow bypass in development mode when VITE_ENABLE_DEBUG is true
    const isDevelopment = import.meta.env.DEV || import.meta.env.VITE_ENV === 'development';
    const debugMode = import.meta.env.VITE_ENABLE_DEBUG === 'true';
    
    // Bypass auth in development/debug mode
    if (isDevelopment && debugMode) {
        return <DattaLayout>{children}</DattaLayout>;
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return <DattaLayout>{children}</DattaLayout>;
};

function App() {
    return (
        <ErrorBoundary>
            <Router>
                <Suspense fallback={<LoadingFallback />}>
                    <Routes>
                        {/* Public Routes */}
                        <Route path="/login" element={<Login />} />

                        {/* Protected Routes */}
                        <Route
                            path="/dashboard"
                            element={
                                <ProtectedRoute>
                                    <Dashboard />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/localization"
                            element={
                                <ProtectedRoute>
                                    <Localization />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/analytics"
                            element={
                                <ProtectedRoute>
                                    <Analytics />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/data-ingestion"
                            element={
                                <ProtectedRoute>
                                    <DataIngestion />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/projects"
                            element={
                                <ProtectedRoute>
                                    <Projects />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/settings"
                            element={
                                <ProtectedRoute>
                                    <Settings />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/profile"
                            element={
                                <ProtectedRoute>
                                    <Profile />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/recording"
                            element={
                                <ProtectedRoute>
                                    <RecordingSession />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/history"
                            element={
                                <ProtectedRoute>
                                    <SessionHistory />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/websdrs"
                            element={
                                <ProtectedRoute>
                                    <WebSDRManagement />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/system-status"
                            element={
                                <ProtectedRoute>
                                    <SystemStatus />
                                </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/sources"
                            element={
                                <ProtectedRoute>
                                    <SourcesManagement />
                                </ProtectedRoute>
                            }
                        />

                        {/* Redirect */}
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="*" element={<Navigate to="/dashboard" replace />} />
                    </Routes>
                </Suspense>
            </Router>
        </ErrorBoundary>
    );
}

export default App;
