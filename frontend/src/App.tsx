import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
    Dashboard,
    Analytics,
    Settings,
    Projects,
    Profile,
    Localization,
    Login,
    RecordingSession,
    SessionHistory,
    WebSDRManagement,
    SystemStatus,
    DataIngestion,
} from './pages';
import { useAuthStore } from './store';
import DattaLayout from './components/layout/DattaLayout';

interface ProtectedRouteProps {
    children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
    const { isAuthenticated } = useAuthStore();

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return <DattaLayout>{children}</DattaLayout>;
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

                {/* Redirect */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
        </Router>
    );
}

export default App;
