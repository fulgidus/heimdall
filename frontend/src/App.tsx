import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { useAuthStore } from './store';
import DattaLayout from './components/layout/DattaLayout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { useTokenRefresh } from './hooks/useTokenRefresh';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { RequireRole } from './components/auth/RequireRole';

// Lazy load all pages for code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Settings = lazy(() => import('./pages/Settings'));
const Profile = lazy(() => import('./pages/Profile'));
const Localization = lazy(() => import('./pages/Localization'));
const Login = lazy(() => import('./pages/Login'));
const RecordingSession = lazy(() => import('./pages/RecordingSession'));
const SessionHistory = lazy(() => import('./pages/SessionHistory'));
const WebSDRManagement = lazy(() => import('./pages/WebSDRManagement'));
const SystemStatus = lazy(() => import('./pages/SystemStatus'));
const DataIngestion = lazy(() => import('./pages/DataIngestion'));
const SourcesManagement = lazy(() => import('./pages/SourcesManagement'));
const ImportExport = lazy(() => import('./pages/ImportExport'));
const Training = lazy(() => import('./pages/Training'));
const TerrainManagement = lazy(() => import('./pages/TerrainManagement'));
const AudioLibrary = lazy(() => import('./pages/AudioLibrary'));

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
  // Enable automatic token refresh
  useTokenRefresh();

  return (
    <ErrorBoundary>
      <WebSocketProvider autoConnect={true}>
        <Router>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {/* Public Routes */}
              <Route path="/login" element={<Login />} />

              {/* Protected Routes */}

              {/* User+ Routes (All Authenticated Users) */}
              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <Dashboard />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/localization"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <Localization />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/analytics"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <Analytics />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/profile"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <Profile />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/recording"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <RecordingSession />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/history"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <SessionHistory />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/system-status"
                element={
                  <ProtectedRoute>
                    <RequireRole role="user">
                      <SystemStatus />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />

              {/* Operator+ Routes */}
              <Route
                path="/websdrs"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <WebSDRManagement />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/sources"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <SourcesManagement />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/import-export"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <ImportExport />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/training"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <Training />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/terrain"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <TerrainManagement />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/audio-library"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <AudioLibrary />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/data-ingestion"
                element={
                  <ProtectedRoute>
                    <RequireRole role="operator">
                      <DataIngestion />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />

              {/* Admin-only Routes */}
              <Route
                path="/settings"
                element={
                  <ProtectedRoute>
                    <RequireRole role="admin">
                      <Settings />
                    </RequireRole>
                  </ProtectedRoute>
                }
              />

              {/* Redirect */}
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Suspense>
        </Router>
      </WebSocketProvider>
    </ErrorBoundary>
  );
    }

export default App;
