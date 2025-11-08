import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

/**
 * Role-based route protection component.
 * Implements role hierarchy: admin > operator > user
 * 
 * Usage:
 * <RequireRole role="admin">
 *   <AdminOnlyPage />
 * </RequireRole>
 */

interface RequireRoleProps {
  role: 'admin' | 'operator' | 'user';
  children: React.ReactNode;
  fallback?: React.ReactNode; // Optional custom fallback component
}

export const RequireRole: React.FC<RequireRoleProps> = ({ role, children, fallback }) => {
  const { isAuthenticated, isAdmin, isOperator, isUser } = useAuth();

  // Not authenticated - redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check role hierarchy
  let hasAccess = false;
  switch (role) {
    case 'admin':
      hasAccess = isAdmin;
      break;
    case 'operator':
      hasAccess = isOperator; // operator OR admin
      break;
    case 'user':
      hasAccess = isUser; // all authenticated users
      break;
  }

  // Access denied - show fallback or default message
  if (!hasAccess) {
    if (fallback) {
      return <>{fallback}</>;
    }

    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="max-w-md w-full px-6 py-8 bg-white dark:bg-gray-800 shadow-lg rounded-lg">
          <div className="text-center">
            <svg
              className="mx-auto h-12 w-12 text-red-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <h2 className="mt-4 text-2xl font-bold text-gray-900 dark:text-white">
              Access Denied
            </h2>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              You don't have permission to access this page.
            </p>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-500">
              Required role: <span className="font-semibold">{role}</span>
            </p>
            <div className="mt-6">
              <button
                onClick={() => window.history.back()}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Go Back
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // User has access
  return <>{children}</>;
};

export default RequireRole;
