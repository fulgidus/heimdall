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
  const { isAuthenticated, isAdmin, isOperator, isUser, user } = useAuth();

  // Debug logging
  console.log('[RequireRole] Debug:', {
    requiredRole: role,
    isAuthenticated,
    isAdmin,
    isOperator,
    isUser,
    userRole: user?.role,
    userRoles: user?.roles,
  });

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
      <div className="d-flex align-items-center justify-content-center min-vh-100 bg-light">
        <div className="card shadow-lg" style={{ maxWidth: '500px', width: '100%' }}>
          <div className="card-body p-5 text-center">
            <div className="mb-4">
              <svg
                style={{ width: '4rem', height: '4rem' }}
                className="mx-auto text-danger"
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
            </div>
            <h2 className="h3 fw-bold mb-3">Access Denied</h2>
            <p className="text-muted mb-2">
              You don't have permission to access this page.
            </p>
            <p className="small text-secondary">
              Required role: <span className="fw-semibold text-uppercase">{role}</span>
            </p>
            <div className="mt-4">
              <button
                onClick={() => window.history.back()}
                className="btn btn-primary"
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
