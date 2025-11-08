/**
 * Error Boundary Component
 *
 * Catches and handles errors gracefully in the component tree
 */

import React, { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="d-flex align-items-center justify-content-center min-vh-100 bg-light text-dark">
          <div className="container">
            <div className="row justify-content-center">
              <div className="col-md-6 col-lg-5">
                <div className="card shadow-lg border-0">
                  <div className="card-body p-5 text-center">
                    {/* Error Icon */}
                    <div className="mb-4">
                      <div className="d-inline-flex align-items-center justify-content-center rounded-circle bg-danger bg-opacity-10 p-3">
                        <i className="ph ph-warning-octagon text-danger" style={{ fontSize: '3rem' }}></i>
                      </div>
                    </div>

                    {/* Error Title */}
                    <h2 className="card-title mb-3 fw-bold text-danger">
                      Something Went Wrong
                    </h2>

                    {/* Error Message */}
                    <div className="alert alert-danger mb-4" role="alert">
                      <i className="ph ph-info me-2"></i>
                      <strong>Error:</strong> {this.state.error?.message || 'An unexpected error occurred'}
                    </div>

                    {/* Error Details (Stack trace in development) */}
                    {import.meta.env.DEV && this.state.error?.stack && (
                      <details className="text-start mb-4">
                        <summary className="btn btn-sm btn-outline-secondary mb-2">
                          <i className="ph ph-bug me-2"></i>
                          Show Technical Details
                        </summary>
                        <pre className="bg-dark text-light p-3 rounded small" style={{ maxHeight: '200px', overflow: 'auto' }}>
                          <code>{this.state.error.stack}</code>
                        </pre>
                      </details>
                    )}

                    {/* Action Buttons */}
                    <div className="d-grid gap-2">
                      <button
                        onClick={this.handleReset}
                        className="btn btn-primary btn-lg"
                      >
                        <i className="ph ph-arrow-clockwise me-2"></i>
                        Try Again
                      </button>
                      <button
                        onClick={() => (window.location.href = '/dashboard')}
                        className="btn btn-outline-secondary"
                      >
                        <i className="ph ph-house me-2"></i>
                        Go to Dashboard
                      </button>
                    </div>

                    {/* Help Text */}
                    <div className="mt-4 text-muted small">
                      <i className="ph ph-lightbulb me-1"></i>
                      If this problem persists, please contact support or check the browser console for more details.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Query Error Component
 *
 * Displays errors from React Query queries
 */
interface QueryErrorProps {
  error: Error | null;
  onRetry?: () => void;
}

export const QueryError: React.FC<QueryErrorProps> = ({ error, onRetry }) => {
  if (!error) return null;

  return (
    <div className="rounded-lg border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20 p-4">
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <i className="ph ph-warning-circle text-red-600 dark:text-red-400 text-xl"></i>
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium text-red-800 dark:text-red-300">Error loading data</h3>
          <div className="mt-2 text-sm text-red-700 dark:text-red-400">{error.message}</div>
          {onRetry && (
            <div className="mt-4">
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/40 hover:bg-red-200 dark:hover:bg-red-900/60 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                <i className="ph ph-arrow-clockwise mr-2"></i>
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Loading Skeleton Component
 *
 * Displays a loading skeleton while data is being fetched
 */
export const LoadingSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`animate-pulse ${className}`}>
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
    </div>
  );
};
