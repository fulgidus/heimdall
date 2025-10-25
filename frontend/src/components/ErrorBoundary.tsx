/**
 * Error Boundary Component
 * 
 * Catches and handles errors gracefully in the component tree
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';

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
                <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
                    <div className="max-w-md w-full">
                        <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6">
                            <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-100 dark:bg-red-900 rounded-full">
                                <i className="ph ph-warning-circle text-red-600 dark:text-red-400 text-2xl"></i>
                            </div>
                            <h1 className="mt-4 text-xl font-semibold text-center text-gray-900 dark:text-gray-100">
                                Something went wrong
                            </h1>
                            <p className="mt-2 text-sm text-center text-gray-600 dark:text-gray-400">
                                {this.state.error?.message || 'An unexpected error occurred'}
                            </p>
                            <div className="mt-6 flex flex-col gap-3">
                                <button
                                    onClick={this.handleReset}
                                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors"
                                >
                                    Try Again
                                </button>
                                <button
                                    onClick={() => window.location.href = '/dashboard'}
                                    className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 font-medium rounded-md transition-colors"
                                >
                                    Go to Dashboard
                                </button>
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
                    <h3 className="text-sm font-medium text-red-800 dark:text-red-300">
                        Error loading data
                    </h3>
                    <div className="mt-2 text-sm text-red-700 dark:text-red-400">
                        {error.message}
                    </div>
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
