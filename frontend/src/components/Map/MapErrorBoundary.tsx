/**
 * MapErrorBoundary Component
 *
 * Error boundary to catch and gracefully handle errors in Mapbox components.
 * Prevents entire application crash when map rendering fails.
 */

import React from 'react';

interface MapErrorBoundaryProps {
  children: React.ReactNode;
}

interface MapErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class MapErrorBoundary extends React.Component<
  MapErrorBoundaryProps,
  MapErrorBoundaryState
> {
  constructor(props: MapErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<MapErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error details for debugging
    console.error('[MapErrorBoundary] Caught error:', error);
    console.error('[MapErrorBoundary] Error info:', errorInfo);

    // Update state with error info
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      const errorMessage = this.state.error?.message || 'An unexpected error occurred';
      const isMapboxError = errorMessage.toLowerCase().includes('mapbox');

      return (
        <div
          className="d-flex align-items-center justify-content-center bg-body-secondary"
          style={{ height: '500px', minHeight: '400px' }}
        >
          <div className="alert alert-danger mb-0 m-4" style={{ maxWidth: '600px' }}>
            <div className="d-flex align-items-start">
              <i className="ph ph-warning-circle fs-3 me-3 text-danger"></i>
              <div className="flex-grow-1">
                <h5 className="alert-heading mb-2">Map Error</h5>
                <p className="mb-2">
                  <strong>{errorMessage}</strong>
                </p>

                {isMapboxError && (
                  <small className="text-muted d-block mb-3">
                    <i className="ph ph-info me-1"></i>
                    Make sure your Mapbox access token is configured correctly in the .env file
                    (VITE_MAPBOX_TOKEN).
                  </small>
                )}

                {/* Error details (collapsed by default in production) */}
                {import.meta.env.DEV && this.state.errorInfo && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-muted">
                      <small>Show technical details</small>
                    </summary>
                    <pre className="bg-light p-2 mt-2 rounded" style={{ fontSize: '0.75rem' }}>
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}

                <div className="d-flex gap-2 mt-3">
                  <button className="btn btn-sm btn-danger" onClick={this.handleReset}>
                    <i className="ph ph-arrow-clockwise me-1"></i>
                    Retry
                  </button>
                  <button
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => window.location.reload()}
                  >
                    <i className="ph ph-arrow-counter-clockwise me-1"></i>
                    Reload Page
                  </button>
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

export default MapErrorBoundary;
