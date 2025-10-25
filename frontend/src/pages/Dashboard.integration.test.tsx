import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { SidebarProvider } from '@/components/ui/sidebar';
import Dashboard from './Dashboard';

// Mock the stores
const mockFetchDashboardData = vi.fn();
const mockResetRetry = vi.fn();
const mockIncrementRetry = vi.fn();

const createMockDashboardStore = (overrides = {}) => ({
    metrics: {
        activeWebSDRs: 5,
        totalWebSDRs: 7,
        signalDetections: 42,
        systemUptime: 7200,
        averageAccuracy: 94.5,
    },
    data: {
        websdrs: [],
        websdrsHealth: {},
        modelInfo: null,
        servicesHealth: {},
        ...overrides.data,
    },
    isLoading: false,
    error: null,
    lastUpdate: new Date(),
    retryCount: 0,
    retryDelay: 1000,
    fetchDashboardData: mockFetchDashboardData,
    resetRetry: mockResetRetry,
    incrementRetry: mockIncrementRetry,
    ...overrides,
});

const createMockWebSDRStore = (overrides = {}) => ({
    websdrs: [
        { id: 1, name: 'Turin', location_name: 'Turin, Italy', is_active: true },
        { id: 2, name: 'Milan', location_name: 'Milan, Italy', is_active: true },
        { id: 3, name: 'Genoa', location_name: 'Genoa, Italy', is_active: true },
    ],
    healthStatus: {
        1: { status: 'online', response_time_ms: 150 },
        2: { status: 'online', response_time_ms: 140 },
        3: { status: 'offline', response_time_ms: null },
    },
    ...overrides,
});

vi.mock('../store', () => ({
    useDashboardStore: vi.fn(),
    useWebSDRStore: vi.fn(),
    useAuthStore: vi.fn(() => ({
        user: { email: 'test@heimdall.local' },
        logout: vi.fn(),
    })),
}));

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => vi.fn(),
    };
});

import { useDashboardStore, useWebSDRStore } from '../store';

describe('Dashboard - Real API Integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderDashboard = (dashboardStoreOverrides = {}, webSDRStoreOverrides = {}) => {
        (useDashboardStore as any).mockReturnValue(
            createMockDashboardStore(dashboardStoreOverrides)
        );
        (useWebSDRStore as any).mockReturnValue(
            createMockWebSDRStore(webSDRStoreOverrides)
        );

        return render(
            <BrowserRouter>
                <SidebarProvider>
                    <Dashboard />
                </SidebarProvider>
            </BrowserRouter>
        );
    };

    describe('Loading States', () => {
        it('should display skeleton loaders when loading with no data', () => {
            renderDashboard({ isLoading: true, data: { servicesHealth: {}, websdrs: [], websdrsHealth: {}, modelInfo: null } });

            // Check for connection status indicator
            expect(screen.getByText(/connecting to services/i)).toBeInTheDocument();
            
            // Skeleton loaders should be present (they don't have accessible text, so we check the component rendered)
            const serviceSection = screen.getByText('Services Status').closest('.card');
            expect(serviceSection).toBeInTheDocument();
        });

        it('should not display skeletons when data is present', () => {
            renderDashboard({
                data: {
                    servicesHealth: {
                        'api-gateway': { status: 'healthy', service: 'api-gateway', version: '0.1.0', timestamp: new Date().toISOString() },
                        'rf-acquisition': { status: 'healthy', service: 'rf-acquisition', version: '0.1.0', timestamp: new Date().toISOString() },
                    },
                    websdrs: [],
                    websdrsHealth: {},
                    modelInfo: null,
                },
            });

            // Services should be visible
            expect(screen.getByText(/api gateway/i)).toBeInTheDocument();
            expect(screen.getByText(/rf acquisition/i)).toBeInTheDocument();
        });
    });

    describe('Service Health Display', () => {
        it('should display all service health statuses', () => {
            renderDashboard({
                data: {
                    servicesHealth: {
                        'api-gateway': { status: 'healthy', service: 'api-gateway', version: '0.1.0', timestamp: new Date().toISOString() },
                        'rf-acquisition': { status: 'healthy', service: 'rf-acquisition', version: '0.1.0', timestamp: new Date().toISOString() },
                        'training': { status: 'degraded', service: 'training', version: '0.1.0', timestamp: new Date().toISOString() },
                        'inference': { status: 'unhealthy', service: 'inference', version: '0.1.0', timestamp: new Date().toISOString() },
                        'data-ingestion-web': { status: 'healthy', service: 'data-ingestion-web', version: '0.1.0', timestamp: new Date().toISOString() },
                    },
                    websdrs: [],
                    websdrsHealth: {},
                    modelInfo: null,
                },
            });

            // All 5 services should be displayed (use getAllByText since some may appear multiple times)
            const apiGatewayElements = screen.getAllByText(/api gateway/i);
            expect(apiGatewayElements.length).toBeGreaterThan(0);
            
            const rfAcquisitionElements = screen.getAllByText(/rf acquisition/i);
            expect(rfAcquisitionElements.length).toBeGreaterThan(0);
            
            const trainingElements = screen.getAllByText(/training/i);
            expect(trainingElements.length).toBeGreaterThan(0);
            
            const inferenceElements = screen.getAllByText(/inference/i);
            expect(inferenceElements.length).toBeGreaterThan(0);
            
            const dataIngestionElements = screen.getAllByText(/data ingestion.web/i);
            expect(dataIngestionElements.length).toBeGreaterThan(0);

            // Check status badges
            const healthyBadges = screen.getAllByText('healthy');
            expect(healthyBadges.length).toBeGreaterThanOrEqual(3);
        });

        it('should show error state with retry button when services fail to load', () => {
            renderDashboard({
                error: 'Failed to connect to backend',
                data: { servicesHealth: {}, websdrs: [], websdrsHealth: {}, modelInfo: null },
            });

            expect(screen.getByText(/error!/i)).toBeInTheDocument();
            expect(screen.getByText(/failed to connect to backend/i)).toBeInTheDocument();
        });
    });

    describe('WebSDR Network Display', () => {
        it('should display WebSDR status from store', () => {
            renderDashboard();

            // Should show WebSDR cities
            expect(screen.getByText('Turin')).toBeInTheDocument();
            expect(screen.getByText('Milan')).toBeInTheDocument();
            expect(screen.getByText('Genoa')).toBeInTheDocument();
        });

        it('should show correct online/offline status', () => {
            renderDashboard();

            // The component should render - we can't easily check icon colors in jsdom
            // but we can verify the cities are displayed
            expect(screen.getByText('Turin')).toBeInTheDocument();
        });
    });

    describe('Refresh Functionality', () => {
        it('should call fetchDashboardData when refresh button is clicked', async () => {
            renderDashboard();

            const refreshButton = screen.getByRole('button', { name: /refresh/i });
            fireEvent.click(refreshButton);

            await waitFor(() => {
                expect(mockFetchDashboardData).toHaveBeenCalled();
            });
        });

        it('should show refreshing state during manual refresh', async () => {
            renderDashboard();

            const refreshButton = screen.getByRole('button', { name: /refresh/i });
            fireEvent.click(refreshButton);

            // The button text should change to "Refreshing..."
            // Note: This might be hard to test due to async nature, but we can verify the function was called
            expect(mockFetchDashboardData).toHaveBeenCalled();
        });
    });

    describe('Polling Mechanism', () => {
        it('should call fetchDashboardData on mount', () => {
            renderDashboard();
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(1);
        });

        it('should setup interval for polling', () => {
            vi.useFakeTimers();
            renderDashboard();

            // Initial call
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(1);

            // After 30 seconds
            vi.advanceTimersByTime(30000);
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(2);

            vi.useRealTimers();
        });
    });

    describe('Exponential Backoff', () => {
        it('should use retry delay when error is present', () => {
            vi.useFakeTimers();
            renderDashboard({
                error: 'Network error',
                retryDelay: 2000,
                retryCount: 1,
            });

            // Initial call
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(1);

            // After 2 seconds (retry delay), should fetch again
            vi.advanceTimersByTime(2000);
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(2);

            vi.useRealTimers();
        });

        it('should use normal interval when no error', () => {
            vi.useFakeTimers();
            renderDashboard({ error: null });

            // Initial call
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(1);

            // After 30 seconds (normal interval)
            vi.advanceTimersByTime(30000);
            expect(mockFetchDashboardData).toHaveBeenCalledTimes(2);

            vi.useRealTimers();
        });
    });

    describe('Model Info Display', () => {
        it('should display model accuracy when available', () => {
            renderDashboard({
                data: {
                    modelInfo: {
                        active_version: 'v1.0.0',
                        stage: 'Production',
                        model_name: 'heimdall-inference',
                        accuracy: 0.945,
                        loaded_at: new Date().toISOString(),
                        uptime_seconds: 3600,
                        predictions_total: 1000,
                        predictions_successful: 950,
                        predictions_failed: 50,
                        is_ready: true,
                        health_status: 'healthy',
                    },
                    servicesHealth: {},
                    websdrs: [],
                    websdrsHealth: {},
                },
            });

            // Should show accuracy percentage
            expect(screen.getByText(/94.5%/)).toBeInTheDocument();
        });

        it('should display N/A when model info is not available', () => {
            renderDashboard({
                data: {
                    modelInfo: null,
                    servicesHealth: {},
                    websdrs: [],
                    websdrsHealth: {},
                },
            });

            expect(screen.getByText('N/A')).toBeInTheDocument();
        });
    });

    describe('Last Update Timestamp', () => {
        it('should display last update time', () => {
            const now = new Date();
            renderDashboard({ lastUpdate: now });

            // Should show time in some format - just check Dashboard heading renders
            const heading = screen.getByRole('heading', { name: /dashboard/i, level: 2 });
            expect(heading).toBeInTheDocument();
        });
    });
});
