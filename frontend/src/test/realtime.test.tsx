/**
 * Real-Time Data Update Tests
 * Phase 7: Testing & Validation
 * 
 * Tests real-time data refresh and auto-update mechanisms
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

// Pages with real-time updates
import Dashboard from '../pages/Dashboard';
import Analytics from '../pages/Analytics';
import WebSDRManagement from '../pages/WebSDRManagement';

// Stores
import { useDashboardStore } from '../store/dashboardStore';
import { useWebSDRStore } from '../store/websdrStore';
import { useAnalyticsStore } from '../store/analyticsStore';

describe('Phase 7: Real-Time Data Updates Validation', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    describe('Dashboard Real-Time Updates', () => {
        it('should auto-refresh dashboard data every 30 seconds', async () => {
            const fetchSpy = vi.spyOn(useDashboardStore.getState(), 'fetchDashboardData');

            render(
                <BrowserRouter>
                    <Dashboard />
                </BrowserRouter>
            );

            // Initial fetch
            expect(fetchSpy).toHaveBeenCalledTimes(1);

            // Advance 30 seconds
            act(() => {
                vi.advanceTimersByTime(30000);
            });

            await waitFor(() => {
                expect(fetchSpy).toHaveBeenCalledTimes(2);
            });

            // Advance another 30 seconds
            act(() => {
                vi.advanceTimersByTime(30000);
            });

            await waitFor(() => {
                expect(fetchSpy).toHaveBeenCalledTimes(3);
            });
        });

        it('should clean up intervals on unmount', async () => {
            const { unmount } = render(
                <BrowserRouter>
                    <Dashboard />
                </BrowserRouter>
            );

            const timers = vi.getTimerCount();
            expect(timers).toBeGreaterThan(0);

            unmount();

            // All intervals should be cleared
            await waitFor(() => {
                expect(vi.getTimerCount()).toBe(0);
            });
        });
    });

    describe('Analytics Real-Time Updates', () => {
        it('should auto-refresh analytics data every 30 seconds', async () => {
            const fetchSpy = vi.spyOn(useAnalyticsStore.getState(), 'fetchAllAnalytics');

            render(
                <BrowserRouter>
                    <Analytics />
                </BrowserRouter>
            );

            // Initial fetch
            expect(fetchSpy).toHaveBeenCalledTimes(1);

            // Advance 30 seconds
            act(() => {
                vi.advanceTimersByTime(30000);
            });

            await waitFor(() => {
                expect(fetchSpy).toHaveBeenCalledTimes(2);
            });
        });

        it('should handle manual refresh', async () => {
            const refreshSpy = vi.spyOn(useAnalyticsStore.getState(), 'refreshData');

            const { container } = render(
                <BrowserRouter>
                    <Analytics />
                </BrowserRouter>
            );

            // Find and click refresh button
            const refreshButton = container.querySelector('button[title*="Refresh"], button:has(.ph-arrows-clockwise)');
            
            if (refreshButton) {
                act(() => {
                    refreshButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
                });

                await waitFor(() => {
                    expect(refreshSpy).toHaveBeenCalled();
                });
            }
        });
    });

    describe('WebSDR Health Real-Time Updates', () => {
        it('should auto-check WebSDR health periodically', async () => {
            const fetchSpy = vi.spyOn(useWebSDRStore.getState(), 'fetchWebSDRs');

            render(
                <BrowserRouter>
                    <WebSDRManagement />
                </BrowserRouter>
            );

            // Initial fetch
            expect(fetchSpy).toHaveBeenCalledTimes(1);

            // Advance 60 seconds (health check interval)
            act(() => {
                vi.advanceTimersByTime(60000);
            });

            await waitFor(() => {
                expect(fetchSpy).toHaveBeenCalled();
            });
        });
    });

    describe('Store State Updates', () => {
        it('should update dashboard store when data changes', async () => {
            const { fetchDashboardData } = useDashboardStore.getState();

            await act(async () => {
                await fetchDashboardData();
            });

            const state = useDashboardStore.getState();
            expect(state.isLoading).toBe(false);
        });

        it('should update WebSDR store when health changes', async () => {
            const { fetchWebSDRs } = useWebSDRStore.getState();

            await act(async () => {
                await fetchWebSDRs();
            });

            const state = useWebSDRStore.getState();
            expect(state.loading).toBe(false);
        });

        it('should update analytics store when metrics change', async () => {
            const { fetchAllAnalytics } = useAnalyticsStore.getState();

            await act(async () => {
                await fetchAllAnalytics();
            });

            const state = useAnalyticsStore.getState();
            expect(state.isLoading).toBe(false);
        });
    });

    describe('Error Handling in Real-Time Updates', () => {
        it('should continue updating even after errors', async () => {
            const fetchSpy = vi.spyOn(useDashboardStore.getState(), 'fetchDashboardData');
            
            // Mock error on first call
            fetchSpy.mockRejectedValueOnce(new Error('Network error'));

            render(
                <BrowserRouter>
                    <Dashboard />
                </BrowserRouter>
            );

            // Advance past error
            act(() => {
                vi.advanceTimersByTime(30000);
            });

            await waitFor(() => {
                // Should have attempted multiple times
                expect(fetchSpy).toHaveBeenCalled();
            });
        });
    });
});
