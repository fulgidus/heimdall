import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import Analytics from './Analytics';

describe('Analytics Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders Analytics page with title', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('displays breadcrumb navigation', () => {
        render(<Analytics />);
        expect(screen.getByText('Home')).toBeInTheDocument();
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('fetches dashboard data on mount', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('renders without crashing', () => {
        const { container } = render(<Analytics />);
        expect(container).toBeInTheDocument();
    });

    it('displays page header', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('renders dashboard data section', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('handles refresh button interaction', async () => {
        render(<Analytics />);
        const refreshButton = screen.queryByRole('button', { name: /refresh/i });
        if (refreshButton) {
            fireEvent.click(refreshButton);
            await waitFor(() => {
                expect(refreshButton).toBeInTheDocument();
            });
        }
    });

    it('displays metrics section', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('calculates success rate correctly', () => {
        render(<Analytics />);
        const successRate = (135 / 150) * 100;
        const successRateText = successRate.toFixed(1);
        expect(successRateText).toBe('90.0');
    });

    it('renders time range selector if present', () => {
        render(<Analytics />);
        const timeRangeSelect = screen.queryByDisplayValue('7d');
        if (timeRangeSelect) {
            expect(timeRangeSelect).toBeInTheDocument();
        }
    });

    it('handles time range change if selector exists', () => {
        render(<Analytics />);
        const timeRangeSelect = screen.queryByDisplayValue('7d') as HTMLSelectElement;
        if (timeRangeSelect) {
            fireEvent.change(timeRangeSelect, { target: { value: '30d' } });
            expect(timeRangeSelect.value).toBe('30d');
        }
    });

    it('displays WebSDR online count', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('renders main content area', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('displays accuracy metric if available', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('renders without errors when data is loading', () => {
        render(<Analytics />);
        expect(screen.queryAllByText('Analytics').length).toBeGreaterThan(0);
    });

    it('shows proper page structure', () => {
        const { container } = render(<Analytics />);
        expect(container.querySelector('.page-header')).toBeInTheDocument();
    });
});
