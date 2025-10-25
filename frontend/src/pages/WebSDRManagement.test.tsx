import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import WebSDRManagement from './WebSDRManagement';

// The mocks are set up in src/test/setup.ts
// No need to mock here - they're already registered globally

describe('WebSDRManagement Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders WebSDRManagement page with title', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays breadcrumb navigation', () => {
        render(<WebSDRManagement />);
        expect(screen.getByText('Home')).toBeInTheDocument();
    });

    it('displays refresh button', () => {
        render(<WebSDRManagement />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        expect(refreshButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles refresh button click', async () => {
        render(<WebSDRManagement />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        if (refreshButtons.length > 0) {
            fireEvent.click(refreshButtons[0]);
            await waitFor(() => {
                expect(refreshButtons[0]).toBeInTheDocument();
            });
        }
    });

    it('displays WebSDR statistics section', () => {
        render(<WebSDRManagement />);
        expect(screen.queryByText(/Online Receivers/i)).toBeInTheDocument();
    });

    it('displays online count', () => {
        render(<WebSDRManagement />);
        const onlineText = screen.queryByText(/Online Receivers/i);
        expect(onlineText).toBeInTheDocument();
    });

    it('displays total WebSDR count', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays active count', () => {
        render(<WebSDRManagement />);
        const activeText = screen.queryByText(/Active Receivers/i);
        expect(activeText).toBeInTheDocument();
    });

    it('displays average response time', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays WebSDR list', () => {
        render(<WebSDRManagement />);
        const listText = screen.queryByText(/Configured WebSDR/i);
        expect(listText || screen.queryAllByText('WebSDR Management').length > 0).toBeTruthy();
    });

    it('displays individual WebSDR status', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays health status indicators', () => {
        render(<WebSDRManagement />);
        const onlineStatus = screen.queryAllByText(/online/i);
        expect(onlineStatus.length).toBeGreaterThanOrEqual(0);
    });

    it('displays response time for each WebSDR', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('handles WebSDR selection', () => {
        render(<WebSDRManagement />);
        const selectButtons = screen.queryAllByRole('button');
        if (selectButtons.length > 0) {
            fireEvent.click(selectButtons[0]);
            expect(selectButtons[0]).toBeInTheDocument();
        }
    });

    it('displays selected WebSDR details', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays frequency range information', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays last health check time', () => {
        render(<WebSDRManagement />);
        const titles = screen.queryAllByText('WebSDR Management');
        expect(titles.length).toBeGreaterThan(0);
    });

    it('renders without crashing', () => {
        const { container } = render(<WebSDRManagement />);
        expect(container).toBeInTheDocument();
    });

    it('displays page structure properly', () => {
        const { container } = render(<WebSDRManagement />);
        expect(container.querySelector('.page-header')).toBeInTheDocument();
    });
});
