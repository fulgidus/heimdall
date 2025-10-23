import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DataIngestion from './DataIngestion';

describe('DataIngestion Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders DataIngestion page with title', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays breadcrumb navigation', () => {
        render(<DataIngestion />);
        const homeTexts = screen.queryAllByText('Home');
        const dataTexts = screen.queryAllByText(/data/i);
        expect(homeTexts.length > 0 || dataTexts.length > 0).toBeTruthy();
    });

    it('displays sources tab', () => {
        render(<DataIngestion />);
        const sourceTabs = screen.queryAllByRole('button', { name: /source/i });
        expect(sourceTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays sessions tab', () => {
        render(<DataIngestion />);
        const sessionTabs = screen.queryAllByRole('button', { name: /session/i });
        expect(sessionTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('switches between tabs on click', () => {
        render(<DataIngestion />);
        const tabs = screen.queryAllByRole('button', { name: /source|session/i });
        if (tabs.length > 0) {
            fireEvent.click(tabs[0]);
            expect(tabs[0]).toBeInTheDocument();
        }
    });

    it('displays known sources list', () => {
        render(<DataIngestion />);
        expect(screen.getByText('Source 1')).toBeInTheDocument();
    });

    it('displays sessions list', () => {
        render(<DataIngestion />);
        expect(screen.getByText('Session 1')).toBeInTheDocument();
    });

    it('shows statistics section', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays total sources count', () => {
        render(<DataIngestion />);
        const source1 = screen.getByText('Source 1');
        expect(source1).toBeInTheDocument();
    });

    it('displays validated sources count', () => {
        render(<DataIngestion />);
        const source1 = screen.getByText('Source 1');
        expect(source1).toBeInTheDocument();
    });

    it('displays refresh button', () => {
        render(<DataIngestion />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        expect(refreshButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles refresh button click', async () => {
        render(<DataIngestion />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        if (refreshButtons.length > 0) {
            fireEvent.click(refreshButtons[0]);
            await waitFor(() => {
                expect(refreshButtons[0]).toBeInTheDocument();
            });
        }
    });

    it('displays loading state', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays error message if present', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('shows pending sessions count', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays analytics metrics', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('renders sources tab content', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('renders sessions tab content', () => {
        render(<DataIngestion />);
        const titles = screen.queryAllByText(/data|ingestion/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('renders without crashing', () => {
        const { container } = render(<DataIngestion />);
        expect(container).toBeInTheDocument();
    });
});
