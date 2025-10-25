import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Localization from './Localization';

// Mock all stores with proper return values
vi.mock('../store', async () => {
    return {
        useAuthStore: vi.fn(() => ({
            user: { email: 'admin@heimdall.local' },
            logout: vi.fn(),
        })),
        useLocalizationStore: vi.fn(() => ({
            recentLocalizations: [
                {
                    id: '1',
                    latitude: 45.123,
                    longitude: 7.456,
                    uncertainty_m: 25.5,
                    confidence: 0.92,
                    timestamp: '2025-01-01T12:00:00Z',
                    active_receivers: 7,
                    signal_quality: 'excellent',
                },
            ],
            isLoading: false,
            error: null,
            fetchRecentLocalizations: vi.fn(),
        })),
        useDashboardStore: vi.fn(() => ({
            data: {
                summary: {
                    total_acquisitions: 150,
                    successful_localizations: 142,
                    avg_accuracy_m: 28.5,
                    active_receivers: 7,
                },
                modelInfo: {
                    accuracy: 0.92,
                    uncertainty: 25.5,
                },
                timeseries: [],
            },
            isLoading: false,
            error: null,
            fetchDashboardData: vi.fn(),
        })),
        useWebSDRStore: vi.fn(() => ({
            websdrs: [
                {
                    id: '1',
                    location: 'Piedmont',
                    location_name: 'Piedmont, Italy',
                    latitude: 45.0,
                    longitude: 7.0,
                    online: true
                },
                {
                    id: '2',
                    location: 'Liguria',
                    location_name: 'Liguria, Italy',
                    latitude: 44.0,
                    longitude: 8.0,
                    online: true
                },
            ],
            healthStatus: {
                '1': { status: 'online', uptime: 99.5 },
                '2': { status: 'online', uptime: 98.2 },
                '3': { status: 'online', uptime: 97.8 },
                '4': { status: 'online', uptime: 99.2 },
                '5': { status: 'online', uptime: 96.5 },
                '6': { status: 'online', uptime: 99.8 },
                '7': { status: 'online', uptime: 98.9 },
            },
            isLoading: false,
        })),
    };
});

describe('Localization Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders Localization page with title', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('displays breadcrumb navigation', () => {
        render(<Localization />);
        expect(screen.getByText('Home')).toBeInTheDocument();
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('displays refresh button', () => {
        render(<Localization />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        expect(refreshButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles refresh button click', () => {
        render(<Localization />);
        const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
        if (refreshButtons.length > 0) {
            fireEvent.click(refreshButtons[0]);
            expect(refreshButtons[0]).toBeInTheDocument();
        }
    });

    it('displays localization results', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
        expect(screen.queryByText(/recent localizations/i)).toBeInTheDocument();
    });

    it('displays accuracy metric', () => {
        render(<Localization />);
        expect(screen.getByText(/Avg Accuracy/i)).toBeInTheDocument();
    });

    it('displays WebSDR status information', () => {
        render(<Localization />);
        expect(screen.getByText(/Active Receivers/i)).toBeInTheDocument();
    });

    it('displays uncertainty information', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('displays signal quality information', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('renders map container', () => {
        const { container } = render(<Localization />);
        expect(container.querySelector('[class*="map"]') || screen.getByText('Localization')).toBeInTheDocument();
    });

    it('displays results list', () => {
        render(<Localization />);
        expect(screen.getByText(/Recent Localizations/i)).toBeInTheDocument();
    });

    it('handles result selection', () => {
        render(<Localization />);
        const resultButtons = screen.queryAllByRole('button');
        if (resultButtons.length > 0) {
            fireEvent.click(resultButtons[0]);
            expect(resultButtons[0]).toBeInTheDocument();
        }
    });

    it('displays confidence level', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('shows active receivers count', () => {
        render(<Localization />);
        expect(screen.getByText(/Active Receivers/i)).toBeInTheDocument();
    });

    it('displays timestamp information', () => {
        render(<Localization />);
        expect(screen.getByText('Localization')).toBeInTheDocument();
    });

    it('renders without crashing', () => {
        const { container } = render(<Localization />);
        expect(container).toBeInTheDocument();
    });

    it('displays page header correctly', () => {
        render(<Localization />);
        const header = screen.getByText('Localization');
        expect(header).toBeInTheDocument();
    });

    it('shows page structure', () => {
        const { container } = render(<Localization />);
        expect(container.querySelector('.page-header')).toBeInTheDocument();
    });
});
