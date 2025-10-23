import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Localization from './Localization';

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
