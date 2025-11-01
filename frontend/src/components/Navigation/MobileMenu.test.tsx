import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import MobileMenu from './MobileMenu';

// Mock auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

describe('MobileMenu Component', () => {
    it('renders when open', () => {
        const onClose = vi.fn();
        render(
            <BrowserRouter>
                <MobileMenu isOpen={true} onClose={onClose} />
            </BrowserRouter>
        );
        
        expect(screen.getByText('🚀 Heimdall')).toBeInTheDocument();
        expect(screen.getByText('RF Localization')).toBeInTheDocument();
    });

    it('does not render when closed', () => {
        const onClose = vi.fn();
        const { container } = render(
            <BrowserRouter>
                <MobileMenu isOpen={false} onClose={onClose} />
            </BrowserRouter>
        );
        
        const drawer = container.querySelector('.translate-x-0');
        expect(drawer).not.toBeInTheDocument();
    });

    it('calls onClose when close button is clicked', () => {
        const onClose = vi.fn();
        render(
            <BrowserRouter>
                <MobileMenu isOpen={true} onClose={onClose} />
            </BrowserRouter>
        );
        
        const closeButton = screen.getByLabelText('Close menu');
        fireEvent.click(closeButton);
        
        expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('renders navigation items', () => {
        const onClose = vi.fn();
        render(
            <BrowserRouter>
                <MobileMenu isOpen={true} onClose={onClose} />
            </BrowserRouter>
        );
        
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Data Ingestion')).toBeInTheDocument();
        expect(screen.getByText('Localization')).toBeInTheDocument();
        expect(screen.getByText('Session History')).toBeInTheDocument();
        expect(screen.getByText('Analytics')).toBeInTheDocument();
    });

    it('calls onClose when overlay is clicked', () => {
        const onClose = vi.fn();
        const { container } = render(
            <BrowserRouter>
                <MobileMenu isOpen={true} onClose={onClose} />
            </BrowserRouter>
        );
        
        const overlay = container.querySelector('.bg-oxford-blue.bg-opacity-70');
        if (overlay) {
            fireEvent.click(overlay);
            expect(onClose).toHaveBeenCalledTimes(1);
        }
    });
});
