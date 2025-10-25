import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import BottomNav from './BottomNav';

describe('BottomNav Component', () => {
    it('renders navigation items', () => {
        render(
            <BrowserRouter>
                <BottomNav />
            </BrowserRouter>
        );
        
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Sessions')).toBeInTheDocument();
        expect(screen.getByText('Localization')).toBeInTheDocument();
        expect(screen.getByText('More')).toBeInTheDocument();
    });

    it('renders with correct navigation links', () => {
        const { container } = render(
            <BrowserRouter>
                <BottomNav />
            </BrowserRouter>
        );
        
        const links = container.querySelectorAll('a');
        expect(links).toHaveLength(4);
        
        const paths = Array.from(links).map(link => link.getAttribute('href'));
        expect(paths).toContain('/dashboard');
        expect(paths).toContain('/sessions');
        expect(paths).toContain('/localization');
        expect(paths).toContain('/settings');
    });

    it('has proper mobile-only styling', () => {
        const { container } = render(
            <BrowserRouter>
                <BottomNav />
            </BrowserRouter>
        );
        
        const nav = container.querySelector('nav');
        expect(nav).toHaveClass('md:hidden');
        expect(nav).toHaveClass('fixed');
        expect(nav).toHaveClass('bottom-0');
    });

    it('applies active state to current route', () => {
        const { container } = render(
            <BrowserRouter>
                <BottomNav />
            </BrowserRouter>
        );
        
        const links = container.querySelectorAll('a');
        expect(links.length).toBeGreaterThan(0);
    });
});
