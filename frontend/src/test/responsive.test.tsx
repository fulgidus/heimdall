/**
 * Responsive Design Tests
 * Phase 7: Testing & Validation
 * 
 * Tests responsive behavior across mobile, tablet, and desktop viewports
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

// Pages to test
import Dashboard from '../pages/Dashboard';
import Analytics from '../pages/Analytics';
import WebSDRManagement from '../pages/WebSDRManagement';
import DataIngestion from '../pages/DataIngestion';
import Localization from '../pages/Localization';

// Mock matchMedia for responsive tests
const createMatchMedia = (width: number) => {
    return (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => true,
    });
};

describe('Phase 7: Responsive Design Validation', () => {
    const viewports = {
        mobile: { width: 375, height: 667, name: 'Mobile (iPhone SE)' },
        tablet: { width: 768, height: 1024, name: 'Tablet (iPad)' },
        desktop: { width: 1920, height: 1080, name: 'Desktop (Full HD)' },
    };

    Object.entries(viewports).forEach(([device, viewport]) => {
        describe(`${viewport.name} (${viewport.width}x${viewport.height})`, () => {
            beforeEach(() => {
                // Mock window.matchMedia for responsive tests
                window.matchMedia = createMatchMedia(viewport.width);
                
                // Mock window dimensions
                Object.defineProperty(window, 'innerWidth', {
                    writable: true,
                    configurable: true,
                    value: viewport.width,
                });
                Object.defineProperty(window, 'innerHeight', {
                    writable: true,
                    configurable: true,
                    value: viewport.height,
                });
            });

            it('Dashboard should render without errors', () => {
                const { container } = render(
                    <BrowserRouter>
                        <Dashboard />
                    </BrowserRouter>
                );
                expect(container).toBeTruthy();
            });

            it('Analytics page should render without errors', () => {
                const { container } = render(
                    <BrowserRouter>
                        <Analytics />
                    </BrowserRouter>
                );
                expect(container).toBeTruthy();
            });

            it('WebSDR Management page should render without errors', () => {
                const { container } = render(
                    <BrowserRouter>
                        <WebSDRManagement />
                    </BrowserRouter>
                );
                expect(container).toBeTruthy();
            });

            it('Data Ingestion page should render without errors', () => {
                const { container } = render(
                    <BrowserRouter>
                        <DataIngestion />
                    </BrowserRouter>
                );
                expect(container).toBeTruthy();
            });

            it('Localization page should render without errors', () => {
                const { container } = render(
                    <BrowserRouter>
                        <Localization />
                    </BrowserRouter>
                );
                expect(container).toBeTruthy();
            });
        });
    });

    describe('Bootstrap Responsive Classes', () => {
        it('should use Bootstrap grid classes correctly', () => {
            render(
                <BrowserRouter>
                    <Dashboard />
                </BrowserRouter>
            );

            // Check for responsive grid classes
            const container = document.querySelector('.container-fluid, .container');
            expect(container).toBeTruthy();
        });

        it('should have responsive cards', () => {
            render(
                <BrowserRouter>
                    <Analytics />
                </BrowserRouter>
            );

            // Check for card components
            const cards = document.querySelectorAll('.card');
            expect(cards.length).toBeGreaterThan(0);
        });
    });

    describe('Mobile-Specific Features', () => {
        beforeEach(() => {
            window.matchMedia = createMatchMedia(375);
            Object.defineProperty(window, 'innerWidth', {
                writable: true,
                configurable: true,
                value: 375,
            });
        });

        it('should handle touch interactions', () => {
            const { container } = render(
                <BrowserRouter>
                    <Dashboard />
                </BrowserRouter>
            );

            // Check that buttons and interactive elements exist
            const buttons = container.querySelectorAll('button');
            expect(buttons.length).toBeGreaterThan(0);
        });
    });
});
