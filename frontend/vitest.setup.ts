import { beforeAll, vi } from 'vitest';
import React from 'react';

// Mock window.matchMedia
beforeAll(() => {
    Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: (query: string) => ({
            matches: false,
            media: query,
            onchange: null,
            addListener: () => { },
            removeListener: () => { },
            addEventListener: () => { },
            removeEventListener: () => { },
            dispatchEvent: () => true,
        }),
    });
});

// Mock Chart.js to prevent DOM errors in tests
vi.mock('chart.js', () => ({
    Chart: class MockChart {
        constructor() { }
        destroy() { }
        update() { }
        resize() { }
    },
}));

// Mock react-chartjs-2
vi.mock('react-chartjs-2', () => ({
    Line: vi.fn(() => React.createElement('div', { 'data-testid': 'mock-line-chart' })),
    Bar: vi.fn(() => React.createElement('div', { 'data-testid': 'mock-bar-chart' })),
    Pie: vi.fn(() => React.createElement('div', { 'data-testid': 'mock-pie-chart' })),
    Doughnut: vi.fn(() => React.createElement('div', { 'data-testid': 'mock-doughnut-chart' })),
}));
