import '@testing-library/jest-dom';
import { beforeEach, beforeAll, vi } from 'vitest';
import React from 'react';
import {
    createMockDashboardStore,
    createMockWebSDRStore,
    createMockSessionStore,
    createMockAuthStore,
} from './mockStoreFactories';

// ============================================
// MAKE REACT AVAILABLE GLOBALLY FOR JSX IN TESTS
// ============================================
// React 19 with new JSX transform doesn't require React import in source files,
// but test files using JSX syntax need React available globally
declare global {
    // Extend globalThis to include React for test JSX
    var React: typeof React;
}
globalThis.React = React;

// ============================================
// SETUP MOCKS - MUST BE BEFORE COMPONENT IMPORTS
// ============================================

const mockDashboardStore = createMockDashboardStore();
const mockWebSDRStore = createMockWebSDRStore();
const mockSessionStore = createMockSessionStore();
const mockAuthStore = createMockAuthStore();

// Mock all stores GLOBALLY
vi.mock('../store', () => ({
    useDashboardStore: () => ({
        ...mockDashboardStore.getState(),
        ...mockDashboardStore,
    }),
    useWebSDRStore: () => ({
        ...mockWebSDRStore.getState(),
        ...mockWebSDRStore,
    }),
    useSessionStore: () => ({
        ...mockSessionStore.getState(),
        ...mockSessionStore,
    }),
    useAuthStore: () => ({
        ...mockAuthStore.getState(),
        ...mockAuthStore,
    }),
}));

vi.mock('../store/sessionStore', () => ({
    useSessionStore: () => ({
        ...mockSessionStore.getState(),
        ...mockSessionStore,
    }),
}));

vi.mock('../store/websdrStore', () => ({
    useWebSDRStore: () => ({
        ...mockWebSDRStore.getState(),
        ...mockWebSDRStore,
    }),
}));

// ============================================
// WINDOW & ENV MOCKS
// ============================================

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

// Mock environment variables
Object.defineProperty(import.meta, 'env', {
    value: {
        VITE_ADMIN_EMAIL: 'admin@heimdall.local',
        VITE_ADMIN_PASSWORD: 'Admin123!@#',
        VITE_API_URL: 'http://localhost:8000/api',
    },
    writable: true,
});

// Clear localStorage before each test
beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
});
