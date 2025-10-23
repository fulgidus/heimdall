import '@testing-library/jest-dom';
import { beforeEach, beforeAll } from 'vitest';

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
});
