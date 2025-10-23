import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    plugins: [react()],
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./src/test/setup.ts'],
        include: ['src/**/*.test.{ts,tsx}'],
        exclude: ['node_modules', 'dist'],
        env: {
            VITE_ADMIN_EMAIL: 'admin@heimdall.local',
            VITE_ADMIN_PASSWORD: 'Admin123!@#',
        },
        mockReset: true,
        restoreMocks: true,
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
});
