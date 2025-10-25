import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

const __dirname = path.dirname(new URL(import.meta.url).pathname);

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
            VITE_ADMIN_PASSWORD: 'admin',
        },
        mockReset: true,
        restoreMocks: true,
        coverage: {
            provider: 'v8',
            reporter: ['text', 'text-summary', 'html'],
            exclude: [
                'node_modules/',
                'src/test/setup.ts',
                '**/*.d.ts',
                '**/*.test.{ts,tsx}',
                '**/*.spec.{ts,tsx}',
                '**/*.integration.test.{ts,tsx}',
                '**/index.ts',
                'e2e/**',
                'e2e-artifacts-*/**',
                'playwright-report/**',
                'test-results/**',
            ],
        },
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
});
