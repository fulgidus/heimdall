import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    server: {
        port: 3001,
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                rewrite: (path) => {
                    // Don't rewrite /api/v1/auth/* - pass directly to backend
                    if (path.startsWith('/api/v1/auth')) {
                        return path;
                    }
                    // Rewrite other /api/* by removing /api prefix
                    return path.replace(/^\/api/, '');
                },
            },
        },
    },
    build: {
        target: 'esnext',
        outDir: 'dist',
        sourcemap: false,
    },
})
