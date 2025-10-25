import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        react(),
        visualizer({
            filename: 'dist/stats.html',
            open: false,
            gzipSize: true,
            brotliSize: true,
        }) as Plugin,
    ],
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
        rollupOptions: {
            output: {
                manualChunks: (id) => {
                    if (id.includes('node_modules')) {
                        if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) {
                            return 'react-vendor';
                        }
                        if (id.includes('chart.js') || id.includes('react-chartjs')) {
                            return 'chart-vendor';
                        }
                        if (id.includes('@radix-ui')) {
                            return 'ui-vendor';
                        }
                        if (id.includes('@tanstack') || id.includes('axios') || id.includes('zustand')) {
                            return 'data-vendor';
                        }
                        return 'vendor';
                    }
                },
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
