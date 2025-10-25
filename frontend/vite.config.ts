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
        port: 5173,
        strictPort: false,
        proxy: {
            '/api': {
                target: process.env.VITE_API_URL || 'http://localhost:8000',
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
        assetsDir: 'assets',
        sourcemap: true, // Required for production debugging
        minify: true, // Let rolldown-vite handle minification with its default minifier
        cssCodeSplit: true,
        rollupOptions: {
            output: {
                manualChunks: (id) => {
                    if (id.includes('node_modules')) {
                        if (id.includes('mapbox-gl')) {
                            return 'mapbox';
                        }
                        if (id.includes('react') || id.includes('react-dom')) {
                            return 'vendor';
                        }
                        if (id.includes('react-router')) {
                            return 'router';
                        }
                        if (id.includes('@radix-ui')) {
                            return 'ui';
                        }
                        if (id.includes('chart.js') || id.includes('react-chartjs')) {
                            return 'charts';
                        }
                        if (id.includes('@tanstack') || id.includes('axios') || id.includes('zustand')) {
                            return 'data';
                        }
                        return 'vendor';
                    }
                },
                chunkFileNames: 'assets/js/[name]-[hash].js',
                entryFileNames: 'assets/js/[name]-[hash].js',
                assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
