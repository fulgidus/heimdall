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
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                pure_funcs: ['console.log', 'console.info'],
            },
        },
        cssCodeSplit: true,
        rollupOptions: {
            output: {
                manualChunks: {
                    'mapbox': ['mapbox-gl'],
                    'vendor': ['react', 'react-dom'],
                    'router': ['react-router-dom'],
                    'ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-label', '@radix-ui/react-separator', '@radix-ui/react-slot', '@radix-ui/react-tooltip'],
                    'charts': ['chart.js', 'react-chartjs-2'],
                    'data': ['@tanstack/react-query', 'axios', 'zustand'],
                },
                chunkFileNames: 'assets/js/[name]-[hash].js',
                entryFileNames: 'assets/js/[name]-[hash].js',
                assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
