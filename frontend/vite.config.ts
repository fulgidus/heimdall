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
        host: '0.0.0.0', // Allow external connections
        proxy: {
            '/api': {
                target: 'http://localhost',
                changeOrigin: true,
                secure: false,
                ws: true, // WebSocket support
                configure: (proxy, _options) => {
                    proxy.on('error', (err, _req, _res) => {
                        console.log('proxy error', err);
                    });
                    proxy.on('proxyReq', (proxyReq, req, _res) => {
                        console.log('Sending Request:', req.method, req.url, '→', proxyReq.path);
                    });
                    proxy.on('proxyRes', (proxyRes, req, _res) => {
                        console.log('Received Response:', proxyRes.statusCode, req.url);
                    });
                },
            },
            '/ws': {
                target: 'ws://localhost',
                ws: true,
                changeOrigin: true,
            },
            '/backend': {
                target: 'http://localhost',
                changeOrigin: true,
            },
            '/training': {
                target: 'http://localhost',
                changeOrigin: true,
            },
            '/inference': {
                target: 'http://localhost',
                changeOrigin: true,
            },
            '/health': {
                target: 'http://localhost',
                changeOrigin: true,
            },
        },
    },
    build: {
        target: 'esnext',
        outDir: 'dist',
        sourcemap: false,
        minify: 'esbuild',
        cssCodeSplit: true,
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
                // Optimize chunk file names for better caching
                chunkFileNames: 'assets/[name]-[hash].js',
                entryFileNames: 'assets/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash].[ext]',
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
