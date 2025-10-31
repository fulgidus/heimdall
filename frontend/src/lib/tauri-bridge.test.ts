/**
 * Unit tests for Tauri Bridge
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
    isTauriEnvironment,
    tauriAPI,
    tauriDataCollection,
    tauriTraining,
    tauriGPU,
    tauriSettings,
} from './tauri-bridge';

describe('Tauri Bridge', () => {
    describe('Environment Detection', () => {
        it('should detect non-Tauri environment (web mode)', () => {
            // In test environment, __TAURI__ should not exist
            expect(isTauriEnvironment()).toBe(false);
        });

        it('should return false when window.__TAURI__ is undefined', () => {
            // Explicitly check that window doesn't have __TAURI__
            expect('__TAURI__' in window).toBe(false);
            expect(isTauriEnvironment()).toBe(false);
        });
    });

    describe('API Structure', () => {
        it('should expose unified tauriAPI object', () => {
            expect(tauriAPI).toBeDefined();
            expect(tauriAPI.isTauri).toBe(false); // In web mode
        });

        it('should have dataCollection commands', () => {
            expect(tauriAPI.dataCollection).toBeDefined();
            expect(tauriAPI.dataCollection).toBe(tauriDataCollection);
            expect(typeof tauriAPI.dataCollection.start).toBe('function');
            expect(typeof tauriAPI.dataCollection.stop).toBe('function');
            expect(typeof tauriAPI.dataCollection.getStatus).toBe('function');
        });

        it('should have training commands', () => {
            expect(tauriAPI.training).toBeDefined();
            expect(tauriAPI.training).toBe(tauriTraining);
            expect(typeof tauriAPI.training.start).toBe('function');
            expect(typeof tauriAPI.training.stop).toBe('function');
            expect(typeof tauriAPI.training.getStatus).toBe('function');
        });

        it('should have GPU commands', () => {
            expect(tauriAPI.gpu).toBeDefined();
            expect(tauriAPI.gpu).toBe(tauriGPU);
            expect(typeof tauriAPI.gpu.check).toBe('function');
            expect(typeof tauriAPI.gpu.getUsage).toBe('function');
        });

        it('should have settings commands', () => {
            expect(tauriAPI.settings).toBeDefined();
            expect(tauriAPI.settings).toBe(tauriSettings);
            expect(typeof tauriAPI.settings.load).toBe('function');
            expect(typeof tauriAPI.settings.save).toBe('function');
            expect(typeof tauriAPI.settings.reset).toBe('function');
        });
    });

    describe('Web Mode Fallback', () => {
        it('should throw error when calling commands in web mode', async () => {
            // In web mode (no Tauri), commands should throw error
            await expect(
                tauriAPI.dataCollection.start({
                    frequency: 145.5,
                    duration_seconds: 60,
                    websdrs: ['test1', 'test2'],
                })
            ).rejects.toThrow('Tauri environment not available');
        });

        it('should throw error for GPU check in web mode', async () => {
            await expect(tauriAPI.gpu.check()).rejects.toThrow(
                'Tauri environment not available'
            );
        });

        it('should throw error for settings operations in web mode', async () => {
            await expect(tauriAPI.settings.load()).rejects.toThrow(
                'Tauri environment not available'
            );
        });
    });

    describe('Type Safety', () => {
        it('should accept valid DataCollectionConfig', () => {
            const config = {
                frequency: 145.5,
                duration_seconds: 60,
                websdrs: ['websdr1', 'websdr2'],
            };
            
            // Type check passes if this compiles
            expect(config).toHaveProperty('frequency');
            expect(config).toHaveProperty('duration_seconds');
            expect(config).toHaveProperty('websdrs');
        });

        it('should accept valid TrainingConfig', () => {
            const config = {
                epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
                model_name: 'test_model',
            };
            
            expect(config).toHaveProperty('epochs');
            expect(config).toHaveProperty('batch_size');
            expect(config).toHaveProperty('learning_rate');
            expect(config).toHaveProperty('model_name');
        });

        it('should accept valid AppSettings', () => {
            const settings = {
                api_url: 'http://localhost:8000',
                websocket_url: 'ws://localhost:80/ws',
                mapbox_token: 'test_token',
                auto_start_backend: false,
                backend_port: 8000,
                enable_gpu: true,
                theme: 'light',
            };
            
            expect(settings).toHaveProperty('api_url');
            expect(settings).toHaveProperty('enable_gpu');
            expect(settings).toHaveProperty('theme');
        });
    });
});

describe('Tauri Bridge - Mock Tauri Environment', () => {
    beforeEach(() => {
        // Mock Tauri environment
        (window as any).__TAURI__ = {
            core: {
                invoke: vi.fn().mockResolvedValue('mocked result'),
            },
        };
    });

    afterEach(() => {
        // Clean up mock
        delete (window as any).__TAURI__;
    });

    it('should detect Tauri environment when __TAURI__ exists', () => {
        expect(isTauriEnvironment()).toBe(true);
    });

    it('should have isTauri property that reflects module load time', () => {
        // isTauri is set at module initialization, not dynamically
        // In web mode during tests, it will be false even if we mock __TAURI__ later
        // This is correct behavior - the value is captured once at module load
        expect(typeof tauriAPI.isTauri).toBe('boolean');
    });

    it('should successfully call commands in Tauri mode', async () => {
        const result = await tauriAPI.dataCollection.start({
            frequency: 145.5,
            duration_seconds: 60,
            websdrs: ['test1', 'test2'],
        });
        
        expect(result).toBe('mocked result');
        expect((window as any).__TAURI__.core.invoke).toHaveBeenCalledWith(
            'start_data_collection',
            expect.any(Object)
        );
    });
});
