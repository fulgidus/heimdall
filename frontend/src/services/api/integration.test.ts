/**
 * API Integration Tests
 * Phase 6: API Integration Verification
 */

import { describe, it, expect } from 'vitest';

describe('Phase 6: API Integration Verification', () => {
    it('should verify test setup is working', () => {
        expect(true).toBe(true);
    });

    it('should import all API services without errors', async () => {
        const { default: webSDRService } = await import('./websdr');
        const { default: acquisitionService } = await import('./acquisition');
        const { default: inferenceService } = await import('./inference');
        const { default: sessionService } = await import('./session');
        const { default: analyticsService } = await import('./analytics');
        const { default: systemService } = await import('./system');

        expect(webSDRService).toBeDefined();
        expect(acquisitionService).toBeDefined();
        expect(inferenceService).toBeDefined();
        expect(sessionService).toBeDefined();
        expect(analyticsService).toBeDefined();
        expect(systemService).toBeDefined();
    }, 15000);  // Increase timeout for dynamic imports
});
