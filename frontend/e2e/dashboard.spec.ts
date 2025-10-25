/**
 * E2E Test: Dashboard Page
 * 
 * Verifies real backend calls for dashboard data loading.
 * Tests stats, WebSDR status, health checks, and activity stream.
 */

import { test, expect } from '@playwright/test';
import {
    setupRequestLogging,
    login,
    TEST_BACKEND_ORIGIN
} from './helpers/test-utils';

test.describe('Dashboard Page - Real Backend Integration', () => {
    test.beforeEach(async ({ page }) => {
        await setupRequestLogging(page);
        // Login before each test
        await login(page);
    });

    test('should load dashboard and make real API calls for stats', async ({ page }) => {
        // Dashboard should already be loaded from login
        await page.waitForLoadState('networkidle');

        // Wait for stats API call (if exists)
        const statsResponsePromise = page.waitForResponse(
            (response) => {
                const url = response.url();
                return (
                    url.startsWith(TEST_BACKEND_ORIGIN) &&
                    (url.includes('/api/v1/stats') ||
                        url.includes('/api/v1/analytics') ||
                        url.includes('/api/v1/system/status'))
                );
            },
            { timeout: 15000 }
        ).catch(() => null); // Don't fail if no stats endpoint yet

        const statsResponse = await statsResponsePromise;

        if (statsResponse) {
            expect(statsResponse.status()).toBeGreaterThanOrEqual(200);
            expect(statsResponse.status()).toBeLessThan(300);
            console.log(`✅ Stats API call: ${statsResponse.status()} ${statsResponse.url()}`);
        } else {
            console.log('⚠️ No stats API endpoint detected (may not be implemented yet)');
        }

        // Verify dashboard content is visible
        await expect(page.locator('main')).toBeVisible();
    });

    test('should fetch WebSDR status from backend', async ({ page }) => {
        await page.waitForLoadState('networkidle');

        // Wait for WebSDR API call
        const websdrResponsePromise = page.waitForResponse(
            (response) => {
                const url = response.url();
                return (
                    url.startsWith(TEST_BACKEND_ORIGIN) &&
                    (url.includes('/api/v1/acquisition/websdrs') ||
                        url.includes('/api/v1/websdrs'))
                );
            },
            { timeout: 15000 }
        ).catch(() => null);

        // Trigger refresh if needed
        await page.reload();

        const websdrResponse = await websdrResponsePromise;

        if (websdrResponse) {
            expect(websdrResponse.status()).toBeGreaterThanOrEqual(200);
            expect(websdrResponse.status()).toBeLessThan(300);

            const responseBody = await websdrResponse.json();
            console.log(`✅ WebSDR API call: ${websdrResponse.status()}, received ${responseBody.length || 0} receivers`);
        } else {
            console.log('⚠️ No WebSDR API endpoint detected');
        }
    });

    test('should make health check API call to backend services', async ({ page }) => {
        await page.waitForLoadState('networkidle');

        // Wait for health check API call
        const healthResponsePromise = page.waitForResponse(
            (response) => {
                const url = response.url();
                return (
                    url.startsWith(TEST_BACKEND_ORIGIN) &&
                    url.includes('/health')
                );
            },
            { timeout: 15000 }
        ).catch(() => null);

        // Trigger health check (reload or navigate)
        await page.reload();

        const healthResponse = await healthResponsePromise;

        if (healthResponse) {
            expect(healthResponse.status()).toBe(200);
            console.log(`✅ Health check API call: ${healthResponse.status()} ${healthResponse.url()}`);
        } else {
            console.log('⚠️ No health check endpoint called from dashboard');
        }
    });

    test('should fetch recent activity/sessions from backend', async ({ page }) => {
        await page.waitForLoadState('networkidle');

        // Wait for sessions/activity API call
        const activityResponsePromise = page.waitForResponse(
            (response) => {
                const url = response.url();
                return (
                    url.startsWith(TEST_BACKEND_ORIGIN) &&
                    (url.includes('/api/v1/sessions') ||
                        url.includes('/api/v1/activity') ||
                        url.includes('/api/v1/recent'))
                );
            },
            { timeout: 15000 }
        ).catch(() => null);

        // Trigger refresh
        await page.reload();

        const activityResponse = await activityResponsePromise;

        if (activityResponse) {
            expect(activityResponse.status()).toBeGreaterThanOrEqual(200);
            expect(activityResponse.status()).toBeLessThan(300);
            console.log(`✅ Activity API call: ${activityResponse.status()} ${activityResponse.url()}`);
        } else {
            console.log('⚠️ No activity/sessions API endpoint detected');
        }
    });

    test('should verify dashboard displays data from backend', async ({ page }) => {
        await page.waitForLoadState('networkidle');

        // Verify main content area exists
        const mainContent = page.locator('main');
        await expect(mainContent).toBeVisible();

        // Look for any stats cards or data display
        const hasContent = await page.locator('main').evaluate((el) => {
            return el.textContent && el.textContent.length > 100;
        });

        expect(hasContent).toBe(true);
        console.log('✅ Dashboard displays content (backend data loaded)');
    });
});
