/**
 * E2E Test Helpers
 * 
 * Utilities for Playwright E2E tests
 */

import { expect } from '@playwright/test';
import type { Page, Response, Request } from '@playwright/test';

/**
 * Backend origin for API requests (via Envoy API Gateway)
 */
export const TEST_BACKEND_ORIGIN = process.env.TEST_BACKEND_ORIGIN || 'http://localhost:80';

/**
 * Wait for a real backend API call and verify response
 * 
 * @param page - Playwright page
 * @param urlPattern - URL pattern to match (e.g., '/api/v1/sessions')
 * @param expectedStatus - Expected HTTP status code (default: 200-299)
 * @returns Response object
 */
export async function waitForBackendCall(
    page: Page,
    urlPattern: string | RegExp,
    expectedStatus: number | { min: number; max: number } = { min: 200, max: 299 }
) {
    console.log(`🔍 Waiting for backend call: ${urlPattern}`);

    const response = await page.waitForResponse(
        (response: Response) => {
            const url = response.url();
            const status = response.status();

            // Check if URL matches pattern
            const urlMatches = typeof urlPattern === 'string'
                ? url.includes(urlPattern)
                : urlPattern.test(url);

            // Check if URL is from backend origin
            const isBackendOrigin = url.startsWith(TEST_BACKEND_ORIGIN);

            // Check status code
            const statusMatches = typeof expectedStatus === 'number'
                ? status === expectedStatus
                : status >= expectedStatus.min && status <= expectedStatus.max;

            if (urlMatches && isBackendOrigin) {
                console.log(`✅ Backend call matched: ${response.request().method()} ${url} -> ${status}`);
            }

            return urlMatches && isBackendOrigin && statusMatches;
        },
        { timeout: 30000 }
    );

    console.log(`✅ Backend call completed: ${response.status()} ${response.url()}`);
    return response;
}

/**
 * Setup request logging (NO mocking, just logging)
 * 
 * @param page - Playwright page
 */
export async function setupRequestLogging(page: Page) {
    page.on('request', (request: Request) => {
        const url = request.url();
        if (url.includes('/api/')) {
            console.log(`📤 Request: ${request.method()} ${url}`);
        }
    });

    page.on('response', (response: Response) => {
        const url = response.url();
        if (url.includes('/api/')) {
            console.log(`📥 Response: ${response.status()} ${url}`);
        }
    });

    page.on('requestfailed', (request: Request) => {
        const url = request.url();
        if (url.includes('/api/')) {
            console.error(`❌ Request failed: ${request.method()} ${url}`);
        }
    });
}

/**
 * Login helper - bypass authentication in debug mode
 * 
 * In debug mode (VITE_ENABLE_DEBUG=true), authentication is bypassed and
 * users can directly access protected routes without login.
 * 
 * In production mode, this would perform real Keycloak authentication.
 * 
 * @param page - Playwright page
 * @param email - User email (optional, used in production mode only)
 * @param password - User password (optional, used in production mode only)
 */
export async function login(
    page: Page,
    email: string = process.env.APP_USER_EMAIL || 'admin@heimdall.local',
    password: string = process.env.APP_USER_PASSWORD || 'admin'
) {
    // Check if we're running in debug mode (auth bypass)
    // Note: VITE_ENABLE_DEBUG is set in playwright.config.ts webServer.env
    const debugMode = process.env.VITE_ENABLE_DEBUG === 'true';
    
    if (debugMode) {
        console.log(`🔓 Debug mode enabled - bypassing authentication`);
        // In debug mode, just navigate directly to dashboard
        // The ProtectedRoute component will allow access because VITE_ENABLE_DEBUG=true in .env.test
        await page.goto('/dashboard');
        await page.waitForLoadState('networkidle');
        console.log(`✅ Access granted in debug mode (navigated to dashboard)`);
        return;
    }

    // Production mode: Real Keycloak authentication
    console.log(`🔐 Logging in as ${email} (production mode)`);

    await page.goto('/login');
    await page.waitForLoadState('networkidle');

    // Fill login form
    await page.fill('input[type="email"]', email);
    await page.fill('input[type="password"]', password);

    // Submit form and wait for Keycloak token endpoint response
    await page.click('button[type="submit"]');

    // Wait for redirect to dashboard
    await page.waitForURL('/dashboard', { timeout: 10000 });

    console.log(`✅ Login successful, redirected to dashboard`);
}

/**
 * Verify backend is reachable before tests
 */
export async function verifyBackendReachable(page: Page) {
    console.log(`🏥 Checking backend health at ${TEST_BACKEND_ORIGIN}/health`);

    try {
        const response = await page.request.get(`${TEST_BACKEND_ORIGIN}/health`, {
            timeout: 10000,
        });

        if (response.ok()) {
            console.log(`✅ Backend is reachable: ${response.status()}`);
            return true;
        } else {
            console.error(`❌ Backend returned non-OK status: ${response.status()}`);
            return false;
        }
    } catch (error) {
        console.error(`❌ Backend not reachable: ${error}`);
        return false;
    }
}

/**
 * Extract HAR network logs for debugging
 */
export async function saveHAR(page: Page, testName: string) {
    const harPath = `playwright-report/${testName}.har`;
    console.log(`💾 Saving HAR to ${harPath}`);
    // HAR is saved automatically via recordHar in config
}

/**
 * Assert that at least one backend call was made
 */
export async function assertBackendCallMade(page: Page, urlPattern: string | RegExp) {
    let callDetected = false;

    page.on('response', (response: Response) => {
        const url = response.url();
        const matches = typeof urlPattern === 'string'
            ? url.includes(urlPattern)
            : urlPattern.test(url);

        if (matches && url.startsWith(TEST_BACKEND_ORIGIN)) {
            callDetected = true;
        }
    });

    // Wait a moment for network activity
    await page.waitForTimeout(2000);

    expect(callDetected).toBe(true);
}
