/**
 * E2E Test: Login Page
 * 
 * Verifies real authentication flow with backend JWT validation.
 * NO mocks/stubs - all HTTP calls must reach the actual backend.
 */

import { test, expect } from '@playwright/test';
import {
    setupRequestLogging,
    waitForBackendCall,
    verifyBackendReachable,
    login,
    TEST_BACKEND_ORIGIN
} from './helpers/test-utils';

test.describe('Login Page - Real Backend Integration', () => {
    test.beforeEach(async ({ page }) => {
        await setupRequestLogging(page);
    });

    test('should verify backend is reachable', async ({ page }) => {
        const isReachable = await verifyBackendReachable(page);
        expect(isReachable).toBe(true);
    });

    test('should load login page and display form', async ({ page }) => {
        await page.goto('/login');
        await page.waitForLoadState('networkidle');

        // Verify form elements exist
        await expect(page.locator('input[type="email"]')).toBeVisible();
        await expect(page.locator('input[type="password"]')).toBeVisible();
        await expect(page.locator('button[type="submit"]')).toBeVisible();
    });

    test('should make real login API call and receive JWT token', async ({ page }) => {
        await page.goto('/login');
        await page.waitForLoadState('networkidle');

        // Fill credentials (reads APP_USER_EMAIL and APP_USER_PASSWORD from .env)
        const email = process.env.APP_USER_EMAIL || 'admin@heimdall.local';
        const password = process.env.APP_USER_PASSWORD || 'admin';

        await page.fill('input[type="email"]', email);
        await page.fill('input[type="password"]', password);

        // Setup listener BEFORE clicking submit
        const loginResponsePromise = waitForBackendCall(page, '/api/v1/auth/login', 200);

        // Submit form
        await page.click('button[type="submit"]');

        // Wait for and verify response
        const loginResponse = await loginResponsePromise;
        expect(loginResponse.status()).toBe(200);

        // Verify response contains access_token (from Keycloak via API Gateway)
        const responseBody = await loginResponse.json();
        expect(responseBody).toHaveProperty('access_token');
        expect(responseBody.access_token).toBeTruthy();

        console.log('✅ Login API call successful, JWT token received');
    });

    test('should redirect to dashboard after successful login', async ({ page }) => {
        await page.goto('/login');
        await page.waitForLoadState('networkidle');

        // Fill credentials (reads APP_USER_EMAIL and APP_USER_PASSWORD from .env)
        const email = process.env.APP_USER_EMAIL || 'admin@heimdall.local';
        const password = process.env.APP_USER_PASSWORD || 'admin';

        await page.fill('input[type="email"]', email);
        await page.fill('input[type="password"]', password);

        // Setup listener
        const loginResponsePromise = waitForBackendCall(page, '/api/v1/auth/login', 200);

        // Submit
        await page.click('button[type="submit"]');

        // Wait for login
        await loginResponsePromise;

        // Verify redirect
        await page.waitForURL('/dashboard', { timeout: 10000 });
        expect(page.url()).toContain('/dashboard');

        console.log('✅ Successfully redirected to dashboard');
    });

    test('should handle invalid credentials with backend error', async ({ page }) => {
        await page.goto('/login');
        await page.waitForLoadState('networkidle');

        // Fill invalid credentials
        await page.fill('input[type="email"]', 'invalid@example.com');
        await page.fill('input[type="password"]', 'wrongpassword');

        // Submit and expect 401 or 403
        const loginResponsePromise = page.waitForResponse(
            (response) =>
                response.url().includes('/api/v1/auth/login') &&
                response.url().startsWith(TEST_BACKEND_ORIGIN),
            { timeout: 10000 }
        );

        await page.click('button[type="submit"]');

        const loginResponse = await loginResponsePromise;

        // Expect 401 Unauthorized or 403 Forbidden
        expect([401, 403]).toContain(loginResponse.status());

        console.log(`✅ Invalid login rejected with status ${loginResponse.status()}`);
    });

    test('should store auth token in localStorage after successful login', async ({ page }) => {
        await page.goto('/login');
        await page.waitForLoadState('networkidle');

        // Fill credentials (reads APP_USER_EMAIL and APP_USER_PASSWORD from .env)
        const email = process.env.APP_USER_EMAIL || 'admin@heimdall.local';
        const password = process.env.APP_USER_PASSWORD || 'admin';

        await page.fill('input[type="email"]', email);
        await page.fill('input[type="password"]', password);

        // Submit
        const loginResponsePromise = waitForBackendCall(page, '/api/v1/auth/login', 200);
        await page.click('button[type="submit"]');
        await loginResponsePromise;

        // Wait for redirect
        await page.waitForURL('/dashboard', { timeout: 10000 });

        // Check localStorage for auth state
        const authState = await page.evaluate(() => {
            const item = localStorage.getItem('auth-store');
            return item ? JSON.parse(item) : null;
        });

        expect(authState).toBeTruthy();
        expect(authState.state.isAuthenticated).toBe(true);
        expect(authState.state.token).toBeTruthy();

        console.log('✅ Auth token stored in localStorage');
    });
});
