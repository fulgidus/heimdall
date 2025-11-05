import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Test Configuration
 * 
 * Tests run with VITE_ENABLE_DEBUG=true to bypass authentication.
 * This allows testing core functionality without Keycloak user setup.
 * 
 * Vite automatically loads .env.test when running with --mode test.
 * 
 * Real authentication testing is handled separately via Keycloak integration tests.
 */

export default defineConfig({
    testDir: './e2e',

    /* Run tests in files in parallel */
    fullyParallel: false,

    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: !!process.env.CI,

    /* Retry on CI only */
    retries: process.env.CI ? 2 : 0,

    /* Opt out of parallel tests on CI. */
    workers: process.env.CI ? 1 : undefined,

    /* Global setup - set environment variables for test runner */
    globalSetup: undefined,
    globalTeardown: undefined,

    /* Reporter to use. See https://playwright.dev/docs/test-reporters */
    reporter: [
        ['html', { outputFolder: 'playwright-report' }],
        ['json', { outputFile: 'playwright-report/results.json' }],
        ['list'],
    ],

    /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
    use: {
        /* Base URL to use in actions like `await page.goto('/')`. */
        baseURL: process.env.BASE_URL || 'http://localhost:3001',

        /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
        trace: 'on-first-retry',

        /* Screenshot on failure */
        screenshot: 'only-on-failure',

        /* Video on failure */
        video: 'retain-on-failure',

        /* Network HAR recording */
        recordHar: {
            mode: 'minimal',
            path: 'playwright-report/network.har',
        },
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },

        /* Uncomment for cross-browser testing
        {
          name: 'firefox',
          use: { ...devices['Desktop Firefox'] },
        },
    
        {
          name: 'webkit',
          use: { ...devices['Desktop Safari'] },
        },
        */
    ],

    /* Run your local dev server before starting the tests */
    webServer: process.env.CI ? undefined : {
        command: 'npm run dev -- --mode test',
        url: 'http://localhost:3001',
        reuseExistingServer: !process.env.CI,
        timeout: 120 * 1000,
        env: {
            // Ensure Vite loads .env.test
            MODE: 'test',
            // Pass debug mode to test environment (not browser)
            VITE_ENABLE_DEBUG: 'true',
        },
    },
});
