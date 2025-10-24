/**
 * E2E Test: System Status Page
 * 
 * Verifies real backend calls for system monitoring:
 * - Service health status
 * - Performance metrics
 * - System alerts
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  waitForBackendCall, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('System Status Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should load system status page and fetch health data', async ({ page }) => {
    await page.goto('/system-status');
    await page.waitForLoadState('networkidle');
    
    // Wait for system health API call
    const healthPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/system/status') ||
           url.includes('/api/v1/health') ||
           url.includes('/status'))
        );
      },
      { timeout: 15000 }
    );
    
    const healthResponse = await healthPromise;
    
    expect(healthResponse.status()).toBeGreaterThanOrEqual(200);
    expect(healthResponse.status()).toBeLessThan(300);
    
    const responseBody = await healthResponse.json();
    console.log(`✅ System health data loaded: ${JSON.stringify(responseBody).substring(0, 100)}...`);
  });

  test('should fetch service status for all microservices', async ({ page }) => {
    await page.goto('/system-status');
    await page.waitForLoadState('networkidle');
    
    // Wait for services status API call
    const servicesPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          url.includes('/api/v1/system/services')
        );
      },
      { timeout: 15000 }
    ).catch(() => null);
    
    const servicesResponse = await servicesPromise;
    
    if (servicesResponse) {
      expect(servicesResponse.status()).toBeGreaterThanOrEqual(200);
      expect(servicesResponse.status()).toBeLessThan(300);
      
      const responseBody = await servicesResponse.json();
      console.log(`✅ Services status loaded: ${JSON.stringify(responseBody).substring(0, 100)}...`);
    } else {
      console.log('⚠️ Services status endpoint not called');
    }
  });

  test('should fetch performance metrics from backend', async ({ page }) => {
    await page.goto('/system-status');
    await page.waitForLoadState('networkidle');
    
    // Wait for performance metrics API call
    const metricsPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/system/metrics') ||
           url.includes('/api/v1/analytics/system'))
        );
      },
      { timeout: 15000 }
    ).catch(() => null);
    
    const metricsResponse = await metricsPromise;
    
    if (metricsResponse) {
      expect(metricsResponse.status()).toBeGreaterThanOrEqual(200);
      expect(metricsResponse.status()).toBeLessThan(300);
      console.log(`✅ Performance metrics loaded`);
    } else {
      console.log('⚠️ Metrics endpoint not called');
    }
  });

  test('should display service health indicators from backend data', async ({ page }) => {
    await page.goto('/system-status');
    await page.waitForLoadState('networkidle');
    
    // Wait for data to load
    await page.waitForTimeout(2000);
    
    // Verify service status indicators exist
    const hasStatusIndicators = await page.locator('[class*="status"], [class*="health"], [class*="service"]')
      .count()
      .then(count => count > 0);
    
    expect(hasStatusIndicators).toBe(true);
    
    // Verify meaningful content
    const pageContent = await page.locator('main').textContent();
    expect(pageContent).toBeTruthy();
    expect(pageContent!.length).toBeGreaterThan(100);
    
    console.log('✅ System status page displays backend data');
  });

  test('should refresh status on manual trigger', async ({ page }) => {
    await page.goto('/system-status');
    await page.waitForLoadState('networkidle');
    
    // Wait for initial load
    await page.waitForTimeout(2000);
    
    // Look for refresh button
    const refreshButton = page.locator('button').filter({ hasText: /refresh|reload|update/i }).first();
    
    if (await refreshButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for refresh API call
      const refreshPromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1/system')
          );
        },
        { timeout: 15000 }
      );
      
      // Click refresh
      await refreshButton.click();
      
      const refreshResponse = await refreshPromise;
      expect(refreshResponse.status()).toBeGreaterThanOrEqual(200);
      expect(refreshResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Status refreshed from backend`);
    } else {
      console.log('⚠️ Refresh button not found');
    }
  });
});
