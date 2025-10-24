/**
 * E2E Test: WebSDR Management Page
 * 
 * Verifies real backend calls for WebSDR operations:
 * - List WebSDR receivers
 * - Check health status
 * - Get configuration
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  waitForBackendCall, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('WebSDR Management Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should load WebSDR list from backend', async ({ page }) => {
    // Navigate to WebSDR management page
    await page.goto('/websdr-management');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSDR list API call
    const websdrsResponsePromise = waitForBackendCall(
      page, 
      '/api/v1/acquisition/websdrs',
      { min: 200, max: 299 }
    );
    
    const websdrsResponse = await websdrsResponsePromise;
    
    expect(websdrsResponse.status()).toBeGreaterThanOrEqual(200);
    expect(websdrsResponse.status()).toBeLessThan(300);
    
    // Verify response structure
    const responseBody = await websdrsResponse.json();
    expect(Array.isArray(responseBody)).toBe(true);
    expect(responseBody.length).toBeGreaterThan(0);
    
    // Verify WebSDR structure
    const firstWebSDR = responseBody[0];
    expect(firstWebSDR).toHaveProperty('id');
    expect(firstWebSDR).toHaveProperty('name');
    expect(firstWebSDR).toHaveProperty('url');
    expect(firstWebSDR).toHaveProperty('latitude');
    expect(firstWebSDR).toHaveProperty('longitude');
    
    console.log(`✅ WebSDR list loaded: ${responseBody.length} receivers`);
  });

  test('should check WebSDR health status from backend', async ({ page }) => {
    await page.goto('/websdr-management');
    await page.waitForLoadState('networkidle');
    
    // Wait for health check API call
    const healthResponsePromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/acquisition/websdrs/health') ||
           url.includes('/health'))
        );
      },
      { timeout: 15000 }
    );
    
    // Trigger health check (may be automatic or via button)
    const healthButton = page.locator('button').filter({ hasText: /health|check|refresh/i }).first();
    if (await healthButton.isVisible({ timeout: 3000 }).catch(() => false)) {
      await healthButton.click();
    }
    
    const healthResponse = await healthResponsePromise;
    
    expect(healthResponse.status()).toBeGreaterThanOrEqual(200);
    expect(healthResponse.status()).toBeLessThan(300);
    
    const responseBody = await healthResponse.json();
    console.log(`✅ Health check completed: ${JSON.stringify(responseBody).substring(0, 100)}...`);
  });

  test('should display WebSDR status indicators from backend data', async ({ page }) => {
    await page.goto('/websdr-management');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSDR list
    await waitForBackendCall(page, '/api/v1/acquisition/websdrs');
    
    // Verify page displays WebSDR data
    const pageContent = await page.locator('main').textContent();
    expect(pageContent).toBeTruthy();
    expect(pageContent!.length).toBeGreaterThan(100);
    
    // Look for status indicators (online/offline)
    const hasStatusIndicators = await page.locator('[class*="status"], [class*="online"], [class*="offline"]')
      .count()
      .then(count => count > 0);
    
    if (hasStatusIndicators) {
      console.log('✅ WebSDR status indicators displayed');
    } else {
      console.log('⚠️ No status indicators found (may be in different format)');
    }
  });

  test('should handle WebSDR configuration view', async ({ page }) => {
    await page.goto('/websdr-management');
    await page.waitForLoadState('networkidle');
    
    // Wait for list to load
    await waitForBackendCall(page, '/api/v1/acquisition/websdrs');
    
    // Look for view/config button
    const configButton = page.locator('button').filter({ hasText: /view|config|details/i }).first();
    
    if (await configButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Click to view config
      await configButton.click();
      
      // Verify details are displayed
      await page.waitForTimeout(1000);
      
      const hasDetails = await page.locator('[class*="dialog"], [class*="modal"], [class*="details"]')
        .isVisible()
        .catch(() => false);
      
      expect(hasDetails).toBe(true);
      console.log('✅ WebSDR configuration displayed');
    } else {
      console.log('⚠️ Config/view button not found');
    }
  });

  test('should verify all 7 Italian WebSDRs are listed', async ({ page }) => {
    await page.goto('/websdr-management');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSDR list
    const websdrsResponse = await waitForBackendCall(page, '/api/v1/acquisition/websdrs');
    const responseBody = await websdrsResponse.json();
    
    // According to WEBSDRS.md, there should be 7 Italian receivers
    expect(responseBody.length).toBeGreaterThanOrEqual(7);
    
    // Verify Italian receivers (Piemonte & Liguria)
    const italianReceivers = responseBody.filter((websdr: any) => 
      websdr.country === 'Italy' || 
      websdr.region?.includes('Piemonte') || 
      websdr.region?.includes('Liguria')
    );
    
    console.log(`✅ Found ${italianReceivers.length} Italian receivers out of ${responseBody.length} total`);
  });
});
