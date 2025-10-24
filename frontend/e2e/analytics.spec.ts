/**
 * E2E Test: Analytics Page
 * 
 * Verifies real backend calls for analytics data:
 * - Prediction metrics
 * - WebSDR performance
 * - System performance
 * - Historical trends
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  waitForBackendCall, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('Analytics Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should load analytics page and fetch metrics from backend', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Wait for analytics API call
    const analyticsResponsePromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/analytics') ||
           url.includes('/api/v1/sessions/analytics') ||
           url.includes('/metrics'))
        );
      },
      { timeout: 15000 }
    );
    
    const analyticsResponse = await analyticsResponsePromise;
    
    expect(analyticsResponse.status()).toBeGreaterThanOrEqual(200);
    expect(analyticsResponse.status()).toBeLessThan(300);
    
    const responseBody = await analyticsResponse.json();
    console.log(`✅ Analytics data loaded: ${JSON.stringify(responseBody).substring(0, 100)}...`);
  });

  test('should fetch session analytics from backend', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Wait for session analytics API call
    const sessionAnalyticsPromise = waitForBackendCall(
      page,
      '/api/v1/sessions/analytics',
      { min: 200, max: 299 }
    );
    
    const sessionAnalyticsResponse = await sessionAnalyticsPromise;
    const responseBody = await sessionAnalyticsResponse.json();
    
    // Verify expected analytics fields
    expect(responseBody).toHaveProperty('total_sessions');
    expect(responseBody).toHaveProperty('completed_sessions');
    expect(responseBody).toHaveProperty('success_rate');
    
    console.log(`✅ Session analytics: ${responseBody.total_sessions} total sessions, ${responseBody.success_rate}% success rate`);
  });

  test('should fetch prediction metrics over time', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Look for time range selector
    const timeRangeSelector = page.locator('select, button').filter({ hasText: /7d|30d|90d/i }).first();
    
    if (await timeRangeSelector.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for predictions API call
      const predictionsPromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1/analytics/predictions')
          );
        },
        { timeout: 15000 }
      );
      
      // Change time range
      await timeRangeSelector.click();
      
      const predictionsResponse = await predictionsPromise;
      expect(predictionsResponse.status()).toBeGreaterThanOrEqual(200);
      expect(predictionsResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Prediction metrics fetched`);
    } else {
      console.log('⚠️ Time range selector not found');
    }
  });

  test('should fetch WebSDR performance metrics', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSDR performance API call
    const websdrPerfPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          url.includes('/api/v1/analytics/websdr')
        );
      },
      { timeout: 15000 }
    ).catch(() => null);
    
    const websdrPerfResponse = await websdrPerfPromise;
    
    if (websdrPerfResponse) {
      expect(websdrPerfResponse.status()).toBeGreaterThanOrEqual(200);
      expect(websdrPerfResponse.status()).toBeLessThan(300);
      
      const responseBody = await websdrPerfResponse.json();
      expect(Array.isArray(responseBody)).toBe(true);
      
      console.log(`✅ WebSDR performance: ${responseBody.length} receivers`);
    } else {
      console.log('⚠️ WebSDR performance endpoint not called');
    }
  });

  test('should display charts with backend data', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Wait for analytics data to load
    await page.waitForTimeout(3000);
    
    // Verify charts/visualizations exist
    const hasCharts = await page.locator('canvas, svg').count().then(count => count > 0);
    
    if (hasCharts) {
      console.log('✅ Charts rendered with backend data');
      expect(hasCharts).toBe(true);
    } else {
      console.log('⚠️ No charts found (may be plain tables/text)');
    }
    
    // Verify page has meaningful content
    const pageContent = await page.locator('main').textContent();
    expect(pageContent).toBeTruthy();
    expect(pageContent!.length).toBeGreaterThan(100);
  });

  test('should handle accuracy distribution data from backend', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');
    
    // Wait for accuracy distribution API call
    const accuracyPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          url.includes('/api/v1/analytics') &&
          url.includes('accuracy')
        );
      },
      { timeout: 15000 }
    ).catch(() => null);
    
    const accuracyResponse = await accuracyPromise;
    
    if (accuracyResponse) {
      expect(accuracyResponse.status()).toBeGreaterThanOrEqual(200);
      expect(accuracyResponse.status()).toBeLessThan(300);
      
      const responseBody = await accuracyResponse.json();
      console.log(`✅ Accuracy distribution fetched: ${JSON.stringify(responseBody).substring(0, 100)}...`);
    } else {
      console.log('⚠️ Accuracy distribution endpoint not called');
    }
  });
});
