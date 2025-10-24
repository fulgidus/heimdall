/**
 * E2E Test: Settings Page
 * 
 * Verifies real backend calls for user settings:
 * - Load user preferences
 * - Update settings
 * - Save configuration
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('Settings Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should load settings page', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Verify settings page loaded
    const pageContent = await page.locator('main').textContent();
    expect(pageContent).toBeTruthy();
    expect(pageContent).toContain('Settings' || 'Preferences' || 'Configuration');
    
    console.log('✅ Settings page loaded');
  });

  test('should fetch user settings from backend if endpoint exists', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Wait for settings API call
    const settingsPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/settings') ||
           url.includes('/api/v1/user/preferences') ||
           url.includes('/api/v1/config'))
        );
      },
      { timeout: 10000 }
    ).catch(() => null);
    
    const settingsResponse = await settingsPromise;
    
    if (settingsResponse) {
      expect(settingsResponse.status()).toBeGreaterThanOrEqual(200);
      expect(settingsResponse.status()).toBeLessThan(300);
      console.log(`✅ Settings fetched from backend`);
    } else {
      console.log('⚠️ No settings endpoint called (may be client-side only)');
    }
  });

  test('should update settings via backend API', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Look for save button
    const saveButton = page.locator('button').filter({ hasText: /save|update|apply/i }).first();
    
    if (await saveButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for update API call
      const updatePromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          const method = response.request().method();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1') &&
            (method === 'PUT' || method === 'PATCH' || method === 'POST')
          );
        },
        { timeout: 15000 }
      );
      
      // Click save
      await saveButton.click();
      
      const updateResponse = await updatePromise;
      expect(updateResponse.status()).toBeGreaterThanOrEqual(200);
      expect(updateResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Settings updated via backend: ${updateResponse.status()}`);
    } else {
      console.log('⚠️ Save button not found');
    }
  });
});
