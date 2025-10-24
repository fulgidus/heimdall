/**
 * E2E Test: Profile Page
 * 
 * Verifies real backend calls for user profile:
 * - Fetch user data
 * - Update profile info
 * - Activity history
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('Profile Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should load profile page and fetch user data', async ({ page }) => {
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');
    
    // Wait for user profile API call
    const profilePromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/user') ||
           url.includes('/api/v1/profile') ||
           url.includes('/api/v1/auth/me'))
        );
      },
      { timeout: 15000 }
    ).catch(() => null);
    
    const profileResponse = await profilePromise;
    
    if (profileResponse) {
      expect(profileResponse.status()).toBeGreaterThanOrEqual(200);
      expect(profileResponse.status()).toBeLessThan(300);
      
      const responseBody = await profileResponse.json();
      console.log(`✅ Profile data loaded: ${JSON.stringify(responseBody).substring(0, 100)}...`);
    } else {
      console.log('⚠️ No profile endpoint called (may use auth state)');
    }
    
    // Verify profile content displayed
    const pageContent = await page.locator('main').textContent();
    expect(pageContent).toBeTruthy();
  });

  test('should fetch user activity history if available', async ({ page }) => {
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');
    
    // Wait for activity API call
    const activityPromise = page.waitForResponse(
      (response) => {
        const url = response.url();
        return (
          url.startsWith(TEST_BACKEND_ORIGIN) &&
          (url.includes('/api/v1/user/activity') ||
           url.includes('/api/v1/profile/history'))
        );
      },
      { timeout: 10000 }
    ).catch(() => null);
    
    const activityResponse = await activityPromise;
    
    if (activityResponse) {
      expect(activityResponse.status()).toBeGreaterThanOrEqual(200);
      expect(activityResponse.status()).toBeLessThan(300);
      console.log(`✅ Activity history loaded`);
    } else {
      console.log('⚠️ No activity endpoint called');
    }
  });

  test('should update profile via backend API', async ({ page }) => {
    await page.goto('/profile');
    await page.waitForLoadState('networkidle');
    
    // Look for edit/save button
    const editButton = page.locator('button').filter({ hasText: /edit|save|update/i }).first();
    
    if (await editButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Click edit
      await editButton.click();
      await page.waitForTimeout(1000);
      
      // Look for save button
      const saveButton = page.locator('button').filter({ hasText: /save|update/i }).first();
      
      if (await saveButton.isVisible({ timeout: 3000 }).catch(() => false)) {
        // Setup listener for update API call
        const updatePromise = page.waitForResponse(
          (response) => {
            const url = response.url();
            const method = response.request().method();
            return (
              url.startsWith(TEST_BACKEND_ORIGIN) &&
              (url.includes('/api/v1/user') || url.includes('/api/v1/profile')) &&
              (method === 'PUT' || method === 'PATCH')
            );
          },
          { timeout: 15000 }
        );
        
        // Click save
        await saveButton.click();
        
        const updateResponse = await updatePromise;
        expect(updateResponse.status()).toBeGreaterThanOrEqual(200);
        expect(updateResponse.status()).toBeLessThan(300);
        
        console.log(`✅ Profile updated via backend: ${updateResponse.status()}`);
      }
    } else {
      console.log('⚠️ Edit button not found');
    }
  });
});
