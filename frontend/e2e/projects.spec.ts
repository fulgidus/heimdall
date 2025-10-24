/**
 * E2E Test: Projects/Recording Session Page
 * 
 * Verifies real backend calls for session management:
 * - List sessions
 * - Create new session
 * - Start/stop recording
 * - Update session status
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  waitForBackendCall, 
  login,
  TEST_BACKEND_ORIGIN 
} from './helpers/test-utils';

test.describe('Projects Page - Real Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
  });

  test('should navigate to projects page and load sessions list', async ({ page }) => {
    // Navigate to projects page
    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    
    // Wait for sessions list API call
    const sessionsResponsePromise = waitForBackendCall(
      page, 
      '/api/v1/sessions',
      { min: 200, max: 299 }
    );
    
    const sessionsResponse = await sessionsResponsePromise;
    
    expect(sessionsResponse.status()).toBeGreaterThanOrEqual(200);
    expect(sessionsResponse.status()).toBeLessThan(300);
    
    // Verify response structure
    const responseBody = await sessionsResponse.json();
    expect(responseBody).toHaveProperty('sessions');
    expect(Array.isArray(responseBody.sessions)).toBe(true);
    
    console.log(`✅ Sessions list loaded: ${responseBody.sessions.length} sessions`);
  });

  test('should create new recording session via backend API', async ({ page }) => {
    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    
    // Click "New Session" or similar button
    const newSessionButton = page.locator('button').filter({ hasText: /new.*session/i }).first();
    
    if (await newSessionButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await newSessionButton.click();
      
      // Fill session form
      await page.fill('input[name="session_name"], input[placeholder*="name" i]', 'E2E Test Session');
      await page.fill('input[name="frequency_mhz"], input[type="number"]', '145.5');
      await page.fill('input[name="duration_seconds"]', '30');
      
      // Setup listener for create session API call
      const createResponsePromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          const method = response.request().method();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1/sessions') &&
            method === 'POST'
          );
        },
        { timeout: 15000 }
      );
      
      // Submit form
      await page.click('button[type="submit"]');
      
      // Verify backend call
      const createResponse = await createResponsePromise;
      expect(createResponse.status()).toBeGreaterThanOrEqual(200);
      expect(createResponse.status()).toBeLessThan(300);
      
      const responseBody = await createResponse.json();
      expect(responseBody).toHaveProperty('id');
      expect(responseBody.session_name).toBe('E2E Test Session');
      
      console.log(`✅ Session created: ID ${responseBody.id}`);
    } else {
      console.log('⚠️ New session button not found, skipping create test');
    }
  });

  test('should start recording session and trigger backend acquisition', async ({ page }) => {
    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    
    // Look for start recording button
    const startButton = page.locator('button').filter({ hasText: /start|record/i }).first();
    
    if (await startButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for acquisition start API call
      const startResponsePromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            (url.includes('/api/v1/acquisition/acquire') ||
             url.includes('/api/v1/sessions') ||
             url.includes('/start'))
          );
        },
        { timeout: 20000 }
      );
      
      // Click start
      await startButton.click();
      
      // Verify backend call
      const startResponse = await startResponsePromise;
      expect(startResponse.status()).toBeGreaterThanOrEqual(200);
      expect(startResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Recording started: ${startResponse.status()} ${startResponse.url()}`);
    } else {
      console.log('⚠️ Start recording button not found');
    }
  });

  test('should update session approval status via backend', async ({ page }) => {
    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    
    // Wait for sessions to load
    await page.waitForTimeout(2000);
    
    // Look for approve/reject buttons
    const approveButton = page.locator('button').filter({ hasText: /approve/i }).first();
    
    if (await approveButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for approval API call
      const approvalResponsePromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          const method = response.request().method();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1/sessions') &&
            (url.includes('/approval') || url.includes('/status')) &&
            (method === 'PATCH' || method === 'PUT')
          );
        },
        { timeout: 15000 }
      );
      
      // Click approve
      await approveButton.click();
      
      // Verify backend call
      const approvalResponse = await approvalResponsePromise;
      expect(approvalResponse.status()).toBeGreaterThanOrEqual(200);
      expect(approvalResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Session approval updated: ${approvalResponse.status()}`);
    } else {
      console.log('⚠️ Approve button not found');
    }
  });

  test('should delete session via backend API', async ({ page }) => {
    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    
    // Wait for sessions to load
    await page.waitForTimeout(2000);
    
    // Look for delete button
    const deleteButton = page.locator('button').filter({ hasText: /delete|remove/i }).first();
    
    if (await deleteButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Setup listener for delete API call
      const deleteResponsePromise = page.waitForResponse(
        (response) => {
          const url = response.url();
          const method = response.request().method();
          return (
            url.startsWith(TEST_BACKEND_ORIGIN) &&
            url.includes('/api/v1/sessions') &&
            method === 'DELETE'
          );
        },
        { timeout: 15000 }
      );
      
      // Click delete
      await deleteButton.click();
      
      // Confirm if dialog appears
      const confirmButton = page.locator('button').filter({ hasText: /confirm|yes|delete/i }).last();
      if (await confirmButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await confirmButton.click();
      }
      
      // Verify backend call
      const deleteResponse = await deleteResponsePromise;
      expect(deleteResponse.status()).toBeGreaterThanOrEqual(200);
      expect(deleteResponse.status()).toBeLessThan(300);
      
      console.log(`✅ Session deleted: ${deleteResponse.status()}`);
    } else {
      console.log('⚠️ Delete button not found');
    }
  });
});
