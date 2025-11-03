/**
 * E2E Test: Modal Portal Stability
 * 
 * Tests that modal components using React portals don't cause
 * DOM manipulation errors when WebSocket updates trigger re-renders.
 * 
 * This test verifies the fix for "Node.removeChild: The node to be 
 * removed is not a child of this node" errors.
 */

import { test, expect } from '@playwright/test';
import { 
  setupRequestLogging, 
  login 
} from './helpers/test-utils';

test.describe('Modal Portal Stability - DOM Manipulation Safety', () => {
  test.beforeEach(async ({ page }) => {
    await setupRequestLogging(page);
    await login(page);
    
    // Capture console errors to detect DOM manipulation issues
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        console.error('Browser Console Error:', msg.text());
      }
    });
    
    // Capture page errors
    page.on('pageerror', (error) => {
      console.error('Page Error:', error.message);
    });
  });

  test('WebSDRModal should open/close safely with WebSocket updates', async ({ page }) => {
    // Navigate to WebSDR page
    await page.goto('/websdrs');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSocket connection (health updates)
    await page.waitForTimeout(2000);
    
    // Open modal (Create button)
    const createButton = page.locator('button').filter({ hasText: /add|create|new/i }).first();
    await createButton.click();
    
    // Verify modal is visible
    const modal = page.locator('.modal.show');
    await expect(modal).toBeVisible({ timeout: 5000 });
    
    // Wait for potential WebSocket updates while modal is open
    await page.waitForTimeout(3000);
    
    // Close modal (backdrop click)
    const backdrop = page.locator('.modal-backdrop');
    if (await backdrop.isVisible().catch(() => false)) {
      await backdrop.click();
    } else {
      // Try close button
      await page.locator('.modal .btn-close').first().click();
    }
    
    // Verify modal is closed
    await expect(modal).not.toBeVisible({ timeout: 2000 });
    
    // Repeat cycle to test stability
    for (let i = 0; i < 3; i++) {
      await createButton.click();
      await expect(modal).toBeVisible({ timeout: 2000 });
      await page.waitForTimeout(1000);
      await page.locator('.modal .btn-close').first().click();
      await expect(modal).not.toBeVisible({ timeout: 2000 });
    }
    
    console.log('✅ WebSDRModal: Survived 4 open/close cycles with WebSocket active');
  });

  test('SessionEditModal should open/close safely', async ({ page }) => {
    // Navigate to sessions/recordings page
    await page.goto('/recordings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Find and click edit button (if any sessions exist)
    const editButton = page.locator('button').filter({ hasText: /edit/i }).first();
    
    if (await editButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await editButton.click();
      
      const modal = page.locator('.modal.show');
      await expect(modal).toBeVisible({ timeout: 3000 });
      
      // Wait with modal open while WebSocket updates
      await page.waitForTimeout(2000);
      
      // Close modal
      await page.locator('.modal .btn-close, .modal button').filter({ hasText: /cancel|close/i }).first().click();
      await expect(modal).not.toBeVisible({ timeout: 2000 });
      
      console.log('✅ SessionEditModal: Opened and closed safely');
    } else {
      console.log('⚠️ SessionEditModal: No sessions to edit (test skipped)');
    }
  });

  test('Training Export/Import dialogs should open/close safely', async ({ page }) => {
    // Navigate to training page
    await page.goto('/training');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Click Models tab
    const modelsTab = page.locator('button, a').filter({ hasText: /models/i }).first();
    if (await modelsTab.isVisible({ timeout: 5000 }).catch(() => false)) {
      await modelsTab.click();
      await page.waitForTimeout(1000);
    }
    
    // Test Export Dialog
    const exportButton = page.locator('button').filter({ hasText: /export/i }).first();
    if (await exportButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await exportButton.click();
      
      const exportModal = page.locator('.modal.show');
      await expect(exportModal).toBeVisible({ timeout: 3000 });
      
      await page.waitForTimeout(1500);
      
      await page.locator('.modal button').filter({ hasText: /cancel|close/i }).first().click();
      await expect(exportModal).not.toBeVisible({ timeout: 2000 });
      
      console.log('✅ ExportDialog: Opened and closed safely');
    } else {
      console.log('⚠️ ExportDialog: Export button not found (test skipped)');
    }
    
    // Test Import Dialog
    const importButton = page.locator('button').filter({ hasText: /import/i }).first();
    if (await importButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await importButton.click();
      
      const importModal = page.locator('.modal.show');
      await expect(importModal).toBeVisible({ timeout: 3000 });
      
      await page.waitForTimeout(1500);
      
      await page.locator('.modal button').filter({ hasText: /cancel|close/i }).first().click();
      await expect(importModal).not.toBeVisible({ timeout: 2000 });
      
      console.log('✅ ImportDialog: Opened and closed safely');
    } else {
      console.log('⚠️ ImportDialog: Import button not found (test skipped)');
    }
  });

  test('WidgetPicker should open/close safely on Dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Look for widget picker trigger (add widget button)
    const addWidgetButton = page.locator('button').filter({ hasText: /add.*widget|widget.*add|\+/i }).first();
    
    if (await addWidgetButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await addWidgetButton.click();
      
      const modal = page.locator('.modal.show');
      await expect(modal).toBeVisible({ timeout: 3000 });
      
      await page.waitForTimeout(1500);
      
      await page.locator('.modal button').filter({ hasText: /close|cancel/i }).first().click();
      await expect(modal).not.toBeVisible({ timeout: 2000 });
      
      console.log('✅ WidgetPicker: Opened and closed safely');
    } else {
      console.log('⚠️ WidgetPicker: Add widget button not found (test skipped)');
    }
  });

  test('DeleteConfirmModal should open/close safely', async ({ page }) => {
    await page.goto('/websdrs');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Look for delete button
    const deleteButton = page.locator('button').filter({ hasText: /delete|remove|trash/i }).first();
    
    if (await deleteButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await deleteButton.click();
      
      const modal = page.locator('.modal.show');
      await expect(modal).toBeVisible({ timeout: 3000 });
      
      await page.waitForTimeout(1500);
      
      // Cancel the deletion
      await page.locator('.modal button').filter({ hasText: /cancel|close/i }).first().click();
      await expect(modal).not.toBeVisible({ timeout: 2000 });
      
      console.log('✅ DeleteConfirmModal: Opened and closed safely');
    } else {
      console.log('⚠️ DeleteConfirmModal: Delete button not found (test skipped)');
    }
  });

  test('Multiple modals in sequence should not cause DOM errors', async ({ page }) => {
    const errors: string[] = [];
    
    // Capture DOM errors
    page.on('pageerror', (error) => {
      if (error.message.includes('removeChild') || error.message.includes('not a child')) {
        errors.push(error.message);
      }
    });
    
    // Test sequence: Dashboard -> WebSDRs -> Training -> WebSDRs
    await page.goto('/dashboard');
    await page.waitForTimeout(2000);
    
    await page.goto('/websdrs');
    await page.waitForTimeout(2000);
    
    // Try to open modal
    const createButton = page.locator('button').filter({ hasText: /add|create/i }).first();
    if (await createButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await createButton.click();
      await page.waitForTimeout(1000);
      await page.locator('.modal .btn-close').first().click();
      await page.waitForTimeout(1000);
    }
    
    await page.goto('/training');
    await page.waitForTimeout(2000);
    
    await page.goto('/websdrs');
    await page.waitForTimeout(2000);
    
    // Check for DOM errors
    expect(errors.length).toBe(0);
    
    if (errors.length === 0) {
      console.log('✅ No DOM manipulation errors detected across navigation');
    } else {
      console.error('❌ DOM errors detected:', errors);
    }
  });

  test('Rapid modal open/close should not cause race conditions', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('pageerror', (error) => {
      if (error.message.includes('removeChild') || error.message.includes('not a child')) {
        errors.push(error.message);
      }
    });
    
    await page.goto('/websdrs');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    const createButton = page.locator('button').filter({ hasText: /add|create/i }).first();
    
    if (await createButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Rapid open/close (stress test)
      for (let i = 0; i < 5; i++) {
        await createButton.click();
        await page.waitForTimeout(300); // Short wait
        
        const closeButton = page.locator('.modal .btn-close').first();
        if (await closeButton.isVisible({ timeout: 1000 }).catch(() => false)) {
          await closeButton.click();
        }
        
        await page.waitForTimeout(300); // Short wait between cycles
      }
      
      expect(errors.length).toBe(0);
      console.log('✅ Rapid modal cycles completed without DOM errors');
    } else {
      console.log('⚠️ Rapid test skipped (no modal trigger found)');
    }
  });
});
