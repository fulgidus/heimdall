/**
 * SampleDetailsPanel Tests
 * 
 * Tests for receiver button color-coding based on SNR values
 */

import { describe, it, expect } from 'vitest';

/**
 * Normalize SNR value to 0-1 range
 */
const normalizeSnr = (snr: number, minSnr: number, maxSnr: number): number => {
    const snrRange = maxSnr - minSnr;
    if (snrRange === 0) return 1; // All receivers have same SNR
    return (snr - minSnr) / snrRange;
};

/**
 * Convert normalized SNR (0-1) to color gradient: red → yellow → green
 * 0.0 = red (#dc3545)
 * 0.5 = yellow (#ffc107)
 * 1.0 = green (#28a745)
 */
const getSnrColor = (normalizedSnr: number): string => {
    // Clamp to 0-1 range
    const value = Math.max(0, Math.min(1, normalizedSnr));
    
    if (value < 0.5) {
        // Interpolate red → yellow (0.0 to 0.5)
        const t = value * 2; // Scale to 0-1
        const r = 220; // Red component stays high
        const g = Math.round(60 + (252 - 60) * t); // 60 → 252
        const b = Math.round(69 - 69 * t); // 69 → 0
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // Interpolate yellow → green (0.5 to 1.0)
        const t = (value - 0.5) * 2; // Scale to 0-1
        const r = Math.round(255 - (255 - 40) * t); // 255 → 40
        const g = Math.round(193 + (167 - 193) * t); // 193 → 167
        const b = Math.round(7 + (69 - 7) * t); // 7 → 69
        return `rgb(${r}, ${g}, ${b})`;
    }
};

describe('SampleDetailsPanel - SNR Color Coding', () => {
    describe('normalizeSnr', () => {
        it('should normalize SNR values correctly', () => {
            // Best receiver (highest SNR)
            expect(normalizeSnr(20, -10, 20)).toBe(1.0);
            
            // Worst receiver (lowest SNR)
            expect(normalizeSnr(-10, -10, 20)).toBe(0.0);
            
            // Middle receiver
            expect(normalizeSnr(5, -10, 20)).toBe(0.5);
        });

        it('should handle equal SNR values', () => {
            // All receivers have same SNR
            expect(normalizeSnr(10, 10, 10)).toBe(1);
        });

        it('should handle edge cases', () => {
            // Single dB difference
            expect(normalizeSnr(0, -1, 0)).toBe(1.0);
            expect(normalizeSnr(-1, -1, 0)).toBe(0.0);
        });
    });

    describe('getSnrColor', () => {
        it('should return red for worst SNR (0.0)', () => {
            const color = getSnrColor(0.0);
            // Red: rgb(220, 60, 69)
            expect(color).toBe('rgb(220, 60, 69)');
        });

        it('should return yellow-ish for medium SNR (0.5)', () => {
            const color = getSnrColor(0.5);
            // At 0.5: rgb(255, 193, 7) - Bootstrap warning yellow
            expect(color).toBe('rgb(255, 193, 7)');
        });

        it('should return green for best SNR (1.0)', () => {
            const color = getSnrColor(1.0);
            // Green: rgb(40, 167, 69)
            expect(color).toBe('rgb(40, 167, 69)');
        });

        it('should interpolate between red and yellow (0.0 to 0.5)', () => {
            const color25 = getSnrColor(0.25);
            // At 0.25, we're halfway between red and yellow
            // r: 220, g: between 60 and 252, b: between 69 and 0
            expect(color25).toMatch(/^rgb\(220, \d+, \d+\)$/);
        });

        it('should interpolate between yellow and green (0.5 to 1.0)', () => {
            const color75 = getSnrColor(0.75);
            // At 0.75, we're halfway between yellow and green
            // r: between 255 and 40, g: between 193 and 167, b: between 7 and 69
            expect(color75).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
        });

        it('should clamp values outside 0-1 range', () => {
            // Values > 1 should be clamped to 1 (green)
            expect(getSnrColor(1.5)).toBe(getSnrColor(1.0));
            
            // Values < 0 should be clamped to 0 (red)
            expect(getSnrColor(-0.5)).toBe(getSnrColor(0.0));
        });
    });

    describe('End-to-end color mapping', () => {
        it('should map receivers with different SNRs to correct colors', () => {
            const receivers = [
                { rx_id: 'RX1', snr_db: 20 },  // Best
                { rx_id: 'RX2', snr_db: 5 },   // Medium
                { rx_id: 'RX3', snr_db: -10 }  // Worst
            ];

            const snrValues = receivers.map(rx => rx.snr_db);
            const minSnr = Math.min(...snrValues);  // -10
            const maxSnr = Math.max(...snrValues);  // 20

            // Best receiver should be green
            const rx1Normalized = normalizeSnr(receivers[0].snr_db, minSnr, maxSnr);
            expect(rx1Normalized).toBe(1.0);
            expect(getSnrColor(rx1Normalized)).toBe('rgb(40, 167, 69)');

            // Medium receiver should be yellow
            const rx2Normalized = normalizeSnr(receivers[1].snr_db, minSnr, maxSnr);
            expect(rx2Normalized).toBe(0.5);
            expect(getSnrColor(rx2Normalized)).toBe('rgb(255, 193, 7)');

            // Worst receiver should be red
            const rx3Normalized = normalizeSnr(receivers[2].snr_db, minSnr, maxSnr);
            expect(rx3Normalized).toBe(0.0);
            expect(getSnrColor(rx3Normalized)).toBe('rgb(220, 60, 69)');
        });
    });
});
