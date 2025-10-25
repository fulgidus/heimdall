/**
 * Tests for Uncertainty Ellipse Utilities
 */

import { describe, it, expect } from 'vitest';
import {
    generateEllipsePoints,
    createEllipseFeature,
    createConfidenceEllipses,
    getConfidenceColor,
    getConfidenceOpacity,
    createCircularUncertainty,
} from './ellipse';

describe('Ellipse Utilities', () => {
    describe('generateEllipsePoints', () => {
        it('should generate correct number of points', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const points = generateEllipsePoints(params, 32);
            expect(points).toHaveLength(33); // numPoints + 1 to close the polygon
        });

        it('should generate points as [lng, lat] pairs', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const points = generateEllipsePoints(params, 8);
            points.forEach((point) => {
                expect(point).toHaveLength(2);
                expect(typeof point[0]).toBe('number'); // lng
                expect(typeof point[1]).toBe('number'); // lat
            });
        });

        it('should scale by confidence level', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 100,
                rotation: 0,
            };

            const points1Sigma = generateEllipsePoints({ ...params, confidence: 1 }, 4);
            const points2Sigma = generateEllipsePoints({ ...params, confidence: 2 }, 4);

            // 2-sigma should be roughly twice as large as 1-sigma
            const distance1 = Math.abs(points1Sigma[0][0] - params.centerLng);
            const distance2 = Math.abs(points2Sigma[0][0] - params.centerLng);

            expect(distance2).toBeGreaterThan(distance1);
            expect(distance2 / distance1).toBeCloseTo(2, 0); // Within 1 decimal place
        });

        it('should close the polygon (first point = last point)', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const points = generateEllipsePoints(params, 16);
            expect(points[0][0]).toBeCloseTo(points[points.length - 1][0], 5);
            expect(points[0][1]).toBeCloseTo(points[points.length - 1][1], 5);
        });
    });

    describe('createEllipseFeature', () => {
        it('should create valid GeoJSON Feature', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const feature = createEllipseFeature(params);

            expect(feature.type).toBe('Feature');
            expect(feature.geometry.type).toBe('Polygon');
            expect(feature.geometry.coordinates).toHaveLength(1);
            expect(feature.properties.centerLat).toBe(44.5);
            expect(feature.properties.centerLng).toBe(9.0);
        });

        it('should include custom properties', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const feature = createEllipseFeature(params, { customProp: 'test' });

            expect(feature.properties.customProp).toBe('test');
        });
    });

    describe('createConfidenceEllipses', () => {
        it('should create multiple ellipses with different confidence levels', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const ellipses = createConfidenceEllipses(params, [1, 2, 3]);

            expect(ellipses).toHaveLength(3);
            expect(ellipses[0].properties.confidenceLevel).toBe(1);
            expect(ellipses[1].properties.confidenceLevel).toBe(2);
            expect(ellipses[2].properties.confidenceLevel).toBe(3);
        });

        it('should use default levels [1, 2, 3] when not specified', () => {
            const params = {
                centerLat: 44.5,
                centerLng: 9.0,
                sigmaX: 100,
                sigmaY: 50,
                rotation: 0,
            };

            const ellipses = createConfidenceEllipses(params);

            expect(ellipses).toHaveLength(3);
        });
    });

    describe('getConfidenceColor', () => {
        it('should return green for 1-sigma', () => {
            expect(getConfidenceColor(1)).toBe('#10b981');
        });

        it('should return yellow/orange for 2-sigma', () => {
            expect(getConfidenceColor(2)).toBe('#f59e0b');
        });

        it('should return light blue for 3-sigma', () => {
            expect(getConfidenceColor(3)).toBe('#3b82f6');
        });

        it('should return gray for unknown levels', () => {
            expect(getConfidenceColor(99)).toBe('#6b7280');
        });
    });

    describe('getConfidenceOpacity', () => {
        it('should return higher opacity for 1-sigma', () => {
            expect(getConfidenceOpacity(1)).toBe(0.3);
        });

        it('should return medium opacity for 2-sigma', () => {
            expect(getConfidenceOpacity(2)).toBe(0.2);
        });

        it('should return low opacity for 3-sigma', () => {
            expect(getConfidenceOpacity(3)).toBe(0.1);
        });
    });

    describe('createCircularUncertainty', () => {
        it('should create circular ellipse parameters', () => {
            const params = createCircularUncertainty(44.5, 9.0, 50);

            expect(params.centerLat).toBe(44.5);
            expect(params.centerLng).toBe(9.0);
            expect(params.sigmaX).toBe(50);
            expect(params.sigmaY).toBe(50); // Same as sigmaX for circular
            expect(params.rotation).toBe(0);
            expect(params.confidence).toBe(1);
        });
    });
});
