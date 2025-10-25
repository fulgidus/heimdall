/**
 * Uncertainty Ellipse Calculation Utilities
 * 
 * Calculate ellipse geometry for uncertainty visualization on maps.
 * Supports 1-sigma (68%), 2-sigma (95%), and 3-sigma (99.7%) confidence levels.
 */

export interface EllipseParams {
    centerLat: number;
    centerLng: number;
    sigmaX: number;      // Semi-major axis in meters
    sigmaY: number;      // Semi-minor axis in meters
    rotation: number;    // Rotation angle in degrees
    confidence?: number; // Confidence level multiplier (1, 2, or 3 for sigma levels)
}

export interface GeoJSONFeature {
    type: 'Feature';
    geometry: {
        type: 'Polygon';
        coordinates: number[][][];
    };
    properties: Record<string, unknown>;
}

// Note: EARTH_RADIUS_M reserved for future geographic calculations
// const EARTH_RADIUS_M = 6378137; // Earth radius in meters (WGS84)

/**
 * Convert meters to degrees latitude
 */
function metersToDegreesLat(meters: number): number {
    return meters / (111320); // 1 degree latitude ≈ 111.32 km
}

/**
 * Convert meters to degrees longitude at given latitude
 */
function metersToDegreesLng(meters: number, latitude: number): number {
    const latRad = (latitude * Math.PI) / 180;
    return meters / (111320 * Math.cos(latRad));
}

/**
 * Generate ellipse points as [lng, lat] pairs
 * 
 * @param params - Ellipse parameters
 * @param numPoints - Number of points to generate (default: 64 for smooth rendering)
 * @returns Array of [lng, lat] coordinates forming the ellipse
 */
export function generateEllipsePoints(
    params: EllipseParams,
    numPoints: number = 64
): [number, number][] {
    const { centerLat, centerLng, sigmaX, sigmaY, rotation, confidence = 1 } = params;
    
    // Scale by confidence level (1σ, 2σ, 3σ)
    const scaledSigmaX = sigmaX * confidence;
    const scaledSigmaY = sigmaY * confidence;
    
    // Convert rotation to radians
    const rotationRad = (rotation * Math.PI) / 180;
    
    const points: [number, number][] = [];
    
    for (let i = 0; i <= numPoints; i++) {
        const angle = (i / numPoints) * 2 * Math.PI;
        
        // Parametric ellipse equations
        const x = scaledSigmaX * Math.cos(angle);
        const y = scaledSigmaY * Math.sin(angle);
        
        // Apply rotation
        const xRotated = x * Math.cos(rotationRad) - y * Math.sin(rotationRad);
        const yRotated = x * Math.sin(rotationRad) + y * Math.cos(rotationRad);
        
        // Convert to lat/lng offsets
        const latOffset = metersToDegreesLat(yRotated);
        const lngOffset = metersToDegreesLng(xRotated, centerLat);
        
        // Add to center coordinates
        const lng = centerLng + lngOffset;
        const lat = centerLat + latOffset;
        
        points.push([lng, lat]);
    }
    
    return points;
}

/**
 * Create GeoJSON Feature for an uncertainty ellipse
 * 
 * @param params - Ellipse parameters
 * @param properties - Additional properties for the feature
 * @returns GeoJSON Feature object
 */
export function createEllipseFeature(
    params: EllipseParams,
    properties: Record<string, unknown> = {}
): GeoJSONFeature {
    const coordinates = generateEllipsePoints(params);
    
    return {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [coordinates],
        },
        properties: {
            centerLat: params.centerLat,
            centerLng: params.centerLng,
            sigmaX: params.sigmaX,
            sigmaY: params.sigmaY,
            rotation: params.rotation,
            confidence: params.confidence || 1,
            ...properties,
        },
    };
}

/**
 * Create multiple confidence level ellipses (1σ, 2σ, 3σ)
 * 
 * @param params - Base ellipse parameters
 * @param levels - Confidence levels to generate (default: [1, 2, 3])
 * @returns Array of GeoJSON Features
 */
export function createConfidenceEllipses(
    params: Omit<EllipseParams, 'confidence'>,
    levels: number[] = [1, 2, 3]
): GeoJSONFeature[] {
    return levels.map((level) =>
        createEllipseFeature(
            { ...params, confidence: level },
            { confidenceLevel: level }
        )
    );
}

/**
 * Get color for confidence level
 * 
 * @param confidenceLevel - Sigma level (1, 2, or 3)
 * @returns Hex color string
 */
export function getConfidenceColor(confidenceLevel: number): string {
    switch (confidenceLevel) {
        case 1:
            return '#10b981'; // Green - 68% confidence
        case 2:
            return '#f59e0b'; // Yellow/Orange - 95% confidence
        case 3:
            return '#3b82f6'; // Light Blue - 99.7% confidence
        default:
            return '#6b7280'; // Gray - fallback
    }
}

/**
 * Get opacity for confidence level
 * 
 * @param confidenceLevel - Sigma level (1, 2, or 3)
 * @returns Opacity value (0-1)
 */
export function getConfidenceOpacity(confidenceLevel: number): number {
    switch (confidenceLevel) {
        case 1:
            return 0.3;
        case 2:
            return 0.2;
        case 3:
            return 0.1;
        default:
            return 0.15;
    }
}

/**
 * Calculate simple circular uncertainty from single uncertainty value
 * 
 * @param centerLat - Center latitude
 * @param centerLng - Center longitude
 * @param uncertaintyM - Uncertainty radius in meters
 * @returns Ellipse parameters for circular uncertainty
 */
export function createCircularUncertainty(
    centerLat: number,
    centerLng: number,
    uncertaintyM: number
): EllipseParams {
    return {
        centerLat,
        centerLng,
        sigmaX: uncertaintyM,
        sigmaY: uncertaintyM,
        rotation: 0,
        confidence: 1,
    };
}
