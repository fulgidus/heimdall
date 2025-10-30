/**
 * WebSDR API Service
 * 
 * Handles all WebSDR-related API calls:
 * - List WebSDR receivers
 * - Check WebSDR health status
 * - Get WebSDR configuration
 */

import { z } from 'zod';
import api from '@/lib/api';
import { WebSDRConfigSchema, WebSDRHealthStatusSchema } from './schemas';
import type { WebSDRConfig, WebSDRHealthStatus } from './schemas';

/**
 * Get list of all configured WebSDR receivers
 */
export async function getWebSDRs(): Promise<WebSDRConfig[]> {
    console.log('📡 WebSDRService.getWebSDRs(): calling GET /api/v1/acquisition/websdrs-all');
    const response = await api.get('/v1/acquisition/websdrs-all');

    console.log('📊 Raw response data:', response.data, 'Type:', typeof response.data);

    // Handle case where response.data might not be an array
    let data = response.data;
    if (data && typeof data === 'object' && 'data' in data) {
        // If response is wrapped in {data: [...]}
        data = data.data;
    }

    if (!Array.isArray(data)) {
        console.error('❌ getWebSDRs(): Response is not an array. Received:', data);
        throw new Error(`Expected array of WebSDRs, got ${typeof data}: ${JSON.stringify(data)}`);
    }

    // Normalize nulls for optional fields so Zod optional/nullable rules behave consistently
    // Backend may return `null` for optional values; convert those to `undefined` which
    // matches `.optional()` semantics and still allows `.nullable()` values when present.
    const normalized = (data as any[]).map(item => ({
        ...item,
        location_description: item.location_description ?? undefined,
        admin_email: item.admin_email ?? undefined,
        altitude_asl: item.altitude_asl ?? undefined,
    }));

    // Validate response with Zod
    try {
        const validated = z.array(WebSDRConfigSchema).parse(normalized);
        console.log('✅ WebSDRService.getWebSDRs(): ricevuti', validated.length, 'WebSDRs');
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in getWebSDRs():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`WebSDR validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Check health status of all WebSDR receivers
 */
export async function checkWebSDRHealth(): Promise<Record<string, WebSDRHealthStatus>> {
    console.log('🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health');
    const response = await api.get('/v1/acquisition/websdrs/health');

    console.log('📊 Raw health response data:', response.data, 'Type:', typeof response.data);

    // Handle case where response.data might be wrapped
    let data = response.data;
    if (data && typeof data === 'object' && 'data' in data) {
        data = data.data;
    }

    if (!data || typeof data !== 'object') {
        console.error('❌ checkWebSDRHealth(): Response is not an object. Received:', data);
        throw new Error(`Expected health status object, got ${typeof data}: ${JSON.stringify(data)}`);
    }

    // Validate response with Zod
    try {
        const validated = z.record(z.string(), WebSDRHealthStatusSchema).parse(data);
        console.log('✅ WebSDRService.checkWebSDRHealth(): ricevuto health status');
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in checkWebSDRHealth():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`WebSDR health validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Get configuration for specific WebSDR
 */
export async function getWebSDRConfig(id: string): Promise<WebSDRConfig> {
    const websdrs = await getWebSDRs();
    const websdr = websdrs.find(w => w.id === id);

    if (!websdr) {
        throw new Error(`WebSDR with id ${id} not found`);
    }

    return websdr;
}

/**
 * Get active WebSDRs only
 */
export async function getActiveWebSDRs(): Promise<WebSDRConfig[]> {
    const websdrs = await getWebSDRs();
    // Ensure websdrs is an array before filtering
    if (!Array.isArray(websdrs)) {
        console.error('❌ getActiveWebSDRs(): websdrs is not an array:', typeof websdrs);
        return [];
    }
    return websdrs.filter(w => w.is_active);
}

/**
 * Create a new WebSDR station
 */
export async function createWebSDR(data: Omit<WebSDRConfig, 'id'>): Promise<WebSDRConfig> {
    console.log('➕ WebSDRService.createWebSDR():', data.name);
    const response = await api.post('/v1/acquisition/websdrs', data);

    // Handle potential response wrapping
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = WebSDRConfigSchema.parse(responseData);
        console.log('✅ WebSDRService.createWebSDR(): created', validated.name);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in createWebSDR():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`WebSDR creation validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Update an existing WebSDR station
 */
export async function updateWebSDR(id: string, data: Partial<WebSDRConfig>): Promise<WebSDRConfig> {
    console.log('✏️ WebSDRService.updateWebSDR():', id);
    const response = await api.put(`/v1/acquisition/websdrs/${id}`, data);

    // Handle potential response wrapping
    let responseData = response.data;
    if (responseData && typeof responseData === 'object' && 'data' in responseData) {
        responseData = responseData.data;
    }

    // Validate response with Zod
    try {
        const validated = WebSDRConfigSchema.parse(responseData);
        console.log('✅ WebSDRService.updateWebSDR(): updated', validated.name);
        return validated;
    } catch (zodError) {
        console.error('❌ Zod validation error in updateWebSDR():', zodError);
        if (zodError instanceof z.ZodError) {
            throw new Error(`WebSDR update validation error: ${zodError.message}`);
        }
        throw zodError;
    }
}

/**
 * Delete a WebSDR station (soft delete by default)
 */
export async function deleteWebSDR(id: string, hardDelete: boolean = false): Promise<void> {
    console.log('🗑️ WebSDRService.deleteWebSDR():', id, hardDelete ? '(HARD DELETE)' : '(soft delete)');
    await api.delete(`/v1/acquisition/websdrs/${id}`, {
        params: { hard_delete: hardDelete }
    });
    console.log('✅ WebSDRService.deleteWebSDR(): deleted');
}

const webSDRService = {
    getWebSDRs,
    checkWebSDRHealth,
    getWebSDRConfig,
    getActiveWebSDRs,
    createWebSDR,
    updateWebSDR,
    deleteWebSDR,
};

export default webSDRService;
