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

    // Validate response with Zod
    const validated = z.array(WebSDRConfigSchema).parse(response.data);

    console.log('✅ WebSDRService.getWebSDRs(): ricevuti', validated.length, 'WebSDRs');
    return validated;
}

/**
 * Check health status of all WebSDR receivers
 */
export async function checkWebSDRHealth(): Promise<Record<string, WebSDRHealthStatus>> {
    console.log('🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health');
    const response = await api.get('/v1/acquisition/websdrs/health');
    
    // Validate response with Zod
    const validated = z.record(z.string(), WebSDRHealthStatusSchema).parse(response.data);
    
    console.log('✅ WebSDRService.checkWebSDRHealth(): ricevuto health status');
    return validated;
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
    
    // Validate response with Zod
    const validated = WebSDRConfigSchema.parse(response.data);
    
    console.log('✅ WebSDRService.createWebSDR(): created', validated.name);
    return validated;
}

/**
 * Update an existing WebSDR station
 */
export async function updateWebSDR(id: string, data: Partial<WebSDRConfig>): Promise<WebSDRConfig> {
    console.log('✏️ WebSDRService.updateWebSDR():', id);
    const response = await api.put(`/api/v1/acquisition/websdrs/${id}`, data);
    
    // Validate response with Zod
    const validated = WebSDRConfigSchema.parse(response.data);
    
    console.log('✅ WebSDRService.updateWebSDR(): updated', validated.name);
    return validated;
}

/**
 * Delete a WebSDR station (soft delete by default)
 */
export async function deleteWebSDR(id: string, hardDelete: boolean = false): Promise<void> {
    console.log('🗑️ WebSDRService.deleteWebSDR():', id, hardDelete ? '(HARD DELETE)' : '(soft delete)');
    await api.delete(`/api/v1/acquisition/websdrs/${id}`, {
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
