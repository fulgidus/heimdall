/**
 * WebSDR API Service
 * 
 * Handles all WebSDR-related API calls:
 * - List WebSDR receivers
 * - Check WebSDR health status
 * - Get WebSDR configuration
 */

import api from '@/lib/api';
import type { WebSDRConfig, WebSDRHealthStatus } from './types';

/**
 * Get list of all configured WebSDR receivers
 */
export async function getWebSDRs(): Promise<WebSDRConfig[]> {
    console.log('📡 WebSDRService.getWebSDRs(): calling GET /api/v1/acquisition/websdrs-all');
    const response = await api.get<WebSDRConfig[]>('/v1/acquisition/websdrs-all');

    // Ensure response.data is an array
    if (!Array.isArray(response.data)) {
        console.error('❌ WebSDRService.getWebSDRs(): Expected array but got:', typeof response.data, response.data);
        return [];
    }

    console.log('✅ WebSDRService.getWebSDRs(): ricevuti', response.data.length, 'WebSDRs');
    return response.data;
}

/**
 * Check health status of all WebSDR receivers
 */
export async function checkWebSDRHealth(): Promise<Record<string, WebSDRHealthStatus>> {
    console.log('🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health');
    const response = await api.get<Record<string, WebSDRHealthStatus>>('/v1/acquisition/websdrs/health');
    console.log('✅ WebSDRService.checkWebSDRHealth(): ricevuto health status');
    return response.data;
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
    const response = await api.post<WebSDRConfig>('/v1/acquisition/websdrs', data);
    console.log('✅ WebSDRService.createWebSDR(): created', response.data.name);
    return response.data;
}

/**
 * Update an existing WebSDR station
 */
export async function updateWebSDR(id: string, data: Partial<WebSDRConfig>): Promise<WebSDRConfig> {
    console.log('✏️ WebSDRService.updateWebSDR():', id);
    const response = await api.put<WebSDRConfig>(`/api/v1/acquisition/websdrs/${id}`, data);
    console.log('✅ WebSDRService.updateWebSDR(): updated', response.data.name);
    return response.data;
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
