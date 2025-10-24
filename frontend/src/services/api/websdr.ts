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
    console.log('📡 WebSDRService.getWebSDRs(): calling GET /api/v1/acquisition/websdrs');
    const response = await api.get<WebSDRConfig[]>('/api/v1/acquisition/websdrs');
    console.log('✅ WebSDRService.getWebSDRs(): ricevuti', response.data.length, 'WebSDRs');
    return response.data;
}

/**
 * Check health status of all WebSDR receivers
 */
export async function checkWebSDRHealth(): Promise<Record<number, WebSDRHealthStatus>> {
    console.log('🏥 WebSDRService.checkWebSDRHealth(): calling GET /api/v1/acquisition/websdrs/health');
    const response = await api.get<Record<number, WebSDRHealthStatus>>('/api/v1/acquisition/websdrs/health');
    console.log('✅ WebSDRService.checkWebSDRHealth(): ricevuto health status');
    return response.data;
}

/**
 * Get configuration for specific WebSDR
 */
export async function getWebSDRConfig(id: number): Promise<WebSDRConfig> {
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
    return websdrs.filter(w => w.is_active);
}

const webSDRService = {
    getWebSDRs,
    checkWebSDRHealth,
    getWebSDRConfig,
    getActiveWebSDRs,
};

export default webSDRService;
