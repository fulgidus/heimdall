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
    const response = await api.get<WebSDRConfig[]>('/v1/acquisition/websdrs');
    return response.data;
}

/**
 * Check health status of all WebSDR receivers
 */
export async function checkWebSDRHealth(): Promise<Record<number, WebSDRHealthStatus>> {
    const response = await api.get<Record<number, WebSDRHealthStatus>>('/v1/acquisition/websdrs/health');
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
