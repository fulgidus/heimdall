/**
 * System API Service
 * 
 * Handles system-level operations:
 * - Service health checks
 * - System metrics
 */

import api from '@/lib/api';
import type { ServiceHealth } from './types';

/**
 * Check health of a specific service
 */
export async function checkServiceHealth(serviceName: string): Promise<ServiceHealth> {
    const response = await api.get<ServiceHealth>(`/api/v1/${serviceName}/health`);
    return response.data;
}

/**
 * Check health of all services
 */
export async function checkAllServicesHealth(): Promise<Record<string, ServiceHealth>> {
    const services = ['api-gateway', 'rf-acquisition', 'inference'];
    const healthChecks = await Promise.allSettled(
        services.map(async (service) => {
            try {
                const health = await checkServiceHealth(service);
                return { service, health };
            } catch (error) {
                return {
                    service,
                    health: {
                        status: 'unhealthy' as const,
                        service,
                        version: 'unknown',
                        timestamp: new Date().toISOString(),
                        details: { error: String(error) },
                    },
                };
            }
        })
    );

    const result: Record<string, ServiceHealth> = {};
    healthChecks.forEach((check, index) => {
        if (check.status === 'fulfilled') {
            result[check.value.service] = check.value.health;
        } else {
            result[services[index]] = {
                status: 'unhealthy',
                service: services[index],
                version: 'unknown',
                timestamp: new Date().toISOString(),
                details: { error: check.reason },
            };
        }
    });

    return result;
}

/**
 * Get API Gateway root status
 */
export async function getAPIGatewayStatus(): Promise<Record<string, unknown>> {
    const response = await api.get('/');
    return response.data;
}

const systemService = {
    checkServiceHealth,
    checkAllServicesHealth,
    getAPIGatewayStatus,
};

export default systemService;
