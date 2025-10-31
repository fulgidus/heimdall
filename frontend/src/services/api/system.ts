/**
 * System API Service
 * 
 * Handles system-level operations:
 * - Service health checks
 * - System metrics
 */

import api from '@/lib/api';
import { ServiceHealthSchema, DetailedHealthResponseSchema } from './schemas';
import type { ServiceHealth, DetailedHealthResponse } from './schemas';

/**
 * Check health of a specific service
 */
export async function checkServiceHealth(serviceName: string): Promise<ServiceHealth> {
    // Health endpoints are served at root level, not under /api/v1
    const response = await api.get(`/${serviceName}/health`);

    // Validate response with Zod
    const validated = ServiceHealthSchema.parse(response.data);
    return validated;
}

/**
 * Check health of all services
 */
export async function checkAllServicesHealth(): Promise<Record<string, ServiceHealth>> {
    const services = ['backend', 'training', 'inference'];
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

/**
 * Get detailed health check with dependency status for backend service
 */
export async function getDetailedHealth(): Promise<DetailedHealthResponse> {
    // Health endpoints are served at root level, not under /api/v1
    const response = await api.get('/health/detailed');

    // Validate response with Zod
    const validated = DetailedHealthResponseSchema.parse(response.data);
    return validated;
}

const systemService = {
    checkServiceHealth,
    checkAllServicesHealth,
    getAPIGatewayStatus,
    getDetailedHealth,
};

export default systemService;
