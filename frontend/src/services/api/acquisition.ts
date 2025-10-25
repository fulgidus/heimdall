/**
 * Acquisition API Service
 * 
 * Handles RF acquisition operations:
 * - Trigger acquisition tasks
 * - Check task status
 * - Get acquisition history
 */

import api from '@/lib/api';
import type {
    AcquisitionRequest,
    AcquisitionTaskResponse,
    AcquisitionStatusResponse,
} from './types';

/**
 * Trigger a new RF acquisition
 */
export async function triggerAcquisition(
    request: AcquisitionRequest
): Promise<AcquisitionTaskResponse> {
    const response = await api.post<AcquisitionTaskResponse>(
        '/api/v1/acquisition/acquire',
        request
    );
    return response.data;
}

/**
 * Get status of an acquisition task
 */
export async function getAcquisitionStatus(
    taskId: string
): Promise<AcquisitionStatusResponse> {
    const response = await api.get<AcquisitionStatusResponse>(
        `/api/v1/acquisition/status/${taskId}`
    );
    return response.data;
}

/**
 * Poll acquisition status until completion
 * 
 * @param taskId - Task ID to poll
 * @param onProgress - Callback for progress updates
 * @param pollInterval - Polling interval in ms (default: 2000)
 * @returns Final acquisition status
 */
export async function pollAcquisitionStatus(
    taskId: string,
    onProgress?: (status: AcquisitionStatusResponse) => void,
    pollInterval: number = 2000
): Promise<AcquisitionStatusResponse> {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const status = await getAcquisitionStatus(taskId);

                if (onProgress) {
                    onProgress(status);
                }

                if (status.status === 'SUCCESS' || status.status === 'FAILURE' || status.status === 'REVOKED') {
                    resolve(status);
                } else {
                    setTimeout(poll, pollInterval);
                }
            } catch (error) {
                reject(error);
            }
        };

        poll();
    });
}

const acquisitionService = {
    triggerAcquisition,
    getAcquisitionStatus,
    pollAcquisitionStatus,
};

export default acquisitionService;
