/**
 * Acquisition API Service
 *
 * Handles RF acquisition operations:
 * - Trigger acquisition tasks
 * - Check task status
 * - Get acquisition history
 */

import api from '@/lib/api';
import { AcquisitionTaskResponseSchema, AcquisitionStatusResponseSchema } from './schemas';
import type {
  AcquisitionRequest,
  AcquisitionTaskResponse,
  AcquisitionStatusResponse,
} from './schemas';

/**
 * Trigger a new RF acquisition
 */
export async function triggerAcquisition(
  request: AcquisitionRequest
): Promise<AcquisitionTaskResponse> {
  const response = await api.post('/v1/acquisition/acquire', request);

  // Validate response with Zod
  const validated = AcquisitionTaskResponseSchema.parse(response.data);
  return validated;
}

/**
 * Get status of an acquisition task
 */
export async function getAcquisitionStatus(taskId: string): Promise<AcquisitionStatusResponse> {
  const response = await api.get(`/v1/acquisition/status/${taskId}`);

  // Validate response with Zod
  const validated = AcquisitionStatusResponseSchema.parse(response.data);
  return validated;
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

        if (
          status.status === 'SUCCESS' ||
          status.status === 'FAILURE' ||
          status.status === 'REVOKED'
        ) {
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
