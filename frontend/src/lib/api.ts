import axios from 'axios';
import { useAuthStore } from '../store';

// Use absolute URL to connect directly to Envoy API Gateway
// Never proxy through frontend Nginx (that would be circular)
// Browser connects directly to http://localhost/api (port 80)
export const getAPIBaseURL = () => {
    // If environment variable is set, use it
    if (import.meta.env.VITE_API_URL && !import.meta.env.VITE_API_URL.startsWith('/')) {
        return import.meta.env.VITE_API_URL;
    }

    // Otherwise, construct URL to connect directly to API Gateway on port 80
    const protocol = window.location.protocol; // http: or https:
    const host = window.location.hostname; // localhost, hostname, etc.
    return `${protocol}//${host}/api`;
};

export const API_BASE_URL = getAPIBaseURL();

// DEBUG: Log della configurazione API
console.log('🔧 API Configuration:', {
    VITE_API_URL: import.meta.env.VITE_API_URL,
    API_BASE_URL,
    environment: import.meta.env.MODE,
    isDev: import.meta.env.DEV,
});

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '10000', 10), // Keep at 10s as requested
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    },
    withCredentials: false, // CORS: Set to true if using cookies
});

// Request interceptor
api.interceptors.request.use(
    config => {
        console.log('📤 API Request:', {
            method: config.method?.toUpperCase(),
            url: config.url,
            fullURL: `${config.baseURL}${config.url}`,
        });
        const { token } = useAuthStore.getState();
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    error => {
        return Promise.reject(error);
    }
);

// Track if we're currently refreshing to avoid multiple simultaneous refresh attempts
let isRefreshing = false;
let failedQueue: Array<{ resolve: (value?: unknown) => void; reject: (reason?: unknown) => void }> =
    [];

const processQueue = (error: unknown = null) => {
    failedQueue.forEach(promise => {
        if (error) {
            promise.reject(error);
        } else {
            promise.resolve();
        }
    });
    failedQueue = [];
};

// Retry configuration - disable in test mode to avoid test timeouts
const IS_TEST_MODE = import.meta.env.MODE === 'test';
const MAX_RETRIES = IS_TEST_MODE ? 0 : 3;
const RETRY_DELAY_MS = 1000;
const RETRYABLE_CODES = ['ECONNABORTED', 'ETIMEDOUT', 'ENOTFOUND', 'ENETUNREACH'];
const RETRYABLE_STATUS = [408, 429, 500, 502, 503, 504];

// Helper to determine if request should be retried
const shouldRetry = (error: any, retryCount: number): boolean => {
    if (retryCount >= MAX_RETRIES) return false;

    // Don't retry on 4xx errors (client errors - bad request, not found, etc.)
    if (error.response?.status && error.response.status >= 400 && error.response.status < 500) {
        return false;
    }

    // Retry on network errors
    if (error.code && RETRYABLE_CODES.includes(error.code)) return true;

    // Retry on specific HTTP status codes (5xx server errors)
    if (error.response?.status && RETRYABLE_STATUS.includes(error.response.status)) return true;

    // Retry on timeout
    if (error.message?.includes('timeout')) return true;

    return false;
};

// Helper to wait with exponential backoff
const waitForRetry = (retryCount: number): Promise<void> => {
    const delay = RETRY_DELAY_MS * Math.pow(2, retryCount);
    console.log(`⏳ Retrying in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`);
    return new Promise(resolve => setTimeout(resolve, delay));
};

// Response interceptor with token refresh and retry logic
api.interceptors.response.use(
    response => {
        console.log('📥 API Response:', {
            status: response.status,
            url: response.config.url,
            dataSize: JSON.stringify(response.data).length,
        });
        return response;
    },
    async error => {
        const originalRequest = error.config;

        // Initialize retry count if not present
        if (!originalRequest._retryCount) {
            originalRequest._retryCount = 0;
        }

        console.error('❌ API Error:', {
            status: error.response?.status,
            url: error.config?.url,
            message: error.message,
            code: error.code,
            data: error.response?.data,
            retryCount: originalRequest._retryCount,
        });

        // Handle 401 errors with token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
            if (isRefreshing) {
                // If already refreshing, queue this request
                return new Promise((resolve, reject) => {
                    failedQueue.push({ resolve, reject });
                })
                    .then(() => {
                        originalRequest.headers.Authorization = `Bearer ${useAuthStore.getState().token}`;
                        return api(originalRequest);
                    })
                    .catch(err => {
                        return Promise.reject(err);
                    });
            }

            originalRequest._retry = true;
            isRefreshing = true;

            try {
                const refreshSuccessful = await useAuthStore.getState().refreshAccessToken();

                if (refreshSuccessful) {
                    const newToken = useAuthStore.getState().token;
                    originalRequest.headers.Authorization = `Bearer ${newToken}`;
                    processQueue(null);
                    return api(originalRequest);
                } else {
                    processQueue(new Error('Token refresh failed'));
                    useAuthStore.getState().logout();
                    window.location.href = '/login';
                    return Promise.reject(error);
                }
            } catch (refreshError) {
                processQueue(refreshError);
                useAuthStore.getState().logout();
                window.location.href = '/login';
                return Promise.reject(refreshError);
            } finally {
                isRefreshing = false;
            }
        }

        // Retry logic for transient failures
        if (shouldRetry(error, originalRequest._retryCount)) {
            originalRequest._retryCount++;
            await waitForRetry(originalRequest._retryCount - 1);
            console.log(`🔄 Retrying request: ${originalRequest.url} (${originalRequest._retryCount}/${MAX_RETRIES})`);
            return api(originalRequest);
        }

        return Promise.reject(error);
    }
);

export default api;
