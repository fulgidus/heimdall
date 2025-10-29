import axios from 'axios';
import { useAuthStore } from '../store';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// DEBUG: Log della configurazione API
console.log('🔧 API Configuration:', {
    VITE_API_URL: import.meta.env.VITE_API_URL,
    API_BASE_URL,
    environment: import.meta.env.MODE,
    isDev: import.meta.env.DEV,
});

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
api.interceptors.request.use((config) => {
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
}, (error) => {
    return Promise.reject(error);
});

// Track if we're currently refreshing to avoid multiple simultaneous refresh attempts
let isRefreshing = false;
let failedQueue: Array<{ resolve: (value?: unknown) => void; reject: (reason?: unknown) => void }> = [];

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

// Response interceptor with token refresh logic
api.interceptors.response.use(
    (response) => {
        console.log('📥 API Response:', {
            status: response.status,
            url: response.config.url,
            dataSize: JSON.stringify(response.data).length,
        });
        return response;
    },
    async (error) => {
        const originalRequest = error.config;

        console.error('❌ API Error:', {
            status: error.response?.status,
            url: error.config?.url,
            message: error.message,
            data: error.response?.data,
        });

        // Handle 401 errors with token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
            if (isRefreshing) {
                // If already refreshing, queue this request
                return new Promise((resolve, reject) => {
                    failedQueue.push({ resolve, reject });
                }).then(() => {
                    originalRequest.headers.Authorization = `Bearer ${useAuthStore.getState().token}`;
                    return api(originalRequest);
                }).catch(err => {
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

        return Promise.reject(error);
    }
);

export default api;
