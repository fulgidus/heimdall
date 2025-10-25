/**
 * React Query Client Configuration
 * 
 * Configures caching, retries, and refetching behavior for API calls
 */

import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            // Refetch on window focus for real-time data
            refetchOnWindowFocus: true,
            // Cache data for 5 minutes
            staleTime: 1000 * 60 * 5,
            // Keep data in cache for 10 minutes
            gcTime: 1000 * 60 * 10,
            // Retry failed requests once
            retry: 1,
            // Don't retry on 4xx errors
            retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
        },
        mutations: {
            // Retry mutations once
            retry: 1,
        },
    },
});
