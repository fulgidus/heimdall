import { useEffect, useRef } from 'react';
import { useAuthStore } from '../store';

/**
 * Hook to automatically refresh access token before it expires.
 * Refreshes token 1 minute before expiration (default JWT lifetime is 5-15 minutes).
 */
export const useTokenRefresh = () => {
  const { token, refreshToken, isAuthenticated } = useAuthStore();
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Only set up refresh if user is authenticated and has tokens
    if (!isAuthenticated || !token || !refreshToken) {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      return;
    }

    try {
      // Decode JWT to get expiration time
      const tokenParts = token.split('.');
      if (tokenParts.length !== 3) {
        console.error('Invalid JWT token format');
        return;
      }

      const payload = JSON.parse(atob(tokenParts[1]));
      const expiresAt = payload.exp * 1000; // Convert to milliseconds
      const now = Date.now();
      const timeUntilExpiry = expiresAt - now;

      // Refresh 60 seconds before expiration, or immediately if already expired/expiring soon
      const refreshIn = Math.max(timeUntilExpiry - 60000, 1000);

      console.log(
        `🕐 Token expires in ${Math.round(timeUntilExpiry / 1000)}s, scheduling refresh in ${Math.round(refreshIn / 1000)}s`
      );

      // Clear any existing timer
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }

      // Schedule refresh
      refreshTimerRef.current = setTimeout(async () => {
        console.log('🔄 Proactively refreshing token...');
        // Get fresh reference from store to avoid stale closure
        const success = await useAuthStore.getState().refreshAccessToken();

        // Don't schedule next refresh here - let the effect handle it when token updates
        if (!success) {
          console.error('❌ Token refresh failed, user will be logged out');
        }
      }, refreshIn);
    } catch (error) {
      console.error('Error scheduling token refresh:', error);
    }

    // Cleanup on unmount
    return () => {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    };
  }, [token, refreshToken, isAuthenticated]);
};
