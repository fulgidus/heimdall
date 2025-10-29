import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAuthStore } from './authStore';

// Mock fetch
global.fetch = vi.fn();

describe('Token Refresh', () => {
    beforeEach(() => {
        // Clear store before each test
        useAuthStore.setState({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
        });
        vi.clearAllMocks();
    });

    it('should refresh token successfully', async () => {
        // Setup initial state with tokens
        useAuthStore.setState({
            token: 'old-access-token',
            refreshToken: 'valid-refresh-token',
            isAuthenticated: true,
            user: {
                id: '1',
                email: 'test@example.com',
                name: 'Test User',
                role: 'user',
            },
        });

        // Mock successful refresh response
        (global.fetch as any).mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                access_token: 'new-access-token',
                refresh_token: 'new-refresh-token',
            }),
        });

        const result = await useAuthStore.getState().refreshAccessToken();

        expect(result).toBe(true);
        expect(useAuthStore.getState().token).toBe('new-access-token');
        expect(useAuthStore.getState().refreshToken).toBe('new-refresh-token');
    });

    it('should logout on refresh failure', async () => {
        useAuthStore.setState({
            token: 'old-access-token',
            refreshToken: 'invalid-refresh-token',
            isAuthenticated: true,
            user: {
                id: '1',
                email: 'test@example.com',
                name: 'Test User',
                role: 'user',
            },
        });

        // Mock failed refresh response
        (global.fetch as any).mockResolvedValueOnce({
            ok: false,
            status: 401,
        });

        const result = await useAuthStore.getState().refreshAccessToken();

        expect(result).toBe(false);
        expect(useAuthStore.getState().token).toBeNull();
        expect(useAuthStore.getState().refreshToken).toBeNull();
        expect(useAuthStore.getState().isAuthenticated).toBe(false);
    });

    it('should return false when no refresh token available', async () => {
        useAuthStore.setState({
            token: 'access-token',
            refreshToken: null,
            isAuthenticated: true,
        });

        const result = await useAuthStore.getState().refreshAccessToken();

        expect(result).toBe(false);
        expect(global.fetch).not.toHaveBeenCalled();
    });
});
