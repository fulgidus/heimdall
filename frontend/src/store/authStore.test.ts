import { describe, it, expect, beforeEach, vi } from 'vitest';

// Unmock the stores module for this test (we want to test the real store)
vi.unmock('@/store');
vi.unmock('@/store/authStore');

// Import after unmocking
import { useAuthStore } from './authStore';

interface JWTPayload {
    sub?: string;
    email?: string;
    [key: string]: unknown;
}

// Helper to create a mock JWT token
function createMockJWT(payload: JWTPayload): string {
    const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
    const payloadEncoded = btoa(JSON.stringify(payload));
    const signature = 'mock-signature';
    return `${header}.${payloadEncoded}.${signature}`;
}

describe('authStore - Authentication Store', () => {
    beforeEach(() => {
        useAuthStore.setState({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
        });
        localStorage.clear();
        vi.clearAllMocks();
    });

    describe('login()', () => {
        it('should successfully login with correct credentials via Keycloak', async () => {
            // Create mock JWT with user claims
            const mockJWT = createMockJWT({
                sub: '123e4567-e89b-12d3-a456-426614174000',
                email: 'admin@heimdall.local',
                name: 'Administrator',
                preferred_username: 'admin',
                realm_access: {
                    roles: ['admin', 'user']
                }
            });

            // Mock successful Keycloak OAuth2 token response
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: async () => ({
                    access_token: mockJWT,
                    refresh_token: 'mock-refresh-token',
                    expires_in: 3600,
                    token_type: 'Bearer',
                }),
            });

            const store = useAuthStore.getState();
            await store.login('admin@heimdall.local', 'Admin123!@#');
            const state = useAuthStore.getState();

            expect(state.isAuthenticated).toBe(true);
            expect(state.token).toBe(mockJWT);
            expect(state.refreshToken).toBe('mock-refresh-token');
            expect(state.user?.email).toBe('admin@heimdall.local');
            expect(state.user?.role).toBe('admin');

            // Verify fetch was called with correct Keycloak OAuth2 token endpoint
            // The authStore calls Keycloak directly through the API gateway/proxy
            expect(global.fetch).toHaveBeenCalledWith(
                expect.stringContaining('/auth/realms/heimdall/protocol/openid-connect/token'),
                expect.objectContaining({
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: expect.stringContaining('grant_type=password'),
                })
            );
        });

        it('should reject invalid credentials', async () => {
            // Mock failed Keycloak response
            global.fetch = vi.fn().mockResolvedValue({
                ok: false,
                status: 401,
                json: async () => ({
                    error: 'invalid_grant',
                    error_description: 'Invalid user credentials'
                }),
            });

            const store = useAuthStore.getState();
            try {
                await store.login('wrong@email.com', 'WrongPassword');
                expect.fail('Should throw error');
            } catch (error) {
                expect((error as Error).message).toContain('Invalid');
            }

            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(false);
            expect(state.token).toBeNull();
        });
    });

    describe('logout()', () => {
        it('should clear auth state on logout', async () => {
            // Create mock JWT
            const mockJWT = createMockJWT({
                sub: '123',
                email: 'admin@heimdall.local',
                name: 'Admin',
            });

            // Mock successful Keycloak login
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: async () => ({
                    access_token: mockJWT,
                    refresh_token: 'mock-refresh-token',
                    expires_in: 3600,
                    token_type: 'Bearer',
                }),
            });

            const store = useAuthStore.getState();
            await store.login('admin@heimdall.local', 'Admin123!@#');

            // Verify login succeeded
            expect(useAuthStore.getState().isAuthenticated).toBe(true);

            // Logout
            store.logout();

            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(false);
            expect(state.token).toBeNull();
            expect(state.refreshToken).toBeNull();
            expect(state.user).toBeNull();
        });
    });

    describe('setUser()', () => {
        it('should set user correctly', () => {
            const store = useAuthStore.getState();
            const user = {
                id: '1',
                email: 'test@test.com',
                name: 'Test',
                role: 'admin' as const,
            };
            store.setUser(user);
            expect(useAuthStore.getState().isAuthenticated).toBe(true);
            expect(useAuthStore.getState().user).toEqual(user);
        });
    });

    describe('setToken()', () => {
        it('should set token correctly', () => {
            const store = useAuthStore.getState();
            store.setToken('my-token');
            expect(useAuthStore.getState().token).toBe('my-token');
        });
    });
});
