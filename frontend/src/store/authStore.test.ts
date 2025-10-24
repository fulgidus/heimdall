import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAuthStore } from './authStore';

describe('authStore - Authentication Store', () => {
    beforeEach(() => {
        useAuthStore.setState({
            user: null,
            token: null,
            isAuthenticated: false,
        });
        localStorage.clear();
        vi.clearAllMocks();
    });

    describe('login()', () => {
        it('should successfully login with correct credentials', async () => {
            // Mock successful API response
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: async () => ({
                    token: 'test-token-123',
                    user: {
                        id: '1',
                        email: 'admin@heimdall.local',
                        name: 'Administrator',
                        role: 'admin',
                    },
                }),
            });

            const store = useAuthStore.getState();
            await store.login('admin@heimdall.local', 'Admin123!@#');
            const state = useAuthStore.getState();
            
            expect(state.isAuthenticated).toBe(true);
            expect(state.token).toBe('test-token-123');
            expect(state.user?.email).toBe('admin@heimdall.local');
            expect(global.fetch).toHaveBeenCalledWith('/api/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: 'admin@heimdall.local', password: 'Admin123!@#' }),
            });
        });

        it('should reject invalid credentials', async () => {
            // Mock failed API response
            global.fetch = vi.fn().mockResolvedValue({
                ok: false,
                status: 401,
            });

            const store = useAuthStore.getState();
            try {
                await store.login('wrong@email.com', 'WrongPassword');
                expect.fail('Should throw error');
            } catch (error) {
                expect((error as Error).message).toBe('Invalid email or password');
            }
        });
    });

    describe('logout()', () => {
        it('should clear auth state on logout', async () => {
            const store = useAuthStore.getState();
            await store.login('admin@heimdall.local', 'Admin123!@#');
            store.logout();
            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(false);
            expect(state.token).toBeNull();
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
