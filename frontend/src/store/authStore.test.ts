import { describe, it, expect, beforeEach } from 'vitest';
import { useAuthStore } from './authStore';

describe('authStore - Authentication Store', () => {
    beforeEach(() => {
        useAuthStore.setState({
            user: null,
            token: null,
            isAuthenticated: false,
        });
        localStorage.clear();
    });

    describe('login()', () => {
        it('should successfully login with correct credentials', async () => {
            const store = useAuthStore.getState();
            await store.login('admin@heimdall.local', 'Admin123!@#');
            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(true);
        });

        it('should reject invalid credentials', async () => {
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
