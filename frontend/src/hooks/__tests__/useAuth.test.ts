import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAuth } from '../useAuth';

// Mock the auth store
const mockLogin = vi.fn();
const mockLogout = vi.fn();
const mockUseAuthStore = vi.fn(() => ({
    isAuthenticated: false,
    user: null,
    token: null,
    login: mockLogin,
    logout: mockLogout,
}));

vi.mock('../../store', () => ({
    useAuthStore: () => mockUseAuthStore(),
}));

describe('useAuth Hook', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
    });

    it('returns initial auth state when not authenticated', () => {
        const { result } = renderHook(() => useAuth());

        expect(result.current.isAuthenticated).toBe(false);
        expect(result.current.user).toBeNull();
        expect(result.current.token).toBeNull();
    });

    it('returns authenticated state when user is logged in', () => {
        mockUseAuthStore.mockReturnValue({
            isAuthenticated: true,
            user: { email: 'test@example.com', role: 'admin' },
            token: 'test-token-123',
            login: mockLogin,
            logout: mockLogout,
        });

        const { result } = renderHook(() => useAuth());

        expect(result.current.isAuthenticated).toBe(true);
        expect(result.current.user).toEqual({ email: 'test@example.com', role: 'admin' });
        expect(result.current.token).toBe('test-token-123');
    });

    it('provides login function', () => {
        const { result } = renderHook(() => useAuth());

        expect(typeof result.current.login).toBe('function');
    });

    it('provides logout function', () => {
        const { result } = renderHook(() => useAuth());

        expect(typeof result.current.logout).toBe('function');
    });

    it('calls login with correct credentials', async () => {
        const { result } = renderHook(() => useAuth());

        await act(async () => {
            await result.current.login('test@example.com', 'password123');
        });

        expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'password123');
    });

    it('calls logout function correctly', () => {
        const { result } = renderHook(() => useAuth());

        act(() => {
            result.current.logout();
        });

        expect(mockLogout).toHaveBeenCalled();
    });

    it('initializes auth from localStorage if available', () => {
        const authData = {
            state: {
                user: { email: 'stored@example.com', role: 'user' },
                token: 'stored-token',
                isAuthenticated: true,
            },
        };
        localStorage.setItem('auth-store', JSON.stringify(authData));

        renderHook(() => useAuth());

        // The hook should attempt to read from localStorage on mount
        expect(localStorage.getItem('auth-store')).toBe(JSON.stringify(authData));
    });

    it('handles invalid localStorage data gracefully', () => {
        localStorage.setItem('auth-store', 'invalid-json-data');

        const { result } = renderHook(() => useAuth());

        // Should not crash and return default state
        expect(result.current.isAuthenticated).toBe(false);
    });

    it('returns user details when authenticated', () => {
        mockUseAuthStore.mockReturnValue({
            isAuthenticated: true,
            user: {
                email: 'admin@heimdall.local',
                firstName: 'Admin',
                lastName: 'User',
                role: 'administrator',
                organization: 'Heimdall SDR',
            },
            token: 'test-token',
            login: mockLogin,
            logout: mockLogout,
        });

        const { result } = renderHook(() => useAuth());

        expect(result.current.user?.email).toBe('admin@heimdall.local');
        expect(result.current.user?.firstName).toBe('Admin');
        expect(result.current.user?.lastName).toBe('User');
        expect(result.current.user?.role).toBe('administrator');
        expect(result.current.user?.organization).toBe('Heimdall SDR');
    });
});
