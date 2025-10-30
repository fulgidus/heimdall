import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
    id: string;
    email: string;
    name: string;
    role: 'admin' | 'user' | 'viewer';
    avatar?: string;
}

interface AuthStore {
    user: User | null;
    token: string | null;
    refreshToken: string | null;
    isAuthenticated: boolean;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
    setUser: (user: User | null) => void;
    setToken: (token: string | null) => void;
    refreshAccessToken: () => Promise<boolean>;
}

// API Gateway configuration (reads from .env)
// Use the same API URL as other calls (proxied through Nginx)
// In development: Vite proxy handles /api/*
// In production: Nginx forwards /api/* to api-gateway

// Keycloak OAuth2/OIDC configuration
const KEYCLOAK_CLIENT_ID = import.meta.env.VITE_KEYCLOAK_CLIENT_ID || 'heimdall-frontend';

export const useAuthStore = create<AuthStore>()(
    persist(
        (set) => ({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,

            login: async (email: string, password: string) => {
                try {
                    // Use relative path for auth (proxied through Nginx in production, Vite proxy in dev)
                    const tokenUrl = '/api/v1/auth/login';

                    const params = new URLSearchParams();
                    params.append('grant_type', 'password');
                    params.append('client_id', KEYCLOAK_CLIENT_ID);
                    params.append('username', email);
                    params.append('password', password);

                    const response = await fetch(tokenUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        credentials: 'omit',
                        mode: 'cors',
                        cache: 'no-cache',
                        body: params.toString(),
                    });

                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(errorData.error_description || 'Invalid email or password');
                    }

                    const data = await response.json();

                    // Decode JWT access token to extract user information
                    // Note: This is safe for client-side as we don't verify signature here
                    // Backend services verify JWT signature using Keycloak public keys
                    const tokenParts = data.access_token.split('.');
                    if (tokenParts.length !== 3) {
                        throw new Error('Invalid JWT token format');
                    }

                    const payload = JSON.parse(atob(tokenParts[1]));

                    // Extract user information from JWT claims
                    const user: User = {
                        id: payload.sub || '1',
                        email: payload.email || email,
                        name: payload.name || payload.preferred_username || email.split('@')[0],
                        role: payload.realm_access?.roles?.includes('admin') ? 'admin' : 'user',
                        avatar: payload.picture,
                    };

                    set({
                        user,
                        token: data.access_token,
                        refreshToken: data.refresh_token,
                        isAuthenticated: true,
                    });
                } catch (error) {
                    console.error('Login failed:', error);
                    set({
                        user: null,
                        token: null,
                        refreshToken: null,
                        isAuthenticated: false,
                    });
                    throw error;
                }
            },

            logout: () => {
                set({
                    user: null,
                    token: null,
                    refreshToken: null,
                    isAuthenticated: false,
                });
            },

            setUser: (user) => {
                set({ user, isAuthenticated: !!user });
            },

            setToken: (token) => {
                set({ token });
            },

            refreshAccessToken: async () => {
                const state = useAuthStore.getState();
                const currentRefreshToken = state.refreshToken;

                if (!currentRefreshToken) {
                    console.warn('No refresh token available');
                    return false;
                }

                try {
                    const refreshUrl = '/api/v1/auth/refresh';

                    const response = await fetch(refreshUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        credentials: 'omit',
                        mode: 'cors',
                        cache: 'no-cache',
                        body: JSON.stringify({
                            refresh_token: currentRefreshToken,
                        }),
                    });

                    if (!response.ok) {
                        console.error('Token refresh failed:', response.status);
                        // Refresh token is invalid or expired, logout user
                        useAuthStore.getState().logout();
                        return false;
                    }

                    const data = await response.json();

                    // Update tokens in store
                    set({
                        token: data.access_token,
                        refreshToken: data.refresh_token,
                    });

                    console.log('✅ Token refreshed successfully');
                    return true;
                } catch (error) {
                    console.error('Token refresh error:', error);
                    useAuthStore.getState().logout();
                    return false;
                }
            },
        }),
        {
            name: 'auth-store',
            partialize: (state) => ({
                user: state.user,
                token: state.token,
                refreshToken: state.refreshToken,
                isAuthenticated: state.isAuthenticated,
            }),
        }
    )
);
