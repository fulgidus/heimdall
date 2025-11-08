import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
    id: string;
    email: string;
    name: string;
    role: 'admin' | 'operator' | 'user';
    roles: string[]; // Complete array of roles from Keycloak
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

/**
 * Extract the highest role from Keycloak realm roles.
 * Implements role hierarchy: admin > operator > user
 * 
 * @param realmRoles - Array of roles from Keycloak JWT (realm_access.roles)
 * @returns The highest role in the hierarchy
 */
function extractUserRole(realmRoles: string[]): 'admin' | 'operator' | 'user' {
    if (realmRoles.includes('admin')) return 'admin';
    if (realmRoles.includes('operator')) return 'operator';
    return 'user';
}

export const useAuthStore = create<AuthStore>()(
    persist(
        set => ({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,

            login: async (email: string, password: string) => {
                try {
                    // Use Envoy proxy for Keycloak auth endpoint
                    // The frontend is served on port 3000 (internal via docker),
                    // but Envoy routes traffic on port 80 to all services.
                    // We need to call Envoy on port 80, not port 3000.
                    const realm = import.meta.env.VITE_KEYCLOAK_REALM || 'heimdall';
                    const clientId = KEYCLOAK_CLIENT_ID;

                    // Build the Keycloak token URL:
                    // - In production/docker: http://localhost/auth/realms/{realm}/protocol/openid-connect/token
                    // - Use window.location.protocol and hostname, but ensure port is 80 or empty (default)
                    const protocol = window.location.protocol; // http: or https:
                    const hostname = window.location.hostname; // localhost
                    const baseUrl = `${protocol}//${hostname}:80`;
                    const tokenUrl = `${baseUrl}/auth/realms/${realm}/protocol/openid-connect/token`;

                    const params = new URLSearchParams();
                    params.append('grant_type', 'password');
                    params.append('client_id', clientId);
                    params.append('username', email);
                    params.append('password', password);

                    const response = await fetch(tokenUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        credentials: 'omit',
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
                    const realmRoles = payload.realm_access?.roles || [];
                    const user: User = {
                        id: payload.sub || '1',
                        email: payload.email || email,
                        name: payload.name || payload.preferred_username || email.split('@')[0],
                        role: extractUserRole(realmRoles),
                        roles: realmRoles,
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

            setUser: user => {
                set({ user, isAuthenticated: !!user });
            },

            setToken: token => {
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
                    const refreshUrl = '/v1/auth/refresh';

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
            partialize: state => ({
                user: state.user,
                token: state.token,
                refreshToken: state.refreshToken,
                isAuthenticated: state.isAuthenticated,
            }),
        }
    )
);
