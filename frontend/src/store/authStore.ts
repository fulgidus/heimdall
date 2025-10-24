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
}

// Keycloak OAuth2/OIDC configuration from environment variables
const KEYCLOAK_URL = import.meta.env.VITE_KEYCLOAK_URL || 'http://localhost:8080';
const KEYCLOAK_REALM = import.meta.env.VITE_KEYCLOAK_REALM || 'heimdall';
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
                    // OAuth2 Resource Owner Password Credentials Grant (for development/E2E tests)
                    // Production should use PKCE or Authorization Code flow
                    const tokenUrl = `${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token`;
                    
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
