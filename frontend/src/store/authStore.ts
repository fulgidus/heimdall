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
    isAuthenticated: boolean;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
    setUser: (user: User | null) => void;
    setToken: (token: string | null) => void;
}

export const useAuthStore = create<AuthStore>()(
    persist(
        (set) => ({
            user: null,
            token: null,
            isAuthenticated: false,

            login: async (email: string, password: string) => {
                // Get credentials from env variables
                const adminEmail = import.meta.env.VITE_ADMIN_EMAIL || 'admin@heimdall.local';
                const adminPassword = import.meta.env.VITE_ADMIN_PASSWORD || 'Admin123!@#';

                // Validate credentials
                if (email !== adminEmail || password !== adminPassword) {
                    throw new Error('Invalid email or password');
                }

                try {
                    // In production, replace with actual API call:
                    // const response = await fetch('/api/auth/login', {
                    //   method: 'POST',
                    //   headers: { 'Content-Type': 'application/json' },
                    //   body: JSON.stringify({ email, password })
                    // })
                    // const data = await response.json()

                    const mockUser: User = {
                        id: '1',
                        email,
                        name: 'Administrator',
                        role: 'admin',
                    };

                    set({
                        user: mockUser,
                        token: `bearer_${Date.now()}_${Math.random().toString(36).substr(2)}`,
                        isAuthenticated: true,
                    });
                } catch (error) {
                    console.error('Login failed:', error);
                    throw error;
                }
            },

            logout: () => {
                set({
                    user: null,
                    token: null,
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
                isAuthenticated: state.isAuthenticated,
            }),
        }
    )
);
