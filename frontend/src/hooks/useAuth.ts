import { useEffect } from 'react';
import { useAuthStore } from '../store';

interface UseAuthReturn {
    isAuthenticated: boolean;
    user: any | null;
    token: string | null;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
}

export const useAuth = (): UseAuthReturn => {
    const { isAuthenticated, user, token, login, logout } = useAuthStore();

    useEffect(() => {
        // Initialize auth on mount
        const storedAuth = localStorage.getItem('auth-store');
        if (storedAuth) {
            try {
                const authData = JSON.parse(storedAuth);
                if (authData.state?.user) {
                    // Already hydrated by Zustand persist
                }
            } catch (error) {
                console.error('Failed to parse auth data:', error);
            }
        }
    }, []);

    return {
        isAuthenticated,
        user,
        token,
        login,
        logout,
    };
};
