import { useEffect } from 'react';
import { useAuthStore } from '../store';

interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'operator' | 'user';
  roles: string[];
  avatar?: string;
}

interface UseAuthReturn {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  // Role hierarchy checks
  isAdmin: boolean;
  isOperator: boolean; // operator OR admin
  isUser: boolean; // all authenticated users
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

  // Implement role hierarchy: admin > operator > user
  const isAdmin = user?.role === 'admin';
  const isOperator = user?.role === 'operator' || user?.role === 'admin';
  const isUser = isAuthenticated; // all authenticated users are 'user' level

  return {
    isAuthenticated,
    user,
    token,
    login,
    logout,
    isAdmin,
    isOperator,
    isUser,
  };
};
