/**
 * User Profile API Service
 *
 * Handles user profile operations:
 * - Get current user profile
 * - Update current user profile
 * - Get other user profiles
 */

import api from '@/lib/api';

export interface UserProfile {
  user_id: string;
  username: string | null;
  email: string | null;
  roles: string[];
  first_name: string | null;
  last_name: string | null;
  phone: string | null;
  organization: string | null;
  location: string | null;
  bio: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface UserProfileUpdate {
  first_name?: string | null;
  last_name?: string | null;
  phone?: string | null;
  organization?: string | null;
  location?: string | null;
  bio?: string | null;
}

/**
 * Get current user's profile
 */
export const getCurrentUserProfile = async (): Promise<UserProfile> => {
  const response = await api.get<UserProfile>('/v1/users/me');
  return response.data;
};

/**
 * Update current user's profile
 */
export const updateCurrentUserProfile = async (
  profile: UserProfileUpdate
): Promise<UserProfile> => {
  const response = await api.put<UserProfile>('/v1/users/me', profile);
  return response.data;
};

/**
 * Get another user's profile by ID
 */
export const getUserProfile = async (userId: string): Promise<UserProfile> => {
  const response = await api.get<UserProfile>(`/v1/users/${userId}`);
  return response.data;
};

/**
 * Search for users by email or username
 */
export const searchUsers = async (query: string): Promise<UserProfile[]> => {
  const response = await api.get<UserProfile[]>('/v1/users/search', {
    params: { q: query },
  });
  return response.data;
};
