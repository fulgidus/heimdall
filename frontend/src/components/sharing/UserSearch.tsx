/**
 * UserSearch Component
 * 
 * Search for users to share resources with.
 * Features:
 * - Debounced search input (300ms delay)
 * - Real-time API search by email/username
 * - Display user info with email and username
 * - Click to select user for sharing
 * - Loading and empty states
 */

import React, { useState, useEffect, useCallback } from 'react';
import { searchUsers, type UserProfile } from '../../services/api/users';

interface UserSearchProps {
  onSelectUser: (user: UserProfile) => void;
  excludeUserIds?: string[]; // Users already shared with (don't show in results)
  placeholder?: string;
}

export const UserSearch: React.FC<UserSearchProps> = ({
  onSelectUser,
  excludeUserIds = [],
  placeholder = 'Search by email or username...',
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<UserProfile[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);

  // Debounced search function
  const performSearch = useCallback(async (searchQuery: string) => {
    if (searchQuery.trim().length < 2) {
      setResults([]);
      setShowResults(false);
      return;
    }

    try {
      setIsSearching(true);
      setError(null);
      const users = await searchUsers(searchQuery);
      
      // Filter out excluded users
      const filteredUsers = users.filter(
        user => !excludeUserIds.includes(user.user_id)
      );
      
      setResults(filteredUsers);
      setShowResults(true);
    } catch (err) {
      console.error('User search error:', err);
      setError('Failed to search users. Please try again.');
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [excludeUserIds]);

  // Debounce search with 300ms delay
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      performSearch(query);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [query, performSearch]);

  const handleSelectUser = (user: UserProfile) => {
    onSelectUser(user);
    setQuery('');
    setResults([]);
    setShowResults(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    if (e.target.value.trim().length < 2) {
      setShowResults(false);
    }
  };

  const handleBlur = () => {
    // Delay hiding results to allow click events to fire
    setTimeout(() => setShowResults(false), 200);
  };

  const getUserDisplayName = (user: UserProfile): string => {
    if (user.first_name && user.last_name) {
      return `${user.first_name} ${user.last_name}`;
    }
    if (user.username) {
      return user.username;
    }
    return user.email || 'Unknown User';
  };

  const getUserSecondaryText = (user: UserProfile): string => {
    const parts: string[] = [];
    if (user.email) parts.push(user.email);
    if (user.username && user.email !== user.username) {
      parts.push(`@${user.username}`);
    }
    if (user.organization) parts.push(user.organization);
    return parts.join(' • ');
  };

  return (
    <div className="position-relative">
      {/* Search Input */}
      <div className="input-group">
        <span className="input-group-text">
          <i className="ph ph-magnifying-glass"></i>
        </span>
        <input
          type="text"
          className="form-control"
          placeholder={placeholder}
          value={query}
          onChange={handleInputChange}
          onFocus={() => query.trim().length >= 2 && setShowResults(true)}
          onBlur={handleBlur}
          aria-label="Search for users"
        />
        {isSearching && (
          <span className="input-group-text">
            <div className="spinner-border spinner-border-sm text-primary" role="status">
              <span className="visually-hidden">Searching...</span>
            </div>
          </span>
        )}
      </div>

      {/* Search Results Dropdown */}
      {showResults && (
        <div
          className="card position-absolute w-100 mt-1 shadow-lg"
          style={{ zIndex: 1050, maxHeight: '300px', overflowY: 'auto' }}
        >
          {error && (
            <div className="alert alert-danger m-2 mb-0" role="alert">
              <i className="ph ph-warning-circle me-2"></i>
              {error}
            </div>
          )}

          {!error && results.length === 0 && query.trim().length >= 2 && !isSearching && (
            <div className="card-body text-center text-muted py-3">
              <i className="ph ph-user-circle-minus d-block mb-2" style={{ fontSize: '2rem' }}></i>
              <p className="mb-0">No users found matching "{query}"</p>
              <small>Try searching by email or username</small>
            </div>
          )}

          {!error && results.length > 0 && (
            <div className="list-group list-group-flush">
              {results.map(user => (
                <button
                  key={user.user_id}
                  type="button"
                  className="list-group-item list-group-item-action d-flex align-items-center"
                  onClick={() => handleSelectUser(user)}
                  onMouseDown={(e) => e.preventDefault()} // Prevent blur before click
                >
                  <div className="flex-shrink-0 me-3">
                    <div className="avtar avtar-s bg-light-primary">
                      <i className="ph ph-user"></i>
                    </div>
                  </div>
                  <div className="flex-grow-1">
                    <h6 className="mb-1">{getUserDisplayName(user)}</h6>
                    <p className="text-muted mb-0 small">{getUserSecondaryText(user)}</p>
                    {user.roles && user.roles.length > 0 && (
                      <div className="mt-1">
                        {user.roles.map(role => (
                          <span key={role} className="badge bg-light-secondary me-1">
                            {role}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex-shrink-0">
                    <i className="ph ph-caret-right"></i>
                  </div>
                </button>
              ))}
            </div>
          )}

          {query.trim().length < 2 && !isSearching && (
            <div className="card-body text-center text-muted py-3">
              <i className="ph ph-magnifying-glass d-block mb-2" style={{ fontSize: '2rem' }}></i>
              <p className="mb-0">Type at least 2 characters to search</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UserSearch;
