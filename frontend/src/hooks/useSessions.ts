/**
 * Custom hook for session operations
 * Provides convenient access to session store actions
 */

import { useSessionStore } from '@/store/sessionStore';

export function useSessions() {
  const {
    sessions,
    currentSession,
    isLoading,
    error,
    totalSessions,
    currentPage,
    perPage,
    statusFilter,
    approvalFilter,
    fetchSessions,
    fetchSession,
    approveSession,
    rejectSession,
    setStatusFilter,
    setApprovalFilter,
    clearError,
  } = useSessionStore();

  const nextPage = () => {
    fetchSessions({ page: currentPage + 1 });
  };

  const previousPage = () => {
    if (currentPage > 1) {
      fetchSessions({ page: currentPage - 1 });
    }
  };

  const goToPage = (page: number) => {
    fetchSessions({ page });
  };

  const totalPages = Math.ceil(totalSessions / perPage);

  return {
    // State
    sessions,
    currentSession,
    isLoading,
    error,

    // Pagination
    currentPage,
    perPage,
    totalSessions,
    totalPages,

    // Filters
    statusFilter,
    approvalFilter,

    // Actions
    fetchSessions,
    fetchSession,
    approveSession,
    rejectSession,
    setStatusFilter,
    setApprovalFilter,
    clearError,
    nextPage,
    previousPage,
    goToPage,
  };
}
