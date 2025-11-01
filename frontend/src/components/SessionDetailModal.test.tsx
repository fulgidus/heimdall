import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SessionDetailModal } from './SessionDetailModal';

// Mock the useSessions hook
vi.mock('@/hooks/useSessions', () => ({
  useSessions: () => ({
    currentSession: {
      id: 1,
      session_name: 'Test Session',
      status: 'completed',
      approval_status: 'pending',
      created_at: '2025-01-01T00:00:00Z',
      duration_seconds: 120,
      source_name: 'Test Source',
      source_frequency: 145000000,
      source_latitude: 45.0,
      source_longitude: 7.5,
      measurements_count: 42,
      celery_task_id: 'task-123',
      notes: 'Test notes',
    },
    fetchSession: vi.fn(),
    approveSession: vi.fn().mockResolvedValue(undefined),
    rejectSession: vi.fn().mockResolvedValue(undefined),
    isLoading: false,
  }),
}));

describe('SessionDetailModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('does not render when isOpen is false', () => {
    const { container } = render(
      <SessionDetailModal isOpen={false} onClose={vi.fn()} sessionId={1} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders session details when open', () => {
    render(<SessionDetailModal isOpen={true} onClose={vi.fn()} sessionId={1} />);

    expect(screen.getByText('Test Session')).toBeInTheDocument();
    expect(screen.getByText(/PENDING APPROVAL/i)).toBeInTheDocument();
  });

  it('displays session metadata correctly', () => {
    render(<SessionDetailModal isOpen={true} onClose={vi.fn()} sessionId={1} />);

    expect(screen.getByText('Test Source')).toBeInTheDocument();
    expect(screen.getByText(/145\.000 MHz/i)).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('Test notes')).toBeInTheDocument();
  });

  it('shows approval buttons for pending sessions', () => {
    render(<SessionDetailModal isOpen={true} onClose={vi.fn()} sessionId={1} />);

    expect(screen.getByText(/Approve for Training/i)).toBeInTheDocument();
    expect(screen.getByText(/Reject/i)).toBeInTheDocument();
  });

  it('displays spectrograms grid', () => {
    render(<SessionDetailModal isOpen={true} onClose={vi.fn()} sessionId={1} />);

    expect(screen.getByText(/WebSDR Spectrograms/i)).toBeInTheDocument();
    expect(screen.getByText(/7 Receivers/i)).toBeInTheDocument();
  });

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn();
    render(<SessionDetailModal isOpen={true} onClose={onClose} sessionId={1} />);

    const closeButton = screen
      .getAllByRole('button')
      .find(btn => btn.querySelector('svg') !== null && btn.textContent === '');

    if (closeButton) {
      fireEvent.click(closeButton);
      expect(onClose).toHaveBeenCalled();
    }
  });

  it('shows reject confirmation dialog', async () => {
    render(<SessionDetailModal isOpen={true} onClose={vi.fn()} sessionId={1} />);

    const rejectButton = screen.getByText(/Reject/i);
    fireEvent.click(rejectButton);

    await waitFor(() => {
      expect(screen.getByText(/Are you sure you want to reject/i)).toBeInTheDocument();
    });
  });

  it('handles approval action', async () => {
    const onClose = vi.fn();
    render(<SessionDetailModal isOpen={true} onClose={onClose} sessionId={1} />);

    const approveButton = screen.getByText(/Approve for Training/i);
    fireEvent.click(approveButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });
});
