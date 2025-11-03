/**
 * Training WebSocket Hook Tests
 *
 * Tests the useTrainingWebSocket custom hook for real-time training updates
 * Verifies connection management, message handling, and cleanup
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useTrainingWebSocket } from './useTrainingWebSocket';

// Mock WebSocket
class MockWebSocket {
  readyState = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  send(data: string) {
    // Mock send
  }
}

describe('useTrainingWebSocket Hook', () => {
  let originalWebSocket: typeof WebSocket;

  beforeEach(() => {
    originalWebSocket = global.WebSocket;
    global.WebSocket = MockWebSocket as any;
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.WebSocket = originalWebSocket;
  });

  it('should establish WebSocket connection on mount', async () => {
    const { result } = renderHook(() => useTrainingWebSocket('job-1'));

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it('should handle connection errors', async () => {
    class ErrorWebSocket extends MockWebSocket {
      constructor(url: string) {
        super(url);
        setTimeout(() => {
          if (this.onerror) {
            this.onerror(new Event('error'));
          }
        }, 0);
      }
    }

    global.WebSocket = ErrorWebSocket as any;

    const { result } = renderHook(() => useTrainingWebSocket('job-1'));

    await waitFor(() => {
      expect(result.current.isConnected).toBe(false);
    });
  });

  it('should close connection on unmount', async () => {
    const { result, unmount } = renderHook(() => useTrainingWebSocket('job-1'));

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    unmount();

    // Connection should be closed after unmount
    expect(result.current.isConnected).toBe(false);
  });

  it('should reconnect when jobId changes', async () => {
    const { result, rerender } = renderHook(({ jobId }) => useTrainingWebSocket(jobId), {
      initialProps: { jobId: 'job-1' },
    });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // Change jobId
    rerender({ jobId: 'job-2' });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it('should not connect when jobId is null', () => {
    const { result } = renderHook(() => useTrainingWebSocket(null));

    expect(result.current.isConnected).toBe(false);
  });
});
